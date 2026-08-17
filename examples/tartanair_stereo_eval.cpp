/**
 * TartanAir V1 Stereo Challenge evaluation runner for Photo-SLAM.
 *
 * Keeps the original Photo-SLAM mapping/training path, while adding:
 *   - direct TartanAir challenge folder loading (image_left / image_right)
 *   - per-frame tracking-state logging with exact source image names
 *   - ORB and Gaussian keyframe manifests
 *   - final-map rendering for every frame present in ORB-SLAM3's final
 *     frame-trajectory bookkeeping (not only keyframes)
 *   - PSNR / SSIM for those final evaluable views
 *   - saved PNG renderings named by the original frame index
 *
 * Failed frames are never filled with GT, interpolation, or last-pose fallback.
 */

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/include/Tracking.h"
#include "include/gaussian_mapper.h"
#include "include/loss_utils.h"
#include "include/tensor_utils.h"
#include "viewer/imgui_viewer.h"

namespace fs = std::filesystem;

struct InputFrame
{
    std::size_t frame_index = 0;
    fs::path left_path;
    fs::path right_path;
    double timestamp = 0.0;
};

struct TrackingRecord
{
    std::size_t frame_index = 0;
    double timestamp = 0.0;
    std::string left_path;
    std::string right_path;
    int tracking_state = ORB_SLAM3::Tracking::SYSTEM_NOT_READY;
    bool pose_set = false;
    bool strict_success = false;
};

static std::string csvQuote(const std::string &s)
{
    std::string out = "\"";
    for (char c : s)
    {
        if (c == '"') out += '"';
        out += c;
    }
    out += "\"";
    return out;
}

static std::string trackingStateName(int state)
{
    switch (state)
    {
        case ORB_SLAM3::Tracking::SYSTEM_NOT_READY: return "SYSTEM_NOT_READY";
        case ORB_SLAM3::Tracking::NO_IMAGES_YET: return "NO_IMAGES_YET";
        case ORB_SLAM3::Tracking::NOT_INITIALIZED: return "NOT_INITIALIZED";
        case ORB_SLAM3::Tracking::OK: return "OK";
        case ORB_SLAM3::Tracking::RECENTLY_LOST: return "RECENTLY_LOST";
        case ORB_SLAM3::Tracking::LOST: return "LOST";
        case ORB_SLAM3::Tracking::OK_KLT: return "OK_KLT";
        default: return "UNKNOWN";
    }
}

static std::size_t frameIndexFromPath(const fs::path &path)
{
    const std::string name = path.filename().string();
    const std::size_t pos = name.find('_');
    const std::string token = (pos == std::string::npos) ? path.stem().string() : name.substr(0, pos);
    return static_cast<std::size_t>(std::stoull(token));
}

static long long timestampKey(double timestamp)
{
    return static_cast<long long>(std::llround(timestamp * 1e9));
}

static std::vector<InputFrame> loadTartanAirStereo(const fs::path &sequence_root, double fps)
{
    const fs::path left_dir = sequence_root / "image_left";
    const fs::path right_dir = sequence_root / "image_right";

    if (!fs::is_directory(left_dir))
        throw std::runtime_error("Missing image_left directory: " + left_dir.string());
    if (!fs::is_directory(right_dir))
        throw std::runtime_error("Missing image_right directory: " + right_dir.string());

    std::vector<fs::path> left_images;
    for (const auto &entry : fs::directory_iterator(left_dir))
    {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() >= 9 && name.rfind("_left.png") == name.size() - 9)
            left_images.push_back(fs::absolute(entry.path()));
    }
    std::sort(left_images.begin(), left_images.end());

    std::vector<InputFrame> frames;
    frames.reserve(left_images.size());
    for (const fs::path &left : left_images)
    {
        const std::size_t frame_index = frameIndexFromPath(left);
        std::ostringstream right_name;
        right_name << std::setw(6) << std::setfill('0') << frame_index << "_right.png";
        const fs::path right = fs::absolute(right_dir / right_name.str());
        if (!fs::exists(right))
            throw std::runtime_error("Missing right image for " + left.string() + ": " + right.string());

        InputFrame frame;
        frame.frame_index = frame_index;
        frame.left_path = left;
        frame.right_path = right;
        frame.timestamp = static_cast<double>(frame_index) / fps;
        frames.push_back(frame);
    }

    if (frames.empty())
        throw std::runtime_error("No *_left.png images found in " + left_dir.string());

    return frames;
}

static void saveTrackingRecords(const std::vector<TrackingRecord> &records, const fs::path &output_dir)
{
    std::ofstream csv(output_dir / "frame_tracking_status.csv");
    csv << "frame_index,timestamp,left_image,right_image,tracking_state,tracking_state_name,pose_set,strict_success\n";

    std::ofstream strict_txt(output_dir / "strict_success_frames.txt");

    for (const auto &r : records)
    {
        csv << r.frame_index << ','
            << std::fixed << std::setprecision(9) << r.timestamp << ','
            << csvQuote(r.left_path) << ','
            << csvQuote(r.right_path) << ','
            << r.tracking_state << ','
            << trackingStateName(r.tracking_state) << ','
            << (r.pose_set ? 1 : 0) << ','
            << (r.strict_success ? 1 : 0) << '\n';

        if (r.strict_success)
            strict_txt << r.frame_index << ' ' << fs::path(r.left_path).filename().string() << '\n';
    }
}

/**
 * Recover final frame poses in the same map coordinate system used by GaussianMapper.
 *
 * This deliberately follows ORB-SLAM3's stored relative-frame-pose/reference-keyframe
 * bookkeeping, but DOES NOT apply SaveTrajectoryTUM's global first-keyframe normalization.
 * GaussianMapper stores keyframe Tcw directly, so omitting that normalization keeps the
 * recovered frame Tcw compatible with the final Gaussian map coordinate system.
 *
 * Frames for which ORB-SLAM3 did not retain a trajectory entry are absent. No fallback
 * pose is synthesized.
 */
static std::map<long long, Sophus::SE3f> collectFinalFramePosesInMapFrame(ORB_SLAM3::Tracking *tracker)
{
    std::map<long long, Sophus::SE3f> result;

    auto lRit = tracker->mlpReferences.begin();
    auto lT = tracker->mlFrameTimes.begin();
    auto lbL = tracker->mlbLost.begin();

    for (auto lit = tracker->mlRelativeFramePoses.begin(),
              lend = tracker->mlRelativeFramePoses.end();
         lit != lend && lRit != tracker->mlpReferences.end() &&
         lT != tracker->mlFrameTimes.end() && lbL != tracker->mlbLost.end();
         ++lit, ++lRit, ++lT, ++lbL)
    {
        if (*lbL)
            continue;

        ORB_SLAM3::KeyFrame *pKF = *lRit;
        if (!pKF)
            continue;

        Sophus::SE3f Trw;
        while (pKF && pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }
        if (!pKF)
            continue;

        Trw = Trw * pKF->GetPose();
        const Sophus::SE3f Tcw = (*lit) * Trw;
        result[timestampKey(*lT)] = Tcw;
    }

    return result;
}

static void saveOrbKeyframeManifest(std::shared_ptr<ORB_SLAM3::System> pSLAM, const fs::path &output_dir)
{
    std::vector<ORB_SLAM3::KeyFrame*> keyframes = pSLAM->getAtlas()->GetAllKeyFrames();
    std::sort(keyframes.begin(), keyframes.end(), [](const auto *a, const auto *b) {
        if (a->mnFrameId != b->mnFrameId) return a->mnFrameId < b->mnFrameId;
        return a->mnId < b->mnId;
    });

    std::ofstream csv(output_dir / "orb_keyframes.csv");
    csv << "orb_keyframe_id,orb_frame_id,timestamp,left_image,is_bad,dataset_id\n";
    std::ofstream txt(output_dir / "orb_keyframe_frames.txt");

    for (auto *kf : keyframes)
    {
        csv << kf->mnId << ','
            << kf->mnFrameId << ','
            << std::fixed << std::setprecision(9) << kf->mTimeStamp << ','
            << csvQuote(kf->mNameFile) << ','
            << (kf->isBad() ? 1 : 0) << ','
            << kf->mnDataset << '\n';

        txt << kf->mnFrameId << ' ' << fs::path(kf->mNameFile).filename().string()
            << " kf_id=" << kf->mnId << " bad=" << (kf->isBad() ? 1 : 0) << '\n';
    }
}

static std::unordered_set<std::string> saveGaussianKeyframeManifest(
    std::shared_ptr<GaussianMapper> mapper,
    const fs::path &output_dir)
{
    std::unordered_set<std::string> image_set;
    std::ofstream csv(output_dir / "gaussian_keyframes.csv");
    csv << "gaussian_keyframe_id,source_frame_index,source_left_image,creation_iteration,final_render_file,final_gt_file\n";
    std::ofstream txt(output_dir / "gaussian_keyframe_frames.txt");

    const int final_iter = mapper->getIteration();
    const fs::path final_dir = output_dir / (std::to_string(final_iter) + "_shutdown");

    for (const auto &item : mapper->scene_->keyframes())
    {
        const auto &pkf = item.second;
        const fs::path source(pkf->img_filename_);
        std::size_t source_index = 0;
        try { source_index = frameIndexFromPath(source); }
        catch (...) { source_index = static_cast<std::size_t>(-1); }

        const fs::path render_file = final_dir / "image" /
            (std::to_string(final_iter) + "_" + std::to_string(pkf->fid_) + ".jpg");
        const fs::path gt_file = final_dir / "image_gt" /
            (std::to_string(final_iter) + "_" + std::to_string(pkf->fid_) + "_gt.jpg");

        csv << pkf->fid_ << ','
            << source_index << ','
            << csvQuote(pkf->img_filename_) << ','
            << pkf->creation_iter_ << ','
            << csvQuote(render_file.string()) << ','
            << csvQuote(gt_file.string()) << '\n';

        txt << source_index << ' ' << source.filename().string()
            << " gaussian_kf_id=" << pkf->fid_ << '\n';
        image_set.insert(pkf->img_filename_);
    }
    return image_set;
}

static cv::Mat loadRgbFloat(const fs::path &path)
{
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (bgr.empty())
        throw std::runtime_error("Failed to read image: " + path.string());

    cv::Mat rgb;
    if (bgr.channels() == 3)
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    else if (bgr.channels() == 4)
        cv::cvtColor(bgr, rgb, cv::COLOR_BGRA2RGB);
    else if (bgr.channels() == 1)
        cv::cvtColor(bgr, rgb, cv::COLOR_GRAY2RGB);
    else
        throw std::runtime_error("Unsupported channel count in: " + path.string());

    if (rgb.type() == CV_8UC3)
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    else if (rgb.type() == CV_16UC3)
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 65535.0);
    else
        rgb.convertTo(rgb, CV_32FC3);

    return rgb;
}

static void saveRgbFloatPng(const cv::Mat &rgb_float, const fs::path &path)
{
    cv::Mat rgb_u8, bgr_u8;
    rgb_float.convertTo(rgb_u8, CV_8UC3, 255.0);
    cv::cvtColor(rgb_u8, bgr_u8, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(path.string(), bgr_u8))
        throw std::runtime_error("Failed to write rendered image: " + path.string());
}

static void evaluateFinalTrackedViews(
    const std::vector<InputFrame> &frames,
    const std::vector<TrackingRecord> &tracking_records,
    const std::map<long long, Sophus::SE3f> &final_poses,
    const std::unordered_set<std::string> &gaussian_keyframe_images,
    std::shared_ptr<GaussianMapper> mapper,
    torch::DeviceType device_type,
    const fs::path &output_dir)
{
    const fs::path eval_dir = output_dir / "final_tracked_view_eval";
    const fs::path render_dir = eval_dir / "rendered";
    fs::create_directories(render_dir);

    std::ofstream metrics(eval_dir / "metrics.csv");
    metrics << "frame_index,timestamp,left_image,live_tracking_state,live_tracking_state_name,"
               "live_pose_set,strict_success,is_gaussian_keyframe,psnr,ssim,rendered_image\n";

    std::ofstream evaluable_txt(eval_dir / "final_evaluable_frames.txt");

    std::map<std::size_t, TrackingRecord> tracking_by_index;
    for (const auto &r : tracking_records)
        tracking_by_index[r.frame_index] = r;

    double sum_psnr = 0.0;
    double sum_ssim = 0.0;
    std::size_t evaluated = 0;
    std::size_t evaluated_keyframes = 0;

    torch::NoGradGuard no_grad;

    for (const auto &frame : frames)
    {
        const auto pose_it = final_poses.find(timestampKey(frame.timestamp));
        if (pose_it == final_poses.end())
            continue;

        cv::Mat gt_rgb = loadRgbFloat(frame.left_path);
        cv::Mat rendered = mapper->renderFromPose(
            pose_it->second, gt_rgb.cols, gt_rgb.rows, true);
        if (rendered.empty())
            continue;
        if (rendered.size() != gt_rgb.size())
            cv::resize(gt_rgb, gt_rgb, rendered.size(), 0.0, 0.0, cv::INTER_LINEAR);

        torch::Tensor rendered_tensor = tensor_utils::cvMat2TorchTensor_Float32(rendered, device_type);
        torch::Tensor gt_tensor = tensor_utils::cvMat2TorchTensor_Float32(gt_rgb, device_type);

        const float psnr = loss_utils::psnr(rendered_tensor, gt_tensor).item().toFloat();
        const float ssim = loss_utils::ssim(rendered_tensor, gt_tensor, device_type).item().toFloat();

        std::ostringstream filename;
        filename << std::setw(6) << std::setfill('0') << frame.frame_index << "_left.png";
        const fs::path rendered_path = render_dir / filename.str();
        saveRgbFloatPng(rendered, rendered_path);

        TrackingRecord tr;
        auto tr_it = tracking_by_index.find(frame.frame_index);
        if (tr_it != tracking_by_index.end()) tr = tr_it->second;

        const bool is_gkf = gaussian_keyframe_images.count(frame.left_path.string()) > 0;
        if (is_gkf) ++evaluated_keyframes;

        metrics << frame.frame_index << ','
                << std::fixed << std::setprecision(9) << frame.timestamp << ','
                << csvQuote(frame.left_path.string()) << ','
                << tr.tracking_state << ','
                << trackingStateName(tr.tracking_state) << ','
                << (tr.pose_set ? 1 : 0) << ','
                << (tr.strict_success ? 1 : 0) << ','
                << (is_gkf ? 1 : 0) << ','
                << std::setprecision(10) << psnr << ','
                << std::setprecision(10) << ssim << ','
                << csvQuote(rendered_path.string()) << '\n';

        evaluable_txt << frame.frame_index << ' ' << frame.left_path.filename().string()
                      << " psnr=" << psnr << " ssim=" << ssim
                      << " keyframe=" << (is_gkf ? 1 : 0) << '\n';

        if (std::isfinite(psnr) && std::isfinite(ssim))
        {
            sum_psnr += psnr;
            sum_ssim += ssim;
            ++evaluated;
        }
    }

    std::ofstream summary(eval_dir / "summary.txt");
    summary << "input_frames " << frames.size() << '\n';
    summary << "final_evaluable_frames " << evaluated << '\n';
    summary << "rendering_coverage "
            << (frames.empty() ? 0.0 : static_cast<double>(evaluated) / static_cast<double>(frames.size()))
            << '\n';
    summary << "evaluable_gaussian_keyframes " << evaluated_keyframes << '\n';
    if (evaluated > 0)
    {
        summary << "mean_psnr " << (sum_psnr / evaluated) << '\n';
        summary << "mean_ssim " << (sum_ssim / evaluated) << '\n';
    }

    std::cout << "[Final tracked-view evaluation] " << evaluated << "/" << frames.size()
              << " views rendered" << std::endl;
    if (evaluated > 0)
    {
        std::cout << "  mean PSNR: " << (sum_psnr / evaluated) << " dB" << std::endl;
        std::cout << "  mean SSIM: " << (sum_ssim / evaluated) << std::endl;
    }
    std::cout << "  renderings: " << render_dir << std::endl;
}

static void saveGpuPeakMemoryUsage(const fs::path &path)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);
    auto reserved_bytes = mem_stats.reserved_bytes[0];
    auto alloc_bytes = mem_stats.allocated_bytes[0];

    std::ofstream out(path);
    out << "Peak reserved (MB): " << reserved_bytes.peak / (1024.0 * 1024.0) << '\n';
    out << "Peak allocated (MB): " << alloc_bytes.peak / (1024.0 * 1024.0) << '\n';
}

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " path_to_vocabulary"
            << " path_to_ORB_SLAM3_settings"
            << " path_to_gaussian_mapping_settings"
            << " path_to_TartanAir_sequence_root"
            << " path_to_output_directory"
            << " [viewer] [--fps=10.0]"
            << std::endl;
        return 1;
    }

    bool use_viewer = false;
    double fps = 10.0;
    for (int i = 6; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "viewer") use_viewer = true;
        else if (arg.rfind("--fps=", 0) == 0) fps = std::stod(arg.substr(6));
        else
        {
            std::cerr << "Unknown optional argument: " << arg << std::endl;
            return 1;
        }
    }
    if (!(fps > 0.0))
    {
        std::cerr << "fps must be > 0" << std::endl;
        return 1;
    }

    const fs::path sequence_root = fs::absolute(fs::path(argv[4]));
    const fs::path output_dir = fs::absolute(fs::path(argv[5]));
    fs::create_directories(output_dir);

    std::vector<InputFrame> frames;
    try
    {
        frames = loadTartanAirStereo(sequence_root, fps);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Dataset error] " << e.what() << std::endl;
        return 1;
    }

    std::cout << "TartanAir stereo sequence: " << sequence_root << std::endl;
    std::cout << "Frames: " << frames.size() << std::endl;
    std::cout << "Synthetic timestamp rate: " << fps << " Hz" << std::endl;

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    if (device_type == torch::kCUDA)
        std::cout << "CUDA available! Training on GPU." << std::endl;
    else
        std::cout << "CUDA unavailable. Training on CPU." << std::endl;

    auto pSLAM = std::make_shared<ORB_SLAM3::System>(
        argv[1], argv[2], ORB_SLAM3::System::STEREO);

    const float imageScale = pSLAM->GetImageScale();

    auto pGausMapper = std::make_shared<GaussianMapper>(
        pSLAM, fs::path(argv[3]), output_dir, 0, device_type);
    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());

    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    std::vector<TrackingRecord> tracking_records;
    tracking_records.reserve(frames.size());

    for (const auto &frame : frames)
    {
        if (pSLAM->isShutDown()) break;

        cv::Mat imLeft = cv::imread(frame.left_path.string(), cv::IMREAD_UNCHANGED);
        cv::Mat imRight = cv::imread(frame.right_path.string(), cv::IMREAD_UNCHANGED);
        if (imLeft.empty() || imRight.empty())
        {
            std::cerr << "Failed to read stereo pair at frame " << frame.frame_index << std::endl;
            return 1;
        }

        if (imageScale != 1.f)
        {
            const int width = static_cast<int>(imLeft.cols * imageScale);
            const int height = static_cast<int>(imLeft.rows * imageScale);
            cv::resize(imLeft, imLeft, cv::Size(width, height));
            cv::resize(imRight, imRight, cv::Size(width, height));
        }

        pSLAM->TrackStereo(
            imLeft,
            imRight,
            frame.timestamp,
            std::vector<ORB_SLAM3::IMU::Point>(),
            frame.left_path.string());

        TrackingRecord record;
        record.frame_index = frame.frame_index;
        record.timestamp = frame.timestamp;
        record.left_path = frame.left_path.string();
        record.right_path = frame.right_path.string();
        record.tracking_state = pSLAM->GetTrackingState();
        record.pose_set = pSLAM->getTracker()->mCurrentFrame.isSet();
        record.strict_success = record.pose_set &&
            (record.tracking_state == ORB_SLAM3::Tracking::OK ||
             record.tracking_state == ORB_SLAM3::Tracking::OK_KLT);
        tracking_records.push_back(record);
    }

    // Finish ORB-SLAM3 first. GaussianMapper then performs its original tail optimization
    // and final keyframe rendering before its thread exits.
    pSLAM->Shutdown();
    training_thd.join();
    if (use_viewer) viewer_thd.join();

    saveTrackingRecords(tracking_records, output_dir);
    saveOrbKeyframeManifest(pSLAM, output_dir);
    const auto gaussian_keyframe_images = saveGaussianKeyframeManifest(pGausMapper, output_dir);

    if (pSLAM->GetNumKeyframes() > 0)
    {
        pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
        pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
        pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
        pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
        pSLAM->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());
    }

    const auto final_poses = collectFinalFramePosesInMapFrame(pSLAM->getTracker());

    if (!pGausMapper->scene_->keyframes().empty())
    {
        try
        {
            evaluateFinalTrackedViews(
                frames,
                tracking_records,
                final_poses,
                gaussian_keyframe_images,
                pGausMapper,
                device_type,
                output_dir);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Final rendering evaluation error] " << e.what() << std::endl;
            return 2;
        }
    }
    else
    {
        std::cerr << "[Evaluation] Gaussian map was never initialized; no final-view rendering metrics." << std::endl;
    }

    if (device_type == torch::kCUDA)
        saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");

    std::size_t strict_success = 0;
    for (const auto &r : tracking_records) strict_success += r.strict_success ? 1 : 0;

    std::ofstream summary(output_dir / "tracking_summary.txt");
    summary << "input_frames " << frames.size() << '\n';
    summary << "processed_frames " << tracking_records.size() << '\n';
    summary << "strict_success_frames " << strict_success << '\n';
    summary << "strict_success_rate "
            << (frames.empty() ? 0.0 : static_cast<double>(strict_success) / static_cast<double>(frames.size()))
            << '\n';
    summary << "final_trajectory_pose_entries " << final_poses.size() << '\n';
    summary << "orb_keyframes " << pSLAM->GetNumKeyframes() << '\n';
    summary << "gaussian_keyframes " << pGausMapper->scene_->keyframes().size() << '\n';
    summary << "final_gaussian_iteration " << pGausMapper->getIteration() << '\n';

    std::cout << "Evaluation metadata saved to: " << output_dir << std::endl;
    std::cout << "Strict tracking success: " << strict_success << "/" << frames.size() << std::endl;
    std::cout << "Final trajectory pose entries: " << final_poses.size() << std::endl;
    std::cout << "Gaussian keyframes: " << pGausMapper->scene_->keyframes().size() << std::endl;

    return 0;
}

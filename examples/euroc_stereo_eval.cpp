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

namespace fs = std::filesystem;

struct InputFrame
{
    std::size_t frame_index = 0;   // index AFTER stride; used by train/test protocol
    std::size_t source_index = 0;  // index in original EuRoC camera stream
    long long timestamp_ns = 0;
    double timestamp = 0.0;
    fs::path left_path;
    fs::path right_path;
};

struct TrackingRecord
{
    std::size_t frame_index = 0;
    std::size_t source_index = 0;
    long long timestamp_ns = 0;
    double timestamp = 0.0;
    int tracking_state = ORB_SLAM3::Tracking::SYSTEM_NOT_READY;
    bool pose_set = false;
    bool strict_success = false;
    bool is_test = false;
};

static long long timestampKey(double timestamp)
{
    return static_cast<long long>(std::llround(timestamp * 1e9));
}

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

static std::vector<InputFrame> loadEuRoC(
    const fs::path &sequence_root,
    const fs::path &timestamps_path,
    int stride)
{
    if (stride <= 0) throw std::runtime_error("stride must be > 0");
    std::ifstream in(timestamps_path);
    if (!in.is_open())
        throw std::runtime_error("Cannot open EuRoC timestamps: " + timestamps_path.string());

    const fs::path cam0 = sequence_root / "mav0" / "cam0" / "data";
    const fs::path cam1 = sequence_root / "mav0" / "cam1" / "data";
    std::vector<InputFrame> frames;
    std::string line;
    std::size_t source_index = 0;
    while (std::getline(in, line))
    {
        if (line.empty()) continue;
        std::istringstream iss(line);
        long long timestamp_ns = 0;
        if (!(iss >> timestamp_ns))
            throw std::runtime_error("Invalid EuRoC timestamp line: " + line);

        if (source_index % static_cast<std::size_t>(stride) == 0)
        {
            InputFrame f;
            f.frame_index = frames.size();
            f.source_index = source_index;
            f.timestamp_ns = timestamp_ns;
            f.timestamp = static_cast<double>(timestamp_ns) / 1e9;
            const std::string name = std::to_string(timestamp_ns) + ".png";
            f.left_path = fs::absolute(cam0 / name);
            f.right_path = fs::absolute(cam1 / name);
            if (!fs::exists(f.left_path) || !fs::exists(f.right_path))
                throw std::runtime_error("Missing EuRoC stereo pair for timestamp " + std::to_string(timestamp_ns));
            frames.push_back(std::move(f));
        }
        ++source_index;
    }
    if (frames.empty()) throw std::runtime_error("No EuRoC frames selected");
    return frames;
}

static std::map<long long, Sophus::SE3f> collectFinalFramePosesInMapFrame(ORB_SLAM3::Tracking *tracker)
{
    std::map<long long, Sophus::SE3f> result;
    auto lRit = tracker->mlpReferences.begin();
    auto lT = tracker->mlFrameTimes.begin();
    auto lbL = tracker->mlbLost.begin();
    for (auto lit = tracker->mlRelativeFramePoses.begin(), lend = tracker->mlRelativeFramePoses.end();
         lit != lend && lRit != tracker->mlpReferences.end() &&
         lT != tracker->mlFrameTimes.end() && lbL != tracker->mlbLost.end();
         ++lit, ++lRit, ++lT, ++lbL)
    {
        if (*lbL) continue;
        ORB_SLAM3::KeyFrame *pKF = *lRit;
        if (!pKF) continue;
        Sophus::SE3f Trw;
        while (pKF && pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }
        if (!pKF) continue;
        Trw = Trw * pKF->GetPose();
        result[timestampKey(*lT)] = (*lit) * Trw;
    }
    return result;
}

static cv::Mat loadRgbFloat(const fs::path &path)
{
    cv::Mat src = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (src.empty()) throw std::runtime_error("Failed to read image: " + path.string());
    cv::Mat rgb;
    if (src.channels() == 1) cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB);
    else if (src.channels() == 3) cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
    else if (src.channels() == 4) cv::cvtColor(src, rgb, cv::COLOR_BGRA2RGB);
    else throw std::runtime_error("Unsupported image channels: " + path.string());
    if (rgb.type() == CV_8UC3) rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    else if (rgb.type() == CV_16UC3) rgb.convertTo(rgb, CV_32FC3, 1.0 / 65535.0);
    else rgb.convertTo(rgb, CV_32FC3);
    return rgb;
}

static void saveRgbFloatPng(const cv::Mat &rgb_float, const fs::path &path)
{
    cv::Mat rgb_u8, bgr_u8;
    rgb_float.convertTo(rgb_u8, CV_8UC3, 255.0);
    cv::cvtColor(rgb_u8, bgr_u8, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(path.string(), bgr_u8))
        throw std::runtime_error("Failed to write image: " + path.string());
}

static std::unordered_set<std::string> saveGaussianKeyframeManifest(
    std::shared_ptr<GaussianMapper> mapper, const fs::path &output_dir)
{
    std::unordered_set<std::string> images;
    std::ofstream out(output_dir / "gaussian_keyframes.csv");
    out << "gaussian_keyframe_id,source_left_image,creation_iteration\n";
    for (const auto &item : mapper->scene_->keyframes())
    {
        const auto &pkf = item.second;
        out << pkf->fid_ << ',' << csvQuote(pkf->img_filename_) << ',' << pkf->creation_iter_ << '\n';
        images.insert(pkf->img_filename_);
    }
    return images;
}

static void evaluateViews(
    const std::vector<InputFrame> &frames,
    const std::vector<TrackingRecord> &tracking_records,
    const std::map<long long, Sophus::SE3f> &poses,
    const std::unordered_set<std::string> &gaussian_keyframe_images,
    std::shared_ptr<GaussianMapper> mapper,
    torch::DeviceType device_type,
    const fs::path &output_dir,
    const std::string &subdir)
{
    const fs::path eval_dir = output_dir / subdir;
    const fs::path render_dir = eval_dir / "rendered";
    fs::create_directories(render_dir);

    std::map<std::size_t, TrackingRecord> tr_by_idx;
    for (const auto &r : tracking_records) tr_by_idx[r.frame_index] = r;

    std::ofstream metrics(eval_dir / "metrics.csv");
    metrics << "frame_index,source_index,timestamp_ns,timestamp,left_image,tracking_state,tracking_state_name,pose_set,strict_success,is_gaussian_keyframe,psnr,ssim,rendered_image\n";

    std::size_t evaluated = 0;
    double sum_psnr = 0.0, sum_ssim = 0.0;
    torch::NoGradGuard no_grad;
    for (const auto &frame : frames)
    {
        auto pose_it = poses.find(frame.timestamp_ns);
        if (pose_it == poses.end()) continue;
        cv::Mat gt = loadRgbFloat(frame.left_path);
        cv::Mat rendered = mapper->renderFromPose(pose_it->second, gt.cols, gt.rows, true);
        if (rendered.empty()) continue;
        if (rendered.size() != gt.size()) cv::resize(gt, gt, rendered.size(), 0, 0, cv::INTER_LINEAR);

        auto pred_t = tensor_utils::cvMat2TorchTensor_Float32(rendered, device_type);
        auto gt_t = tensor_utils::cvMat2TorchTensor_Float32(gt, device_type);
        const float psnr = loss_utils::psnr(pred_t, gt_t).item().toFloat();
        const float ssim = loss_utils::ssim(pred_t, gt_t, device_type).item().toFloat();

        std::ostringstream fn;
        fn << std::setw(6) << std::setfill('0') << frame.frame_index << ".png";
        const fs::path render_path = render_dir / fn.str();
        saveRgbFloatPng(rendered, render_path);

        TrackingRecord tr;
        auto tr_it = tr_by_idx.find(frame.frame_index);
        if (tr_it != tr_by_idx.end()) tr = tr_it->second;
        const bool is_gkf = gaussian_keyframe_images.count(frame.left_path.string()) > 0;
        metrics << frame.frame_index << ',' << frame.source_index << ',' << frame.timestamp_ns << ','
                << std::fixed << std::setprecision(9) << frame.timestamp << ','
                << csvQuote(frame.left_path.string()) << ',' << tr.tracking_state << ','
                << trackingStateName(tr.tracking_state) << ',' << (tr.pose_set ? 1 : 0) << ','
                << (tr.strict_success ? 1 : 0) << ',' << (is_gkf ? 1 : 0) << ','
                << std::setprecision(10) << psnr << ',' << ssim << ','
                << csvQuote(render_path.string()) << '\n';
        if (std::isfinite(psnr) && std::isfinite(ssim))
        {
            sum_psnr += psnr; sum_ssim += ssim; ++evaluated;
        }
    }

    std::ofstream summary(eval_dir / "summary.txt");
    summary << "input_frames " << frames.size() << '\n';
    summary << "final_evaluable_frames " << evaluated << '\n';
    if (evaluated)
    {
        summary << "mean_psnr " << sum_psnr / evaluated << '\n';
        summary << "mean_ssim " << sum_ssim / evaluated << '\n';
    }
    std::cout << '[' << subdir << "] " << evaluated << '/' << frames.size() << " rendered" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0]
                  << " vocabulary orb_yaml gaussian_yaml sequence_root timestamps_txt output_dir"
                  << " [--stride=5] [--test-every=5] [--test-offset=4] [--skip-final-eval]\n";
        return 1;
    }

    int stride = 5, test_every = 5, test_offset = 4;
    bool skip_final_eval = false;
    for (int i = 7; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg.rfind("--stride=", 0) == 0) stride = std::stoi(arg.substr(9));
        else if (arg.rfind("--test-every=", 0) == 0) test_every = std::stoi(arg.substr(13));
        else if (arg.rfind("--test-offset=", 0) == 0) test_offset = std::stoi(arg.substr(14));
        else if (arg == "--skip-final-eval") skip_final_eval = true;
        else { std::cerr << "Unknown argument: " << arg << std::endl; return 1; }
    }
    if (stride <= 0 || test_every <= 0 || test_offset < 0 || test_offset >= test_every)
        throw std::runtime_error("Invalid stride/test split parameters");

    const fs::path sequence_root = fs::absolute(fs::path(argv[4]));
    const fs::path timestamps_path = fs::absolute(fs::path(argv[5]));
    const fs::path output_dir = fs::absolute(fs::path(argv[6]));
    fs::create_directories(output_dir);

    std::vector<InputFrame> frames = loadEuRoC(sequence_root, timestamps_path, stride);
    std::ofstream selected(output_dir / "selected_frames.csv");
    selected << "frame_index,source_index,timestamp_ns,timestamp,left_image,right_image,split\n";
    std::ofstream train_ids(output_dir / "train_frame_ids.txt");
    std::ofstream test_ids(output_dir / "test_frame_ids.txt");
    std::size_t train_count = 0, test_count = 0;
    for (const auto &f : frames)
    {
        const bool is_test = static_cast<int>(f.frame_index % static_cast<std::size_t>(test_every)) == test_offset;
        (is_test ? test_ids : train_ids) << f.frame_index << '\n';
        if (is_test) ++test_count; else ++train_count;
        selected << f.frame_index << ',' << f.source_index << ',' << f.timestamp_ns << ','
                 << std::fixed << std::setprecision(9) << f.timestamp << ','
                 << csvQuote(f.left_path.string()) << ',' << csvQuote(f.right_path.string()) << ','
                 << (is_test ? "test" : "train") << '\n';
    }

    std::cout << "EuRoC sequence: " << sequence_root << '\n'
              << "Original timestamps: " << timestamps_path << '\n'
              << "Stride: " << stride << " -> selected frames: " << frames.size() << '\n'
              << "Held-out split after stride: train=" << train_count << ", test=" << test_count << std::endl;

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    auto pSLAM = std::make_shared<ORB_SLAM3::System>(argv[1], argv[2], ORB_SLAM3::System::STEREO);
    const float imageScale = pSLAM->GetImageScale();
    auto pGausMapper = std::make_shared<GaussianMapper>(pSLAM, fs::path(argv[3]), output_dir, 0, device_type);
    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());

    std::vector<TrackingRecord> tracking_records;
    tracking_records.reserve(frames.size());
    const auto stream_start = std::chrono::steady_clock::now();
    for (const auto &frame : frames)
    {
        if (pSLAM->isShutDown()) break;
        cv::Mat left = cv::imread(frame.left_path.string(), cv::IMREAD_UNCHANGED);
        cv::Mat right = cv::imread(frame.right_path.string(), cv::IMREAD_UNCHANGED);
        if (left.empty() || right.empty()) throw std::runtime_error("Failed to read EuRoC stereo pair");
        if (imageScale != 1.f)
        {
            cv::resize(left, left, cv::Size(static_cast<int>(left.cols * imageScale), static_cast<int>(left.rows * imageScale)));
            cv::resize(right, right, cv::Size(static_cast<int>(right.cols * imageScale), static_cast<int>(right.rows * imageScale)));
        }

        const bool is_test = static_cast<int>(frame.frame_index % static_cast<std::size_t>(test_every)) == test_offset;
        pSLAM->getTracker()->SuppressKeyFrameInsertion(is_test);
        pSLAM->TrackStereo(left, right, frame.timestamp, std::vector<ORB_SLAM3::IMU::Point>(), frame.left_path.string());

        TrackingRecord r;
        r.frame_index = frame.frame_index;
        r.source_index = frame.source_index;
        r.timestamp_ns = frame.timestamp_ns;
        r.timestamp = frame.timestamp;
        r.tracking_state = pSLAM->GetTrackingState();
        r.pose_set = pSLAM->getTracker()->mCurrentFrame.isSet();
        r.strict_success = r.pose_set && (r.tracking_state == ORB_SLAM3::Tracking::OK || r.tracking_state == ORB_SLAM3::Tracking::OK_KLT);
        r.is_test = is_test;
        tracking_records.push_back(r);
    }
    pSLAM->getTracker()->SuppressKeyFrameInsertion(false);
    const auto stream_end = std::chrono::steady_clock::now();

    pSLAM->Shutdown();
    training_thd.join();
    const auto mapper_done = std::chrono::steady_clock::now();

    long long online_iteration = -1, online_steady_clock_ns = -1, online_sh_degree = -1;
    {
        std::ifstream meta(output_dir / "online_checkpoint_metadata.txt");
        std::string key; long long value;
        while (meta >> key >> value)
        {
            if (key == "online_iteration") online_iteration = value;
            else if (key == "online_steady_clock_ns") online_steady_clock_ns = value;
            else if (key == "online_sh_degree") online_sh_degree = value;
        }
    }
    const long long stream_start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stream_start.time_since_epoch()).count();
    const double online_sec = online_steady_clock_ns >= stream_start_ns ? static_cast<double>(online_steady_clock_ns - stream_start_ns) / 1e9 : -1.0;
    const double stream_sec = std::chrono::duration_cast<std::chrono::duration<double>>(stream_end - stream_start).count();
    const double pipeline_sec = std::chrono::duration_cast<std::chrono::duration<double>>(mapper_done - stream_start).count();
    {
        std::ofstream t(output_dir / "timing_summary.txt");
        t << std::fixed << std::setprecision(9);
        t << "input_frames " << frames.size() << '\n';
        t << "processed_frames " << tracking_records.size() << '\n';
        t << "stride " << stride << '\n';
        t << "train_frames " << train_count << '\n';
        t << "test_frames " << test_count << '\n';
        t << "stream_wall_sec " << stream_sec << '\n';
        t << "online_pipeline_wall_sec " << online_sec << '\n';
        t << "online_pipeline_fps " << (online_sec > 0 ? tracking_records.size() / online_sec : 0.0) << '\n';
        t << "pipeline_until_gaussian_mapper_exit_wall_sec " << pipeline_sec << '\n';
        t << "note_no_realtime_playback_sleep 1\n";
    }

    std::ofstream trcsv(output_dir / "frame_tracking_status.csv");
    trcsv << "frame_index,source_index,timestamp_ns,timestamp,tracking_state,tracking_state_name,pose_set,strict_success,split\n";
    std::size_t strict_success = 0;
    for (const auto &r : tracking_records)
    {
        trcsv << r.frame_index << ',' << r.source_index << ',' << r.timestamp_ns << ','
              << std::fixed << std::setprecision(9) << r.timestamp << ',' << r.tracking_state << ','
              << trackingStateName(r.tracking_state) << ',' << (r.pose_set ? 1 : 0) << ','
              << (r.strict_success ? 1 : 0) << ',' << (r.is_test ? "test" : "train") << '\n';
        strict_success += r.strict_success ? 1 : 0;
    }

    const auto gaussian_keyframe_images = saveGaussianKeyframeManifest(pGausMapper, output_dir);
    if (pSLAM->GetNumKeyframes() > 0)
    {
        pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
        pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
        pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    }
    const auto final_poses = collectFinalFramePosesInMapFrame(pSLAM->getTracker());

    if (!skip_final_eval && !pGausMapper->scene_->keyframes().empty())
    {
        evaluateViews(frames, tracking_records, final_poses, gaussian_keyframe_images,
                      pGausMapper, device_type, output_dir, "final_tracked_view_eval");
        if (online_iteration >= 0)
        {
            const fs::path online_ply = output_dir / (std::to_string(online_iteration) + "_online") /
                "ply" / "point_cloud" / ("iteration_" + std::to_string(online_iteration)) / "point_cloud.ply";
            if (fs::exists(online_ply))
            {
                pGausMapper->loadPly(online_ply);
                if (online_sh_degree >= 0) pGausMapper->gaussians_->setShDegree(static_cast<int>(online_sh_degree));
                evaluateViews(frames, tracking_records, final_poses, gaussian_keyframe_images,
                              pGausMapper, device_type, output_dir, "online_tracked_view_eval");
            }
        }
    }

    std::ofstream summary(output_dir / "tracking_summary.txt");
    summary << "dataset EuRoC\n";
    summary << "stride " << stride << '\n';
    summary << "input_frames " << frames.size() << '\n';
    summary << "processed_frames " << tracking_records.size() << '\n';
    summary << "train_frames " << train_count << '\n';
    summary << "test_frames " << test_count << '\n';
    summary << "strict_success_frames " << strict_success << '\n';
    summary << "strict_success_rate " << (frames.empty() ? 0.0 : static_cast<double>(strict_success) / frames.size()) << '\n';
    summary << "final_trajectory_pose_entries " << final_poses.size() << '\n';
    summary << "orb_keyframes " << pSLAM->GetNumKeyframes() << '\n';
    summary << "gaussian_keyframes " << pGausMapper->scene_->keyframes().size() << '\n';

    std::cout << "Strict tracking success: " << strict_success << '/' << frames.size() << std::endl;
    std::cout << "Result: " << output_dir << std::endl;
    return 0;
}

#include <torch/torch.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "include/gaussian_mapper.h"
#include "include/loss_utils.h"
#include "include/tensor_utils.h"

namespace fs = std::filesystem;

struct PoseRecord
{
    std::size_t frame_index = 0;
    double timestamp = 0.0;
    Sophus::SE3f Tcw;
};

static std::vector<std::string> splitCsv(const std::string &line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) out.push_back(token);
    return out;
}

static std::vector<PoseRecord> loadPoseCsv(const fs::path &path)
{
    std::ifstream in(path);
    if (!in.is_open()) throw std::runtime_error("Cannot open pose CSV: " + path.string());

    std::string line;
    std::getline(in, line); // header
    std::vector<PoseRecord> records;
    while (std::getline(in, line))
    {
        if (line.empty()) continue;
        auto cols = splitCsv(line);
        if (cols.size() != 18)
            throw std::runtime_error("Expected 18 columns in pose CSV, got " + std::to_string(cols.size()));

        PoseRecord rec;
        rec.frame_index = static_cast<std::size_t>(std::stoull(cols[0]));
        rec.timestamp = std::stod(cols[1]);

        Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
        int k = 2;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c, ++k)
                M(r, c) = std::stof(cols[k]);

        Eigen::Matrix3f R = M.block<3, 3>(0, 0);
        Eigen::Vector3f t = M.block<3, 1>(0, 3);
        rec.Tcw = Sophus::SE3f(Sophus::SO3f(R), t);
        records.push_back(rec);
    }
    return records;
}

static cv::Mat loadRgbFloat(const fs::path &path)
{
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (bgr.empty()) throw std::runtime_error("Failed to read image: " + path.string());

    cv::Mat rgb;
    if (bgr.channels() == 3) cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    else if (bgr.channels() == 4) cv::cvtColor(bgr, rgb, cv::COLOR_BGRA2RGB);
    else if (bgr.channels() == 1) cv::cvtColor(bgr, rgb, cv::COLOR_GRAY2RGB);
    else throw std::runtime_error("Unsupported channel count: " + path.string());

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

static fs::path leftImagePath(const fs::path &sequence_root, std::size_t frame_index)
{
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << frame_index;
    if (fs::is_directory(sequence_root / "image_left"))
        return sequence_root / "image_left" / (name.str() + "_left.png");
    if (fs::is_directory(sequence_root / "image_lcam_front"))
        return sequence_root / "image_lcam_front" / (name.str() + "_lcam_front.png");
    throw std::runtime_error("Unsupported TartanAir stereo layout: " + sequence_root.string());
}

static fs::path writeTartanAirLoaderCameraYaml(const fs::path &output_dir,
                                                const fs::path &sequence_root)
{
    // GaussianMapper::loadPly() expects its second argument to be an OpenCV
    // FileStorage camera YAML (Camera.w/h/type/fx/...), NOT cameras.json.
    // The latter stores checkpoint keyframe poses and is used only by the Python
    // recovery step to recover the Gaussian-map coordinate frame.
    //
    // TartanAir V1 Challenge: 640x480, fx=fy=320, cx=320, cy=240.
    // TartanAir V2 front stereo: 640x640, fx=fy=320, cx=cy=320.
    int width = 0, height = 0;
    float fx = 320.0f, fy = 320.0f, cx = 320.0f, cy = 0.0f;
    if (fs::is_directory(sequence_root / "image_left"))
    {
        width = 640; height = 480; cy = 240.0f;
    }
    else if (fs::is_directory(sequence_root / "image_lcam_front"))
    {
        width = 640; height = 640; cy = 320.0f;
    }
    else
    {
        throw std::runtime_error("Unsupported TartanAir stereo layout: " + sequence_root.string());
    }

    const fs::path path = output_dir / "_loader_camera.yaml";
    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("Cannot write loader camera YAML: " + path.string());
    out << "%YAML:1.0\n";
    out << "Camera.type: \"Pinhole\"\n";
    out << "Camera.w: " << width << "\n";
    out << "Camera.h: " << height << "\n";
    out << "Camera.fx: " << fx << "\n";
    out << "Camera.fy: " << fy << "\n";
    out << "Camera.cx: " << cx << "\n";
    out << "Camera.cy: " << cy << "\n";
    out << "Camera.k1: 0.0\n";
    out << "Camera.k2: 0.0\n";
    out << "Camera.p1: 0.0\n";
    out << "Camera.p2: 0.0\n";
    out << "Camera.k3: 0.0\n";
    out.close();
    return path;
}

int main(int argc, char **argv)
{
    if (argc != 8)
    {
        std::cerr << "Usage: " << argv[0]
                  << " gaussian_config checkpoint_ply cameras_json sequence_root pose_csv output_dir sh_degree\n";
        return 1;
    }

    const fs::path gaussian_cfg(argv[1]);
    const fs::path checkpoint_ply(argv[2]);
    const fs::path cameras_json(argv[3]); // recovery provenance only; not OpenCV camera config
    const fs::path sequence_root(argv[4]);
    const fs::path pose_csv(argv[5]);
    const fs::path output_dir(argv[6]);
    const int sh_degree = std::stoi(argv[7]);

    if (!fs::exists(cameras_json))
        throw std::runtime_error("Checkpoint cameras.json not found: " + cameras_json.string());

    fs::create_directories(output_dir);
    const fs::path render_dir = output_dir / "rendered";
    fs::create_directories(render_dir);

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    auto mapper = std::make_shared<GaussianMapper>(
        nullptr, gaussian_cfg, output_dir / "_loader_tmp", 0, device_type);

    const fs::path loader_camera_yaml = writeTartanAirLoaderCameraYaml(output_dir, sequence_root);
    mapper->loadPly(checkpoint_ply, loader_camera_yaml);
    mapper->gaussians_->setShDegree(sh_degree);

    const auto poses = loadPoseCsv(pose_csv);
    std::ofstream metrics(output_dir / "metrics.csv");
    metrics << "frame_index,timestamp,left_image,psnr,ssim,rendered_image\n";

    double sum_psnr = 0.0;
    double sum_ssim = 0.0;
    std::size_t evaluated = 0;
    torch::NoGradGuard no_grad;

    for (const auto &rec : poses)
    {
        const fs::path left_path = leftImagePath(sequence_root, rec.frame_index);
        cv::Mat gt_rgb = loadRgbFloat(left_path);
        cv::Mat rendered = mapper->renderFromPose(rec.Tcw, gt_rgb.cols, gt_rgb.rows, true);
        if (rendered.empty()) continue;
        if (rendered.size() != gt_rgb.size())
            cv::resize(gt_rgb, gt_rgb, rendered.size(), 0.0, 0.0, cv::INTER_LINEAR);

        torch::Tensor rendered_tensor = tensor_utils::cvMat2TorchTensor_Float32(rendered, device_type);
        torch::Tensor gt_tensor = tensor_utils::cvMat2TorchTensor_Float32(gt_rgb, device_type);
        const float psnr = loss_utils::psnr(rendered_tensor, gt_tensor).item().toFloat();
        const float ssim = loss_utils::ssim(rendered_tensor, gt_tensor, device_type).item().toFloat();

        std::ostringstream filename;
        filename << std::setw(6) << std::setfill('0') << rec.frame_index << "_left.png";
        const fs::path rendered_path = render_dir / filename.str();
        saveRgbFloatPng(rendered, rendered_path);

        metrics << rec.frame_index << ',' << std::fixed << std::setprecision(9) << rec.timestamp << ','
                << '"' << left_path.string() << '"' << ','
                << std::setprecision(10) << psnr << ',' << ssim << ','
                << '"' << rendered_path.string() << '"' << '\n';

        if (std::isfinite(psnr) && std::isfinite(ssim))
        {
            sum_psnr += psnr;
            sum_ssim += ssim;
            ++evaluated;
        }
    }

    std::ofstream summary(output_dir / "summary.txt");
    summary << "pose_frames " << poses.size() << '\n';
    summary << "evaluated_frames " << evaluated << '\n';
    if (evaluated)
    {
        summary << "mean_psnr " << (sum_psnr / evaluated) << '\n';
        summary << "mean_ssim " << (sum_ssim / evaluated) << '\n';
    }
    summary << "evaluation_type recovered_online_checkpoint_largest_map_only\n";
    summary << "camera_loader_yaml " << loader_camera_yaml.string() << '\n';
    summary << "checkpoint_cameras_json " << cameras_json.string() << '\n';

    std::cout << "[Recovered checkpoint evaluation] " << evaluated << "/" << poses.size() << " rendered\n";
    if (evaluated)
        std::cout << "  mean PSNR: " << (sum_psnr / evaluated)
                  << " dB, SSIM: " << (sum_ssim / evaluated) << '\n';
    std::cout << "  output: " << output_dir << std::endl;
    return 0;
}

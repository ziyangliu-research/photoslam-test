cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(photo_slam LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(realsense2 QUIET)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

set(ORB_SLAM3_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ORB-SLAM3)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/viewer
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/tinyply
    ${PROJECT_SOURCE_DIR}/submodules/simple-knn
    ${PROJECT_SOURCE_DIR}/submodules/diff-gaussian-rasterization
    ${ORB_SLAM3_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/include/CameraModels
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS})

link_directories(
    ${PROJECT_SOURCE_DIR}/lib
    ${ORB_SLAM3_SOURCE_DIR}/lib)

add_library(simple_knn SHARED
    submodules/simple-knn/simple_knn.cu
    submodules/simple-knn/spatial.cu
    submodules/simple-knn/ext.cpp)
set_target_properties(simple_knn PROPERTIES CUDA_ARCHITECTURES "120")
target_include_directories(simple_knn PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(simple_knn
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen)

add_library(cuda_rasterizer SHARED
    submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h
    submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu
    submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.h
    submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h
    submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
    submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h
    submodules/diff-gaussian-rasterization/cuda_rasterizer/operate_points.h
    submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h
    submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
    submodules/diff-gaussian-rasterization/cuda_rasterizer/cuda_rasterizer/rasterizer_impl.h
    submodules/diff-gaussian-rasterization/cuda_rasterizer/stereo_vision.h)
set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "120")
target_include_directories(cuda_rasterizer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_rasterizer
    glm::glm
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen)

add_library(imgui SHARED
    viewer/imgui/imconfig.h
    viewer/imgui/imgui_demo.cpp
    viewer/imgui/imgui_draw.cpp
    viewer/imgui/imgui_impl_glfw.cpp
    viewer/imgui/imgui_impl_glfw.h
    viewer/imgui/imgui_impl_opengl3_loader.h
    viewer/imgui/imgui_impl_opengl3.cpp
    viewer/imgui/imgui_impl_opengl3.h
    viewer/imgui/imgui_internal.h
    viewer/imgui/imgui_tables.cpp
    viewer/imgui/imgui_widgets.cpp
    viewer/imgui/imgui.cpp
    viewer/imgui/imgui.h
    viewer/imgui/imstb_rectpack.h
    viewer/imgui/imstb_textedit.h
    viewer/imgui/imstb_truetype.h)
target_link_libraries(imgui glfw OpenGL::GL)

add_library(gaussian_mapper SHARED
    third_party/tinyply/tinyply.h
    third_party/tinyply/tinyply.cpp
    include/gaussian_keyframe.h
    include/gaussian_model.h
    include/gaussian_parameters.h
    include/gaussian_rasterizer.h
    include/gaussian_renderer.h
    include/gaussian_scene.h
    include/gaussian_trainer.h
    include/gaussian_mapper.h
    include/general_utils.h
    include/graphics_utils.h
    include/loss_utils.h
    include/sh_utils.h
    include/tensor_utils.h
    include/camera.h
    include/point_cloud.h
    include/point2d.h
    include/point3d.h
    include/types.h
    src/gaussian_keyframe.cpp
    src/gaussian_model.cpp
    src/gaussian_parameters.cpp
    src/gaussian_rasterizer.cpp
    src/gaussian_renderer.cpp
    src/gaussian_scene.cpp
    src/gaussian_trainer.cpp
    src/gaussian_mapper.cpp)
target_link_libraries(gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen
    simple_knn
    cuda_rasterizer)

add_library(gaussian_viewer SHARED
    viewer/drawer_utils.h
    viewer/imgui_viewer.cpp
    viewer/imgui_viewer.h
    viewer/map_drawer.cpp
    viewer/map_drawer.h)
target_link_libraries(gaussian_viewer
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    gaussian_mapper
    imgui
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    glm::glm
    glfw
    OpenGL::GL)

add_executable(train_colmap examples/train_colmap.cpp)
target_link_libraries(train_colmap gaussian_viewer gaussian_mapper)

add_executable(view_result examples/view_result.cpp)
target_link_libraries(view_result gaussian_viewer gaussian_mapper)

add_executable(replica_mono examples/replica_mono.cpp)
target_link_libraries(replica_mono gaussian_viewer gaussian_mapper ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

add_executable(replica_rgbd examples/replica_rgbd.cpp)
target_link_libraries(replica_rgbd gaussian_viewer gaussian_mapper ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

add_executable(tum_mono examples/tum_mono.cpp)
target_link_libraries(tum_mono gaussian_viewer gaussian_mapper ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

add_executable(tum_rgbd examples/tum_rgbd.cpp)
target_link_libraries(tum_rgbd gaussian_viewer gaussian_mapper ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

add_executable(euroc_stereo examples/euroc_stereo.cpp)
target_link_libraries(euroc_stereo gaussian_viewer gaussian_mapper ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# EuRoC stereo held-out/paper evaluation runner. Uses Photo-SLAM's official EuRoC configs.
add_executable(euroc_stereo_eval examples/euroc_stereo_eval.cpp)
target_link_libraries(euroc_stereo_eval
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# TartanAir V1/V1-challenge Stereo evaluation
add_executable(tartanair_stereo_eval examples/tartanair_stereo_eval.cpp)
target_link_libraries(tartanair_stereo_eval
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# Offline recovery/evaluation of an already-saved TartanAir Gaussian checkpoint.
add_executable(tartanair_checkpoint_eval examples/tartanair_checkpoint_eval.cpp)
target_link_libraries(tartanair_checkpoint_eval
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

if(realsense2_FOUND)
    add_executable(realsense_rgbd examples/realsense_rgbd.cpp)
    target_include_directories(realsense_rgbd PUBLIC ${realsense_INCLUDE_DIR})
    target_link_libraries(realsense_rgbd
        gaussian_viewer
        gaussian_mapper
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
        ${realsense2_LIBRARY})
endif()

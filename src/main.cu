#include "cpu_scene.cuh"
#include "gpu_scene.cuh"

#include <stdexcept>
#include <utility.cuh>
#include <scene.cuh>

#include <iostream>
#include <memory>
#include <vector_types.h>

int main(int argc, char* argv[])
{
    auto device = render::ProcessArgs(argc, argv);
    std::unique_ptr<render::IScene> scene;
    switch (device)
    {
        case render::Device::CPU:
            scene.reset(new render::CpuScene);
            std::cin >> (static_cast<render::CpuScene&>(*scene));
            break;
        case render::Device::GPU:
            scene.reset(new render::GpuScene);
            std::cin >> (static_cast<render::GpuScene&>(*scene));
            break;
        case render::Device::DEFAULT:
            scene.reset(new render::GpuScene);
            std::cin >> (static_cast<render::GpuScene&>(*scene));
            break;
        default:
            throw std::runtime_error{"unknown device type"};
    }

    scene->GenerateScene();
    scene->Render();
    
    return 0;
}
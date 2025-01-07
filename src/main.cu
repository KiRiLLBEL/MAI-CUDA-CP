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
            std::cout << "300\nout/img_%d.data\n1920 1080 70\n7.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0\n2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0\n2.0 0.0 0.0 1.0 0.0 0.0 1.0 0.95 0.1 10\n0.0 2.0 0.0 0.0 1.0 0.0 1.0 0.5 0.5 5\n-2.0 -2.0 0.0 0.0 0.7 0.7 1.5 0.1 0.9 2\n-5.0 -5.0 -1.0 -5.0 5.0 -1.0 5.0 5.0 -1.0 5.0 -5.0 -1.0 res/floor.data 0.0 1.0 0.0 0.5\n2\n-10.0 0.0 10.0 1.0 1.0 1.0\n1.0 0.0 10.0 0.54 0.0 1.0\n5 2\n";
            return 0;
            break;
        default:
            throw std::runtime_error{"unknown device type"};
    }

    scene->GenerateScene();
    scene->Render();
    
    return 0;
}
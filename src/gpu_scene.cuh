#pragma once

#include <scene.cuh>

namespace render 
{
class GpuScene : public IScene
{
private:
    Floor floor_m;
    uint64_t frames_m;
    std::string savePath_m;
    Camera camera_m;
    std::vector<Object> objects_m{OBJECTS_COUNT};
    std::vector<Light> lights_m;
    std::vector<Polygon> polygons_m;
    Light* deviceLights_m;
    Polygon* devicePolygons_m;
    uint64_t maxDepth;
    uint64_t upscaleFactor;
private:
    int64_t GenerateFrame(int UUID, uchar4* deviceSSAA);
public:

    ~GpuScene()
    {
        if (deviceLights_m)
        {
            cudaFree(deviceLights_m);
        }
        if (devicePolygons_m)
        {
            cudaFree(devicePolygons_m);
        }
    }

    friend std::istream& operator>>(std::istream& in, GpuScene& scene)
    {
        in >> scene.frames_m
        >> scene.savePath_m
        >> scene.camera_m;

        for(uint64_t i = 0; i < OBJECTS_COUNT; ++i)
        {
            in >> scene.objects_m[i];
        }

        in >> scene.floor_m;
        uint64_t countOfLights{0};
        in >> countOfLights;
        scene.lights_m.resize(countOfLights);
        for(uint64_t i = 0; i < countOfLights; ++i)
        {
            in >> scene.lights_m[i];
        }
        cudaMalloc(&scene.deviceLights_m, countOfLights * sizeof(Light));
        cudaMemcpy(scene.deviceLights_m, scene.lights_m.data(), countOfLights * sizeof(Light), cudaMemcpyHostToDevice);

        scene.devicePolygons_m = nullptr;
        in >> scene.maxDepth;
        return in >> scene.upscaleFactor;
    }

    void GenerateScene() override;
    void Render() override;
};
};

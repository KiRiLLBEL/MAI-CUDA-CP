#pragma once

#include <cstdint>
#include <scene.cuh>

namespace render 
{
class CpuScene : public IScene
{
private:
    Floor floor_m;
    uint64_t frames_m;
    std::string savePath_m;
    Camera camera_m;
    std::vector<Object> objects_m{OBJECTS_COUNT};
    std::vector<Light> lights_m;
    std::vector<Polygon> polygons_m;
    uint64_t maxDepth;
    uint64_t upscaleFactor;
private:
    int64_t GenerateFrame(int UUID, std::vector<uchar4>& image);
    Vector3f TraceRay(const Ray& ray, int depth, uint64_t& countOfRays);
public:
    
    friend std::istream& operator>>(std::istream& in, CpuScene& scene)
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

        in >> scene.maxDepth;
        return in >> scene.upscaleFactor;
    }

    void GenerateScene() override;
    void Render() override;
};
};
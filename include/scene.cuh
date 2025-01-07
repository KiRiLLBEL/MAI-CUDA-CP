#pragma once

#include <camera.cuh>
#include <cstdint>
#include <istream>
#include <object.cuh>
#include <light.cuh>
#include <floor.cuh>
#include <polygon.cuh>
#include <ssaa.cuh>

#include <array>
#include <vector>

#define OBJECTS_COUNT 3
#define SHIFT 1e-3
const dim3 BLOCKS_2D(64, 64);
const dim3 THREADS_2D(32, 32);
const uint64_t BLOCKS{1024};
const uint64_t THREADS{512};

constexpr std::array<const char*, 3> ObjectsPaths
{
    "res/cube.obj",
    "res/dodec.obj",
    "res/tetraedr.obj"
};

namespace render
{
class IScene
{
public:
    virtual void GenerateScene() = 0;
    virtual void Render() = 0;
};

__host__ __device__ inline Vector3f Phong(
    const Ray& ray,
    const Vector3d& intersection,
    int polygonIdx,
    const Light* lights,
    const int countOfLights,
    const Polygon* polygons,
    int countOfPolygons)
{
    auto pidx = polygons[polygonIdx];
    auto color = pidx.ComputeColorInPoint(ray, intersection);
    float coef = 1.0 - pidx.reflection() - pidx.transparency();
    auto phongColor = (0.25 * coef) * ray.color() * color;
    for(int i = 0; i < countOfLights; ++i) {
        double maxT = (lights[i].position() - intersection).length();
        auto lightRay = Ray(intersection, lights[i].position() - intersection, ray.uid());
        auto visibility = Vector3f(1, 1, 1);
        for(int polygonId = 0; polygonId < countOfPolygons; ++polygonId)
        {
            if(polygonId == polygonIdx)
            {
                continue;
            }
            bool flag{false};
            auto t = polygons[polygonId].MollerTrumboreIntersection(lightRay, flag);
            if(flag && t < maxT)
            {
                visibility *= polygons[polygonId].transparency();
            }
        }

        Vector3f colorIntensity = 
            (1.0f - pidx.reflection() - pidx.transparency())
            * ray.color() 
            * visibility
            * lights[i].intensity()
            * color;

        double diffusial = pidx.triangle().n() ^ lightRay.view();
        double specullar = 0;
        if(diffusial < 0)
        {
            diffusial = 0;
        }
        else
        {
            Vector3d reflected = Vector3d::reflect(lightRay.view(), pidx.triangle().n());
            specullar = reflected ^ ray.view();
            if(specullar < 0)
            {
                specullar = 0;
            }
            else
            {
                specullar = std::pow(specullar, 9);
            }
        }

        phongColor += (diffusial + 0.5 * specullar) * colorIntensity;
    }
    phongColor.range(0.0, 1.0);
    return phongColor;
}
};
#pragma once
#include <istream>
#include <vector3.cuh>
namespace render
{
class Light
{
private:
    Vector3d position_m;
    Vector3f intensity_m;
public:
    friend std::istream& operator>>(std::istream& in, Light& light)
    {
        return in >> light.position_m
        >> light.intensity_m;
    }

    __host__ __device__ Vector3d position() const
    {
        return position_m;
    }

    __host__ __device__ Vector3f intensity() const
    {
        return intensity_m;
    }
};
};
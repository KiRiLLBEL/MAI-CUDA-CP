#pragma once

#include <vector3.cuh>
#include <light.cuh>
#include <cstdint>

namespace render {

class Ray 
{
private:
    Vector3d position_m;
    Vector3d view_m;
    uint64_t id;
    Vector3f color_m;
public:
    __host__ __device__ Ray() : position_m(), view_m(), id()
    {

    }

    __host__ __device__ Ray(const Ray& other)
        : position_m(other.position_m),
        view_m(other.view_m),
        id(other.id),
        color_m(other.color_m)
    {

    }

    __host__ __device__ Ray& operator=(const Ray& other)
    {
        if (this != &other) {
            position_m = other.position_m;
            view_m = other.view_m;
            id = other.id;
            color_m = other.color_m;
        }
        return *this;
    }

    __host__ __device__ Ray(const Vector3d& position, const Vector3d& view, const uint64_t uuid) 
        : position_m(position),
        view_m(view),
        id(uuid),
        color_m(1, 1, 1)
    {
        view_m.norm();
    }

    __host__ __device__ Ray(const Vector3d& position, const Vector3d& view, const uint64_t uuid, const Vector3f& color) 
        : position_m(position),
        view_m(view),
        id(uuid),
        color_m(color)
    {
        view_m.norm();
    }

    __host__ __device__ Vector3d position() const
    {
        return position_m;
    }

    __host__ __device__ Vector3d view() const
    {
        return view_m;
    }

    __host__ __device__ Vector3f color() const
    {
        return color_m;
    }

    __host__ __device__ uint64_t uid() const
    {
        return id;
    }
};

};


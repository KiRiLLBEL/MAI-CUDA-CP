#pragma once

#include <cstdint>
#include <memory>

#include <ray.cuh>
#include <triangle.cuh>
#include <textures.cuh>

namespace render
{
class Polygon
{
private:
    Triangle base_m;
    Vector3f color_m;
    double a_m, b_m, c_m, d_m;
    double reflection_m{0}, transparency_m{0};
    int64_t countOfLights_m{0};
    Vector3d v1_m;
    Vector3d v2_m;
    Vector3d v3_m;
    Texture texture_m;
    bool hasTexture = false;
private:
    void CreatePlane()
    {
        Vector3d p0 = base_m.a();
        Vector3d v1 = base_m.b() - p0;
        Vector3d v2 = base_m.c() - p0;
        auto product = v1 | v2;
        a_m = product.x();
        b_m = product.y();
        c_m = product.z();
        d_m = -p0.x() * a_m + p0.y() * (-b_m) + (-p0.z()) * c_m;
    }
public:
    Polygon() = default;

    Polygon& operator=(const render::Polygon& other) = default;
    
    Polygon(const Triangle& base, const Vector3f& color)
        : base_m(base), color_m(color)
    {
        CreatePlane();
    }

    Polygon(const Triangle& base, const Vector3f& color, double reflection, double transparency)
        : base_m(base), color_m(color), reflection_m(reflection), transparency_m(transparency)
    {
        CreatePlane();
    }

    Polygon(
        const Triangle& base, 
        const Vector3f& color, 
        const int64_t countOfLights,
        const Vector3d& v1,
        const Vector3d& v2)
        : base_m(base),
        color_m(color),
        countOfLights_m(countOfLights),
        v1_m(v1),
        v2_m(v2)
    {
        CreatePlane();
    }

    Polygon(
        const Triangle& base, 
        const Vector3f& color, 
        double reflection,
        double transparency,
        const Vector3d& v1,
        const Vector3d& v2,
        const Vector3d& v3,
        const Texture& texture)
        : base_m(base),
        color_m(color),
        reflection_m(reflection),
        transparency_m(transparency),
        v1_m(v1),
        v2_m(v2),
        v3_m(v3),
        texture_m(texture),
        hasTexture(true)
    {
        CreatePlane();
    }

    __host__ __device__ Triangle triangle() const
    {
        return base_m;
    }

    __host__ __device__ double reflection() const
    {
        return reflection_m;
    }

    __host__ __device__ double transparency() const
    {
        return transparency_m;
    }

    __host__ __device__ Vector3f color() const
    {
        return color_m;
    }

    __host__ __device__ Vector3f ComputeColorInPoint(const Ray& ray, const Vector3d& intersection) const
    {
        if(hasTexture)
        {
            auto p = intersection - v3_m;
            double x = (p.x() * v1_m.y() - p.y() * v1_m.x()) / (v2_m.x() * v1_m.y() - v2_m.y() * v1_m.x());
            double y = (p.x() * v2_m.y() - p.y() * v2_m.x()) / (v1_m.x() * v2_m.y() - v1_m.y() * v2_m.x());
            return texture_m.getColor(x, y);
        }
        else if(countOfLights_m > 0 && (base_m.n() ^ ray.view()) > 0.0)
        {
            Vector3d vl = (v2_m - v1_m) / (countOfLights_m + 1);
            for (int i = 1; i <= countOfLights_m; ++i)
            {
                Vector3d pointLight = v1_m + i * vl;
                if ((pointLight - intersection).length() < 0.025)
                {
                    return Vector3f(16, 16, 16);
                }
            }
        }

        return color_m;
    }

    __host__ __device__ double RayIntersection(const Ray& ray) const
    {
        return -(a_m * ray.position().x() + b_m * ray.position().y() + c_m * ray.position().z() + d_m) / 
                (a_m * ray.view().x() + b_m * ray.view().y() + c_m * ray.view().z());
    }
    __host__ __device__ double MollerTrumboreIntersection(const Ray& ray, bool& flag) const
    {
        flag = false;
        auto P = ray.view() | base_m.e2();
        auto div = P ^ base_m.e1();
        if(std::abs(div) < EPS)
        {
            return 0;
        }

        auto T = ray.position() - base_m.a();
        auto u = (P ^ T) / div;
        if(u < 0.0 || u > 1.0)
        {
            return 0;
        }

        auto Q = T | base_m.e1();
        auto v = (Q ^ ray.view()) / div;
        if(v < 0.0 || u + v > 1.0)
        {
            return 0;
        }

        double t = (Q ^ base_m.e2()) / div;
        if(t < 0.0)
        {
            flag = false;
        }
        else
        {
            flag = true;
        }

        return t;
    }
};

};
#pragma once
#include "utility.cuh"
#include <vector3.cuh>

namespace render {
class Triangle
{
private:
    // Points
    Vector3d a_m, b_m, c_m;
    // Vectors
    Vector3d e1_m, e2_m, n_m;
public:
    Triangle() = default;
    Triangle(const Vector3d& a, const Vector3d& b, const Vector3d& c)
        : a_m(a),
        b_m(b), 
        c_m(c), 
        e1_m(b - a), 
        e2_m(c - a), 
        n_m(e1_m | e2_m)
    {
        n_m.norm();
    }

    __host__ __device__ Vector3d a() const
    {
        return a_m;
    }

    __host__ __device__ Vector3d b() const
    {
        return b_m;
    }

    __host__ __device__ Vector3d c() const
    {
        return c_m;
    }

    __host__ __device__ Vector3d e1() const
    {
        return e1_m;
    }

    __host__ __device__ Vector3d e2() const
    {
        return e2_m;
    }
    
    __host__ __device__ Vector3d n() const
    {
        return n_m;
    }

    __host__ __device__ void shift(const Vector3d& other)
    {
        a_m += other;
        b_m += other;
        c_m += other;
    }

    __host__ __device__ void rotate(const Vector3d& n)
    {
        if((n ^ n_m) < -EPS)
        {
            auto temp = a_m;
            a_m = c_m;
            c_m = temp;
            e1_m = b_m - a_m;
            e2_m = c_m - a_m; 
            n_m = e1_m | e2_m;
            n_m.norm();
        }
    }
};
};
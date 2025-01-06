#pragma once
#include <cstdint>
#include <istream>
#include <sys/types.h>
#include <vector3.cuh>

namespace render
{
class Camera
{
private:
    Vector3d position_m;
    Vector3d view_m;
    uint64_t width_m;
    uint64_t height_m;
    double angle_m;
    double r_0c, z_0c, phi_0c, A_rc, A_zc, w_rc, w_zc, w_phic, p_rc, p_zc;
    double r_0n, z_0n, phi_0n, A_rn, A_zn, w_rn, w_zn, w_phin, p_rn, p_zn;
    uint64_t widthSSAA_m;
    uint64_t heightSSAA_m;
public:
    __host__ __device__ Camera() : width_m{0}, height_m{0}, angle_m{0}
    {

    }

    __host__ __device__ Camera(const Camera& other)
    {
        position_m = other.position_m;
        view_m = other.view_m;
        width_m = other.width_m;
        height_m = other.height_m;
        angle_m = other.angle_m;
        r_0c = other.r_0c;
        z_0c = other.z_0c;
        phi_0c = other.phi_0c;
        A_rc = other.A_rc;
        A_zc = other.A_zc;
        w_rc = other.w_rc;
        w_zc = other.w_zc;
        w_phic = other.w_phic;
        p_rc = other.p_rc;
        p_zc = other.p_zc;
        r_0n = other.r_0n;
        z_0n = other.z_0n;
        phi_0n = other.phi_0n;
        A_rn = other.A_rn;
        A_zn = other.A_zn;
        w_rn = other.w_rn;
        w_zn = other.w_zn;
        w_phin = other.w_phin;
        p_rn = other.p_rn;
        p_zn = other.p_zn;
        widthSSAA_m = other.widthSSAA_m;
        heightSSAA_m = other.heightSSAA_m;
    }

    __host__ __device__ Camera& operator=(const Camera& other)
    {
        if (this == &other)
            return *this;

        position_m = other.position_m;
        view_m = other.view_m;
        width_m = other.width_m;
        height_m = other.height_m;
        angle_m = other.angle_m;
        r_0c = other.r_0c;
        z_0c = other.z_0c;
        phi_0c = other.phi_0c;
        A_rc = other.A_rc;
        A_zc = other.A_zc;
        w_rc = other.w_rc;
        w_zc = other.w_zc;
        w_phic = other.w_phic;
        p_rc = other.p_rc;
        p_zc = other.p_zc;
        r_0n = other.r_0n;
        z_0n = other.z_0n;
        phi_0n = other.phi_0n;
        A_rn = other.A_rn;
        A_zn = other.A_zn;
        w_rn = other.w_rn;
        w_zn = other.w_zn;
        w_phin = other.w_phin;
        p_rn = other.p_rn;
        p_zn = other.p_zn;
        widthSSAA_m = other.widthSSAA_m;
        heightSSAA_m = other.heightSSAA_m;

        return *this;
    }

    __host__ __device__ Camera(const uint64_t width, const uint64_t height, const double angle)
        : width_m(width), height_m(height), angle_m(angle)
    {

    }



    __host__ __device__ void updatePosition(const double time)
    {
        auto r = r_0c + A_rc * std::sin(w_rc * time + p_rc);
        auto z = z_0c + A_zc * std::sin(w_zc * time + p_zc);
        auto phi = phi_0c + w_phic * time;
        position_m = {r * std::cos(phi), r * std::sin(phi), z};
    }

    __host__ __device__ void updateView(const double time)
    {
        auto r = r_0n + A_rn * std::sin(w_rn * time + p_rn);
        auto z = z_0n + A_zn * std::sin(w_zn * time + p_zn);
        auto phi = phi_0n + w_phin * time;
        view_m = {r * std::cos(phi), r * std::sin(phi), z};
    }

    __host__ __device__ Vector3d position() const
    {
        return position_m;
    }

    __host__ __device__ Vector3d view() const
    {
        return view_m;
    }

    __host__ __device__ uint64_t w() const
    {
        return width_m;
    }

    __host__ __device__ uint64_t h() const
    {
        return height_m;
    }

    __host__ __device__ uint64_t wSSAA() const
    {
        return widthSSAA_m;
    }

    __host__ __device__ uint64_t hSSAA() const
    {
        return heightSSAA_m;
    }

    __host__ __device__ double angle() const
    {
        return angle_m;
    }

    __host__ __device__ void setSSAA(const double upscaleFactor)
    {
        widthSSAA_m = width_m * upscaleFactor;
        heightSSAA_m = height_m * upscaleFactor;
    }

    friend std::istream& operator>>(std::istream& in, Camera& camera)
    {
        return in >> camera.width_m
        >> camera.height_m
        >> camera.angle_m
        >> camera.r_0c 
        >> camera.z_0c
        >> camera.phi_0c
        >> camera.A_rc
        >> camera.A_zc
        >> camera.w_rc
        >> camera.w_zc
        >> camera.w_phic
        >> camera.p_rc
        >> camera.p_zc
        >> camera.r_0n
        >> camera.z_0n
        >> camera.phi_0n
        >> camera.A_rn
        >> camera.A_zn
        >> camera.w_rn
        >> camera.w_zn
        >> camera.w_phin
        >> camera.p_rn
        >> camera.p_zn;
    }


};
}
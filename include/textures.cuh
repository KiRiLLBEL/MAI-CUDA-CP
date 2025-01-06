#pragma once
#include <utility.cuh>
#include <vector3.cuh>

namespace render
{
class Texture
{
private:
    int width_m, height_m;
    uchar4* data_m{nullptr};
    uchar4* deviceData_m{nullptr};
public:
    __host__ __device__ Texture() : width_m{0}, height_m{0}
    {

    }

    __host__ Texture(const std::string& path)
    {
        data_m = LoadFile(path, width_m, height_m);
    }

    __host__ __device__ Texture(const Texture& other)
        : width_m(other.width_m),
        height_m(other.height_m),
        data_m(other.data_m),
        deviceData_m(other.deviceData_m)
    {

    }

    Texture& operator=(const Texture& other) = default;

    __host__ __device__ Texture(Texture&& other)
        : width_m(std::move(other.width_m)),
        height_m(std::move(other.height_m)),
        data_m(std::move(other.data_m)),
        deviceData_m(std::move(other.deviceData_m))
    {

    }

    __host__ void LoadTextureToDevice()
    {
        cudaMalloc(&deviceData_m, sizeof(uchar4) * width_m * height_m);
        cudaMemcpy(deviceData_m, data_m, sizeof(uchar4) * width_m * height_m, cudaMemcpyHostToDevice);
    }

    __host__ __device__ Vector3f getColor(double x, double y) const
    {
        int xPos = x * width_m;
        int yPos = y * height_m;
        xPos = max(0, min(xPos, width_m - 1));
        yPos = max(0, min(yPos, height_m - 1));
        uchar4 pos;
        if(deviceData_m)
        {
            pos = deviceData_m[yPos * width_m + xPos];
        }
        else
        {
            pos = data_m[yPos * width_m + xPos];
        }

        return Vector3f(pos.x, pos.y, pos.z) / 255.0f;
    }

};

};
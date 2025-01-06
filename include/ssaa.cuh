#pragma once

namespace render
{
__global__ void ssaaGPUKernel(
    const uchar4* deviceSSAA,
    uchar4* deviceData,
    int w,
    int h,
    int upscaleFactor)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int offsetx = blockDim.x * gridDim.x;
    const int offsety = blockDim.y * gridDim.y;

    const int wSSAA = w * upscaleFactor;

    for(int i = idx; i < w; i += offsetx)
    {
        for(int j = idy; j < h; j += offsety)
        {
            double x = 0, y = 0, z = 0;
            for(int si = 0; si < upscaleFactor; ++si)
            {
                for(int sj = 0; sj < upscaleFactor; ++sj)
                {
                    uchar4 data = deviceSSAA[(upscaleFactor * i + si) + wSSAA * (upscaleFactor * j + sj)];
                    x += data.x;
                    y += data.y;
                    z += data.z;
                }
            }
            x /= upscaleFactor * upscaleFactor;
            y /= upscaleFactor * upscaleFactor;
            z /= upscaleFactor * upscaleFactor;
            deviceData[i + j * w] = make_uchar4(x, y, z, 255);
        }
    }
}
};
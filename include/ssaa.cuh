#pragma once

#include <vector>

namespace render
{
__global__ void ssaaGPUKernel(
    const uchar4* deviceSSAA,
    uchar4* deviceData,
    int w,
    int h,
    int upscaleFactor);

void ApplySSAA(
    const std::vector<uchar4>& input, 
    std::vector<uchar4>& output, 
    int w, int h, int upscaleFactor);
}
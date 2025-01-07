#include "gpu_scene.cuh"
#include <cstdint>
#include <iostream>
#include <limits>

namespace render
{

__global__ void PrepareBuffer(Vector3f* deviceData, const uint64_t size) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t offset = gridDim.x * blockDim.x;
    for (uint64_t i = idx; i < size; i += offset) {
        deviceData[i] = Vector3f(0.0, 0.0, 0.0);
    }
}

void GpuScene::GenerateScene() 
{
    auto texture = Texture(floor_m.TexturePath());
    texture.LoadTextureToDevice();
    auto floor1 = Triangle(floor_m.p1(), floor_m.p3(), floor_m.p2());
    auto floor2 = Triangle(floor_m.p3(), floor_m.p1(), floor_m.p4());

    polygons_m.push_back(
        Polygon(
            floor1, 
            floor_m.color(), 
            floor_m.reflection(), 
            0.0,
            floor_m.p2() - floor_m.p3(),
            floor_m.p2() - floor_m.p1(),
            floor_m.p1() + floor_m.p3() - floor_m.p2(),
            texture));
    polygons_m.push_back(
        Polygon(
            floor2, 
            floor_m.color(), 
            floor_m.reflection(), 
            0.0,
            floor_m.p1() - floor_m.p4(),
            floor_m.p3() - floor_m.p4(),
            floor_m.p4(),
            texture));

    for(uint64_t i = 0; i < OBJECTS_COUNT; ++i)
    {
        objects_m[i].CreateObjectFromFile(ObjectsPaths[i], polygons_m);
    }
    cudaMalloc(&devicePolygons_m, polygons_m.size() * sizeof(Polygon));
    cudaMemcpy(devicePolygons_m, polygons_m.data(), polygons_m.size() * sizeof(Polygon), cudaMemcpyHostToDevice);
}

__global__ void CreateRaysKernel(const Camera* camera, Ray* deviceRays)
{
    double dw = 2.0 / (camera->wSSAA() - 1.0);
    double dh = 2.0 / (camera->hSSAA() - 1.0);
    double z = 1.0 / std::tan(camera->angle() * M_PI / 360.0);

    Vector3d bz = camera->view() - camera->position();
    Vector3d bx = bz | Vector3d(0.0, 0.0, 1.0);
    Vector3d by = bx | bz;

    bx.norm();
    by.norm();
    bz.norm();

    const uint64_t offsetx = blockDim.x * gridDim.x;
    const uint64_t offsety = blockDim.y * gridDim.y;

    for (uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < camera->wSSAA(); idx += offsetx)
    {
        for (uint64_t idy = blockDim.y * blockIdx.y + threadIdx.y; idy < camera->hSSAA(); idy += offsety)
        {
            Vector3d v(-1.0 + dw * idx, (-1.0 + dh * idy) * camera->hSSAA() / camera->wSSAA(), z);
            auto dir = Vector3d::transposeMultiplication(bx, by, bz, v);
            uint64_t rayId = (camera->hSSAA() - 1 - idy) * camera->wSSAA() + idx;
            deviceRays[idx * camera->hSSAA() + idy] = Ray(camera->position(), dir, rayId);
        }
    }    
}

__global__ void ProcessRaysKernel(
    const Ray* deviceRaysInput,
    const uint64_t inputSize,
    Ray* deviceRaysOutput,
    uint64_t* outputSize,
    Vector3f* deviceData,
    const Light* deviceLights,
    const int lightsCount,
    const Polygon* devicePolygons,
    int polygonsCount)
{
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t offset = gridDim.x * blockDim.x;
    for(uint64_t k = idx; k < inputSize; k += offset)
    {
        int minPolygonIndex = -1;
        double minT = 1e18;
        for(int i = 0; i < polygonsCount; ++i)
        {
            bool flag{false};
            auto dpi = devicePolygons[i];
            auto t = dpi.MollerTrumboreIntersection(deviceRaysInput[k], flag);
            if(flag && t < minT)
            {
                minPolygonIndex = i;
                minT = t;
            }
        }
        if(minPolygonIndex < 0)
        {
            continue;
        }
        auto intersection = deviceRaysInput[k].position() + minT * deviceRaysInput[k].view();
        auto dpm = devicePolygons[minPolygonIndex];
        auto color = dpm.ComputeColorInPoint(deviceRaysInput[k], intersection);
        auto phongColor = Phong(
            deviceRaysInput[k],
            intersection,
            minPolygonIndex,
            deviceLights,
            lightsCount,
            devicePolygons,
            polygonsCount
        );
        deviceData[deviceRaysInput[k].uid()] &= phongColor;

        if(dpm.transparency() > 0)
        {
            deviceRaysOutput[atomicAdd((unsigned long long int*)outputSize, 1)] = Ray(
                intersection + SHIFT * deviceRaysInput[k].view(),
                deviceRaysInput[k].view(), 
                deviceRaysInput[k].uid(),
                dpm.transparency() * deviceRaysInput[k].color() * color);
        }

        if(dpm.reflection() > 0)
        {
            auto reflected = Vector3d::reflect(deviceRaysInput[k].view(), dpm.triangle().n());
            deviceRaysOutput[atomicAdd((unsigned long long int*)outputSize, 1)] = Ray(
                intersection + SHIFT * reflected,
                reflected, 
                deviceRaysInput[k].uid(),
                dpm.reflection() * deviceRaysInput[k].color() * color);
        }
    }
}

__global__ void NormalizeDataKernel(Vector3f* deviceData, uchar4* deviceSSAA, uint64_t size)
{
    const uint64_t offset = gridDim.x * blockDim.x;
    for(uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += offset)
    {
        deviceData[idx].range(0.0, 1.0);
        deviceData[idx] *= 255.0;
        deviceSSAA[idx] = make_uchar4(deviceData[idx].x(), deviceData[idx].y(), deviceData[idx].z(), 255);
    }
}


int64_t GpuScene::GenerateFrame(uint64_t UUID, uchar4* deviceSSAA)
{
    uint64_t w = camera_m.wSSAA();
    uint64_t h = camera_m.hSSAA();
    uint64_t inputSize = w * h;
    Vector3f* deviceDataPtr;
    cudaMalloc(&deviceDataPtr, inputSize * sizeof(Vector3f));
    PrepareBuffer<<<BLOCKS, THREADS>>>(deviceDataPtr, inputSize);
    cudaDeviceSynchronize();
    Ray* deviceRaysInputPtr;
    cudaMalloc(&deviceRaysInputPtr, inputSize * sizeof(Ray));
    Camera* deviceCameraPtr;
    cudaMalloc(&deviceCameraPtr, sizeof(Camera));
    cudaMemcpy(deviceCameraPtr, &camera_m, sizeof(Camera), cudaMemcpyHostToDevice);

    CreateRaysKernel<<<BLOCKS_2D, THREADS_2D>>>(deviceCameraPtr, deviceRaysInputPtr);
    cudaDeviceSynchronize();
    int64_t countRays{0};
    for(uint64_t depth = 0; depth < maxDepth; ++depth)
    {
        countRays += inputSize;
        Ray* deviceRaysOutputPtr;
        cudaMalloc(&deviceRaysOutputPtr, 2 * inputSize * sizeof(Ray));
        
        uint64_t zeroValue = 0;
        uint64_t* deviceOutputSizePtr;
        cudaMalloc(&deviceOutputSizePtr, sizeof(uint64_t));
        cudaMemcpy(deviceOutputSizePtr, &zeroValue, sizeof(uint64_t), cudaMemcpyHostToDevice);

        ProcessRaysKernel<<<BLOCKS, THREADS>>>(
            deviceRaysInputPtr, 
            inputSize,
            deviceRaysOutputPtr,
            deviceOutputSizePtr,
            deviceDataPtr,
            deviceLights_m,
            lights_m.size(),
            devicePolygons_m,
            polygons_m.size());
        cudaDeviceSynchronize();
        cudaMemcpy(&inputSize, deviceOutputSizePtr, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaFree(deviceRaysInputPtr);
        cudaFree(deviceOutputSizePtr);
        deviceRaysInputPtr = deviceRaysOutputPtr;
    }
    NormalizeDataKernel<<<BLOCKS, THREADS>>>(deviceDataPtr, deviceSSAA, w * h);
    cudaDeviceSynchronize();
    cudaFree(deviceDataPtr);
    cudaFree(deviceRaysInputPtr);
    cudaFree(deviceCameraPtr);
    return countRays;
} 

void GpuScene::Render() 
{
    double dt = 2 * M_PI / frames_m;
    camera_m.setSSAA(upscaleFactor);
    uchar4* deviceSSAA;
    cudaMalloc(&deviceSSAA, camera_m.wSSAA() * camera_m.hSSAA() * sizeof(uchar4));
    uchar4* deviceData;
    cudaMalloc(&deviceData, camera_m.w() * camera_m.h() * sizeof(uchar4));
    for(uint64_t k = 0; k < frames_m; ++k)
    {
        Timer timer;
        timer.begin();
        double time = k * dt;
        camera_m.updatePosition(time);
        camera_m.updateView(time);
        auto countRays = GenerateFrame(k, deviceSSAA);
        ssaaGPUKernel<<<BLOCKS_2D, THREADS_2D>>>(
            deviceSSAA, 
            deviceData,
            camera_m.w(),
            camera_m.h(),
            upscaleFactor);
        cudaDeviceSynchronize();
        std::vector<uchar4> output(camera_m.w() * camera_m.h());
        cudaMemcpy(output.data(), deviceData, camera_m.w() * camera_m.h() * sizeof(uchar4), cudaMemcpyDeviceToHost);
        SaveFile(savePath_m, output, camera_m.w(), camera_m.h(), k);
        timer.finish();
        std::cout << k + 1 << '\t' << timer.elapsed() << '\t' << countRays << "\n";
    }

    cudaFree(deviceSSAA);
    cudaFree(deviceData);
}

}
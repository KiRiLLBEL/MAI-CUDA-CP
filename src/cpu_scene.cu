#include "cpu_scene.cuh"
#include "polygon.cuh"
#include "triangle.cuh"

#include <iostream>
#include <algorithm>
#include <cstdint>

namespace render
{
void CpuScene::GenerateScene() 
{
    auto texture = Texture(floor_m.TexturePath());
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
}

Vector3f CpuScene::TraceRay(const Ray& ray, uint64_t depth, uint64_t& countOfRays)
{
    if (depth > maxDepth) {
        return Vector3f(0.0, 0.0, 0.0);
    }

    double minT = 1e18;
    int minPolygonIndex = -1;

    for (size_t i = 0; i < polygons_m.size(); ++i) {
        bool intersect = false;
        double t = polygons_m[i].MollerTrumboreIntersection(ray, intersect);
        if (intersect && t < minT) {
            minT = t;
            minPolygonIndex = i;
        }
    }

    if (minPolygonIndex < 0) {
        return Vector3f(0.0, 0.0, 0.0);
    }

    const Polygon& polygon = polygons_m[minPolygonIndex];
    Vector3d intersection = ray.position() + minT * ray.view();
    Vector3f localColor = polygon.ComputeColorInPoint(ray, intersection);

    Vector3f finalColor = Phong(
        ray,
        intersection,
        minPolygonIndex,
        lights_m.data(),
        lights_m.size(), 
        polygons_m.data(),
        polygons_m.size());

    if (polygon.reflection() > 0) {
        Vector3d reflectedDir = Vector3d::reflect(ray.view(), polygon.triangle().n());
        Ray reflectedRay(intersection + SHIFT * reflectedDir, reflectedDir, ray.uid(), polygon.transparency() * ray.color() * localColor);
        ++countOfRays;
        finalColor += polygon.reflection() * TraceRay(reflectedRay, depth + 1, countOfRays);
    }

    if (polygon.transparency() > 0) {
        Ray transparentRay(intersection + SHIFT * ray.view(), ray.view(), ray.uid(), polygon.reflection() * ray.color() * localColor);
        finalColor += polygon.transparency() * TraceRay(transparentRay, depth + 1, countOfRays);
        ++countOfRays;
    }

    return finalColor;
}

int64_t CpuScene::GenerateFrame(int UUID, std::vector<uchar4>& image)
{
    int w = camera_m.wSSAA();
    int h = camera_m.hSSAA();
    std::vector<Vector3f> frameBuffer(w * h, Vector3f(0.0, 0.0, 0.0));

    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / std::tan(camera_m.angle() * M_PI / 360.0);

    Vector3d bz = camera_m.view() - camera_m.position();
    Vector3d bx = bz | Vector3d(0.0, 0.0, 1.0);
    Vector3d by = bx | bz;

    bx.norm();
    by.norm();
    bz.norm();

    uint64_t countOfRays = 0;
    for (int idx = 0; idx < w; ++idx) {
        for (int idy = 0; idy < h; ++idy) {
            Vector3d v(-1.0 + dw * idx, (-1.0 + dh * idy) * h / w, z);
            auto dir = Vector3d::transposeMultiplication(bx, by, bz, v);
            Ray ray(camera_m.position(), dir, idy * w + idx);
            ++countOfRays;
            Vector3f color = TraceRay(ray, 0, countOfRays);
            frameBuffer[idy * w + idx] = color;
        }
    }

    image.resize(w * h);
    for (int i = 0; i < w * h; ++i) {
        frameBuffer[i].range(0.0, 1.0);
        frameBuffer[i] *= 255.0;
        image[i] = make_uchar4(frameBuffer[i].x(), frameBuffer[i].y(), frameBuffer[i].z(), 255);
    }
    return countOfRays;
}

void CpuScene::Render() 
{
    double dt = 2 * M_PI / frames_m;
    camera_m.setSSAA(upscaleFactor);
    std::vector<uchar4> image;
    std::vector<uchar4> output;

    for (uint64_t k = 0; k < frames_m; ++k) {
        Timer timer;
        timer.begin();
        double time = k * dt;
        camera_m.updatePosition(time);
        camera_m.updateView(time);
        auto countRays = GenerateFrame(k, image);
        ApplySSAA(image, output, camera_m.w(), camera_m.h(), upscaleFactor);
        for (uint64_t i = 0; i < camera_m.w() * camera_m.h() / 2; ++i) {
            std::swap(output[i], output[camera_m.w() * camera_m.h() - i - 1]);
        }
        SaveFile(savePath_m, output, camera_m.w(), camera_m.h(), k);
        timer.finish();
        std::cout << k + 1 << '\t' << timer.elapsed() << '\t' << countRays << "\n";
    }
}
}
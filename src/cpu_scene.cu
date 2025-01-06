#include "cpu_scene.cuh"
#include "polygon.cuh"
#include "triangle.cuh"
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

void CpuScene::Render() 
{

}
}
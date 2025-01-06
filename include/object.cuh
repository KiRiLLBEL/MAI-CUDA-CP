#pragma once

#include <polygon.cuh>
#include <vector3.cuh>

#include <istream>

namespace render
{
class Object
{
private:
    Vector3d center_m;
    Vector3f color_m;
    double radius_m;
    double transparency_m;
    double reflection_m;
    uint64_t lightCount_m;
public:
    friend std::istream& operator>>(std::istream& in, Object& obj)
    {
        return in >> obj.center_m
        >> obj.color_m
        >> obj.radius_m
        >> obj.reflection_m
        >> obj.transparency_m
        >> obj.lightCount_m;
    }

    void CreateObjectFromFile(const std::string& path, std::vector<Polygon>& polygons);
};
};
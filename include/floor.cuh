#pragma once

#include <istream>
#include <vector3.cuh>

namespace render
{

class Floor
{
private:
    Vector3d p1_m, p2_m, p3_m, p4_m;
    std::string pathToTexture_m;
    Vector3f color_m;
    double reflection_m;
public:
    Floor() = default;

    Floor(
        const Vector3d& p1,
        const Vector3d& p2,
        const Vector3d& p3,
        const Vector3d& p4,
        const Vector3f& color)
        : p1_m(p1), p2_m(p2), p3_m(p3), p4_m(p4), color_m(color)
    {

    }

    Vector3d p1() const
    {
        return p1_m;
    }

    Vector3d p2() const
    {
        return p2_m;
    }

    Vector3d p3() const
    {
        return p3_m;
    }

    Vector3d p4() const
    {
        return p4_m;
    }

    Vector3f color() const
    {
        return color_m;
    }

    double reflection() const
    {
        return reflection_m;
    }

    std::string TexturePath() const
    {
        return pathToTexture_m;
    }

    friend std::istream& operator>>(std::istream& in, Floor& floor)
    {
        return in >> floor.p1_m
        >> floor.p2_m
        >> floor.p3_m
        >> floor.p4_m
        >> floor.pathToTexture_m
        >> floor.color_m
        >> floor.reflection_m;
    }
};

};
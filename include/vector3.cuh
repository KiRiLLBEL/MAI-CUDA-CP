#pragma once

#include <cmath>
#include <istream>
#include <ostream>

namespace render {

template <typename T>
class Vector3
{

private:
    T x_m;
    T y_m;
    T z_m;

public:
    using V3Type = Vector3<T>;

    __host__ __device__ Vector3(): x_m{0}, y_m{0}, z_m{0}
    {

    }

    __host__ __device__ Vector3(const T& x, const T& y, const T& z) : x_m(x), y_m(y), z_m(z)
    {

    }

    __host__ __device__ Vector3(const V3Type& other) 
        : x_m(other.x_m), 
        y_m(other.y_m), 
        z_m(other.z_m)
    {

    }

    __host__ __device__ V3Type& operator+=(const V3Type& other)
    {
        x_m += other.x_m;
        y_m += other.y_m;
        z_m += other.z_m;
        return *this;
    }

    __host__ __device__ friend V3Type operator+(const V3Type& lhs, const V3Type& rhs)
    {
        return V3Type(lhs) += rhs;
    }

    __host__ __device__ V3Type& operator+=(const T& value)
    {
        x_m += value;
        y_m += value;
        z_m += value;
        return *this;
    }

    __host__ __device__ friend V3Type operator+(const V3Type& lhs, const T& value)
    {
        return V3Type(lhs) += value;
    }

    __host__ __device__ V3Type& operator-=(const V3Type& other)
    {
        x_m -= other.x_m;
        y_m -= other.y_m;
        z_m -= other.z_m;
        return *this;
    }

    __host__ __device__ friend V3Type operator-(const V3Type& lhs, const V3Type& rhs)
    {
        return V3Type(lhs) -= rhs;
    }

    __host__ __device__ V3Type& operator-=(const T& value)
    {
        x_m -= value;
        y_m -= value;
        z_m -= value;
        return *this;
    }

    __host__ __device__ friend V3Type operator-(const V3Type& lhs, const T& value)
    {
        return V3Type(lhs) -= value;
    }

    __host__ __device__ V3Type& operator*=(const V3Type& other)
    {
        x_m *= other.x_m;
        y_m *= other.y_m;
        z_m *= other.z_m;
        return *this;
    }

    __host__ __device__ friend V3Type operator*(const V3Type& lhs, const V3Type& rhs)
    {
        return V3Type(lhs) *= rhs;
    }

    __host__ __device__ V3Type& operator*=(const T& value)
    {
        x_m *= value;
        y_m *= value;
        z_m *= value;
        return *this;
    }

    __host__ __device__ friend V3Type operator*(const T& value, const V3Type& rhs)
    {
        return V3Type(rhs) *= value;
    }

    __host__ __device__ friend V3Type operator*(const V3Type& lhs, const T& value)
    {
        return V3Type(lhs) *= value;
    }

    __host__ __device__ V3Type& operator/=(const V3Type& other)
    {
        x_m /= other.x_m;
        y_m /= other.y_m;
        z_m /= other.z_m;
        return *this;
    }

    __host__ __device__ friend V3Type operator/(const V3Type& lhs, const V3Type& rhs)
    {
        return V3Type(lhs) /= rhs;
    }

    __host__ __device__ V3Type& operator/=(const T& value)
    {
        x_m /= value;
        y_m /= value;
        z_m /= value;
        return *this;
    }

    __host__ __device__ friend V3Type operator/(const V3Type& lhs, const T& value)
    {
        return V3Type(lhs) /= value;
    }

    // dot
    __host__ __device__ friend T operator^(const V3Type& lhs, const V3Type& rhs)
    {
        return lhs.x_m * rhs.x_m + lhs.y_m * rhs.y_m + lhs.z_m * rhs.z_m;
    }

    // cross product
    __host__ __device__ friend V3Type operator|(const V3Type& lhs, const V3Type& rhs)
    {
        return {
            lhs.y_m * rhs.z_m - lhs.z_m * rhs.y_m, 
            lhs.z_m * rhs.x_m - lhs.x_m * rhs.z_m,
            lhs.x_m * rhs.y_m - lhs.y_m * rhs.x_m
        };
    }

    __host__ __device__ V3Type& operator&=(const V3Type& other)
    {
        atomicAdd(&x_m, other.x_m);
        atomicAdd(&y_m, other.y_m);
        atomicAdd(&z_m, other.z_m);
        return *this;
    }

    __host__ __device__ double length() const
    {
        return sqrt(*this ^ *this);
    }

    __host__ __device__ void norm()
    {
        auto len = length();
        *this /= len;
    }

    __host__ __device__ static V3Type reflect(const V3Type& lhs, const V3Type& rhs)
    {
        V3Type reflected = lhs - 2 * (rhs ^ lhs) * rhs;
        reflected.norm();
        return reflected;
    }

    __host__ __device__ static V3Type transposeMultiplication(
        const V3Type& a,
        const V3Type& b,
        const V3Type& c,
        const V3Type& v)
    {
        return {a.x() * v.x() + b.x() * v.y() + c.x() * v.z(),
                a.y() * v.x() + b.y() * v.y() + c.y() * v.z(),
                a.z() * v.x() + b.z() * v.y() + c.z() * v.z()};
    }

    __host__ friend std::istream& operator>>(std::istream& in, V3Type& vec)
    {
        return in >> vec.x_m >> vec.y_m >> vec.z_m;
    }

    __host__ friend std::ostream& operator<<(std::ostream& out, V3Type& vec)
    {
        return out << '[' << vec.x_m << ", " << vec.y_m << ", " << vec.z_m << ']';
    }

    __host__ __device__ T x() const
    {
        return x_m;
    }

    __host__ __device__ T y() const
    {
        return y_m;
    }

    __host__ __device__ T z() const
    {
        return z_m;
    }

    __host__ __device__ static T setRange(const T value, const T start, const T end)
    {
        if (value > end)
        {
            return end;
        }
        if (value < start)
        {
            return start;
        }
        return value;
    }

    __host__ __device__ void range(const T start, const T end)
    {
        x_m = setRange(x_m, start, end);
        y_m = setRange(y_m, start, end);
        z_m = setRange(z_m, start, end);
    }
};

using Vector3d = Vector3<double>;
using Vector3l = Vector3<int>;
using Vector3f = Vector3<float>;
using Vector3ll = Vector3<long long>;
using Vector3ull = Vector3<unsigned long long>;

}; // namespace::render
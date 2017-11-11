#pragma once

#include <cmath>
#include <limits>

namespace EdgeDetector {

struct CurvePoint {
    CurvePoint():
        x(std::numeric_limits<float>::quiet_NaN()),
        y(std::numeric_limits<float>::quiet_NaN()),
        valid(false){}
    CurvePoint(float x, float y, bool valid):
        x(x), y(y), valid(valid)
    {}

    float inline distance(CurvePoint const& other) const {
        return std::hypotf(other.x-x, other.y-y);
    }

    CurvePoint inline operator-(CurvePoint const& rhs) const {
        return CurvePoint(x-rhs.x, y-rhs.y, valid&&rhs.valid);
    }

    float x;
    float y;
    bool valid;
};

}

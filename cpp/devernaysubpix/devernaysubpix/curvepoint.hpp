#pragma once

namespace EdgeDetector {

struct CurvePoint {
    CurvePoint():
        x(0), y(0), valid(false){}
    CurvePoint(float x, float y, bool valid):
        x(x), y(y), valid(valid)
    {}

    float x;
    float y;
    bool valid;
};

}

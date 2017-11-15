#pragma once

#include <cmath>
#include <limits>

namespace EdgeDetector {

/**
 * @brief The CurvePoint struct is used for designating coordinates in image space, possibly linked on a curve.
 */
struct CurvePoint {
    /**
     * @brief Creates a new invalid CurvePoint instance with its coordinates set to not-a-number
     */
    CurvePoint()
        : x(std::numeric_limits<float>::quiet_NaN()),
          y(std::numeric_limits<float>::quiet_NaN()), valid(false) {}
    /**
     * @brief Creates a new CurvePoint instance with coordinates and validity flag
     * @param x The point location on x-axis
     * @param y The point location on y-axis
     * @param valid The validity flag checked in linking algorithm
     */
    CurvePoint(float x, float y, bool valid) : x(x), y(y), valid(valid) {}

    /**
     * @brief Computes metric distance between this and another point
     * @param other The other point
     * @return The distance between both points
     */
    float inline distance(CurvePoint const &other) const {
        return std::hypotf(other.x - x, other.y - y);
    }

    CurvePoint inline operator-(CurvePoint const &rhs) const {
        return CurvePoint(x - rhs.x, y - rhs.y, valid && rhs.valid);
    }

    float x;
    float y;
    bool valid;
};

} // namespace EdgeDetector

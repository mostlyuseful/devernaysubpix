#pragma once

#include "curvepoint.hpp"
#include "linkmap.hpp"
#include "neighborhoodbitmap.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv/cv.hpp>
#include <stdexcept>

namespace EdgeDetector {

struct PartialImageGradients {
    cv::Mat horz;
    cv::Mat vert;


    /**
     * @brief Evaluates gradient tuple at location (row,col)
     * @param row Row index
     * @param col Column index
     * @return Gradient value as CurvePoint with valid:=false
     */
    template <typename T>
    CurvePoint inline at(int const row, int const col) const {
        return CurvePoint(horz.at<float>(row, col), vert.at<float>(row, col),
                          false);
    }

    /**
     * @brief Evaluates gradient tuple at location of point. Coordinates are rounded by casting to int
     * @param p The point
     * @return Gradient value as CurvePoint with valid:=false
     */
    template <typename T>
    CurvePoint inline at(CurvePoint const &p) const {
        return at<T>(static_cast<int>(p.y), static_cast<int>(p.x));
    }

    /**
     * @brief Computes gradient magnitude at location of point. Coordinates are rounded by casting to int
     * @param p The point
     * @return The gradient magnitude
     */
    template <typename T>
    inline T magnitude(CurvePoint const &p) const {
        int const row = static_cast<int>(p.y);
        int const col = static_cast<int>(p.x);
        T const dx = horz.at<T>(row, col);
        T const dy = vert.at<T>(row, col);
        return std::hypot(dx, dy);
    }

    /**
     * @brief Computes magnitude for all points in gradient field
     * @return The magnitudes of whole gradient field
     */
    cv::Mat inline magnitude() const {
        cv::Mat magn;
        cv::magnitude(horz, vert, magn);
        return magn;
    }

    /**
     * @brief Thresholds gradient magnitude, returns !=0 where local_magnitude > min_magnitude
     * @param min_magnitude The lower magnitude threshold. All gradient magnitudes higher than this will be set to non-zero
     * @return A mask image with matching locations set to !=0
     */
    cv::Mat inline threshold(double min_magnitude) const {
        return this->magnitude() > min_magnitude;
    }
};

/**
 * @brief Computes gradient field for image using a gaussian smoothing kernel
 * @param image The input image
 * @param sigma Sigma of gaussian smoothing kernel
 * @return The gradient field for the input image
 */
inline PartialImageGradients image_gradient(cv::Mat image, double sigma) {
    if(image.channels()>1) {
        throw std::runtime_error("image must have exactly one channel");
    }
    // Convert to float to prevent integer rounding errors
    cv::Mat image32f;    
    image.convertTo(image32f, CV_32F);

    // Smooth image before derivative computation to reduce spurious kinks
    cv::Mat smoothed;
    cv::GaussianBlur(image32f, smoothed, cv::Size(0, 0), sigma);

    // Compute derivative with optimized sobel operator
    cv::Mat grad_horz;
    cv::Mat grad_vert;
    cv::Sobel(smoothed, grad_horz, CV_32F, 1, 0);
    cv::Sobel(smoothed, grad_vert, CV_32F, 0, 1);

    return PartialImageGradients{grad_horz, grad_vert};
}

/**
 * @brief The PossibleCurvePoint struct bundles a point and signifies whether it has local extremal gradient magnitude
 */
struct PossibleCurvePoint {
    bool isLocalExtremum;
    CurvePoint point;
};

inline PossibleCurvePoint
compute_single_edge_point(PartialImageGradients const &gradients, int row,
                          int col) {

    auto const mag = [&gradients](int row, int col) -> float {
        return std::hypotf(gradients.horz.at<float>(row, col),
                           gradients.vert.at<float>(row, col));
    };

    float const center_mag = mag(row, col);
    float const left_mag = mag(row, col - 1);
    float const right_mag = mag(row, col + 1);
    float const top_mag = mag(row - 1, col);
    float const bottom_mag = mag(row + 1, col);

    float const abs_gx = std::abs(gradients.horz.at<float>(row, col));
    float const abs_gy = std::abs(gradients.vert.at<float>(row, col));

    int theta_x = 0;
    int theta_y = 0;
    if ((left_mag < center_mag) && (center_mag >= right_mag) &&
        (abs_gx >= abs_gy)) {
        theta_x = 1;
    } else if ((top_mag < center_mag) && (center_mag >= bottom_mag) &&
               (abs_gx <= abs_gy)) {
        theta_y = 1;
    }
    if (theta_x || theta_y) {
        float const a = mag(row - theta_y, col - theta_x);
        float const b = mag(row, col);
        float const c = mag(row + theta_y, col + theta_x);
        float const lamda = (a - c) / (2 * (a - (2 * b) + c));
        /*if (lamda>1 || lamda < -1) {
            return PossibleCurvePoint{false, {}};
        }*/
        float const ex = col + lamda * theta_x;
        float const ey = row + lamda * theta_y;
        return PossibleCurvePoint{true, CurvePoint(ex, ey, false)};
    } else {
        return PossibleCurvePoint{false, {}};
    }
}

inline std::vector<CurvePoint>
compute_edge_points(PartialImageGradients gradients, cv::Mat mask) {

    if (gradients.horz.size() != gradients.vert.size()) {
        throw std::runtime_error("Image gradients differ in size");
    }

    if (gradients.horz.size() != mask.size()) {
        throw std::runtime_error("Mask and image gradients differ in size");
    }

    if (gradients.horz.channels() != 1) {
        throw std::runtime_error("Horizontal image must have single channel");
    }

    if (gradients.vert.channels() != 1) {
        throw std::runtime_error("Vertical image must have single channel");
    }

    int const rows = gradients.horz.rows;
    int const columns = gradients.horz.cols;

    if ((rows < 3) || (columns < 3)) {
        throw std::runtime_error("Input must be at least 3x3 pixels big");
    }

    std::vector<CurvePoint> edges;

    // TODO: Parallelize
    for (int row = 1; row < (rows - 1); ++row) {
        for (int col = 1; col < (columns - 1); ++col) {
            if (mask.at<uint8_t>(row, col)) {
                PossibleCurvePoint p =
                    compute_single_edge_point(gradients, row, col);
                if (p.isLocalExtremum) {
                    if (p.point.x < 0 || p.point.y < 0) {
                        throw std::runtime_error("BELOW 0");
                    }
                    if (std::isnan(p.point.x) || std::isnan(p.point.y)) {
                        throw std::runtime_error("NAN");
                    }
                    edges.push_back(p.point);
                }
            }
        }
    }

    return edges;
}

struct NearestNeighborhoodPoints {
    bool forward_valid;
    bool backward_valid;
    CurvePoint forward;
    CurvePoint backward;
};

/**
 * @brief Computes vector perpendicular to input vector
 * @param p The input vector
 * @return The perpendicular vector with same validity flag as input
 */
CurvePoint inline perpendicular_vec(CurvePoint const &p) {
    return CurvePoint(p.y, -p.x, p.valid);
}

/**
 * @brief Computes dot product (scalar product) between two points
 * @param p1 Left-hand point
 * @param p2 Right-hand point
 * @return The dot product of p1 and p2
 */
float inline dot(CurvePoint const &p1, CurvePoint const &p2) {
    return (p1.x * p2.x) + (p1.y * p2.y);
}

NearestNeighborhoodPoints inline find_nearest_forward_and_backward(
    std::vector<CurvePoint> const &neighborhood,
    PartialImageGradients const &gradients, CurvePoint const &reference) {

    // Default: invalid points
    NearestNeighborhoodPoints out{false, false, CurvePoint(), CurvePoint()};

    if (neighborhood.size() == 0) {
        return out;
    }

    // Gradient values at g(e.x, e.y)
    CurvePoint ge(gradients.at<float>(reference));

    float min_distance_forward = std::numeric_limits<float>::max();
    float min_distance_backward = std::numeric_limits<float>::max();
    for (auto const &p : neighborhood) {
        // Gradient values at g(n.x, n.y)
        CurvePoint gn(gradients.at<float>(p));
        float distance = reference.distance(p);
        auto const angle_lower_than_90_degrees = dot(ge, gn) > 0;
        if (!angle_lower_than_90_degrees) {
            continue;
        }

        auto const angle_perp = dot(p - reference, perpendicular_vec(ge));
        bool const is_forward = angle_perp > 0;
        bool const is_backward = angle_perp < 0;
        if (is_forward) {
            out.forward_valid = true;
            if (distance < min_distance_forward) {
                out.forward = p;
                min_distance_forward = distance;
            }
        } else if (is_backward) {
            out.backward_valid = true;
            if (distance < min_distance_backward) {
                out.backward = p;
                min_distance_backward = distance;
            }
        }
    }
    return out;
}

NeighborhoodBitmap inline precompute_neighborhood_bitmap(
    cv::Size const imageSize, std::vector<CurvePoint> const &edges) {
    NeighborhoodBitmap bitmap(imageSize);
    for (auto const &e : edges) {
        bitmap.set(e);
    }
    return bitmap;
}

std::vector<CurvePoint> inline get_neighborhood_bitmap(
    NeighborhoodBitmap const &bitmap, CurvePoint const &p,
    unsigned int const max_distance) {
    int const px = static_cast<int>(p.x);
    int const py = static_cast<int>(p.y);
    int const max_distancei = static_cast<int>(max_distance);
    std::vector<CurvePoint> out;
    for (int row = py - max_distancei; row <= py + max_distancei; ++row) {
        for (int col = px - max_distancei; col <= px + max_distancei; ++col) {
            if (bitmap.has(row, col)) {
                out.push_back(bitmap.get(row, col));
            }
        }
    }
    return out;
}

std::vector<CurvePoint> inline get_neighborhood_direct(
    std::vector<CurvePoint> const &edges, CurvePoint const &p,
    float max_distance) {
    // TODO: Optimize by precomputing neighborhood and put into 2d array
    std::vector<CurvePoint> hood;
    for (auto const &edge : edges) {
        if (std::abs(edge.x - p.x) <= max_distance &&
            std::abs(edge.y - p.y) <= max_distance) {
            hood.push_back(edge);
        }
    }
    return hood;
}

LinkMap inline chain_edge_points(std::vector<CurvePoint> const &edges,
                                 PartialImageGradients const &gradients) {
    LinkMap links;
    auto const neighborhood_bitmap =
        precompute_neighborhood_bitmap(gradients.horz.size(), edges);
    for (auto const &e : edges) {
        auto const neighborhood =
            get_neighborhood_bitmap(neighborhood_bitmap, e, 2);
        // auto const neighborhood = get_neighborhood_direct(edges, e, 2);
        auto const nearest =
            find_nearest_forward_and_backward(neighborhood, gradients, e);
        if (nearest.forward_valid) {
            auto const &f = nearest.forward;
            // L6
            if (!links.hasRight(f) ||
                (e.distance(f) < links.getByRight(f).first.distance(f))) {
                links.unlinkByRight(f);
                links.unlinkByLeft(e);
                links.link(e, f);
            }
        }
        if (nearest.backward_valid) {
            auto const &b = nearest.backward;
            // L10
            if (!links.hasLeft(b) ||
                (b.distance(e) < b.distance(links.getByLeft(b).second))) {
                links.unlinkByLeft(b);
                links.unlinkByRight(e);
                links.link(b, e);
            }
        }
    }
    return links;
}

using Chain = std::vector<CurvePoint>;
using Chains = std::vector<Chain>;

Chains inline thresholds_with_hysteresis(std::vector<CurvePoint> &edges,
                                         LinkMap &links,
                                         PartialImageGradients const &grads,
                                         float const high_threshold,
                                         float const low_threshold) {
    Chains chains;
    // Ensure all edges are invalid at first so all edges are considered for
    // chaining
    for (auto &e : edges) {
        e.valid = false;
    }

    for (auto &e : edges) {
        if (!e.valid && grads.magnitude<float>(e) >= high_threshold) {
            Chain forward;
            Chain backward;
            e.valid = true;
            auto f = e;
            while (links.hasLeft(f) && !(links.getByLeft(f).second.valid) &&
                   (grads.magnitude<float>(links.getByLeft(f).second) >=
                    low_threshold)) {
                auto n = links.getByLeft(f).second;
                n.valid = true;
                links.replace(f, n);
                f = n;
                forward.push_back(f);
            }
            auto b = e;
            while (links.hasRight(b) && !(links.getByRight(b).first.valid) &&
                   (grads.magnitude<float>(links.getByRight(b).first) >=
                    low_threshold)) {
                auto n = links.getByRight(b).first;
                n.valid = true;
                links.replace(n, b);
                b = n;
                backward.insert(backward.begin(), b);
            }
            Chain chain;
            std::copy(backward.begin(), backward.end(),
                      std::back_inserter(chain));
            chain.push_back(e);
            std::copy(forward.begin(), forward.end(),
                      std::back_inserter(chain));
            if (chain.size() > 1) {
                chains.push_back(std::move(chain));
            }
        }
    }
    return chains;
}

} // namespace EdgeDetector

#pragma once

#include "curvepoint.hpp"
#include "linkmap.hpp"

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <opencv/cv.hpp>

namespace EdgeDetector {

struct DerivativeGaussianKernels {
    cv::Mat horz;
    cv::Mat vert;
};

struct PartialImageGradients {
    cv::Mat horz;
    cv::Mat vert;
};

inline DerivativeGaussianKernels
derivative_gaussian_kernels(double sigma, uint max_ksize = 64) {
    int const min_ksize = 3; // Pixels
    int const ksize = std::max<int>(
                min_ksize, static_cast<int>(std::min<double>(
                                                2 * (((sigma - 0.8) / 0.3) + 1) + 1, max_ksize)));
    cv::Mat gauss_kernel_horz = cv::getGaussianKernel(ksize, sigma, CV_32F);
    cv::Mat gauss_kernel_horz_deriv;
    cv::Mat diff_kernel{0.5, 0.0, -0.5};
    // Convolving the gaussian kernel with a sobel kernel results in a first-order
    // Gaussian derivative filter
    cv::filter2D(gauss_kernel_horz, gauss_kernel_horz_deriv, -1, diff_kernel,
    {-1, -1}, 0, cv::BORDER_ISOLATED);
    cv::Mat gauss_kernel_vert_deriv = gauss_kernel_horz_deriv.t();
    return DerivativeGaussianKernels{gauss_kernel_horz_deriv,
                gauss_kernel_vert_deriv};
}

inline PartialImageGradients image_gradient(cv::Mat image, double sigma) {
    auto const kernels = derivative_gaussian_kernels(sigma);
    cv::Mat grad_horz;
    cv::Mat grad_vert;
    cv::filter2D(image, grad_horz, -1, kernels.horz);
    cv::filter2D(image, grad_vert, -1, kernels.vert);
    return PartialImageGradients{grad_horz, grad_vert};
}

struct PossibleCurvePoint{
    bool isLocalExtremum;
    CurvePoint point;
};

inline PossibleCurvePoint compute_single_edge_point(PartialImageGradients const& gradients, int row, int col) {

    auto const mag = [&gradients](int row, int col) -> float {
        return std::hypotf(gradients.horz.at<float>(row,col),
                           gradients.vert.at<float>(row,col));
    };

    float const center_mag = mag(row, col);
    float const left_mag = mag(row, col - 1);
    float const right_mag = mag(row, col + 1);
    float const top_mag = mag(row - 1, col);
    float const bottom_mag = mag(row + 1, col);

    float const abs_gx = std::abs(gradients.horz.at<float>(row,col));
    float const abs_gy = std::abs(gradients.vert.at<float>(row,col));

    int theta_x = 0;
    int theta_y = 0;
    if ((left_mag < center_mag >= right_mag) && abs_gx >= abs_gy) {
        theta_x = 1;
    }else if((top_mag < center_mag >= bottom_mag) && abs_gx <= abs_gy){
        theta_y = 1;
    }
    if(theta_x || theta_y) {
        float const a = mag(row - theta_y, row - theta_x);
        float const                 b = mag(row, col);
                    float const     c = mag(row + theta_y, col + theta_x);
                        float const lamda = (a - c) / (2 * (a - 2 * b + c));
                        float const ex =col + lamda * theta_x;
                        float const ey = row + lamda * theta_y;
                        return PossibleCurvePoint{ true, CurvePoint(ex,ey,false)};
    }else {
        return PossibleCurvePoint{false,{}};
    }

}

inline std::vector<CurvePoint> compute_edge_points(PartialImageGradients gradients, cv::Mat mask) {

    if(gradients.horz.size() != gradients.vert.size()) {
        throw std::runtime_error("Image gradients differ in size");
    }

    if(gradients.horz.size() != mask.size()) {
        throw std::runtime_error("Mask and image gradients differ in size");
    }

    if(gradients.horz.channels()!=2) {
        throw std::runtime_error("Images have to be two-dimensional");
    }

    int const rows = gradients.horz.rows;
    int const columns = gradients.horz.cols;

    if((rows < 3) || (columns<3)){
        throw std::runtime_error("Input must be at least 3x3 pixels big");
    }

    std::vector<CurvePoint> edges;

    // TODO: Parallelize
    for(int row=1;row<(rows-1);++row){
        for(int col=1;col<(columns-1);++col){
            if(mask.at<uint8_t>(row,col)) {
                PossibleCurvePoint p = compute_single_edge_point(gradients, row, col);
                if(p.isLocalExtremum){
                    edges.push_back(std::move(p.point));
                }
            }
        }
    }

    return edges;
}

LinkMap chain_edge_points(std::vector<CurvePoint> const& edges, PartialImageGradients& gradients) {
    // TODO: Optimize by precomputing neighborhood and put into 2d array
    auto const neighborhood = [&edges](float const row, float const col, float const max_distance) {
        std::vector<CurvePoint> hood;
        for(auto const& edge : edges) {
            if (std::abs(edge.x - col) <= max_distance && std::abs(edge.y - row) <= max_distance) {
                hood.push_back(edge);
            }
        }
        return hood;
    };

    LinkMap links;

    for(auto const& edge:edges) {
        // TODO: Line 89
#error "HERE"
    }
}

} // namespace EdgeDetector

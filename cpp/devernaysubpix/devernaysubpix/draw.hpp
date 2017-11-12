#pragma once

#include <cmath>
#include <cv.hpp>

namespace Draw {

cv::Vec3b inline blend(cv::Vec3b const &src, cv::Vec3b const &dst, double const alpha) {
  cv::Scalar out = (src * alpha) + (dst * (1 - alpha));
  return cv::Vec3b(out[0],out[1],out[2]);
}

void inline pixel_aa(cv::Mat &canvas, cv::Point2d const location, cv::Vec3b const &color) {
  for (int rounded_x = std::floor(location.x);
       rounded_x <= std::ceil(location.x); ++rounded_x) {
    for (int rounded_y = std::floor(location.y);
         rounded_y <= std::ceil(location.y); ++rounded_y) {
      double const percent_x = 1 - std::abs(location.x - rounded_x);
      double const percent_y = 1 - std::abs(location.y - rounded_y);
      double const percent = percent_x * percent_y;
      auto const bg_color = canvas.at<cv::Vec3b>(location);
      auto const blended = blend(color, bg_color, percent);
      canvas.at<cv::Vec3b>(rounded_y, rounded_x) = blended;
    }
  }
}

} // namespace Draw

#include "draw.hpp"
#include "edge_detector.hpp"

#include <cv.hpp>
#include <fstream>
#include <iostream>
#include <vector>

cv::Mat draw_edge_points(cv::Mat const &background_image,
                         std::vector<EdgeDetector::CurvePoint> const &edges) {
  cv::Mat canvas;
  cv::cvtColor(background_image, canvas, CV_GRAY2BGR);
  for (auto const &e : edges) {
    Draw::pixel_aa(canvas, cv::Point2d(e.x, e.y), {0, 0, 255});
  }
  return canvas;
}

cv::Mat draw_chains(cv::Mat const &background_image,
                    EdgeDetector::Chains const &chains) {

  std::vector<cv::Vec3b> const colors = {{255, 0, 0},   {0, 255, 0},
                                         {0, 0, 255},   {255, 255, 0},
                                         {0, 255, 255}, {255, 0, 255}};
  size_t color_idx = 0;

  cv::Mat canvas;
  cv::cvtColor(background_image, canvas, CV_GRAY2BGR);
  for (auto const &chain : chains) {
    auto const &color = colors.at(color_idx);
    color_idx = (color_idx + 1) % colors.size();
    for (size_t i = 1; i < chain.size(); ++i) {
      auto const pt1 = chain.at(i - 1);
      auto const pt2 = chain.at(i);
      cv::arrowedLine(canvas, cv::Point2f(pt1.x, pt1.y),
                      cv::Point2f(pt2.x, pt2.y), color, 1, CV_AA);
    }
  }
  return canvas;
}

cv::Mat generateSmoothedCircle() {
  cv::Mat canvas(127, 127, CV_8U);
  canvas = 0;
  cv::circle(canvas, cv::Point(64, 64), 40, cv::Scalar(255, 255, 255),
             CV_FILLED);
  cv::GaussianBlur(canvas, canvas, cv::Size(0, 0), 3.0);
  return canvas;
}

int main(int argc, char *argv[]) {
  namespace E = EdgeDetector;

  cv::Mat source = generateSmoothedCircle();
  auto const grads = E::image_gradient(source, 1.0);
  auto const mask = grads.threshold(10);
  auto edge_points = E::compute_edge_points(grads, mask);
  auto links = E::chain_edge_points(edge_points, grads);
  auto chains =
      E::thresholds_with_hysteresis(edge_points, links, grads, 1, 0.1f);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(mask.clone(), contours, hierarchy, CV_RETR_CCOMP,
                   CV_CHAIN_APPROX_NONE);

  cv::Mat contours_img(source.size(), CV_8U);
  contours_img = 0;
  cv::drawContours(contours_img, contours, -1, 255, CV_FILLED, 8, hierarchy);
  cv::dilate(contours_img, contours_img, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8,8)));

  cv::namedWindow("contours");
  cv::imshow("contours", contours_img);

  /*cv::namedWindow("source");
  cv::imshow("source", source);

  cv::namedWindow("mask");
  cv::imshow("mask", mask);

  cv::namedWindow("edge_points");
  cv::imshow("edge_points", draw_edge_points(source, edge_points));

  cv::namedWindow("chains");
  cv::imshow("chains", draw_chains(source, chains));*/

  cv::waitKey();

  return 0;
}

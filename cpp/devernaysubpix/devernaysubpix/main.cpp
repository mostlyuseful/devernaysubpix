#include "mainwindow.h"
#include <QApplication>

#include "edge_detector.hpp"
#include <cv.hpp>
#include <fstream>
#include <iostream>

void write_edge_points(cv::Mat const &background_image,
                       std::vector<EdgeDetector::CurvePoint> const &edges) {
  cv::Mat canvas;
  cv::cvtColor(background_image, canvas, CV_GRAY2BGR);
  for (auto const &e : edges) {
    canvas.at<cv::Vec3b>(cv::Point(e.x, e.y)) = {0, 0, 255};
    // cv::circle(canvas, cv::Point(e.x,e.y), 3, {0,0,255});
  }
  cv::imwrite("edges.tif", canvas);

  std::ofstream f("edges.txt", std::ios_base::out);
  for (auto const &e : edges) {
    f << e.x << " " << e.y << std::endl;
  }
}

void write_chains(cv::Mat const &background_image,
                  EdgeDetector::Chains const &chains) {

  std::vector<cv::Vec3b> const colors = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255},
                                         {255,255,0},{0,255,255},{255,0,255}};
  size_t color_idx = 0;

  cv::Mat canvas;
  cv::cvtColor(background_image, canvas, CV_GRAY2BGR);
  for (auto const &chain : chains) {
    auto const &color = colors.at(color_idx);
    color_idx = (color_idx + 1) % colors.size();
    for (size_t i = 1; i < chain.size(); ++i) {
      auto const pt1 = chain.at(i - 1);
      auto const pt2 = chain.at(i);
      cv::arrowedLine(canvas, cv::Point(pt1.x, pt1.y), cv::Point(pt2.x, pt2.y),
                      color);
    }
  }
  cv::imwrite("chains.tif", canvas);
}

int main(int argc, char *argv[]) {

  namespace E = EdgeDetector;

  auto g = cv::imread("zebra_256.tif", 0);
  // auto g = cv::imread("kreis.png", 0);

  auto grads = E::image_gradient(g, 1.0);
  auto mag = grads.magnitude();
  double min, max;
  cv::minMaxLoc(mag, &min, &max);
  cv::Mat mag_u8;
  mag.convertTo(mag_u8, CV_8U, 255 / (max - min), -min);
  cv::imwrite("magn.tif", mag_u8);

  auto mask = grads.threshold(50);
  cv::imwrite("mask.tif", mask);

  auto edges = E::compute_edge_points(grads, mask);
  write_edge_points(g, edges);

  auto links = E::chain_edge_points(edges, grads);
  auto chains = E::thresholds_with_hysteresis(edges, links, grads, 1, 0.1f);

  write_chains(g, chains);

  return 0;

  /*QApplication a(argc, argv);
  MainWindow w;
  w.show();
  return a.exec();*/
}

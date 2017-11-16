#include "draw.hpp"
#include "edge_detector.hpp"

#include <cv.hpp>
#include <fstream>
#include <iostream>

// Only used in this example, feel free to remove boost depedencies
#include <boost/timer/timer.hpp>

void write_edge_points(cv::Mat const &background_image,
                       std::vector<EdgeDetector::CurvePoint> const &edges) {
    cv::Mat canvas;
    cv::cvtColor(background_image, canvas, CV_GRAY2BGR);
    for (std::vector<EdgeDetector::CurvePoint>::const_iterator e=edges.cbegin();e<edges.cend();++e){
        Draw::pixel_aa(canvas, cv::Point2d(e->x, e->y), cv::Vec3b(0, 0, 255));
    }
    cv::imwrite("edges.tif", canvas);

    std::ofstream f("edges.txt", std::ios_base::out);
    for (std::vector<EdgeDetector::CurvePoint>::const_iterator e=edges.cbegin();e<edges.cend();++e){
        f << e->x << " " << e->y << std::endl;
    }
}

void write_chains(cv::Mat const &background_image,
                  EdgeDetector::Chains const &chains) {
    std::vector<cv::Vec3b> colors;
    colors.push_back(cv::Vec3b(255,0,0));
    colors.push_back(cv::Vec3b(0, 255, 0));
    colors.push_back(cv::Vec3b(0, 0, 255));
    colors.push_back(cv::Vec3b(255, 255, 0));
    colors.push_back(cv::Vec3b(0, 255, 255));
    colors.push_back(cv::Vec3b(255, 0, 255));
    size_t color_idx = 0;

    cv::Mat canvas;
    cv::cvtColor(background_image, canvas, CV_GRAY2BGR);
    for (EdgeDetector::Chains::const_iterator chain=chains.cbegin();chain<chains.cend();++chain){
        cv::Vec3b const &color = colors.at(color_idx);
        color_idx = (color_idx + 1) % colors.size();
        for (size_t i = 1; i < chain->size(); ++i) {
            EdgeDetector::CurvePoint const pt1 = chain->at(i - 1);
            EdgeDetector::CurvePoint const pt2 = chain->at(i);
            cv::arrowedLine(canvas, cv::Point2f(pt1.x, pt1.y),
                            cv::Point2f(pt2.x, pt2.y), color, 1, CV_AA);
        }
    }
    cv::imwrite("chains.tif", canvas);
}

int main(int argc, char *argv[]) {

    namespace E = EdgeDetector;

    cv::Mat g = cv::imread("zebra_256.tif", 0);
    // auto g = cv::imread("kreis.png", 0);
    // auto g = cv::imread("edge.png", 0);
    // auto g = cv::imread("kreis_gross.png", 0);

    boost::timer::auto_cpu_timer *t = new boost::timer::auto_cpu_timer();
    E::PartialImageGradients grads = E::image_gradient(g, 1.0);
    std::cout << "image_gradient:" << std::endl;
    delete t;

    t = new boost::timer::auto_cpu_timer();
    cv::Mat mask = grads.threshold(50);
    std::cout << "grads.threshold:" << std::endl;
    delete t;

    t = new boost::timer::auto_cpu_timer();
    std::vector<E::CurvePoint> edges = E::compute_edge_points(grads, mask);
    std::cout << "compute_edge_points:" << std::endl;
    delete t;

    t = new boost::timer::auto_cpu_timer();
    E::LinkMap links = E::chain_edge_points(edges, grads);
    std::cout << "chain_edge_points:" << std::endl;
    delete t;

    t = new boost::timer::auto_cpu_timer();
    E::Chains chains = E::thresholds_with_hysteresis(edges, links, grads, 1, 0.1f);
    std::cout << "thresholds_with_hysteresis:" << std::endl;
    delete t;

    // Now write out informational data to disk

    // Raw edge magnitude image, normalized to [0..255]
    {
        cv::Mat mag = grads.magnitude();
        double min, max;
        cv::minMaxLoc(mag, &min, &max);
        cv::Mat mag_u8;
        mag.convertTo(mag_u8, CV_8U, 255 / (max - min), -min);
        cv::imwrite("magn.tif", mag_u8);
    }

    cv::imwrite("mask.tif", mask);
    write_edge_points(g, edges);
    write_chains(g, chains);

    return 0;
}

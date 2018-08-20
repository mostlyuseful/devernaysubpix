#include "edge_detector.hpp"

using namespace cv;
using namespace std;
namespace E = EdgeDetector;

void DrawEdges(cv::Mat &rgb, cv::Mat &gray,
               const std::vector<E::CurvePoint> &edge_points,
               const cv::Scalar &color, const int scaleFactor) {
    cv::Mat gray2;

    cv::resize(gray, gray2, gray.size() * scaleFactor, 0, 0, INTER_LINEAR);
    cv::cvtColor(gray2, rgb, CV_GRAY2BGR);

    cv::Point2f offset(scaleFactor / 2. - 0.5, scaleFactor / 2. - 0.5);
    for (size_t i = 0; i < edge_points.size(); i++) {
        cv::Point2f b =
            scaleFactor * Point2f(edge_points[i].x, edge_points[i].y) + offset;
        cv::line(rgb, b, b, color);
    }
}

int main(int argc, char *argv[]) {
    const String keys =
        "{help h usage ? |          | print this message            }"
        "{@image         |          | image for edge detection      }"
        "{@output        |edge.tiff | image for draw edges          }"
        "{data           |          | edges data in txt format      }"
        "{low            |40        | low threshold                 }"
        "{high           |100       | high threshold                }"
        "{alpha          |1.0       | gaussian alpha                }";
    CommandLineParser parser(argc, argv, keys);
    parser.about("subpixel edge detection");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    if (!parser.has("@image")) {
        parser.printMessage();
        return 0;
    }

    String imageFile = parser.get<String>(0);
    String outputFile = parser.get<String>("@output");
    int low = parser.get<int>("low");
    int high = parser.get<int>("high");
    double alpha = parser.get<double>("alpha");

    Mat image = imread(imageFile, IMREAD_GRAYSCALE);
    int64 t0 = getCPUTickCount();
    auto grads = E::image_gradient(image, alpha);
    auto mask = grads.threshold(low);
    auto edges = E::compute_edge_points(grads, mask);
    int64 t1 = getCPUTickCount();
    cout << "execution time is " << (t1 - t0) / (double)getTickFrequency()
         << " seconds" << endl;

    if (parser.has("data")) {
        FileStorage fs(parser.get<String>("data"),
                       FileStorage::WRITE | FileStorage::FORMAT_YAML);
        fs << "edges"
           << "[";
        for (size_t i = 0; i < edges.size(); ++i) {
            fs << "{:";
            fs << "x" << edges[i].x;
            fs << "y" << edges[i].y;
            fs << "}";
        }
        fs << "]";
        fs.release();
    }

    cv::Mat rgb;
    DrawEdges(rgb, image, edges, cv::Scalar(0, 0, 255), 10);

    cv::imwrite(outputFile, rgb);

    return 0;
}

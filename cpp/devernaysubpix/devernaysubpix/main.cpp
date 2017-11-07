#include "mainwindow.h"
#include <QApplication>

#include "edge_detector.hpp"

int main(int argc, char *argv[])
{

    auto g = EdgeDetector::image_gradient(cv::Mat(), 3.141);

    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}

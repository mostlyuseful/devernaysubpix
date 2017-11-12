#pragma once

#include "curvepoint.hpp"

#include <cv.hpp>
#include <memory>

namespace EdgeDetector {

class NeighborhoodBitmap {
public:
  NeighborhoodBitmap(cv::Size const size);
  void set(CurvePoint const& p);
  bool has(unsigned int row, unsigned int col) const;
  CurvePoint get(unsigned int row, unsigned int col) const;

protected:
  int indexForLocation(int row, int col) const;
  bool isValidIndex(int index) const;

protected:
  unsigned int m_rows;
  unsigned int m_columns;
  std::vector<std::unique_ptr<CurvePoint>> m_bitmap;
};

} // namespace EdgeDetector

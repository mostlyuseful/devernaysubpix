#pragma once

#include "curvepoint.hpp"

#include <cv.hpp>
#include <memory>

namespace EdgeDetector {

class NeighborhoodBitmap {
public:
    NeighborhoodBitmap(cv::Size const size):
        m_rows(static_cast<unsigned int>(size.height)),
        m_columns(static_cast<unsigned int>(size.width)),
        m_bitmap(static_cast<size_t>(size.area())) {}
    void set(CurvePoint const& p)
    {
        int const row = static_cast<int>(p.y);
        int const col = static_cast<int>(p.x);
        auto const idx = this->indexForLocation(row, col);
        if(!this->isValidIndex(idx)){
            throw std::runtime_error("Not a valid location");
        }

        // If using >= C++14, please use std::make_unique!
        m_bitmap[static_cast<size_t>(idx)]= std::unique_ptr<CurvePoint>(new CurvePoint(p));
    }
    bool has(unsigned int row, unsigned int col) const
    {
        auto const idx = this->indexForLocation(static_cast<int>(row), static_cast<int>(col));
        if(!this->isValidIndex(idx)){
            return false;
        }
        return static_cast<bool>(m_bitmap[static_cast<size_t>(idx)]);
    }
    CurvePoint get(unsigned int row, unsigned int col) const
    {
        auto const idx = this->indexForLocation(static_cast<int>(row), static_cast<int>(col));
        if(!this->isValidIndex(idx)){
            throw std::runtime_error("Not a valid location");
        }
        return *(m_bitmap[static_cast<size_t>(idx)]);
    }

protected:
    int indexForLocation(int row, int col) const
    {
        return (row * m_columns) + col;
    }
    bool isValidIndex(int index) const
    {
        return (index >= 0) && (index < m_bitmap.size());
    }

protected:
  unsigned int m_rows;
  unsigned int m_columns;
  std::vector<std::unique_ptr<CurvePoint>> m_bitmap;
};

} // namespace EdgeDetector

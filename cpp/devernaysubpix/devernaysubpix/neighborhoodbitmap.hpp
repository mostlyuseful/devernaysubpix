#pragma once

#include "curvepoint.hpp"

#include <cv.hpp>
#include <memory>
#include <cstdint>

namespace EdgeDetector {

/**
 * @brief The NeighborhoodBitmap class stores information about neighbors inside a flat vector
 */
class NeighborhoodBitmap {
  public:
    NeighborhoodBitmap(cv::Size const size)
        : m_rows(static_cast<unsigned int>(size.height)),
          m_columns(static_cast<unsigned int>(size.width)),
          m_bitmap(static_cast<size_t>(size.area())) {}
    /**
     * @brief Stores point in bitmap
     * @param p The point to store
     */
    void set(CurvePoint const &p) {
        int const row = static_cast<int>(p.y);
        int const col = static_cast<int>(p.x);
        int const idx = this->indexForLocation(row, col);
        if (!this->isValidIndex(idx)) {
            throw std::runtime_error("Not a valid location");
        }

        // If using >= C++14, please use std::make_unique!
        m_bitmap[static_cast<size_t>(idx)] =
            std::unique_ptr<CurvePoint>(new CurvePoint(p));
    }

    /**
     * @brief Tests (row,col) location for membership
     * @param row
     * @param col
     * @return True if bitmap contains point for (row,col), false if not
     */
    bool has(int row, int col) const {
        int const idx = this->indexForLocation(row,col);
        if (!this->isValidIndex(idx)) {
            return false;
        }
        return static_cast<bool>(m_bitmap[static_cast<size_t>(idx)]);
    }

    /**
     * @brief Returns point stored at location (row,col)
     * @param row
     * @param col
     * @return The stored point
     * @throw std::runtime_error if location not stored. Test by using has(...) before.
     */
    CurvePoint get(int row, int col) const {
        int const idx = this->indexForLocation(row, col);
        if (!this->isValidIndex(idx)) {
            throw std::runtime_error("Not a valid location");
        }
        return *(m_bitmap[static_cast<size_t>(idx)]);
    }

  protected:
    int indexForLocation(int row, int col) const {
        return (row * static_cast<int>(m_columns)) + col;
    }
    bool isValidIndex(int index) const {
        return (index >= 0) && (index < static_cast<int>(m_bitmap.size()));
    }

  protected:
    unsigned int m_rows;
    unsigned int m_columns;
    std::vector<std::unique_ptr<CurvePoint> > m_bitmap;
};

} // namespace EdgeDetector

#pragma once

#include "curvepoint.hpp"
#include <functional>
#include <unordered_map>

namespace EdgeDetector {

/**
 * @brief The hashCurvePointLocationOnly struct is a function object that hashes
 * only the 'x' and 'y' parts, ignoring 'valid': 'valid' members are not needed
 * while linking
 */
struct hashCurvePointLocationOnly {
    std::size_t inline operator()(CurvePoint const &p) const noexcept {
        std::size_t const h1 = std::hash<float>{}(p.x);
        std::size_t const h2 = std::hash<float>{}(p.y);
        std::size_t const combinedHash = h1 ^ (h2 << 1);
        return combinedHash;
    }
};

/**
 * @brief The equalsCurvePointLocationOnly struct is a function object that compares
 * only the x and y members, ignoring valid
 */
struct equalsCurvePointLocationOnly {
    bool inline operator()(CurvePoint const &p1, CurvePoint const &p2) const
        noexcept {
        return (p1.x == p2.x) && (p1.y == p2.y);
    }
};

using OneSidedMap =
    std::unordered_map<CurvePoint, CurvePoint,
                       hashCurvePointLocationOnly,
                       equalsCurvePointLocationOnly>;

using Link = std::pair<CurvePoint, CurvePoint>;

class LinkMap {
  public:
    LinkMap(){}
    bool has(CurvePoint const &left, CurvePoint const &right) const {
        auto it = m_leftRight.find(left);
        if (it == m_leftRight.cend()) {
            return false;
        }
        return equalsCurvePointLocationOnly()(it->second, right);
    }
    bool hasLeft(CurvePoint const &p) const {
        return m_leftRight.find(p) != m_leftRight.cend();
    }
    bool hasRight(CurvePoint const &p) const {
        return m_rightLeft.find(p) != m_rightLeft.cend();
    }
    Link getByLeft(CurvePoint const &p) const {
        auto it = m_leftRight.find(p);
        return std::make_pair(it->first, it->second);
    }
    Link getByRight(CurvePoint const &p) const {
        auto it = m_rightLeft.find(p);
        return std::make_pair(it->second, it->first);
    }
    void link(CurvePoint const &left, CurvePoint const &right) {
        if (this->hasLeft(left)) {
            throw std::runtime_error("Left already registered");
        }
        if (this->hasRight(right)) {
            throw std::runtime_error("Right already registered");
        }
        m_leftRight.insert(std::make_pair(left, right));
        m_rightLeft.insert(std::make_pair(right, left));
    }
    void unlink(const EdgeDetector::CurvePoint &left,
                const EdgeDetector::CurvePoint &right) {
        m_leftRight.erase(left);
        m_rightLeft.erase(right);
    }
    void
    unlink(std::pair<EdgeDetector::CurvePoint, EdgeDetector::CurvePoint> item) {
        return this->unlink(item.first, item.second);
    }
    void unlinkByLeft(const EdgeDetector::CurvePoint &left) {
        if (!this->hasLeft(left)) {
            return;
        }
        auto const pair = this->getByLeft(left);
        this->unlink(pair);
    }
    void unlinkByRight(const EdgeDetector::CurvePoint &right) {
        if (!this->hasRight(right)) {
            return;
        }
        auto const pair = this->getByRight(right);
        this->unlink(pair);
    }
    void replace(CurvePoint const &left, CurvePoint const &right) {
        this->unlink(left, right);
        this->link(left, right);
    }

  protected:
    OneSidedMap m_leftRight;
    OneSidedMap m_rightLeft;
};

} // namespace EdgeDetector

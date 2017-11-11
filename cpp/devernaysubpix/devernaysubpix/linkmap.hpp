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

struct equalsCurvePointLocationOnly {
  bool inline operator()(CurvePoint const &p1, CurvePoint const &p2) const noexcept {
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
  LinkMap();
  bool has(CurvePoint const& left, CurvePoint const& right) const;
  bool hasLeft(CurvePoint const& p) const;
  bool hasRight(CurvePoint const& p) const;
  Link getByLeft(CurvePoint const& p) const;
  Link getByRight(CurvePoint const& p) const;
  void link(CurvePoint const& left, CurvePoint const& right);
  void unlink(const EdgeDetector::CurvePoint &left, const EdgeDetector::CurvePoint &right);
  void unlink(std::pair<EdgeDetector::CurvePoint,EdgeDetector::CurvePoint> item);
  void unlinkByLeft(const EdgeDetector::CurvePoint &left);
  void unlinkByRight(const EdgeDetector::CurvePoint &right);
  void replace(CurvePoint const& left, CurvePoint const& right);

protected:
  OneSidedMap m_leftRight;
  OneSidedMap m_rightLeft;
};

} // namespace EdgeDetector

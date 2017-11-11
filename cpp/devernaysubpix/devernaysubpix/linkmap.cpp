#include "linkmap.hpp"

EdgeDetector::LinkMap::LinkMap() {}

bool EdgeDetector::LinkMap::has(const EdgeDetector::CurvePoint &left,
                                const EdgeDetector::CurvePoint &right) const {
  auto it = m_leftRight.find(left);
  if (it == m_leftRight.cend()) {
    return false;
  }
  return equalsCurvePointLocationOnly()(it->second, right);
}

bool EdgeDetector::LinkMap::hasLeft(const EdgeDetector::CurvePoint &p) const {
  return m_leftRight.find(p) != m_leftRight.cend();
}

bool EdgeDetector::LinkMap::hasRight(const EdgeDetector::CurvePoint &p) const {
  return m_rightLeft.find(p) != m_leftRight.cend();
}

EdgeDetector::Link
EdgeDetector::LinkMap::getByLeft(const EdgeDetector::CurvePoint &p) const {
  auto it = m_leftRight.find(p);
  return std::make_pair(it->first, it->second);
}

EdgeDetector::Link
EdgeDetector::LinkMap::getByRight(const EdgeDetector::CurvePoint &p) const {
  auto it = m_rightLeft.find(p);
  return std::make_pair(it->second, it->first);
}

void EdgeDetector::LinkMap::link(const EdgeDetector::CurvePoint &left,
                                 const EdgeDetector::CurvePoint &right) {
  if (this->hasLeft(left)) {
    throw std::runtime_error("Left already registered");
  }
  if (this->hasRight(right)) {
    throw std::runtime_error("Right already registered");
  }
  m_leftRight.insert(std::make_pair(left, right));
  m_rightLeft.insert(std::make_pair(right, left));
}

void EdgeDetector::LinkMap::unlink(EdgeDetector::CurvePoint const &left,
                                   EdgeDetector::CurvePoint const &right) {
    m_leftRight.erase(left);
    m_rightLeft.erase(right);
}

void EdgeDetector::LinkMap::unlink(std::pair<EdgeDetector::CurvePoint,EdgeDetector::CurvePoint> item) {
    return this->unlink(item.first, item.second);
}

void EdgeDetector::LinkMap::unlinkByLeft(const EdgeDetector::CurvePoint &left) {
  if (!this->hasLeft(left)) {
    return;
  }
  auto const pair = this->getByLeft(left);
  this->unlink(pair);
}

void EdgeDetector::LinkMap::unlinkByRight(const EdgeDetector::CurvePoint &right) {
  if (!this->hasRight(right)) {
    return;
  }
  auto const pair = this->getByRight(right);
  this->unlink(pair);
}

void EdgeDetector::LinkMap::replace(const EdgeDetector::CurvePoint &left, const EdgeDetector::CurvePoint &right)
{
    this->unlink(left, right);
    this->link(left, right);
}

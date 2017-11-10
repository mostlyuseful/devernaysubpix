#include "linkmap.hpp"


EdgeDetector::LinkMap::LinkMap()
{

}

bool EdgeDetector::LinkMap::has(const EdgeDetector::CurvePoint &left, const EdgeDetector::CurvePoint &right) const
{
    auto it = m_leftRight.find(left);
    if (it==m_leftRight.cend()) {
        return false;
    }
    return equalsCurvePointLocationOnly()(it->second, right);
}

bool EdgeDetector::LinkMap::hasLeft(const EdgeDetector::CurvePoint &p) const
{
    return m_leftRight.find(p) != m_leftRight.cend();
}

bool EdgeDetector::LinkMap::hasRight(const EdgeDetector::CurvePoint &p) const
{
    return m_rightLeft.find(p) != m_leftRight.cend();
}

EdgeDetector::Link EdgeDetector::LinkMap::byLeft(const EdgeDetector::CurvePoint &p) const
{
    auto it = m_leftRight.find(p);
    return std::make_pair(it->first, it->second);
}

EdgeDetector::Link EdgeDetector::LinkMap::byRight(const EdgeDetector::CurvePoint &p) const
{
    auto it = m_rightLeft.find(p);
    return std::make_pair(it->second, it->first);
}

void EdgeDetector::LinkMap::link(const EdgeDetector::CurvePoint &left, const EdgeDetector::CurvePoint &right)
{
    if(this->hasLeft(left)) {
        throw std::runtime_error("Left already registered");
    }
    if(this->hasRight(right)) {
        throw std::runtime_error("Right already registered");
    }
    m_leftRight.insert(std::make_pair(left, right));
    m_rightLeft.insert(std::make_pair(right, left));
}

void EdgeDetector::LinkMap::unlink(const EdgeDetector::CurvePoint &left, const EdgeDetector::CurvePoint &right)
{

}

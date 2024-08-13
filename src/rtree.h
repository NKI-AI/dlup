#ifndef RTREE_H
#define RTREE_H

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>

#include "exceptions.h"
#include "geometry.h"
#include "geometry_utils.h"
#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// #define DLUPDEBUG

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
namespace py = pybind11;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostBox = bg::model::box<BoostPoint>;
using BoostRing = bg::model::ring<BoostPoint>;
using BoostLineString = bg::model::linestring<BoostPoint>;
using BoostMultiPolygon = bg::model::multi_polygon<BoostPolygon>;

class GeometryCollection; // Forward declaration

class RTreeWrapper {
public:
    using RTreeType = bgi::rtree<std::pair<BoostBox, size_t>, bgi::quadratic<16>>;

    RTreeWrapper(GeometryCollection *geometryCollection)
        : geometryCollection(geometryCollection), rTreeInvalidated(true) {}

    void insert(const BoostBox &box, size_t index) {
        rtree.insert(std::make_pair(box, index));
        rTreeInvalidated = false;
    }

    template <typename QueryType, typename OutputIterator>
    void query(const QueryType &query, OutputIterator out) {
        if (rTreeInvalidated) {
            rebuild();
        }
        rtree.query(query, out);
    }

    void invalidate() { rTreeInvalidated = true; }

    void clear() {
        rtree.clear();
        rTreeInvalidated = true;
    }

    bool isInvalidated() const { return rTreeInvalidated; }

    void rebuild();

private:
    RTreeType rtree;
    bool rTreeInvalidated;
    GeometryCollection *geometryCollection; // Pointer to GeometryCollection
};

#endif // RTREE_H
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>

#include "exceptions.h"
#include "geometry.h"
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

namespace py = pybind11;

class FactoryGuard {
public:
    FactoryGuard(py::function &factory_ref, py::function new_factory)
        : factory_ref_(factory_ref), original_factory_(factory_ref) {
        factory_ref_ = new_factory;
    }

    ~FactoryGuard() { factory_ref_ = original_factory_; }

private:
    py::function &factory_ref_;
    py::function original_factory_;
};

class RTreeWrapper {
public:
    using RTreeType = bgi::rtree<std::pair<BoostBox, size_t>, bgi::quadratic<16>>;

    RTreeWrapper() : rTreeInvalidated(true) {}

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

private:
    void rebuild() {
        // Rebuild the tree based on existing polygons and points (if available)
        // This is left as a placeholder since the actual data to rebuild with is managed externally
        rtree.clear();
        // Example: Add logic to rebuild rtree using stored polygons and points
        rTreeInvalidated = false;
    }

    RTreeType rtree;
    bool rTreeInvalidated;
};

class BaseGeometry {
public:
    virtual ~BaseGeometry() = default;
    std::unordered_map<std::string, py::object> parameters;

    void setField(const std::string &name, py::object value) { parameters[name] = value; }

    std::optional<py::object> getField(const std::string &name) const {
        if (auto it = parameters.find(name); it != parameters.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    auto getFields() const {
        std::vector<std::string> fieldNames;
        fieldNames.reserve(parameters.size());
        std::transform(parameters.begin(), parameters.end(), std::back_inserter(fieldNames),
                       [](const auto &param) { return param.first; });
        return fieldNames;
    }

    std::uintptr_t getPointerId() const { return reinterpret_cast<std::uintptr_t>(this); }
    virtual std::string toWkt() const = 0; // Force derived classes to provide the WKT

protected:
    template <typename GeometryType>
    std::string convertToWkt(const GeometryType &geometry) const {
        std::stringstream ss;
        ss << boost::geometry::wkt(geometry);
        return ss.str();
    }
};

class Polygon : public BaseGeometry {
public:
    ~Polygon() override = default;
    std::shared_ptr<BoostPolygon> polygon;

    Polygon() : polygon(std::make_shared<BoostPolygon>()) {}
    Polygon(const BoostPolygon &p) : polygon(std::make_shared<BoostPolygon>(p)) {}
    // This doesn't work, but is probably
    // Polygon(BoostPolygon &&p) : polygon(std::make_shared<BoostPolygon>(std::move(p))) {}
    Polygon(std::shared_ptr<BoostPolygon> p) : polygon(p) {}

    Polygon(const std::vector<std::pair<double, double>> &exterior,
            const std::vector<std::vector<std::pair<double, double>>> &interiors = {})
        : polygon(std::make_shared<BoostPolygon>()) {
        setExterior(std::move(exterior));
        setInteriors(std::move(interiors));
    }

    // TODO: Box is probably sufficient.
    std::vector<std::shared_ptr<Polygon>> intersection(const BoostPolygon &otherPolygon) const;

    std::string toWkt() const override { return convertToWkt(*polygon); }

    std::vector<std::pair<double, double>> getExterior() const;
    std::vector<std::vector<std::pair<double, double>>> getInteriors() const;

    double getArea() const { return bg::area(*polygon); }

private:
    void setExterior(const std::vector<std::pair<double, double>> &coordinates);
    void setInteriors(const std::vector<std::vector<std::pair<double, double>>> &interiors);
};

std::vector<std::pair<double, double>> Polygon::getExterior() const {
    std::vector<std::pair<double, double>> result;
    result.reserve(bg::exterior_ring(*polygon).size());
    for (const auto &point : bg::exterior_ring(*polygon)) {
        result.emplace_back(bg::get<0>(point), bg::get<1>(point));
    }
    return result;
}

std::vector<std::vector<std::pair<double, double>>> Polygon::getInteriors() const {
    std::vector<std::vector<std::pair<double, double>>> result;
    result.reserve(polygon->inners().size());
    for (const auto &inner : polygon->inners()) {
        std::vector<std::pair<double, double>> inner_result;
        for (const auto &point : inner) {
            inner_result.emplace_back(bg::get<0>(point), bg::get<1>(point));
        }
        result.emplace_back(inner_result);
    }
    return result;
}

void Polygon::setExterior(const std::vector<std::pair<double, double>> &coordinates) {
    bg::exterior_ring(*polygon).clear();
    bg::exterior_ring(*polygon).reserve(coordinates.size());
    for (const auto &coord : coordinates) {
        bg::append(*polygon, BoostPoint(coord.first, coord.second));
    }
    // Close the ring if it's not already closed
    if (coordinates.front() != coordinates.back()) {
        bg::append(*polygon, BoostPoint(coordinates.front().first, coordinates.front().second));
    }
}

void Polygon::setInteriors(const std::vector<std::vector<std::pair<double, double>>> &interiors) {
    bg::interior_rings(*polygon).clear();
    bg::exterior_ring(*polygon).reserve(interiors.size());
    polygon->inners().resize(interiors.size());
    for (size_t i = 0; i < interiors.size(); ++i) {
        const auto &interior_coords = interiors[i];
        auto &inner = polygon->inners()[i];
        inner.clear();

        // Process the interior ring in reverse order
        for (auto it = interior_coords.rbegin(); it != interior_coords.rend(); ++it) {
            bg::append(inner, BoostPoint(it->first, it->second));
        }
        // Close the ring if it's not already closed
        if (interior_coords.front() != interior_coords.back()) {
            bg::append(inner, BoostPoint(interior_coords.back().first, interior_coords.back().second));
        }
    }
}

std::vector<std::shared_ptr<Polygon>> Polygon::intersection(const BoostPolygon &otherPolygon) const {
    // Make the polygon valid if needed before performing the intersection
    // TODO: This simplifies the polygon!!
    BoostPolygon validPolygon = GeometryUtils::makeValid(*polygon);

    std::vector<BoostPolygon> intersectionResult;
    bg::intersection(validPolygon, otherPolygon, intersectionResult);

    std::vector<std::shared_ptr<Polygon>> result;
    for (const auto &intersectedBoostPolygon : intersectionResult) {
        auto intersectedPolygon = std::make_shared<Polygon>(intersectedBoostPolygon);
        // Copy the parameters from this polygon to the new one

        for (const auto &param : parameters) {
            intersectedPolygon->setField(param.first, param.second);
        }

        result.emplace_back(intersectedPolygon);
    }

    return result;
}

class Point : public BaseGeometry {
public:
    ~Point() override = default;
    std::shared_ptr<BoostPoint> point;

    Point() : point(std::make_shared<BoostPoint>()) {}
    Point(const BoostPoint &p) : point(std::make_shared<BoostPoint>(p)) {}
    Point(std::shared_ptr<BoostPoint> p) : point(p) {}
    Point(double x, double y) : point(std::make_shared<BoostPoint>(x, y)) {}

    Point(const Point &other) : BaseGeometry(other), point(std::make_shared<BoostPoint>(*other.point)) {
        parameters = other.parameters; // Copy parameters
    }

    // Factory function for creating points from Python
    static std::shared_ptr<Point> create(double x, double y) { return std::make_shared<Point>(x, y); }

    std::string toWkt() const override { return convertToWkt(*point); }

    void setCoordinates(double x, double y) {
        bg::set<0>(*point, x);
        bg::set<1>(*point, y);
    }
    std::pair<double, double> getCoordinates() const { return std::make_pair(bg::get<0>(*point), bg::get<1>(*point)); }
    inline double getX() const { return bg::get<0>(*point); }
    inline double getY() const { return bg::get<1>(*point); }
    double distanceTo(const Point &other) const { return bg::distance(*point, *(other.point)); }
    bool equals(const Point &other) const { return bg::equals(*point, *(other.point)); }
    bool within(const Polygon &polygon) const { return bg::within(*point, *(polygon.polygon)); }

    std::shared_ptr<Point> centroid(const Polygon &polygon) const {
        BoostPoint centroid;
        bg::centroid(*(polygon.polygon), centroid);
        return std::make_shared<Point>(centroid);
    }

    double azimuth(const Point &other) const { return bg::azimuth(*point, *(other.point)); }

    std::shared_ptr<Point> translate(double dx, double dy) const {
        BoostPoint translated;
        bg::strategy::transform::translate_transformer<double, 2, 2> translate(dx, dy);
        bg::transform(*point, translated, translate);
        return std::make_shared<Point>(translated);
    }

    std::shared_ptr<Point> rotate(double angle, const Point &origin = Point(0, 0)) const {
        BoostPoint rotated;
        bg::strategy::transform::rotate_transformer<bg::degree, double, 2, 2> rotate(angle);
        bg::transform(*point, rotated, rotate);
        return std::make_shared<Point>(rotated);
    }

    std::shared_ptr<Point> scale(double scaling, const Point &origin = Point(0, 0)) const {
        BoostPoint scaled;
        double dx = getX() - origin.getX();
        double dy = getY() - origin.getY();

        bg::strategy::transform::scale_transformer<double, 2, 2> scale(scaling);
        bg::transform(BoostPoint(dx, dy), scaled, scale);

        return std::make_shared<Point>(scaled.get<0>() + origin.getX(), scaled.get<1>() + origin.getY());
    }
};

cv::Mat generateMaskFromAnnotations(const std::vector<std::shared_ptr<Polygon>> &annotations, cv::Size region_size,
                                    const std::unordered_map<std::string, int> &index_map, int default_value) {
    // Create the mask and initialize with the default value
    cv::Mat mask(region_size, CV_32S, cv::Scalar(default_value));

    std::vector<cv::Point> exterior_cv_points;
    std::vector<std::vector<cv::Point>> interiors_cv_points;

    for (const auto &annotation : annotations) {
        int index_value = index_map.at(annotation->getField("label")->cast<std::string>());

        // Convert exterior points
        exterior_cv_points.clear();
        const auto &exterior = annotation->getExterior();
        exterior_cv_points.reserve(exterior.size());
        for (const auto &[x, y] : exterior) {
            exterior_cv_points.emplace_back(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
        }

        // Convert interior points
        interiors_cv_points.clear();
        const auto &interiors = annotation->getInteriors();
        interiors_cv_points.reserve(interiors.size());
        for (const auto &interior : interiors) {
            std::vector<cv::Point> interior_cv;
            interior_cv.reserve(interior.size());
            for (const auto &[x, y] : interior) {
                interior_cv.emplace_back(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
            }
            interiors_cv_points.push_back(std::move(interior_cv));
        }

        // Only clone mask if necessary
        cv::Mat original_values;
        if (!interiors_cv_points.empty()) {
            original_values = mask.clone();
        }

        // Create a mask for holes if necessary
        cv::Mat holes_mask;
        if (!interiors_cv_points.empty()) {
            holes_mask = cv::Mat::zeros(region_size, CV_8U);
            cv::fillPoly(holes_mask, interiors_cv_points, cv::Scalar(1));
        }

        // Fill the exterior polygon in the mask
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{exterior_cv_points}, cv::Scalar(index_value));

        // If interiors exist, reset the holes in the mask using the backup
        if (!interiors_cv_points.empty()) {
            original_values.copyTo(mask, holes_mask);
        }
    }

    return mask;
}


py::array_t<int> maskToPyArray(const cv::Mat &mask) {
    // Ensure the mask is of type CV_32S (int type)
    if (mask.type() != CV_32S) {
        throw std::runtime_error("Mask must be of type CV_32S (int).");
    }

    // Create a buffer info that describes the numpy array
    py::buffer_info buf_info(mask.data,                             // Pointer to buffer
                             sizeof(int),                           // Size of one scalar element
                             py::format_descriptor<int>::format(),  // Python struct-style format descriptor
                             2,                                     // Number of dimensions
                             {mask.rows, mask.cols},                // Buffer dimensions
                             {sizeof(int) * mask.cols, sizeof(int)} // Strides (in bytes) for each dimension
    );

    // Create the numpy array from the buffer info
    return py::array_t<int>(buf_info);
}

class AnnotationRegion {
public:
    AnnotationRegion(std::vector<std::shared_ptr<Polygon>> polygons, std::vector<std::shared_ptr<Point>> points)
        : polygons_(std::move(polygons)), points_(std::move(points)) {}

    static void setPolygonFactory(py::function factory) { polygonFactory() = std::move(factory); }
    static void setPointFactory(py::function factory) { pointFactory() = std::move(factory); }

    static FactoryGuard createPolygonFactoryGuard(py::function factory) {
        return FactoryGuard(polygonFactory(), factory);
    }

    static FactoryGuard createPointFactoryGuard(py::function factory) { return FactoryGuard(pointFactory(), factory); }

    static py::object callFactoryFunction(const std::shared_ptr<Polygon> &polygon) {
        return invokeFactoryFunction(polygonFactory(), polygon);
    }

    static py::object callFactoryFunction(const std::shared_ptr<Point> &point) {
        return invokeFactoryFunction(pointFactory(), point);
    }

    py::list getPolygons() const {
#ifdef DLUPDEBUG
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#endif
        py::list py_polygons;
        for (const auto &polygon : polygons_) {
            py_polygons.append(callFactoryFunction(polygon));
        }
#ifdef DLUPDEBUG
        std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
        std::cout << "Elapsed time in AnnotationRegion::getPolygons: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - end).count() << " ms" << std::endl;
#endif
        return py_polygons;
    }

    py::list getPoints() const {
        py::list py_points;
        for (const auto &point : points_) {
            py_points.append(callFactoryFunction(point));
        }
        return py_points;
    }

    py::array_t<int> toMask(std::tuple<int, int> mask_size, const std::unordered_map<std::string, int> &index_map,
                            int default_value = 0) const {
#ifdef DLUPDEBUG
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif
        cv::Size region_size(std::get<1>(mask_size), std::get<0>(mask_size));
        cv::Mat mask = generateMaskFromAnnotations(polygons_, region_size, index_map, default_value);
#ifdef DLUPDEBUG
        std::cout
            << "AnnotationRegion::toMask: mask generated in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count()
            << " ms" << std::endl;
#endif
        return maskToPyArray(mask);
    }

private:
    std::vector<std::shared_ptr<Polygon>> polygons_;
    std::vector<std::shared_ptr<Point>> points_;

    static py::function &polygonFactory() {
        static py::function instance;
        return instance;
    }

    static py::function &pointFactory() {
        static py::function instance;
        return instance;
    }

    template <typename T>
    static py::object invokeFactoryFunction(py::function factoryFunction, const std::shared_ptr<T> &object) {
        if (factoryFunction != py::function()) {
            try {
                py::object result = factoryFunction(object);
                if (result.ptr() != nullptr) {
                    return result;
                } else {
                    throw GeometryFactoryFunctionError("Factory function returned null object");
                }
            } catch (const std::exception &e) {
                throw GeometryFactoryFunctionError(std::string("Exception in factory function: ") + e.what());
            } catch (...) {
                throw GeometryFactoryFunctionError("Unknown exception in factory function");
            }
        }
        return py::cast(object);
    }
};

class GeometryContainer {
public:
    // Whatever any LLM says this has to be a shared pointer as we share it with the python interpreter
    using PolygonPtr = std::shared_ptr<Polygon>;
    using PointPtr = std::shared_ptr<Point>;

    std::vector<PolygonPtr> polygons;
    std::vector<PointPtr> points;
    RTreeWrapper rtreeWrapper;

    void addPolygon(const PolygonPtr &p) {
        // Print the parameters of the polygon being added
        BoostBox box;
        bg::envelope(*(p->polygon), box);
        rtreeWrapper.insert(box, polygons.size());
        polygons.emplace_back(p);
    }

    void addPoint(const PointPtr &p) {
        BoostBox box(*(p->point), *(p->point));
        rtreeWrapper.insert(box, polygons.size() + points.size());
        points.emplace_back(p);
    }

    py::list getPolygons() {
#ifdef DLUPDEBUG
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#endif
        py::list py_polygons;
        for (const auto &polygon : polygons) {
            py_polygons.append(AnnotationRegion::callFactoryFunction(polygon));
        }
#ifdef DLUPDEBUG
        std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
        std::cout << "Elapsed time in GeometryContainer::getPolygons: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - end).count() << " ms" << std::endl;
#endif
        return py_polygons;
    }

    void removePolygon(const PolygonPtr &p);
    void removePolygon(size_t index);

    void removePoint(const PointPtr &p);
    void removePoint(size_t index);

    void scale(double scaling);
    void setOffset(std::pair<double, double> offset);
    void rebuildRTree();

    std::uintptr_t getPointerId() const { return reinterpret_cast<std::uintptr_t>(this); }

    bool isRTreeInvalidated() const { return rtreeWrapper.isInvalidated(); }

    AnnotationRegion readRegion(const std::pair<double, double> &coordinates, double scaling,
                                const std::pair<double, double> &size);

    // TODO: Rethink the need for this function.
    void reindexPolygons(const std::map<std::string, int> &indexMap) {
        for (auto &polygon : polygons) {
            std::optional<py::object> label_opt = polygon->getField("label");

            if (label_opt.has_value()) {
                std::string label = label_opt->cast<std::string>();
                auto it = indexMap.find(label);
                if (it != indexMap.end()) {
                    polygon->setField("index", py::int_(it->second));
                } else {
                    throw std::invalid_argument("Label '" + label + "' not found in indexMap");
                }
            } else {
                throw std::invalid_argument("Polygon does not have a value for the 'label' field");
            }
        }
    }
};

void GeometryContainer::scale(double scaling) {
    for (auto &point : points) {
        GeometryUtils::applyAffineTransformation(*point->point, {0.0, 0.0}, scaling);
    }
    for (auto &polygon : polygons) {
        GeometryUtils::applyAffineTransformation(*polygon->polygon, {0.0, 0.0}, scaling);
    }
    rtreeWrapper.invalidate();
}

void GeometryContainer::setOffset(std::pair<double, double> offset) {
    for (auto &point : points) {
        GeometryUtils::applyAffineTransformation(*point->point, offset, 1.0);
    }
    for (auto &polygon : polygons) {
        GeometryUtils::applyAffineTransformation(*polygon->polygon, offset, 1.0);
    }
    rtreeWrapper.invalidate();
}

void GeometryContainer::rebuildRTree() {
    rtreeWrapper.clear();
    for (size_t i = 0; i < polygons.size(); ++i) {
        BoostBox box;
        bg::envelope(*(polygons[i]->polygon), box);
        rtreeWrapper.insert(box, i);
    }
    for (size_t i = 0; i < points.size(); ++i) {
        BoostBox box(*(points[i]->point), *(points[i]->point));
        rtreeWrapper.insert(box, polygons.size() + i);
    }
}

void GeometryContainer::removePolygon(const PolygonPtr &p) {
    auto it = std::find(polygons.begin(), polygons.end(), p);
    if (it != polygons.end()) {
        polygons.erase(it);
        rtreeWrapper.invalidate();
    } else {
        throw GeometryNotFoundError("Polygon not found");
    }
}

void GeometryContainer::removePolygon(size_t index) {
    if (index >= polygons.size()) {
        throw std::out_of_range("Polygon index out of range");
    }

    polygons.erase(polygons.begin() + index);
    rtreeWrapper.invalidate();
}

void GeometryContainer::removePoint(const PointPtr &p) {
    auto it = std::find(points.begin(), points.end(), p);
    if (it != points.end()) {
        points.erase(it);
        rtreeWrapper.invalidate();
    } else {
        throw GeometryNotFoundError("Point not found");
    }
}

void GeometryContainer::removePoint(size_t index) {
    if (index >= points.size()) {
        throw std::out_of_range("Point index out of range");
    }

    points.erase(points.begin() + index);
    rtreeWrapper.invalidate();
}

AnnotationRegion GeometryContainer::readRegion(const std::pair<double, double> &coordinates, double scaling,
                                               const std::pair<double, double> &size) {

#ifdef DLUPDEBUG
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif
    BoostPoint topLeft(coordinates.first / scaling, coordinates.second / scaling);
    BoostPoint bottomRight((coordinates.first + size.first) / scaling, (coordinates.second + size.second) / scaling);
    BoostBox queryBox(topLeft, bottomRight);

    BoostPolygon intersectionPolygon;
    bg::convert(queryBox, intersectionPolygon);
    std::vector<std::pair<BoostBox, size_t>> results;
    rtreeWrapper.query(bgi::intersects(queryBox), std::back_inserter(results));

    std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

    std::vector<std::shared_ptr<Polygon>> intersectedPolygons;
    std::vector<std::shared_ptr<Point>> intersectedPoints;

    for (const auto &result : results) {
        size_t index = result.second;
        if (index < polygons.size()) {
            auto &polygon = polygons[index];
            auto intersections = polygon->intersection(intersectionPolygon);
            for (const auto &intersectedPolygon : intersections) {
                GeometryUtils::applyAffineTransformation(*intersectedPolygon->polygon, coordinates, scaling);
                intersectedPolygons.push_back(intersectedPolygon);
            }
        } else {
            auto &point = points[index - polygons.size()];
            auto transformedPoint = std::make_shared<Point>(*point);
            GeometryUtils::applyAffineTransformation(*transformedPoint->point, coordinates, scaling);
            intersectedPoints.push_back(transformedPoint);
        }
    }
#ifdef DLUPDEBUG
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in GeometryContainer::readRegion: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
#endif
    auto returnValue = AnnotationRegion(std::move(intersectedPolygons), std::move(intersectedPoints));

    return returnValue;
}

PYBIND11_MODULE(_geometry, m) {
    py::class_<BaseGeometry, std::shared_ptr<BaseGeometry>>(m, "BaseGeometry")
        .def("set_field", &BaseGeometry::setField)
        .def("get_field", &BaseGeometry::getField)
        .def_property_readonly("fields", &BaseGeometry::getFields)
        .def_property_readonly("pointer_id", &BaseGeometry::getPointerId);

    py::class_<Polygon, BaseGeometry, std::shared_ptr<Polygon>>(m, "Polygon")
        .def(py::init<>())
        .def(py::init<const BoostPolygon &>())
        .def(py::init<const std::vector<std::pair<double, double>> &,
                      const std::vector<std::vector<std::pair<double, double>>> &>())
        .def(py::init([](const std::shared_ptr<Polygon> &p) {
            // Share the same C++ object, not creating a new one
            return p;
        }))
        .def(py::init([](const Polygon &other) {
            // Explicitly copy parameters when copying the polygon
            auto newPolygon = std::make_shared<Polygon>(*other.polygon);
            newPolygon->parameters = other.parameters; // Copy the parameters
            return newPolygon;
        }))
        .def("get_exterior", &Polygon::getExterior)
        .def("get_interiors", &Polygon::getInteriors)
        .def_property_readonly("wkt", &Polygon::toWkt)
        .def_property_readonly("area", &Polygon::getArea);

    py::class_<Point, BaseGeometry, std::shared_ptr<Point>>(m, "Point")
        .def(py::init<>())
        .def(py::init<const BoostPoint &>())
        .def(py::init<double, double>())
        .def(py::init([](const std::shared_ptr<Point> &p) {
            // Share the same C++ object, not creating a new one
            return p;
        }))
        .def(py::init([](const Point &other) {
            // Explicitly copy parameters when copying the polygon
            auto newPoint = std::make_shared<Point>(*other.point);
            newPoint->parameters = other.parameters; // Copy the parameters
            return newPoint;
        }))
        .def("set_coordinates", &Point::setCoordinates)
        .def("get_coordinates", &Point::getCoordinates)
        .def_property_readonly("x", &Point::getX)
        .def_property_readonly("y", &Point::getY)
        .def("distance_to", &Point::distanceTo)
        .def("equals", &Point::equals)
        .def("within", &Point::within)
        .def("centroid", &Point::centroid)
        .def("azimuth", &Point::azimuth)
        .def("translate", &Point::translate)
        .def("rotate", &Point::rotate, py::arg("angle"), py::arg("origin") = Point(0, 0))
        .def("scale", &Point::scale, py::arg("scaling"), py::arg("origin") = Point(0, 0))
        .def_property_readonly("wkt", &Point::toWkt);

    m.def("set_polygon_factory", &AnnotationRegion::setPolygonFactory);
    m.def("set_point_factory", &AnnotationRegion::setPointFactory);

    py::class_<GeometryContainer, std::shared_ptr<GeometryContainer>>(m, "GeometryContainer")
        .def(py::init<>())
        .def("add_polygon", &GeometryContainer::addPolygon)
        .def("add_point", &GeometryContainer::addPoint)

        // Overload remove_polygon to handle both object and index
        .def("remove_polygon", py::overload_cast<const std::shared_ptr<Polygon> &>(&GeometryContainer::removePolygon),
             "Remove a polygon by passing the Polygon object")
        .def("remove_polygon", py::overload_cast<size_t>(&GeometryContainer::removePolygon),
             "Remove a polygon by its index")
        .def("reindex_polygons", &GeometryContainer::reindexPolygons)

        // Overload remove_point to handle both object and index
        .def("remove_point", py::overload_cast<const std::shared_ptr<Point> &>(&GeometryContainer::removePoint),
             "Remove a point by passing the Point object")
        .def("remove_point", py::overload_cast<size_t>(&GeometryContainer::removePoint), "Remove a point by its index")
        .def("read_region", &GeometryContainer::readRegion)
        .def("rebuild_rtree", &GeometryContainer::rebuildRTree, "Rebuild the R-tree index manually")
        .def("scale", &GeometryContainer::scale, "Scale all geometries by a factor")
        .def("set_offset", &GeometryContainer::setOffset, "Set an offset for all geometries")
        .def_property_readonly("rtree_invalidated", &GeometryContainer::isRTreeInvalidated)
        .def_property_readonly("pointer_id", &GeometryContainer::getPointerId)
        .def_property_readonly("polygons", &GeometryContainer::getPolygons)
        .def_property_readonly("points", [](const GeometryContainer &self) { return self.points; });

    py::class_<AnnotationRegion, std::shared_ptr<AnnotationRegion>>(m, "RegionResult")
        .def_property_readonly("polygons", &AnnotationRegion::getPolygons)
        .def_property_readonly("points", &AnnotationRegion::getPoints)
        .def("to_mask", &AnnotationRegion::toMask, py::arg("mask_size"), py::arg("index_map"),
             py::arg("default_value") = 0);

    py::register_exception<GeometryError>(m, "GeometryError");
    py::register_exception<GeometryIntersectionError>(m, "GeometryIntersectionError");
    py::register_exception<GeometryTransformationError>(m, "GeometryTransformationError");
    py::register_exception<GeometryFactoryFunctionError>(m, "GeometryFactoryFunctionError");
    py::register_exception<GeometryNotFoundError>(m, "GeometryNotFoundError");
}
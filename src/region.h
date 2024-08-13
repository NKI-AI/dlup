#ifndef DLUP_REGION_H
#define DLUP_REGION_H

#include "geometry.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

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

class AnnotationRegion {
public:
    AnnotationRegion(std::vector<std::shared_ptr<Polygon>> polygons, std::vector<std::shared_ptr<Point>> points,
                     std::tuple<int, int> mask_size)
        : polygons_(std::move(polygons)), points_(std::move(points)), mask_size_(std::move(mask_size)) {}

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

    py::list getPolygons() const;
    py::list getPoints() const;

    py::array_t<int> toMask(int default_value = 0) const {
#ifdef DLUPDEBUG
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif
        cv::Size region_size(std::get<0>(mask_size_), std::get<1>(mask_size_));
        cv::Mat mask = generateMaskFromAnnotations(polygons_, region_size, default_value);
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
    std::tuple<int, int> mask_size_;

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

#endif
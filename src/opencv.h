#ifndef DLUP_OPENCV_H
#define DLUP_OPENCV_H

#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <vector>

cv::Mat generateMaskFromAnnotations(const std::vector<std::shared_ptr<Polygon>> &annotations, cv::Size region_size,
                                    int default_value) {
    // Create the mask and initialize with the default value
    cv::Mat mask(region_size, CV_32S, cv::Scalar(default_value));

    std::vector<cv::Point> exterior_cv_points;
    std::vector<std::vector<cv::Point>> interiors_cv_points;

    for (const auto &annotation : annotations) {
        auto index_value_field = annotation->getField("index");
        if (!index_value_field) {
            auto label = annotation->getField("label");
            throw std::runtime_error("Annotation with label '" + label->cast<std::string>() +
                                     "' does not have an index.");
        }
        // Cast index_value to int
        int index_value = index_value_field->cast<int>();

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

#endif // DLUP_OPENCV_H
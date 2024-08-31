#ifndef DLUP_OPENCV_H
#define DLUP_OPENCV_H

#include "geometry/polygon.h"
#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <vector>

std::shared_ptr<std::vector<int>> generateMaskFromAnnotations(const std::vector<std::shared_ptr<Polygon>> &annotations,
                                                              const std::tuple<int, int> &mask_size,
                                                              int default_value) {

  int width = std::get<0>(mask_size);
  int height = std::get<1>(mask_size);

  // Use a shared_ptr to manage the mask's lifetime
  auto mask = std::make_shared<std::vector<int>>(width * height, default_value);

  cv::Mat mask_view(height, width, CV_32S, mask->data());

  std::vector<cv::Point> exterior_cv_points;
  std::vector<std::vector<cv::Point>> interiors_cv_points;

  for (const auto &annotation : annotations) {
    auto index_value_field = annotation->getField("index");
    if (!index_value_field) {
      auto label = annotation->getField("label");
      throw std::runtime_error("Annotation with label '" + label->cast<std::string>() + "' does not have an index.");
    }
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
      interiors_cv_points.emplace_back(); // Create a new vector in place
      interiors_cv_points.back().reserve(interior.size());
      for (const auto &[x, y] : interior) {
        interiors_cv_points.back().emplace_back(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
      }
    }

    if (!interiors_cv_points.empty()) {
      // Create a mask for holes
      cv::Mat holes_mask = cv::Mat::zeros(height, width, CV_8U);
      cv::fillPoly(holes_mask, interiors_cv_points, cv::Scalar(1));

      // Apply exterior mask first, then restore original values in holes
      cv::Mat original_values = mask_view.clone();
      cv::fillPoly(mask_view, std::vector<std::vector<cv::Point>>{exterior_cv_points}, cv::Scalar(index_value));
      original_values.copyTo(mask_view, holes_mask);
    } else {
      // Directly fill the exterior mask if no interiors exist
      cv::fillPoly(mask_view, std::vector<std::vector<cv::Point>>{exterior_cv_points}, cv::Scalar(index_value));
    }
  }

  return mask;
}

#endif // DLUP_OPENCV_H

// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/estimators/rig_calibration_distortion.h"

#include "colmap/math/polynomial.h"
#include "colmap/sensor/models.h"

#include <ceres/ceres.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

bool FullOpenCVProjectionIsValid(const Camera& camera) {
  THROW_CHECK_EQ(camera.model_id, FullOpenCVCameraModel::model_id);
  const auto& params = camera.params;
  const double k1 = params[4], k2 = params[5], p1 = params[6], p2 = params[7];
  const double k3 = params[8], k4 = params[9], k5 = params[10], k6 = params[11];
  double max_squared_radius = 0.0;
  for (int row = 0; row <= 150; ++row) {
    for (int column = 0; column <= 150; ++column) {
      const Eigen::Vector2d pixel(0.5 + (camera.width - 1.0) * column / 150.0,
                                  0.5 + (camera.height - 1.0) * row / 150.0);
      const auto ray = camera.CamFromImg(pixel);
      if (!ray.has_value() || !ray->allFinite()) {
        return false;
      }
      const auto projected = camera.ImgFromCam(ray->homogeneous());
      if (!projected.has_value() || !projected->allFinite() ||
          (*projected - pixel).norm() > 1e-4) {
        return false;
      }
      const double x = ray->x(), y = ray->y(), radius = ray->squaredNorm();
      max_squared_radius = std::max(max_squared_radius, radius);
      const double numerator =
          1.0 + radius * (k1 + radius * (k2 + radius * k3));
      const double denominator =
          1.0 + radius * (k4 + radius * (k5 + radius * k6));
      const double numerator_derivative =
          k1 + radius * (2 * k2 + 3 * radius * k3);
      const double denominator_derivative =
          k4 + radius * (2 * k5 + 3 * radius * k6);
      const double radial = numerator / denominator;
      const double derivative = (numerator_derivative * denominator -
                                 numerator * denominator_derivative) /
                                (denominator * denominator);
      const double horizontal =
          radial + 2 * x * x * derivative + 2 * p1 * y + 6 * p2 * x;
      const double vertical =
          radial + 2 * y * y * derivative + 6 * p1 * y + 2 * p2 * x;
      const double cross = 2 * x * y * derivative + 2 * p1 * x + 2 * p2 * y;
      if (!(horizontal > 0 && horizontal * vertical - cross * cross > 0)) {
        return false;
      }
    }
  }
  if (k4 == 0.0 && k5 == 0.0 && k6 == 0.0) {
    return true;  // A constant denominator has no roots.
  }
  Eigen::VectorXd real, imaginary;
  THROW_CHECK(FindPolynomialRootsCompanionMatrix(
      Eigen::Vector4d(k6, k5, k4, 1.0), &real, &imaginary));
  for (Eigen::Index index = 0; index < real.size(); ++index) {
    if (std::abs(imaginary[index]) < 1e-10 && real[index] > 0 &&
        real[index] <= max_squared_radius) {
      return false;
    }
  }
  return true;
}

Camera CarrotCamera(const bool valid) {
  Camera camera = Camera::CreateFromModelId(
      1, FullOpenCVCameraModel::model_id, 1440, 2062, 2472);
  // Type 17 cam12: the accepted scan-361 model and the folded scan-349 fit.
  camera.params = valid ? std::vector<double>{1443.3599260460865,
                                              1441.1283648802673,
                                              968.2543230141835,
                                              1240.1594358459704,
                                              -0.39202946424484253,
                                              -0.28028514981269836,
                                              -0.0003090359678026289,
                                              3.631979052443057e-05,
                                              0.12659968435764313,
                                              -0.22963164746761322,
                                              -0.4370480477809906,
                                              0.16610059142112732}
                        : std::vector<double>{1441.0992886850963,
                                              1439.8694544636392,
                                              967.2593988675724,
                                              1240.8043272408092,
                                              -0.5220213532447815,
                                              -0.26459869742393494,
                                              -0.0003212183655705303,
                                              -3.4396303817629814e-05,
                                              0.13156360387802124,
                                              -0.36154797673225403,
                                              -0.43635544180870056,
                                              0.17703060805797577};
  return camera;
}

struct LensObservation {
  Eigen::Vector3d ray;
  Eigen::Vector2d pixel;

  template <typename T>
  bool operator()(const T* params, T* residuals) const {
    FullOpenCVCameraModel::ImgFromCam(
        params, T(ray.x()), T(ray.y()), T(1.0), residuals, residuals + 1);
    residuals[0] -= T(pixel.x());
    residuals[1] -= T(pixel.y());
    return true;
  }
};

std::pair<Camera, ceres::Solver::Summary> FitCentralObservations(
    bool regularize) {
  const Camera reference = CarrotCamera(true);
  const Camera folded = CarrotCamera(false);
  Camera candidate = reference;
  ceres::Problem problem;
  for (int row = 1; row < 10; ++row) {
    for (int column = 1; column < 10; ++column) {
      const auto ray = reference.CamFromImg(Eigen::Vector2d(
          reference.width * column / 10.0, reference.height * row / 10.0));
      EXPECT_TRUE(ray.has_value());
      const auto pixel = folded.ImgFromCam(ray->homogeneous());
      EXPECT_TRUE(pixel.has_value());
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<LensObservation, 2, 12>(
              new LensObservation{ray->homogeneous(), *pixel}),
          nullptr,
          candidate.params.data());
    }
  }
  if (regularize) {
    AddFullOpenCVDistortionPrior(problem, candidate, 81);
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 200;
  options.function_tolerance = 1e-12;
  options.gradient_tolerance = 1e-12;
  options.parameter_tolerance = 1e-12;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  return {candidate, summary};
}

TEST(RigCalibrationDistortion, KeepsFitInvertibleWhenTracksFavorFoldedModel) {
  const auto [unregularized, unregularized_summary] =
      FitCentralObservations(false);
  EXPECT_FALSE(FullOpenCVProjectionIsValid(unregularized));
  const auto [candidate, summary] = FitCentralObservations(true);

  EXPECT_TRUE(summary.IsSolutionUsable()) << summary.FullReport();
  EXPECT_TRUE(FullOpenCVProjectionIsValid(candidate));
  const Camera reference = CarrotCamera(true);
  const Camera folded = CarrotCamera(false);
  double squared_error = 0.0;
  for (int row = 1; row < 10; ++row) {
    for (int column = 1; column < 10; ++column) {
      const auto ray = reference.CamFromImg(Eigen::Vector2d(
          reference.width * column / 10.0, reference.height * row / 10.0));
      squared_error += (*candidate.ImgFromCam(ray->homogeneous()) -
                        *folded.ImgFromCam(ray->homogeneous()))
                           .squaredNorm();
    }
  }
  EXPECT_LT(std::sqrt(squared_error / 81), 0.05);
  EXPECT_LT(summary.final_cost, summary.initial_cost * 0.01);
}

}  // namespace
}  // namespace colmap

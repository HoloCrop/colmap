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

#include <algorithm>
#include <cmath>

#include <ceres/sized_cost_function.h>

namespace colmap {
namespace {

class RationalDenominatorPrior : public ceres::SizedCostFunction<3, 12> {
 public:
  RationalDenominatorPrior(double radius_squared, size_t num_observations) {
    // A unit denominator term at the sensor radius costs 0.1 pixels per
    // observation. Weak image evidence cannot justify nearly cancelling poles.
    const double weight =
        0.1 * std::sqrt(static_cast<double>(num_observations));
    scales_ = weight *
              Eigen::Vector3d(radius_squared,
                              radius_squared * radius_squared,
                              radius_squared * radius_squared * radius_squared);
  }
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    for (int index = 0; index < 3; ++index) {
      residuals[index] = scales_[index] * parameters[0][9 + index];
    }
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      std::fill_n(jacobians[0], 36, 0.0);
      for (int index = 0; index < 3; ++index) {
        jacobians[0][12 * index + 9 + index] = scales_[index];
      }
    }
    return true;
  }

 private:
  Eigen::Vector3d scales_;
};

}  // namespace

void AddFullOpenCVDistortionPrior(ceres::Problem& problem,
                                  Camera& camera,
                                  size_t num_observations) {
  double squared_radius = 0.0;
  for (double x : {0.5, camera.width - 0.5}) {
    for (double y : {0.5, camera.height - 0.5}) {
      squared_radius =
          std::max(squared_radius,
                   Eigen::Vector2d(
                       (x - camera.PrincipalPointX()) / camera.FocalLengthX(),
                       (y - camera.PrincipalPointY()) / camera.FocalLengthY())
                       .squaredNorm());
    }
  }
  problem.AddResidualBlock(
      new RationalDenominatorPrior(squared_radius, num_observations),
      nullptr,
      camera.params.data());
}

}  // namespace colmap

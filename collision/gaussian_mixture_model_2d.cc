// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "collision/gaussian_mixture_model_2d.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "eigenmath/normal_distribution.h"
#include "eigenmath/types.h"
#include "eigenmath/vector_utils.h"
#include "genit/transform_iterator.h"

namespace mobility::collision {

GaussianMixtureModel2d::GaussianMixtureModel2d(int num_components)
    : components_(
          std::max(0, num_components),
          {eigenmath::EuclideanNormalDistribution<double, 2>::CreateStandard(),
           1.0 / num_components}) {
  CHECK_GT(num_components, 0);
}

void GaussianMixtureModel2d::SetMean(const eigenmath::Vector2d& mean,
                                     int num_component) {
  CHECK_GT(components_.size(), num_component);
  components_[num_component].dist.SetMean(mean);
}

void GaussianMixtureModel2d::SetCovariance(
    const eigenmath::Matrix2d& covariance, int num_component) {
  CHECK_GT(components_.size(), num_component);
  components_[num_component].dist.SetCovariance(covariance);
}

void GaussianMixtureModel2d::SetWeight(const double weight, int num_component) {
  CHECK_GT(components_.size(), num_component);
  components_[num_component].weight = weight;
}

const eigenmath::Vector2d& GaussianMixtureModel2d::Mean(
    int num_component) const {
  CHECK_GT(components_.size(), num_component);
  return components_[num_component].dist.Mean();
}

const eigenmath::Matrix2d& GaussianMixtureModel2d::Covariance(
    int num_component) const {
  CHECK_GT(components_.size(), num_component);
  return components_[num_component].dist.Covariance();
}

const double& GaussianMixtureModel2d::Weight(int num_component) const {
  CHECK_GT(components_.size(), num_component);
  return components_[num_component].weight;
}

double GaussianMixtureModel2d::Probability(const eigenmath::Vector2d& point,
                                           int num_component) const {
  CHECK_GT(components_.size(), num_component);
  return Weight(num_component) *
         components_[num_component].dist.Probability(point);
}

std::vector<eigenmath::Vector2d> GaussianMixtureModel2d::Means() const {
  const auto& f = [&](const Component& component) {
    return component.dist.Mean();
  };
  return genit::CopyRange<std::vector<eigenmath::Vector2d>>(
      genit::TransformRange(components_, std::cref(f)));
}

std::vector<eigenmath::Matrix2d> GaussianMixtureModel2d::Covariances() const {
  const auto& f = [&](const Component& component) {
    return component.dist.Covariance();
  };
  return genit::CopyRange<std::vector<eigenmath::Matrix2d>>(
      genit::TransformRange(components_, std::cref(f)));
}

std::vector<double> GaussianMixtureModel2d::Weights() const {
  const auto& f = [&](const Component& component) { return component.weight; };
  return genit::CopyRange<std::vector<double>>(
      genit::TransformRange(components_, std::cref(f)));
}

GaussianMixtureModel2d FitGaussianMixture(
    const std::vector<eigenmath::Vector2d>& samples, const int num_components,
    const int max_iterations, const double log_prob_change_thresh,
    const std::vector<double>& weights) {
  CHECK_GE(samples.size(), num_components);
  CHECK_GT(num_components, 0);
  CHECK_GT(max_iterations, 0);

  absl::AnyInvocable<double(int) const&> sample_weights = [](int index) {
    return 1.0;
  };
  if (!weights.empty()) {
    CHECK_EQ(weights.size(), samples.size());
    for (const auto weight : weights) {
      CHECK_GE(weight, 0.0);
    }
    const double norm_factor =
        std::accumulate(weights.begin(), weights.end(), 0.0) / weights.size();
    CHECK_GT(norm_factor, 1e-6);
    sample_weights = [weights = &weights, norm_factor](int index) {
      return (*weights)[index] / norm_factor;
    };
  }

  constexpr double kReg = 1e-6;

  GaussianMixtureModel2d model(num_components);

  std::vector<std::vector<double>> probs(samples.size());
  double prev_log_prob = 0.0;

  // Initialize modes from the samples.
  const int stride = samples.size() / num_components;
  for (int j = 0; j < num_components; ++j) {
    model.SetMean(samples[stride * j], j);
  }

  for (int i = 0; i < max_iterations; ++i) {
    double log_prob = 0.0;

    // Expectation step
    for (int k = 0; k < samples.size(); ++k) {
      double sum = 0.0;
      probs[k].resize(num_components, 0.0);

      for (int j = 0; j < num_components; ++j) {
        probs[k][j] = model.Probability(samples[k], j);
        sum += probs[k][j];
      }

      for (int j = 0; j < num_components; ++j) {
        // Normalize probabilities.
        probs[k][j] /= sum;
      }
      log_prob += std::log(sum);
    }
    log_prob /= samples.size();

    // Maximization step
    for (int j = 0; j < num_components; ++j) {
      eigenmath::Vector2d mean = eigenmath::Vector2d::Zero();
      eigenmath::Matrix2d covariance = eigenmath::Matrix2d::Zero();
      double cum_prob = 0.0;

      for (int k = 0; k < samples.size(); ++k) {
        mean += sample_weights(k) * probs[k][j] * samples[k];
        cum_prob += probs[k][j];
      }
      mean /= cum_prob;
      model.SetMean(mean, j);

      for (int k = 0; k < samples.size(); ++k) {
        const eigenmath::Vector2d diff = samples[k] - mean;
        covariance += sample_weights(k) * probs[k][j] * diff * diff.transpose();
      }
      covariance /= cum_prob;
      // Add regularization to covariance diagonal
      covariance += eigenmath::Matrix2d::Identity() * kReg;
      model.SetCovariance(covariance, j);
      model.SetWeight(cum_prob / samples.size(), j);
    }

    // Break early if change in mean of log probabilities is below threshold.
    if (std::abs(log_prob - prev_log_prob) < log_prob_change_thresh) {
      break;
    }

    prev_log_prob = log_prob;
  }

  return model;
}

std::vector<std::vector<int>> Cluster(
    const GaussianMixtureModel2d& model,
    const std::vector<eigenmath::Vector2d>& points) {
  std::vector<std::vector<int>> component_assignment(model.num_components());

  int i = 0;
  for (const eigenmath::Vector2d& point : points) {
    const auto to_prob = [&](int j) { return model.Probability(point, j); };
    const auto prob_range = genit::TransformRange(
        genit::IndexRange(0, model.num_components()), std::cref(to_prob));
    const int best_component =
        std::max_element(prob_range.begin(), prob_range.end()) -
        prob_range.begin();
    component_assignment[best_component].push_back(i++);
  }
  return component_assignment;
}

eigenmath::Matrix2d ScaleAndExpandEllipse(
    const eigenmath::Vector2d& mean, const eigenmath::Matrix2d& covariance,
    const std::vector<int>& assignments,
    const std::vector<eigenmath::Vector2d>& points, double margin) {
  Eigen::SelfAdjointEigenSolver<eigenmath::Matrix2d> solver(covariance);

  // Pre-compute square difference vectors.
  std::vector<eigenmath::Vector2d> sqr_diffs;
  sqr_diffs.reserve(assignments.size());
  for (const auto& point : assignments) {
    const eigenmath::Vector2d diff =
        solver.eigenvectors().transpose() * (points[point] - mean);
    const eigenmath::Vector2d sqr_diff = diff.cwiseProduct(diff);
    const double sqr_sqr_dist = sqr_diff.dot(sqr_diff);
    if (sqr_sqr_dist < std::numeric_limits<double>::epsilon()) {
      continue;
    }
    sqr_diffs.emplace_back(sqr_diff);
  }

  // Find best starting pivots.
  int pivots[] = {-1, -1};
  for (int i = 0; i < sqr_diffs.size(); ++i) {
    if (pivots[0] < 0 || sqr_diffs[i].x() > sqr_diffs[pivots[0]].x() ||
        (sqr_diffs[i].x() == sqr_diffs[pivots[0]].x() &&
         sqr_diffs[i].y() > sqr_diffs[pivots[0]].y())) {
      pivots[0] = i;
    }
    if (pivots[1] < 0 || sqr_diffs[i].y() > sqr_diffs[pivots[1]].y() ||
        (sqr_diffs[i].y() == sqr_diffs[pivots[1]].y() &&
         sqr_diffs[i].x() > sqr_diffs[pivots[1]].x())) {
      pivots[1] = i;
    }
  }

  // Walk from both pivots to the center.
  const int dirs[] = {1, -1};
  int bases[] = {pivots[0], pivots[1]};
  double sqr_areas[] = {std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()};
  eigenmath::Vector2d vertices[] = {
      eigenmath::Vector2d{sqr_diffs[pivots[0]].x(), 0.0},
      eigenmath::Vector2d{0.0, sqr_diffs[pivots[1]].y()}};

  for (int side = 0; side < 2; ++side) {
    while (true) {
      int prev_basis = bases[side];
      // Find next vertex in current direction.
      double min_pivot_cross_v = std::numeric_limits<double>::infinity();
      for (int i = 0; i < sqr_diffs.size(); ++i) {
        const double pivot_cross_p =
            eigenmath::CrossProduct(sqr_diffs[pivots[side]], sqr_diffs[i]) *
            dirs[side];
        if (i == pivots[side] ||
            pivot_cross_p <= std::numeric_limits<double>::epsilon()) {
          continue;
        }
        // Solve for intersection: [sd[pivot]^T; sd[i]^T] v = [1; 1]
        const eigenmath::Matrix2d A =
            (eigenmath::Matrix2d{} << sqr_diffs[pivots[side]].transpose(),
             sqr_diffs[i].transpose())
                .finished();
        const eigenmath::Vector2d v = A.inverse() * eigenmath::Vector2d::Ones();
        const double pivot_cross_v =
            eigenmath::CrossProduct(sqr_diffs[pivots[side]], v) * dirs[side];
        if (v.allFinite() && v.x() >= 0.0 && v.y() >= 0.0 &&
            pivot_cross_v < min_pivot_cross_v) {
          min_pivot_cross_v = pivot_cross_v;
          vertices[side] = v;
          bases[side] = i;
        }
      }
      const double new_sqr_area =
          1.0 / (vertices[side].x() * vertices[side].y());
      if (bases[side] == prev_basis || new_sqr_area >= sqr_areas[side]) {
        bases[side] = prev_basis;
        break;
      }
      sqr_areas[side] = new_sqr_area;
      std::swap(bases[side], pivots[side]);
    }
  }
  // At this point, pivots[side] is the leading edge, with bases[side]
  // lagging.
  eigenmath::Vector2d sqrt_diag = eigenmath::Vector2d::Zero();
  // Solve for min area: [sd[pivot]^T; -sd[pivot].x, sd[pivot].y] v = [1; 0]
  //  which is just v = 0.5*[1/sd[pivot].x; 1/sd[pivot].y].
  const eigenmath::Vector2d v0{0.5 / sqr_diffs[pivots[0]].x(),
                               0.5 / sqr_diffs[pivots[0]].y()};
  const eigenmath::Vector2d v1{0.5 / sqr_diffs[pivots[1]].x(),
                               0.5 / sqr_diffs[pivots[1]].y()};
  if (!std::isfinite(v0.x()) && !std::isfinite(v1.x())) {
    // Only y-aligned values.
    sqrt_diag = eigenmath::Vector2d(
        margin, std::sqrt(sqr_diffs[pivots[1]].y()) + margin);
  } else if (!std::isfinite(v0.y()) && !std::isfinite(v1.y())) {
    // Only x-aligned values.
    sqrt_diag = eigenmath::Vector2d(
        std::sqrt(sqr_diffs[pivots[0]].x()) + margin, margin);
  } else if (v0.dot(sqr_diffs[bases[0]]) <= 1.0 &&
             v1.dot(sqr_diffs[bases[1]]) <= 1.0) {
    const double sqr_area0 = 1.0 / (v0.x() * v0.y());
    const double sqr_area1 = 1.0 / (v1.x() * v1.y());
    // The largest area must be the one that lies in the feasible range.
    if (v0.allFinite() && v0.x() > 0.0 && v0.y() > 0.0 &&
        sqr_area0 > sqr_area1) {
      sqrt_diag = eigenmath::Vector2d(std::sqrt(1.0 / v0.x()) + margin,
                                      std::sqrt(1.0 / v0.y()) + margin);
    } else {
      sqrt_diag = eigenmath::Vector2d(std::sqrt(1.0 / v1.x()) + margin,
                                      std::sqrt(1.0 / v1.y()) + margin);
    }
  } else {
    // Solve for intersection: [sd[pivot]^T; sd[i]^T] v = [1; 1]
    const eigenmath::Matrix2d A =
        (eigenmath::Matrix2d{} << sqr_diffs[pivots[0]].transpose(),
         sqr_diffs[bases[0]].transpose())
            .finished();
    const eigenmath::Vector2d v = A.inverse() * eigenmath::Vector2d::Ones();
    sqrt_diag = eigenmath::Vector2d(std::sqrt(1.0 / v.x()) + margin,
                                    std::sqrt(1.0 / v.y()) + margin);
  }

  return solver.eigenvectors() *
         sqrt_diag.array().square().matrix().asDiagonal() *
         solver.eigenvectors().transpose();
}

}  // namespace mobility::collision

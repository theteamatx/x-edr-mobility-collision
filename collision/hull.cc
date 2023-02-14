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

#include "collision/hull.h"

#include <limits>
#include <numeric>
#include <vector>

namespace mobility::collision {

namespace {

template <typename HullContainer>
bool Contains(const HullContainer &hulls, const eigenmath::Vector2d &point) {
  for (const auto &hull : hulls) {
    if (hull.Contains(point)) {
      return true;
    }
  }
  return false;
}

template <typename HullContainer>
double Distance(const HullContainer &hulls, const eigenmath::Vector2d &point) {
  double min_positive = std::numeric_limits<double>::max();
  double max_negative = std::numeric_limits<double>::lowest();
  bool is_inside = false;
  for (const auto &hull : hulls) {
    double dist;
    if (is_inside) {
      dist = hull.DistanceIfPenetrating(point);
    } else {
      dist = hull.Distance(point);
    }
    if (dist > 0.0) {
      if (dist < min_positive) {
        min_positive = dist;
      }
    } else {
      if (dist > max_negative) {
        max_negative = dist;
      }
    }
    is_inside = is_inside || (dist <= 0.0);
  }
  return (is_inside ? max_negative : min_positive);
}

template <typename HullContainer>
double DistanceIfLessThan(const HullContainer &hulls,
                          const eigenmath::Vector2d &point,
                          double current_distance) {
  double max_negative = std::numeric_limits<double>::lowest();
  bool is_inside = false;
  for (const auto &hull : hulls) {
    const double dist = hull.DistanceIfLessThan(point, current_distance);
    if (dist <= current_distance) {
      if (dist > max_negative) {
        max_negative = dist;
      }
    }
    is_inside = is_inside || (dist < current_distance);
  }
  return (is_inside ? max_negative : 1.0);
}

template <typename HullContainer>
double ClosestPointIfPenetrating(const HullContainer &hulls,
                                 const eigenmath::Vector2d &point,
                                 eigenmath::Vector2d *closest_point) {
  double max_negative = std::numeric_limits<double>::lowest();
  bool is_inside = false;
  for (const auto &hull : hulls) {
    eigenmath::Vector2d temp_closest_point = eigenmath::Vector2d::Zero();
    const double dist =
        hull.ClosestPointIfPenetrating(point, &temp_closest_point);
    if (dist < 0.0 && dist > max_negative) {
      max_negative = dist;
      if (closest_point) {
        *closest_point = temp_closest_point;
      }
    }
    is_inside = is_inside || (dist < 0.0);
  }
  return (is_inside ? max_negative : 1.0);
}

template <typename LhsHulls, typename RhsHulls>
bool AreOverlappingImpl(const LhsHulls &lhs, const RhsHulls &rhs) {
  for (auto &lhs_chull : lhs) {
    for (auto &rhs_chull : rhs) {
      if (AreOverlapping(lhs_chull, rhs_chull)) {
        return true;
      }
    }
  }
  return false;
}

template <typename LhsHulls, typename RhsHulls>
double DistanceBetweenImpl(const LhsHulls &lhs, const RhsHulls &rhs,
                           double min_distance, double max_distance) {
  double result_min_dist = max_distance;
  for (auto &lhs_chull : lhs) {
    for (auto &rhs_chull : rhs) {
      const double dist =
          DistanceBetween(lhs_chull, rhs_chull, min_distance, result_min_dist);
      if (dist < result_min_dist) {
        result_min_dist = dist;
        if (result_min_dist < min_distance) {
          return result_min_dist;
        }
      }
    }
  }
  return result_min_dist;
}

template <typename DestHullVector, typename SrcHullVector,
          typename... ExtraArgs>
void ResetHullBaseImpl(DestHullVector *dest_hulls,
                       const SrcHullVector &src_hulls,
                       ExtraArgs &&...extra_args) {
  auto dest_it = dest_hulls->begin();
  for (auto &src : src_hulls) {
    if (dest_it == dest_hulls->end()) {
      dest_it = dest_hulls->emplace(
          dest_it, std::forward<ExtraArgs>(extra_args)..., &src);
    } else {
      dest_it->ResetHullBase(std::forward<ExtraArgs>(extra_args)..., &src);
    }
    ++dest_it;
  }
  if (dest_it != dest_hulls->end()) {
    dest_hulls->erase(dest_it, dest_hulls->end());
  }
}

}  // namespace

void Hull::Add(const std::vector<eigenmath::Vector2d> &points) {
  convex_hulls_.emplace_back(points);
}

void Hull::RemoveInternalHulls() {
  // Go through each hull and remove any convex hull that is entirely
  // contained in another.
  for (int i = 0; i < convex_hulls_.size(); ++i) {
    for (int j = 0; j < convex_hulls_.size(); ++j) {
      if (i == j) {
        continue;
      }
      if (convex_hulls_[j].Contains(convex_hulls_[i])) {
        convex_hulls_.erase(convex_hulls_.begin() + i);
        --i;
        break;
      }
    }
  }
}

void Hull::SimplifyHulls() {
  for (int i = 0; i < convex_hulls_.size(); ++i) {
    for (int j = 0; j < convex_hulls_.size(); ++j) {
      if (i == j) {
        continue;
      }
      if (convex_hulls_[j].ContainsWithCircle(convex_hulls_[i])) {
        std::vector<eigenmath::Vector2d> new_points;
        new_points.insert(new_points.end(), convex_hulls_[j].points_.begin(),
                          convex_hulls_[j].points_.end());
        new_points.insert(new_points.end(), convex_hulls_[i].points_.begin(),
                          convex_hulls_[i].points_.end());
        convex_hulls_[j] = ConvexHull(new_points);
        convex_hulls_.erase(convex_hulls_.begin() + i);
        --i;
        break;
      }
    }
  }
}

template <typename ConvexHullType>
bool HullBase<ConvexHullType>::Contains(
    const eigenmath::Vector2d &point) const {
  return mobility::collision::Contains(convex_hulls_, point);
}

template <typename ConvexHullType>
template <typename OtherType>
bool HullBase<ConvexHullType>::IsApprox(const HullBase<OtherType> &rhs,
                                        double tolerance) const {
  if (convex_hulls_.size() != rhs.convex_hulls_.size()) {
    return false;
  }
  std::vector<int> unmatched_hulls(rhs.convex_hulls_.size());
  std::iota(unmatched_hulls.begin(), unmatched_hulls.end(), 0);
  for (int i = 0; i < convex_hulls_.size(); ++i) {
    bool found_match = false;
    for (int j = 0; j < unmatched_hulls.size(); ++j) {
      if (convex_hulls_[i].IsApprox(rhs.convex_hulls_[unmatched_hulls[j]],
                                    tolerance)) {
        unmatched_hulls.erase(unmatched_hulls.begin() + j);
        found_match = true;
        break;
      }
    }
    if (!found_match) {
      return false;
    }
  }
  return unmatched_hulls.empty();
}

template <typename ConvexHullType>
double HullBase<ConvexHullType>::Distance(
    const eigenmath::Vector2d &point) const {
  return mobility::collision::Distance(convex_hulls_, point);
}

template <typename ConvexHullType>
double HullBase<ConvexHullType>::DistanceIfPenetrating(
    const eigenmath::Vector2d &point) const {
  return mobility::collision::DistanceIfLessThan(convex_hulls_, point, 0.0);
}

template <typename ConvexHullType>
double HullBase<ConvexHullType>::DistanceIfLessThan(
    const eigenmath::Vector2d &point, double current_distance) const {
  return mobility::collision::DistanceIfLessThan(convex_hulls_, point,
                                                 current_distance);
}

template <typename ConvexHullType>
double HullBase<ConvexHullType>::ClosestPointIfPenetrating(
    const eigenmath::Vector2d &point,
    eigenmath::Vector2d *closest_point) const {
  return mobility::collision::ClosestPointIfPenetrating(convex_hulls_, point,
                                                        closest_point);
}

bool Hull::TransformAndCopy(const eigenmath::Pose2d &rel_pose,
                            const Hull &orig_hull) {
  if (convex_hulls_.size() != orig_hull.convex_hulls_.size()) {
    return false;
  }
  for (int i = 0; i < convex_hulls_.size(); ++i) {
    if (!convex_hulls_[i].TransformAndCopy(rel_pose,
                                           orig_hull.convex_hulls_[i])) {
      return false;
    }
  }
  return true;
}

template <typename ConvexHullType>
double HullBase<ConvexHullType>::GetMaxSquaredRadiusAround(
    const eigenmath::Vector2d &point) const {
  double max_radius_sqr = 0.0;
  for (int i = 0; i < convex_hulls_.size(); ++i) {
    const double ch_radius_sqr =
        convex_hulls_[i].GetMaxSquaredRadiusAround(point);
    if (ch_radius_sqr > max_radius_sqr) {
      max_radius_sqr = ch_radius_sqr;
    }
  }
  return max_radius_sqr;
}

template <typename ConvexHullType>
double HullBase<ConvexHullType>::GetMaxRadiusAround(
    const eigenmath::Vector2d &point) const {
  return std::sqrt(GetMaxSquaredRadiusAround(point));
}

LazyTransformedHull::LazyTransformedHull(const Hull &orig_hull) {
  ResetHullBaseImpl(&convex_hulls_, orig_hull.convex_hulls_);
}

void LazyTransformedHull::ResetHullBase(const Hull &orig_hull) {
  ResetHullBaseImpl(&convex_hulls_, orig_hull.convex_hulls_);
}

void LazyTransformedHull::ApplyTransform(
    const eigenmath::Pose2d &transformed_pose_orig) {
  for (auto &hull : convex_hulls_) {
    hull.ApplyTransform(transformed_pose_orig);
  }
}

LazyTransformedAugmentedHull::LazyTransformedAugmentedHull(
    const LazyAugmentedHull &orig_hull) {
  ResetHullBaseImpl(&convex_hulls_, orig_hull.convex_hulls_);
}

void LazyTransformedAugmentedHull::ResetHullBase(
    const LazyAugmentedHull &orig_hull) {
  ResetHullBaseImpl(&convex_hulls_, orig_hull.convex_hulls_);
}

void LazyTransformedAugmentedHull::ApplyTransform(
    const eigenmath::Pose2d &transformed_pose_orig) {
  for (auto &hull : convex_hulls_) {
    hull.ApplyTransform(transformed_pose_orig);
  }
}

LazyAugmentedHull::LazyAugmentedHull(const diff_drive::DynamicLimits *limits,
                                     const Hull &orig_hull) {
  ResetHullBaseImpl(&convex_hulls_, orig_hull.convex_hulls_, limits);
}

void LazyAugmentedHull::ResetHullBase(const diff_drive::DynamicLimits *limits,
                                      const Hull &orig_hull) {
  ResetHullBaseImpl(&convex_hulls_, orig_hull.convex_hulls_, limits);
}

void LazyAugmentedHull::ApplyMotion(const diff_drive::State &state) {
  for (auto &convex_hull : convex_hulls_) {
    convex_hull.ApplyMotion(state);
  }
}

void LazyAugmentedHull::SetDangerMargin(double danger_margin) {
  for (auto &convex_hull : convex_hulls_) {
    convex_hull.SetDangerMargin(danger_margin);
  }
}

template <typename Lhs, typename Rhs>
double DistanceBetween(const HullBase<Lhs> &lhs, const HullBase<Rhs> &rhs,
                       double min_distance, double max_distance) {
  return DistanceBetweenImpl(lhs.GetConvexHulls(), rhs.GetConvexHulls(),
                             min_distance, max_distance);
}

template <typename Lhs, typename Rhs>
bool AreOverlapping(const HullBase<Lhs> &lhs, const HullBase<Rhs> &rhs) {
  return AreOverlappingImpl(lhs.GetConvexHulls(), rhs.GetConvexHulls());
}

template class HullBase<EagerConvexHull>;
template class HullBase<LazyTransformedConvexHull>;
template class HullBase<LazyAugmentedConvexHull>;
template class HullBase<LazyTransformedAugmentedConvexHull>;

// Function template instantiations.
#define INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(lhs, rhs)        \
  template double DistanceBetween(const HullBase<lhs> &,                      \
                                  const HullBase<rhs> &, double, double);     \
  template bool AreOverlapping(const HullBase<lhs> &, const HullBase<rhs> &); \
  template bool HullBase<lhs>::IsApprox(const HullBase<rhs> &, double) const;

INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(EagerConvexHull,
                                                     EagerConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(EagerConvexHull,
                                                     LazyAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(EagerConvexHull,
                                                     LazyTransformedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    EagerConvexHull, LazyTransformedAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyAugmentedConvexHull,
                                                     EagerConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyAugmentedConvexHull,
                                                     LazyAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyAugmentedConvexHull,
                                                     LazyTransformedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyAugmentedConvexHull, LazyTransformedAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyTransformedConvexHull,
                                                     EagerConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyTransformedConvexHull,
                                                     LazyAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(LazyTransformedConvexHull,
                                                     LazyTransformedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyTransformedConvexHull, LazyTransformedAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyTransformedAugmentedConvexHull, EagerConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyTransformedAugmentedConvexHull, LazyAugmentedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyTransformedAugmentedConvexHull, LazyTransformedConvexHull);
INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES(
    LazyTransformedAugmentedConvexHull, LazyTransformedAugmentedConvexHull);

#undef INSTANTIATE_FUNCTION_TEMPLATES_FOR_CONVEX_HULL_TYPES

}  // namespace mobility::collision

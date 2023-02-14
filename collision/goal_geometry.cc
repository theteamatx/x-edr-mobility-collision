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

#include "collision/goal_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <variant>
#include <vector>

#include "eigenmath/so2.h"

namespace mobility::collision {

namespace {
// Returns a minimally shifted copy of `frame` so that `point` lies exactly on
// it.
LatticeFrame ShiftFrameToMatchPoint(const LatticeFrame& frame,
                                    const eigenmath::Vector2d& point) {
  const eigenmath::Vector2d closest_point =
      frame.GridToFrame(frame.FrameToGrid(point));
  auto shifted = frame;
  shifted.origin.translation() += (point - closest_point);
  return shifted;
}

}  // namespace

bool GoalGeometry::IsValid() const {
  return inclusion_zone_.index() != kNoGoalRegion ||
         attraction_field_.index() == kAttractionPoint;
}

GridRange GoalGeometry::GetGridRangeForGoalRegion(
    const GridFrame& frame, const GoalRegionVariant& goal_region) const {
  GridRange result;
  switch (goal_region.index()) {
    case kNoGoalRegion:
      break;
    case kRadialSegment: {
      auto& radial_segment = std::get<kRadialSegment>(goal_region);
      result = frame.FrameCircleToGridRange(radial_segment.center,
                                            radial_segment.outer_radius);
      break;
    }
    case kRegionHull: {
      auto& hull = std::get<kRegionHull>(goal_region);
      for (auto& chull : hull.GetConvexHulls()) {
        result.SpanningUnion(frame.FrameCircleToGridRange(chull.GetCentroid(),
                                                          chull.GetRadius()));
      }
      break;
    }
  }
  return result;
}

bool GoalGeometry::IsPointInGoalRegion(const eigenmath::Vector2d& point,
                                       const GoalRegionVariant& goal_region,
                                       double buffer_radius) const {
  switch (goal_region.index()) {
    case kNoGoalRegion:
      return false;
    case kRadialSegment: {
      auto& radial_segment = std::get<kRadialSegment>(goal_region);
      const eigenmath::Vector2d center_diff = point - radial_segment.center;
      const double center_dist = center_diff.norm();
      if (center_dist < 1e-6) {
        // If at the center of the radial segment, the angle range no longer
        // matters.
        return ((center_dist <= radial_segment.outer_radius + buffer_radius) &&
                (center_dist >= radial_segment.inner_radius - buffer_radius));
      } else {
        const eigenmath::Vector2d center_dir = center_diff / center_dist;
        const eigenmath::SO2d radial_dir(center_dir.x(), center_dir.y(), false);
        return (IsInInterval(
                    radial_dir,
                    radial_segment.start_angle - buffer_radius / center_dist,
                    radial_segment.end_angle + buffer_radius / center_dist) &&
                (center_dist <= radial_segment.outer_radius + buffer_radius) &&
                (center_dist >= radial_segment.inner_radius - buffer_radius));
      }
    }
    case kRegionHull: {
      auto& hull = std::get<kRegionHull>(goal_region);
      return hull.Distance(point) <= buffer_radius;
    }
  }
  return false;
}

eigenmath::Vector2d GoalGeometry::ComputeBestPossibleGoal() const {
  if (attraction_field_.index() == kAttractionPoint) {
    // Find the point in the inclusion zone that is the closest to the
    // attraction point.
    auto& attraction_point = std::get<kAttractionPoint>(attraction_field_);
    switch (inclusion_zone_.index()) {
      case kNoGoalRegion:
        // If no inclusion zone, the best point is just the attraction point.
        return attraction_point;
      case kRadialSegment: {
        auto& radial_segment = std::get<kRadialSegment>(inclusion_zone_);
        const eigenmath::Vector2d dir =
            attraction_point - radial_segment.center;
        const double dir_norm = dir.norm();
        if (dir_norm < radial_segment.outer_radius) {
          // If the attraction point is within the outer radius of the
          // inclusion zone, then it is the best point.
          return attraction_point;
        } else {
          // If not, the best point is the one on the outer circle of the
          // inclusion zone, towards the attraction point.
          return radial_segment.center +
                 (radial_segment.outer_radius / dir_norm) * dir;
        }
      }
      case kRegionHull: {
        // For simplicity, treat each convex hull as a circle and use the same
        // logic as for the radial segments to determine the best point.
        auto& hull = std::get<kRegionHull>(inclusion_zone_);
        double best_dist = std::numeric_limits<double>::max();
        eigenmath::Vector2d best_point = attraction_point;
        for (auto& chull : hull.GetConvexHulls()) {
          // Best point on the circle around the convex hull:
          const eigenmath::Vector2d dir =
              attraction_point - chull.GetCentroid();
          const double dir_norm = dir.norm();
          if (dir_norm - chull.GetRadius() < best_dist) {
            best_dist = dir_norm - chull.GetRadius();
            if (dir_norm < chull.GetRadius()) {
              best_point = attraction_point;
              break;
            } else {
              best_point =
                  chull.GetCentroid() + (chull.GetRadius() / dir_norm) * dir;
            }
          }
        }
        return best_point;
      }
    }
  }
  if (attraction_field_.index() == kNoAttraction) {
    // Without an attraction field, pick the center of the inclusion zone.
    switch (inclusion_zone_.index()) {
      case kNoGoalRegion:
        // Nothing to output here (probably not a valid goal geometry anyways).
        return eigenmath::Vector2d(0.0, 0.0);
      case kRadialSegment: {
        // Use the center of the radial segment.
        auto& radial_segment = std::get<kRadialSegment>(inclusion_zone_);
        return radial_segment.center;
      }
      case kRegionHull: {
        // Use the centroid of the largest convex hull.
        auto& hull = std::get<kRegionHull>(inclusion_zone_);
        double max_radius = std::numeric_limits<double>::lowest();
        eigenmath::Vector2d best_point(0.0, 0.0);
        for (auto& chull : hull.GetConvexHulls()) {
          if (chull.GetRadius() > max_radius) {
            max_radius = chull.GetRadius();
            best_point = chull.GetCentroid();
          }
        }
        return best_point;
      }
    }
  }
  if (attraction_field_.index() == kAttractionDirection) {
    // Find the point in the inclusion zone that is the farthest along the
    // attraction direction.
    auto& dir = std::get<kAttractionDirection>(attraction_field_);
    switch (inclusion_zone_.index()) {
      case kNoGoalRegion:
        // Nothing to output here (probably not a valid goal geometry anyways).
        return eigenmath::Vector2d(0.0, 0.0);
      case kRadialSegment: {
        // Use the outer edge of the radial segment, in attraction direction.
        auto& radial_segment = std::get<kRadialSegment>(inclusion_zone_);
        return radial_segment.center + radial_segment.outer_radius * dir;
      }
      case kRegionHull: {
        // For simplicity, treat each convex hull as a circle and use the same
        // logic as for the radial segments to determine the best point.
        auto& hull = std::get<kRegionHull>(inclusion_zone_);
        double best_dist = std::numeric_limits<double>::lowest();
        eigenmath::Vector2d best_point(0.0, 0.0);
        for (auto& chull : hull.GetConvexHulls()) {
          // Best point on the circle around the convex hull:
          const eigenmath::Vector2d candidate_point =
              chull.GetCentroid() + chull.GetRadius() * dir;
          const double candidate_dist = candidate_point.dot(dir);
          if (candidate_dist > best_dist) {
            best_dist = candidate_dist;
            best_point = candidate_point;
          }
        }
        return best_point;
      }
    }
  }
  return eigenmath::Vector2d(0.0, 0.0);
}

double GoalGeometry::ComputeGoalCost(
    const eigenmath::Vector2d& point,
    const eigenmath::Vector2d& best_point) const {
  switch (attraction_field_.index()) {
    case kNoAttraction:
      return 0.0;
    case kAttractionPoint: {
      // Use the L1 norm to be above any distance based cost (such as an A*
      // search distance).
      auto& attraction_point = std::get<kAttractionPoint>(attraction_field_);
      return (point - attraction_point).lpNorm<1>();
    }
    case kAttractionDirection: {
      auto& attraction_dir = std::get<kAttractionDirection>(attraction_field_);
      // Scale to be larger than any distance we would use.
      const double norm_scaling = attraction_dir.lpNorm<1>();
      return (best_point - point).dot(attraction_dir) * norm_scaling;
    }
  }
  return 0.0;
}

double GoalGeometry::ComputeGoalCostForPose(
    const eigenmath::Pose2d& pose) const {
  if (inclusion_zone_.index() == kNoGoalRegion) {
    return 0.0;
  }
  const eigenmath::Vector2d best_goal = ComputeBestPossibleGoal();
  const eigenmath::Vector2d world_point = pose.translation();
  return ComputeGoalCost(world_point, best_goal);
}

Interval<double> GoalGeometry::AnglesAtGoal(const GridFrame& frame,
                                            const GridIndex& goal_index) const {
  // Default covers the full circle.
  Interval<double> angles(0.0, 10.0);
  switch (orientation_.index()) {
    case kNoOrientation: {
      // Use the full circle.
      break;
    }
    case kOrientation: {
      const auto& angle = std::get<kOrientation>(orientation_);
      angles = {angle - orientation_range_, angle + orientation_range_};
      break;
    }
    case kFaceTowardsPoint: {
      const auto& point = std::get<kFaceTowardsPoint>(orientation_);
      const eigenmath::Vector2d central_diff =
          point - frame.GridToFrame(goal_index);
      const double central_dist = central_diff.norm();
      if (central_dist <= 1e-6) {
        // Use the full circle.
        break;
      }
      const eigenmath::Vector2d central_dir = central_diff / central_dist;
      const eigenmath::SO2d central_so2(central_dir.x(), central_dir.y(),
                                        false);
      const double angle = central_so2.angle();
      angles = {angle - orientation_range_, angle + orientation_range_};
      break;
    }
  }
  return angles;
}

bool GoalGeometry::IsPointInGoalGeometry(
    const eigenmath::Vector2d& point) const {
  if (inclusion_zone_.index() == kNoGoalRegion &&
      attraction_field_.index() == kAttractionPoint) {
    auto& attraction_point = std::get<kAttractionPoint>(attraction_field_);
    return (attraction_point - point).squaredNorm() <
           distance_tolerance_ * distance_tolerance_;
  }
  if (!IsPointInGoalRegion(point, inclusion_zone_) ||
      IsPointInGoalRegion(point, exclusion_zone_)) {
    return false;
  }
  return true;
}

bool GoalGeometry::IsPointInRadiusOfGoalGeometry(
    const eigenmath::Vector2d& point, double buffer_radius) const {
  if (inclusion_zone_.index() == kNoGoalRegion &&
      attraction_field_.index() == kAttractionPoint) {
    auto& attraction_point = std::get<kAttractionPoint>(attraction_field_);
    return (attraction_point - point).squaredNorm() <
           (distance_tolerance_ + buffer_radius) *
               (distance_tolerance_ + buffer_radius);
  }
  if (!IsPointInGoalRegion(point, inclusion_zone_, buffer_radius) ||
      IsPointInGoalRegion(point, exclusion_zone_, -buffer_radius)) {
    return false;
  }
  return true;
}

// Returns the angle difference of the pose's orientation to the target range
// of orientations at the pose's position.
double GoalGeometry::DistanceToTargetOrientation(
    const eigenmath::Pose2d& pose) const {
  // Center grid at the point to get the orientation angles at the point.
  const GridFrame frame("dummy", pose, 1.0);
  const Interval<double> angles = AnglesAtGoal(frame, {0, 0});
  // Check distance to the boundary points.
  if (eigenmath::IsInInterval(pose.so2(), angles.min(), angles.max())) {
    return 0.0;
  }
  const eigenmath::SO2d relative_orientations[] = {
      eigenmath::SO2d(angles.min()).inverse() * pose.so2(),
      eigenmath::SO2d(angles.max()).inverse() * pose.so2()};
  return std::min(relative_orientations[0].norm(),
                  relative_orientations[1].norm());
}

bool GoalGeometry::IntersectsGridRange(const GridFrame& frame,
                                       const GridRange& range) const {
  bool found_valid_goal = false;
  ForEachGoalPoint(frame, range, [&](const GridIndex& goal_index) {
    found_valid_goal = true;
  });
  return found_valid_goal;
}

GridRange GoalGeometry::GetGoalGridRange(const GridFrame& frame) const {
  if (inclusion_zone_.index() == kNoGoalRegion &&
      attraction_field_.index() != kAttractionPoint) {
    // No goal.
    return GridRange();
  }
  if (inclusion_zone_.index() == kNoGoalRegion &&
      attraction_field_.index() == kAttractionPoint) {
    auto& point = std::get<kAttractionPoint>(attraction_field_);
    return frame.FrameCircleToGridRange(point, distance_tolerance_);
  }
  return GetGridRangeForGoalRegion(frame, inclusion_zone_);
}

void GoalGeometry::SampleGoalsOnGrid(const GridFrame& frame,
                                     std::vector<GridIndex>* goals,
                                     std::vector<double>* costs) const {
  SampleGoalsOnGrid(frame, GridRange::Unlimited(), goals, costs);
}

void GoalGeometry::SampleGoalsOnGrid(const GridFrame& frame,
                                     const GridRange& range,
                                     std::vector<GridIndex>* goals,
                                     std::vector<double>* costs) const {
  goals->clear();
  costs->clear();
  const eigenmath::Vector2d best_goal = ComputeBestPossibleGoal();
  ForEachGoalPoint(frame, range, [&](const GridIndex& goal_index) {
    goals->push_back(goal_index);
    const eigenmath::Vector2d world_point = frame.GridToFrame(goal_index);
    costs->push_back(ComputeGoalCost(world_point, best_goal));
  });
}

void GoalGeometry::SampleGoalsOnLattice(const LatticeFrame& frame,
                                        std::vector<LatticePose>* goals,
                                        std::vector<double>* costs) const {
  SampleGoalsOnLattice(frame, GridRange::Unlimited(), goals, costs);
}

void GoalGeometry::SampleGoalsOnLattice(const LatticeFrame& frame,
                                        const GridRange& range,
                                        std::vector<LatticePose>* goals,
                                        std::vector<double>* costs) const {
  goals->clear();
  costs->clear();
  const eigenmath::Vector2d best_goal = ComputeBestPossibleGoal();
  ForEachGoalPoint(frame, range, [&](const GridIndex& goal_index) {
    const eigenmath::Vector2d goal_point = frame.GridToFrame(goal_index);
    const double cost = ComputeGoalCost(goal_point, best_goal);
    const Interval<double> angles = AnglesAtGoal(frame, goal_index);
    if (angles.Length() >= 2 * M_PI) {
      for (int i = 0; i < frame.num_angle_divisions; ++i) {
        goals->emplace_back(goal_index, i);
        costs->push_back(cost);
      }
    } else {
      const double central_angle = (angles.max() + angles.min()) / 2;
      const eigenmath::SO2d central_orientation(central_angle);
      const int central_angle_id =
          frame.FrameSO2ToLatticeAngleIndex(central_orientation);
      goals->emplace_back(goal_index, central_angle_id);
      costs->push_back(cost);
      for (int i = central_angle_id + 1;
           i < central_angle_id + frame.num_angle_divisions; ++i) {
        const int angle_id = i % frame.num_angle_divisions;
        const eigenmath::SO2d orientation =
            frame.LatticeAngleIndexToFrameSO2(angle_id);
        const double relative_angle =
            (central_orientation.inverse() * orientation).angle();
        if (angles.Contains(central_angle + relative_angle)) {
          goals->emplace_back(goal_index, angle_id);
          costs->emplace_back(cost);
        }
      }
    }
  });
}

void GoalGeometry::SampleGoalsWithLattice(const LatticeFrame& frame,
                                          std::vector<eigenmath::Pose2d>* goals,
                                          std::vector<double>* costs) const {
  return SampleGoalsWithLattice(frame, GridRange::Unlimited(), goals, costs);
}

void GoalGeometry::SampleGoalsWithLattice(const LatticeFrame& frame,
                                          const GridRange& range,
                                          std::vector<eigenmath::Pose2d>* goals,
                                          std::vector<double>* costs) const {
  goals->clear();
  costs->clear();
  // Translate frame to align with goal geometry definition.
  // We could also rotate the frame to make the sampled points depend only on
  // the frame (distance and angular) resolution.
  auto adjusted_frame = frame;
  if (IsAttractedToSinglePoint()) {
    adjusted_frame =
        ShiftFrameToMatchPoint(adjusted_frame, GetAttractionPoint());
  } else if (IsToFaceTowardsPoint()) {
    adjusted_frame =
        ShiftFrameToMatchPoint(adjusted_frame, GetOrientationTarget());
  } else if (HasInclusionRadialSegment()) {
    adjusted_frame = ShiftFrameToMatchPoint(adjusted_frame,
                                            GetInclusionRadialSegment().center);
  }

  const eigenmath::Vector2d best_goal = ComputeBestPossibleGoal();
  ForEachGoalPoint(adjusted_frame, range, [&](const GridIndex& goal_index) {
    const eigenmath::Vector2d goal_point =
        adjusted_frame.GridToFrame(goal_index);
    const double cost = ComputeGoalCost(goal_point, best_goal);
    const Interval<double> angles = AnglesAtGoal(adjusted_frame, goal_index);
    if (angles.Length() >= 2 * M_PI) {
      // Use all lattice orientations.
      for (int i = 0; i < adjusted_frame.num_angle_divisions; ++i) {
        goals->emplace_back(goal_point,
                            adjusted_frame.LatticeAngleIndexToFrameSO2(i));
        costs->push_back(cost);
      }
    } else {
      // Sample angles evenly in the interval, centering the samples at the
      // central angle.  Keep the boundary samples at `orientation_tolerance`
      // from the interval boundary.  Use up to `frame.num_angle_divisions`
      // samples.
      const double central_angle = (angles.max() + angles.min()) / 2;
      const double half_range =
          std::max(0.0, angles.Length() / 2 - orientation_tolerance_);
      // Samples to each direction of `central_angle`.
      const int n_samples_bound =
          std::max(0, (frame.num_angle_divisions - 1) / 2);
      const int n_samples = std::min<int>(
          n_samples_bound, std::lrint(half_range / orientation_tolerance_));
      const double angle_step = n_samples > 0 ? half_range / n_samples : 0.0;
      for (int i = -n_samples; i <= n_samples; ++i) {
        const double angle = central_angle + i * angle_step;
        const eigenmath::SO2d orientation(angle);
        goals->emplace_back(goal_point, orientation);
        costs->push_back(cost);
      }
    }
  });
}

eigenmath::SO2d GoalGeometry::ComputeOrientation(
    const eigenmath::Vector2d& point) const {
  switch (orientation_.index()) {
    case kNoOrientation:
      return eigenmath::SO2d();
    case kOrientation:
      return eigenmath::SO2d(std::get<kOrientation>(orientation_));
    case kFaceTowardsPoint: {
      auto& target_point = std::get<kFaceTowardsPoint>(orientation_);
      const eigenmath::Vector2d central_diff = target_point - point;
      const double central_dist = central_diff.norm();
      eigenmath::SO2d central_so2;
      if (central_dist > 1e-6) {
        const eigenmath::Vector2d central_dir = central_diff / central_dist;
        central_so2 = eigenmath::SO2d(central_dir.x(), central_dir.y(), false);
      }
      return central_so2;
    }
  }
  return eigenmath::SO2d();
}

void GoalGeometry::ApplyTransform(const eigenmath::Pose2d& new_pose_old) {
  switch (attraction_field_.index()) {
    case kNoAttraction:
      break;
    case kAttractionPoint: {
      auto& point = std::get<kAttractionPoint>(attraction_field_);
      point = new_pose_old * point;
      break;
    }
    case kAttractionDirection: {
      auto& dir = std::get<kAttractionDirection>(attraction_field_);
      dir = new_pose_old.so2() * dir;
      break;
    }
  }

  switch (orientation_.index()) {
    case kNoOrientation:
      break;
    case kOrientation: {
      auto& angle = std::get<kOrientation>(orientation_);
      angle = (new_pose_old.so2() * eigenmath::SO2d(angle)).angle();
      break;
    }
    case kFaceTowardsPoint: {
      auto& point = std::get<kFaceTowardsPoint>(orientation_);
      point = new_pose_old * point;
      break;
    }
  }

  switch (inclusion_zone_.index()) {
    case kNoGoalRegion:
      break;
    case kRadialSegment: {
      auto& radial_segment = std::get<kRadialSegment>(inclusion_zone_);
      radial_segment.center = new_pose_old * radial_segment.center;
      const double delta_angle =
          radial_segment.end_angle - radial_segment.start_angle;
      radial_segment.start_angle =
          (new_pose_old.so2() * eigenmath::SO2d(radial_segment.start_angle))
              .angle();
      radial_segment.end_angle = radial_segment.start_angle + delta_angle;
      break;
    }
    case kRegionHull: {
      auto& hull = std::get<kRegionHull>(inclusion_zone_);
      hull.ApplyTransform(new_pose_old);
      break;
    }
  }

  switch (exclusion_zone_.index()) {
    case kNoGoalRegion:
      break;
    case kRadialSegment: {
      auto& radial_segment = std::get<kRadialSegment>(exclusion_zone_);
      radial_segment.center = new_pose_old * radial_segment.center;
      const double delta_angle =
          radial_segment.end_angle - radial_segment.start_angle;
      radial_segment.start_angle =
          (new_pose_old.so2() * eigenmath::SO2d(radial_segment.start_angle))
              .angle();
      radial_segment.end_angle = radial_segment.start_angle + delta_angle;
      break;
    }
    case kRegionHull: {
      auto& hull = std::get<kRegionHull>(exclusion_zone_);
      hull.ApplyTransform(new_pose_old);
      break;
    }
  }
}

void GoalGeometry::ForEachGoalPoint(
    const GridFrame& frame, const GridRange& range,
    absl::FunctionRef<void(const GridIndex&)> goal_visitor) const {
  if (inclusion_zone_.index() == kNoGoalRegion &&
      attraction_field_.index() != kAttractionPoint) {
    // No goal.
    return;
  }
  if (inclusion_zone_.index() == kNoGoalRegion &&
      attraction_field_.index() == kAttractionPoint) {
    auto& point = std::get<kAttractionPoint>(attraction_field_);
    const GridIndex index = frame.FrameToGrid(point);
    if (range.Contains(index)) {
      goal_visitor(index);
    }
    return;
  }
  const GridRange inclusion_range = GridRange::Intersect(
      GetGridRangeForGoalRegion(frame, inclusion_zone_), range);
  const GridRange exclusion_range =
      GetGridRangeForGoalRegion(frame, exclusion_zone_);
  int goal_points_found = 0;
  inclusion_range.ForEachGridCoord([&](const GridIndex& index) {
    const eigenmath::Vector2d world_point = frame.GridToFrame(index);
    if (!IsPointInGoalRegion(world_point, inclusion_zone_)) {
      return;
    }
    if (exclusion_range.Contains(index) &&
        IsPointInGoalRegion(world_point, exclusion_zone_)) {
      return;
    }
    ++goal_points_found;
    goal_visitor(index);
  });
  if (goal_points_found == 0) {
    inclusion_range.ForEachGridCoord([&](const GridIndex& index) {
      const eigenmath::Vector2d world_point = frame.GridToFrame(index);
      if (!IsPointInGoalRegion(world_point, inclusion_zone_,
                               frame.resolution)) {
        return;
      }
      if (exclusion_range.Contains(index) &&
          IsPointInGoalRegion(world_point, exclusion_zone_)) {
        return;
      }
      ++goal_points_found;
      goal_visitor(index);
    });
  }
}

}  // namespace mobility::collision

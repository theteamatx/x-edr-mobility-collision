/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MOBILITY_COLLISION_COLLISION_GOAL_GEOMETRY_H_
#define MOBILITY_COLLISION_COLLISION_GOAL_GEOMETRY_H_

#include <algorithm>
#include <memory>
#include <variant>
#include <vector>

#include "absl/functional/function_ref.h"
#include "collision/grid_common.h"
#include "collision/hull.h"
#include "collision/lattice_pose.h"
#include "diff_drive/interval.h"
#include "eigenmath/pose2.h"
#include "eigenmath/so2.h"
#include "eigenmath/types.h"

namespace mobility::collision {

// A radial segment is a complete or partial donut shape.
struct RadialSegment {
  eigenmath::Vector2d center = eigenmath::Vector2d::Zero();
  double inner_radius = 0.0;
  double outer_radius = 0.0;

  // The end_angle and start_angle values do not have to be in [-pi, pi]
  // interval and will be treated as going ccw from start to end.
  // end_angle must be greater than start_angle (not wrapping around).
  double start_angle = 0.0;
  double end_angle = 10.0;

  RadialSegment() = default;

  RadialSegment(const eigenmath::Vector2d& center_, double inner_radius_,
                double outer_radius_, double start_angle_ = 0.0,
                double end_angle_ = 10.0)
      : center(center_),
        inner_radius(inner_radius_),
        outer_radius(outer_radius_),
        start_angle(start_angle_),
        end_angle(end_angle_) {}
};

// This class stores the common goal (or planning objective) for the planners
// in terms of:
//  - A goal region that covers all the areas within which a planner could
//    consider the goal to have been achieved (success). This is specified as
//    an inclusion zone and an exclusion zone.
//  - An attraction field that determines, through a bias, where in the goal
//    region it is most desirable to end up (e.g., closest to a point, farthest
//    in a direction).
//  - An orientation specification that determines what the desirable
//    orientation is (which may depend on the robot's position).
//
// Like any other geometric element, you have to know in what frame you have
// it in. This is no different than a 2d vector or a convex hull. This class
// is agnostic to what frame its geometry is expressed in. There is an
// ApplyTransform function to transform the geometry from one frame to another.
class GoalGeometry {
 public:
  // Clear the goal geometry to the default-constructed value (empty).
  void Clear() { *this = GoalGeometry(); }

  // Check if the goal geometry is valid.
  bool IsValid() const;

  // Checks if the given point is within the goal geometry.
  bool IsPointInGoalGeometry(const eigenmath::Vector2d& point) const;

  // Checks if the given point is within a given radius of the goal geometry.
  bool IsPointInRadiusOfGoalGeometry(const eigenmath::Vector2d& point,
                                     double buffer_radius) const;

  // Returns the angle difference of the pose's orientation to the target range
  // of orientations at the pose's position.
  double DistanceToTargetOrientation(const eigenmath::Pose2d& pose) const;

  // Checks if the goal regions intersect with a given grid frame and range:
  bool IntersectsGridRange(const GridFrame& frame,
                           const GridRange& range) const;

  // Gets the grid range needed in a grid to include all the goal geometry:
  GridRange GetGoalGridRange(const GridFrame& frame) const;

  // Sample goal points for a given grid frame:
  void SampleGoalsOnGrid(const GridFrame& frame, std::vector<GridIndex>* goals,
                         std::vector<double>* costs) const;

  // Sample goal points for a given grid frame and range:
  void SampleGoalsOnGrid(const GridFrame& frame, const GridRange& range,
                         std::vector<GridIndex>* goals,
                         std::vector<double>* costs) const;

  // Sample goal points for a given lattice frame:
  void SampleGoalsOnLattice(const LatticeFrame& frame,
                            std::vector<LatticePose>* goals,
                            std::vector<double>* costs) const;

  // Sample goal points for a given lattice frame and range:
  void SampleGoalsOnLattice(const LatticeFrame& frame, const GridRange& range,
                            std::vector<LatticePose>* goals,
                            std::vector<double>* costs) const;

  // Aligns the frame with the goal and samples goal poses.
  void SampleGoalsWithLattice(const LatticeFrame& frame, const GridRange& range,
                              std::vector<eigenmath::Pose2d>* goals,
                              std::vector<double>* costs) const;

  // Aligns the frame with the goal and samples goal poses.
  void SampleGoalsWithLattice(const LatticeFrame& frame,
                              std::vector<eigenmath::Pose2d>* goals,
                              std::vector<double>* costs) const;

  // Applies a given transform from old to new to the geometric elements
  // contained in this object.
  void ApplyTransform(const eigenmath::Pose2d& new_pose_old);

  // Sets the attraction field to nothing.
  void SetAttractionToNothing() { attraction_field_.emplace<kNoAttraction>(); }

  // Sets the attraction field to point towards a single point.
  void SetAttractionPoint(const eigenmath::Vector2d& attraction_point) {
    attraction_field_.emplace<kAttractionPoint>(attraction_point);
  }

  // Gets the attraction point.
  const eigenmath::Vector2d& GetAttractionPoint() const {
    return std::get<kAttractionPoint>(attraction_field_);
  }

  // Sets the attraction field to point towards a single point.
  void SetAttractionDirection(const eigenmath::Vector2d& attraction_dir) {
    attraction_field_.emplace<kAttractionDirection>(
        attraction_dir.normalized());
  }

  // Gets the attraction direction.
  const eigenmath::Vector2d& GetAttractionDirection() const {
    return std::get<kAttractionDirection>(attraction_field_);
  }

  // Checks if the attraction field is set to nothing.
  bool IsAttractedToNothing() const {
    return attraction_field_.index() == kNoAttraction;
  }

  // Checks if the attraction field is set to a single point.
  bool IsAttractedToSinglePoint() const {
    return attraction_field_.index() == kAttractionPoint;
  }

  // Checks if the attraction field is set to a direction.
  bool IsAttractedInDirection() const {
    return attraction_field_.index() == kAttractionDirection;
  }

  // Sets the desired orientation to be arbitrary.
  void SetArbitraryOrientation() { orientation_.emplace<kNoOrientation>(); }

  // Sets the desired orientation to a given fixed angle [rad] and a given
  // tolerance on the final orientation.
  void SetFixedOrientation(double orientation, double orientation_range) {
    orientation_range_ = std::max(orientation_range, kMinimumOrientationRange);
    if (orientation_range_ >= M_PI) {
      orientation_.emplace<kNoOrientation>();
    } else {
      orientation_.emplace<kOrientation>(orientation);
    }
  }

  // Gets the desired orientation as a fixed angle.
  double GetFixedOrientation() const {
    return std::get<kOrientation>(orientation_);
  }

  // Sets the desired orientation towards a given target point and a given
  // tolerance on the final orientation.
  void SetOrientationTarget(const eigenmath::Vector2d& orientation_target,
                            double orientation_range) {
    orientation_range_ = std::max(orientation_range, kMinimumOrientationRange);
    if (orientation_range_ >= M_PI) {
      orientation_.emplace<kNoOrientation>();
    } else {
      orientation_.emplace<kFaceTowardsPoint>(orientation_target);
    }
  }

  // Gets the desired orientation as a target point.
  const eigenmath::Vector2d& GetOrientationTarget() const {
    return std::get<kFaceTowardsPoint>(orientation_);
  }

  // Checks if the desired orientation is arbitrary.
  bool IsOrientationArbitrary() const {
    return orientation_.index() == kNoOrientation;
  }

  // Checks if the desired orientation is fixed to some angle.
  bool IsOrientationFixed() const {
    return orientation_.index() == kOrientation;
  }

  // Checks if the desired orientation is to face towards a point.
  bool IsToFaceTowardsPoint() const {
    return orientation_.index() == kFaceTowardsPoint;
  }

  // Computes the orientation desired when robot is at a given point.
  eigenmath::SO2d ComputeOrientation(const eigenmath::Vector2d& point) const;

  // Returns the symmetric range that applies around the target orientation.
  double GetOrientationRange() const {
    if (IsOrientationArbitrary()) {
      return 10.0;  // Greater than pi.
    }
    return orientation_range_;
  }

  // Sets no inclusion zone.
  void SetNoInclusionZone() { inclusion_zone_.emplace<kNoGoalRegion>(); }

  // Checks if there is no inclusion zone.
  bool HasNoInclusionZone() const {
    return inclusion_zone_.index() == kNoGoalRegion;
  }

  // Sets the inclusion zone to a radial segment (donut segment).
  //  - center is the center of donut.
  //  - inner_radius is the radius of the donut "hole".
  //  - outer_radius is the radius of the donut overall.
  //  - start_angle is the start angle of a donut segment.
  //  - end_angle is the end angle of a donut segment.
  // The end_angle and start_angle values do not have to be in [-pi, pi]
  // interval and will be treated as going ccw from start to end.
  // If start and end angles are equal, it is considered as a full circle.
  void SetInclusionRadialSegment(const eigenmath::Vector2d& center,
                                 double inner_radius, double outer_radius,
                                 double start_angle = 0.0,
                                 double end_angle = 10.0) {
    inclusion_zone_.emplace<RadialSegment>(center, inner_radius, outer_radius,
                                           start_angle, end_angle);
  }

  // Gets the goal region as a radial segment (donut segment).
  const RadialSegment& GetInclusionRadialSegment() const {
    return std::get<kRadialSegment>(inclusion_zone_);
  }

  // Checks if the inclusion zone is a radial segment.
  bool HasInclusionRadialSegment() const {
    return inclusion_zone_.index() == kRadialSegment;
  }

  // Sets no exclusion zone.
  void SetNoExclusionZone() { exclusion_zone_.emplace<kNoGoalRegion>(); }

  // Checks if there is no exclusion zone.
  bool HasNoExclusionZone() const {
    return exclusion_zone_.index() == kNoGoalRegion;
  }

  // Sets the exclusion zone to a radial segment (donut segment).
  //  - center is the center of donut.
  //  - inner_radius is the radius of the donut "hole".
  //  - outer_radius is the radius of the donut overall.
  //  - start_angle is the start angle of a donut segment.
  //  - end_angle is the end angle of a donut segment.
  // The end_angle and start_angle values do not have to be in [-pi, pi]
  // interval and will be treated as going ccw from start to end.
  // If start and end angles are equal, it is considered as a full circle.
  void SetExclusionRadialSegment(const eigenmath::Vector2d& center,
                                 double inner_radius, double outer_radius,
                                 double start_angle = 0.0,
                                 double end_angle = 0.0) {
    exclusion_zone_.emplace<RadialSegment>(center, inner_radius, outer_radius,
                                           start_angle, end_angle);
  }

  // Gets the exclusion zone as a radial segment (donut segment).
  const RadialSegment& GetExclusionRadialSegment() const {
    return std::get<kRadialSegment>(exclusion_zone_);
  }

  // Checks if the exclusion zone is a radial segment.
  bool HasExclusionRadialSegment() const {
    return exclusion_zone_.index() == kRadialSegment;
  }

  // Sets the inclusion zone to a hull.
  void SetInclusionHull(const Hull& hull) {
    inclusion_zone_.emplace<Hull>(hull);
  }

  // Gets the goal region as a radial segment (donut segment).
  const Hull& GetInclusionHull() const {
    return std::get<kRegionHull>(inclusion_zone_);
  }

  // Checks if the inclusion zone is a radial segment.
  bool HasInclusionHull() const {
    return inclusion_zone_.index() == kRegionHull;
  }

  // Sets the exclusion zone to a hull.
  void SetExclusionHull(const Hull& hull) {
    exclusion_zone_.emplace<Hull>(hull);
  }

  // Gets the exclusion zone as a hull.
  const Hull& GetExclusionHull() const {
    return std::get<kRegionHull>(exclusion_zone_);
  }

  // Checks if the exclusion zone is a hull.
  bool HasExclusionHull() const {
    return exclusion_zone_.index() == kRegionHull;
  }

  // Sets the distance tolerance around the achievable position within the goal
  // region.
  void SetDistanceTolerance(double distance_tolerance) {
    distance_tolerance_ = distance_tolerance;
  }
  // Returns the distance tolerance around the achievable position within the
  // goal region.
  double GetDistanceTolerance() const { return distance_tolerance_; }

  // Sets the orientation tolerance around the achievable position within the
  // goal region.
  void SetOrientationTolerance(double orientation_tolerance) {
    orientation_tolerance_ = orientation_tolerance;
  }

  // Returns the orientation tolerance around the desired orientation.
  double GetOrientationTolerance() const { return orientation_tolerance_; }

  // Computes the goal point that is the most attractive, based on the
  // attraction field configured, within the goal geometry.
  // Note that the best possible goal point might not be always be within the
  // goal region (inclusion - exclusion), but sometimes only near it.
  eigenmath::Vector2d ComputeBestPossibleGoal() const;

  // Computes the cost of a given goal pose wrt/ this goal geometry.
  double ComputeGoalCostForPose(const eigenmath::Pose2d& pose) const;

 private:
  // Minimum value for orientation range, since the orientation range is used in
  // trajectory generation primitives as a convergence tolerance.
  constexpr static double kMinimumOrientationRange = 1e-2;
  struct EmptyType {};

  // Defines the tolerances for the final reachable goal pose.
  double distance_tolerance_ = 0.05;
  double orientation_tolerance_ = 0.05;

  // Defines the shape of the attraction field.
  enum AttractionFieldCase {
    kNoAttraction = 0,
    kAttractionPoint,
    kAttractionDirection,
  };
  std::variant<EmptyType, eigenmath::Vector2d, eigenmath::Vector2d>
      attraction_field_ = EmptyType();

  // Defines the orientation that is desired at the goal.
  enum OrientationCase {
    kNoOrientation = 0,
    kOrientation,
    kFaceTowardsPoint,
  };
  std::variant<EmptyType, double, eigenmath::Vector2d> orientation_ =
      EmptyType();
  // Defines the range of orientations [target - range, target + range] around
  // the target value.
  double orientation_range_ = kMinimumOrientationRange;

  // Defines the region within which the final reachable goal pose should be.
  enum GoalRegionCase {
    kNoGoalRegion = 0,
    kRadialSegment,
    kRegionHull,
  };
  using GoalRegionVariant = std::variant<EmptyType, RadialSegment, Hull>;
  GoalRegionVariant inclusion_zone_ = EmptyType();
  GoalRegionVariant exclusion_zone_ = EmptyType();

  // Gets the grid range corresponding to a given goal region (incl/excl).
  GridRange GetGridRangeForGoalRegion(
      const GridFrame& frame, const GoalRegionVariant& goal_region) const;

  // Checks if a point is inside a given goal region (incl/excl).
  bool IsPointInGoalRegion(const eigenmath::Vector2d& point,
                           const GoalRegionVariant& goal_region,
                           double buffer_radius = 0.0) const;

  // Computes the cost of a given goal point. The most attractive goal point
  // should also be passed in to give a point of reference to the cost.
  double ComputeGoalCost(const eigenmath::Vector2d& point,
                         const eigenmath::Vector2d& best_point) const;

  // Returns the interval of angles included at the goal position.
  Interval<double> AnglesAtGoal(const GridFrame& frame,
                                const GridIndex& goal_index) const;

  // Visits every sampled goal point on the grid.
  void ForEachGoalPoint(
      const GridFrame& frame, const GridRange& range,
      absl::FunctionRef<void(const GridIndex&)> goal_visitor) const;
};

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_GOAL_GEOMETRY_H_

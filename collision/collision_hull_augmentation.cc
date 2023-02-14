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

#include "collision/collision_hull_augmentation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "absl/log/check.h"
#include "eigenmath/pose2.h"
#include "eigenmath/scalar_utils.h"
#include "eigenmath/types.h"
#include "eigenmath/utils.h"
#include "eigenmath/vector_utils.h"
#include "genit/adjacent_circular_iterator.h"

namespace mobility::collision {

namespace {

// Expresses a rigid body motion for a given arc vector as an affine transform.
// The motion can be applied to a position in the body fixed frame.
Eigen::Affine2d RigidBodyMotion(const ArcVector &arc_vector) {
  Eigen::Affine2d rigid_body_motion = Eigen::Affine2d::Identity();
  rigid_body_motion.translation() =
      eigenmath::Vector2d(arc_vector.Translation(), 0.0);
  rigid_body_motion.linear().col(0) =
      eigenmath::Vector2d(1.0, arc_vector.Rotation());
  rigid_body_motion.linear().col(1) =
      eigenmath::Vector2d(-arc_vector.Rotation(), 1.0);

  return rigid_body_motion;
}

// Explicitly delete overload to ensure the function is called with the correct
// type.
void RigidBodyMotion(const eigenmath::Vector2d &arc_vector) = delete;

// Returns the relative arc motion to stop.  If the limits are too restrictive,
// return zero motion.
ArcVector StoppingArcMotion(const diff_drive::DynamicLimits &limits,
                            const diff_drive::State &state) {
  // For small velocities, and for unconstrained acceleration, can stop
  // immediately.
  const double acceleration_upper_bound =
      std::max(limits.MinArcAcceleration().lpNorm<1>(),
               limits.MaxArcAcceleration().lpNorm<1>());
  if ((state.GetArcVelocity().squaredNorm() <
       std::numeric_limits<double>::epsilon()) ||
      !std::isfinite(acceleration_upper_bound)) {
    return eigenmath::Vector2d::Zero();
  }

  const ArcVector arc_direction = state.GetArcVelocity().normalized();

  // Get maximum acceleration opposing arc velocity.  Extend beyond constraints,
  // and bound to limits.
  const ArcVector arc_acceleration = -arc_direction * acceleration_upper_bound;
  ArcVector constrained_arc_acceleration;
  limits.AccelerationLimits().BringInBounds(arc_acceleration,
                                            &constrained_arc_acceleration);

  // Acceleration and velocity are colinear.  Calculate stopping motion as (v *
  // v) / (2 * a).
  const double constrained_acceleration =
      constrained_arc_acceleration.lpNorm<Eigen::Infinity>();
  if (constrained_acceleration < std::numeric_limits<double>::epsilon()) {
    return eigenmath::Vector2d::Zero();
  }
  const double stopping_cord_length =
      0.5 *
      eigenmath::Square(state.GetArcVelocity().lpNorm<Eigen::Infinity>()) /
      constrained_acceleration;
  return stopping_cord_length * arc_direction;
}

// Finds the smallest circle containing both passed circles.
void GetMinimalEnclosingCircle(const eigenmath::Vector2d &center1,
                               double radius1,
                               const eigenmath::Vector2d &center2,
                               double radius2,
                               eigenmath::Vector2d *center_union,
                               double *radius_union) {
  const double distance = (center2 - center1).norm();
  const double scaled_param = 0.5 * (distance + radius1 - radius2);
  const double clamped_scaled_param = std::clamp(scaled_param, 0.0, distance);
  const double clamped_param =
      (distance < std::numeric_limits<double>::epsilon())
          ? 0.5
          : clamped_scaled_param / distance;

  *center_union = clamped_param * center1 + (1 - clamped_param) * center2;
  *radius_union = std::max((1 - clamped_param) * distance + radius1,
                           clamped_param * distance + radius2);
}

}  // namespace

eigenmath::Vector2d GetVelocityAtPoint(const diff_drive::State &state,
                                       const eigenmath::Vector2d &position) {
  const Eigen::Affine2d rigid_body_motion =
      RigidBodyMotion(state.GetArcVelocity());
  return rigid_body_motion * position - position;
}

eigenmath::Vector2d GetMomentaryStoppingPoint(
    const diff_drive::DynamicLimits &limits, const diff_drive::State &state,
    const eigenmath::Vector2d &position) {
  const ArcVector stopping_arc = StoppingArcMotion(limits, state);
  const Eigen::Affine2d rigid_body_motion = RigidBodyMotion(stopping_arc);
  return rigid_body_motion * position;
}

void AppendMomentaryStoppingPoints(
    const diff_drive::DynamicLimits &limits, const diff_drive::State &state,
    const std::vector<eigenmath::Vector2d> &original_points,
    std::vector<eigenmath::Vector2d> *stopping_points) {
  CHECK_NE(stopping_points, nullptr);
  CHECK_NE(stopping_points, &original_points);

  // Duplicate calculation from GetMomentaryStoppingPoint in order to re-use the
  // point independent quantities.
  const ArcVector stopping_arc = StoppingArcMotion(limits, state);
  const Eigen::Affine2d rigid_body_motion = RigidBodyMotion(stopping_arc);

  // Do not augment the hull if it makes little difference.
  if (stopping_arc.squaredNorm() < std::numeric_limits<double>::epsilon()) {
    return;
  }

  // Add transformed points.
  for (const eigenmath::Vector2d &point : original_points) {
    stopping_points->push_back(rigid_body_motion * point);
  }
}

ConvexHull AugmentConvexHull(const diff_drive::DynamicLimits &limits,
                             const diff_drive::State &state,
                             const ConvexHull &convex_hull) {
  std::vector<eigenmath::Vector2d> augmented_points;
  // Use all points and their stopping points to construct a new convex hull.
  augmented_points = convex_hull.GetPoints();
  AppendMomentaryStoppingPoints(limits, state, convex_hull.GetPoints(),
                                &augmented_points);
  return ConvexHull(augmented_points);
}

void ApproximatelyAugmentConvexHull(
    const diff_drive::DynamicLimits &limits, const diff_drive::State &state,
    const std::vector<eigenmath::Vector2d> &original_points,
    std::vector<eigenmath::Vector2d> *augmented_points) {
  CHECK_NE(augmented_points, nullptr);
  CHECK_NE(augmented_points, &original_points);

  // Duplicate calculation from GetMomentaryStoppingPoint in order to re-use the
  // point independent quantities.
  const ArcVector stopping_arc = StoppingArcMotion(limits, state);
  const Eigen::Affine2d rigid_body_motion = RigidBodyMotion(stopping_arc);

  // Calculate the stopping motion relative to a point.
  Eigen::Affine2d relative_motion = rigid_body_motion;
  relative_motion.linear() -= Eigen::Matrix2d::Identity();

  augmented_points->resize(original_points.size());
  auto inserter = augmented_points->begin();
  for (auto triad : genit::AdjacentElementsCircularRange<3>(original_points)) {
    // Project stopping point of center point p into cone / Voronoi region at p.
    // The Voronoi region is spanned by vectors orthogonal to the neighbouring
    // edges.
    const eigenmath::Vector2d v0 = triad[0] - triad[1];
    const eigenmath::Vector2d v1 = triad[2] - triad[1];
    const eigenmath::Vector2d relative_stopping_point =
        relative_motion * triad[1];
    const eigenmath::Vector2d projected =
        triad[1] +
        eigenmath::ProjectPointOutsideVertex(v0, v1, relative_stopping_point);
    *inserter = projected;
    ++inserter;
  }
}

void AugmentBoundingCircle(const diff_drive::DynamicLimits &limits,
                           const diff_drive::State &state,
                           const eigenmath::Vector2d &center, double radius,
                           eigenmath::Vector2d *modified_center,
                           double *modified_radius) {
  CHECK_NE(modified_center, nullptr);
  CHECK_NE(modified_radius, nullptr);

  const ArcVector stopping_arc = StoppingArcMotion(limits, state);
  const Eigen::Affine2d rigid_body_motion = RigidBodyMotion(stopping_arc);

  // The rigid motion is an affine transformation.  That is, a point p maps to
  // Tp + b.  For a circle of radius r around a center c, or
  // {p where |p - c| < r}, the transformed circle is given by the ellipse
  // {Tp + b where |p - c| < r}.  This is enclosed in the larger circle
  // {q where |q - (Tc + b)| < |T| r}.

  // Calculate the transformed circle.  Use the operator norm relative to the L2
  // norm.
  const eigenmath::Vector2d transformed_center = rigid_body_motion * center;
  const double operator_norm =
      std::sqrt(1.0 + eigenmath::Square(stopping_arc.Rotation()));
  const double transformed_radius = operator_norm * radius;

  // Take the union of the two circles.
  GetMinimalEnclosingCircle(center, radius, transformed_center,
                            transformed_radius, modified_center,
                            modified_radius);
}

}  // namespace mobility::collision

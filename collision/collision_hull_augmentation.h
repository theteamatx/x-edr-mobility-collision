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

#ifndef MOBILITY_COLLISION_COLLISION_COLLISION_HULL_AUGMENTATION_H_
#define MOBILITY_COLLISION_COLLISION_COLLISION_HULL_AUGMENTATION_H_

#include <vector>

#include "collision/convex_hull.h"
#include "diff_drive/dynamic_limits.h"
#include "diff_drive/state.h"

namespace mobility::collision {

// Calculate the velocity vector at position given in the body fixed frame.  The
// velocity at position p is given as omega x p + v, where omega is the angular
// velocity (around the z axis) and v is the translational velocity.
eigenmath::Vector2d GetVelocityAtPoint(const diff_drive::State &state,
                                       const eigenmath::Vector2d &position);

// Assuming the momentary velocity given by state, and the maximum acceleration
// within the given limits, find the stopping point for a particle at position
// expressed in the body fixed frame.
eigenmath::Vector2d GetMomentaryStoppingPoint(
    const diff_drive::DynamicLimits &limits, const diff_drive::State &state,
    const eigenmath::Vector2d &position);

// Same as above, but applies the same motion to a range of points.  Transforms
// points in order of original_points and appends to stopping_points.  If the
// motion is negligible, the output is not modified.
void AppendMomentaryStoppingPoints(
    const diff_drive::DynamicLimits &limits, const diff_drive::State &state,
    const std::vector<eigenmath::Vector2d> &original_points,
    std::vector<eigenmath::Vector2d> *stopping_points);

// Augments a convex hull to include the stopping points of all its extreme
// points. All points are given in a body fixed frame.  This is intended for
// testing purposes.
ConvexHull AugmentConvexHull(const diff_drive::DynamicLimits &limits,
                             const diff_drive::State &state,
                             const ConvexHull &convex_hull);

// Calculates a simple approximation to the convex hull augmentation which
// maintains the number of points in the convex hull.  The output describes a
// convex hull, but it returns only the points in counter clockwise order, and
// not a convex hull object.
void ApproximatelyAugmentConvexHull(
    const diff_drive::DynamicLimits &limits, const diff_drive::State &state,
    const std::vector<eigenmath::Vector2d> &original_points,
    std::vector<eigenmath::Vector2d> *augmented_points);

// Augments a bounding circle to include its stopping points.  Points are given
// in a body fixed frame,
void AugmentBoundingCircle(const diff_drive::DynamicLimits &limits,
                           const diff_drive::State &state,
                           const eigenmath::Vector2d &center, double radius,
                           eigenmath::Vector2d *modified_center,
                           double *modified_radius);

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_COLLISION_HULL_AUGMENTATION_H_

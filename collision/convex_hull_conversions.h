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

#ifndef MOBILITY_COLLISION_COLLISION_CONVEX_HULL_CONVERSIONS_H_
#define MOBILITY_COLLISION_COLLISION_CONVEX_HULL_CONVERSIONS_H_

#include <memory>
#include <vector>

#include "collision/convex_hull.h"
#include "eigenmath/eigenmath.pb.h"

namespace mobility::collision {

// Creates an `EagerConvexHull` object from a point-cloud of `Vector2dProto`.
EagerConvexHull CreateEagerConvexHull(
    const std::vector<eigenmath::Vector2dProto> &pointcloud);

// Returns the points of `convex_hull`.
std::vector<eigenmath::Vector2dProto> GetPoints(
    const EagerConvexHull &convex_hull);

// Returns the centroid of `convex_hull`.
eigenmath::Vector2dProto GetCentroid(const EagerConvexHull &convex_hull);

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_CONVEX_HULL_CONVERSIONS_H_

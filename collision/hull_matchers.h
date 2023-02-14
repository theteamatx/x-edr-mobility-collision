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

#ifndef MOBILITY_COLLISION_COLLISION_HULL_MATCHERS_H_
#define MOBILITY_COLLISION_COLLISION_HULL_MATCHERS_H_

#include "absl/strings/str_format.h"
#include "cmock/gmock.h"
#include "collision/hull.h"

namespace mobility::collision {

// Tests if two Hulls are approximately equal.
MATCHER_P(IsApproxHull, hull, "") {
  *result_listener << " got Hull with " << arg.GetConvexHulls().size()
                   << " convex hulls:\n";
  for (const auto& chull : arg.GetConvexHulls()) {
    *result_listener << "{";
    for (const eigenmath::Vector2d& pt : chull.GetPoints()) {
      *result_listener << absl::StrFormat("{%f, %f}", pt.x(), pt.y());
    }
    *result_listener << "}\n";
  }
  return hull.IsApprox(arg);
}

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_HULL_MATCHERS_H_

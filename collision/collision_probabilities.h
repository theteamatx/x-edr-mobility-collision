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

#ifndef MOBILITY_COLLISION_COLLISION_COLLISION_PROBABILITIES_H_
#define MOBILITY_COLLISION_COLLISION_COLLISION_PROBABILITIES_H_

#include "collision/danger_hull.h"

namespace mobility::collision {
// Returns a heuristic collision probability between two danger hulls.
double GetCollisionProbabilityBetweenHulls(const DangerHull& ego_danger_hull,
                                           const DangerHull& agent_danger_hull,
                                           double max_collision_prob = 1.0);

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_COLLISION_PROBABILITIES_H_

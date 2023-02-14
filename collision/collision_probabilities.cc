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

#include "collision/collision_probabilities.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "eigenmath/scalar_utils.h"

namespace mobility::collision {

double GetCollisionProbabilityBetweenHulls(const DangerHull& ego_danger_hull,
                                           const DangerHull& agent_danger_hull,
                                           double max_collision_prob) {
  const double max_collision_prob_sqrt = std::sqrt(max_collision_prob);
  const double combined_danger_margin =
      ego_danger_hull.GetDangerMargin() + agent_danger_hull.GetDangerMargin();
  double min_pen_dist = 0.0;
  for (auto& ego_chull : ego_danger_hull.GetHull().GetConvexHulls()) {
    for (const auto& agent_chull :
         agent_danger_hull.GetHull().GetConvexHulls()) {
      const double pen_dist =
          DistanceBetween(ego_chull, agent_chull,
                          std::numeric_limits<double>::lowest(), min_pen_dist);
      if (pen_dist < min_pen_dist) {
        min_pen_dist = pen_dist;
        if (-min_pen_dist >= max_collision_prob_sqrt * combined_danger_margin) {
          return max_collision_prob;
        }
      }
    }
  }
  // Probability of collision is calculated here as just a square function of
  // the penetration distance into the danger hull.
  return std::min(max_collision_prob,
                  eigenmath::Square(-min_pen_dist / combined_danger_margin));
}

}  // namespace mobility::collision

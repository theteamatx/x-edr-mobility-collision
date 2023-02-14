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

#ifndef MOBILITY_COLLISION_COLLISION_DANGER_HULL_H_
#define MOBILITY_COLLISION_COLLISION_DANGER_HULL_H_

#include "collision/hull.h"
#include "diff_drive/dynamic_limits.h"
#include "diff_drive/state.h"

namespace mobility::collision {

// Provides a convenience wrapper to represent the collision hull of the robot,
// with modifiers to augment and transform it.  There are two methods to augment
// the convex hull:
// 1. Use a fixed margin, danger_margin,
// 2. Use a velocity dependent margin, based on reaction_time.
class DangerHull {
 public:
  DangerHull();

  // Constructs a danger hull referencing the original collision hull.
  // Note: The original hull's lifetime has to exceed that of the danger hull.
  DangerHull(const Hull& collision_hull,
             const diff_drive::DynamicLimits* limits, double danger_margin,
             double reaction_time, bool use_velocity_dependence);

  // Requires special handling because of internal references between members.
  DangerHull(const DangerHull&);
  DangerHull& operator=(const DangerHull&);

  DangerHull(DangerHull&&);
  DangerHull& operator=(DangerHull&&);

  // Augments and transforms the collision hull using the pose and velocity of
  // the state, and sets the danger margin.
  // Note: the state object has to stay valid as long as any of the data inside
  // this class is used.
  void ApplyState(const diff_drive::State& state);

  // Clear the danger hull.
  void Clear() { ApplyState(diff_drive::State{}); }

  // Accesses the outer wrapper of the danger hull.
  const LazyTransformedAugmentedHull& GetHull() const {
    return transformed_hull_;
  }

  // Accesses the untransformed augmented hull.
  const LazyAugmentedHull& GetUntransformedHull() const {
    return augmented_hull_;
  }

  // Accesses the collision hull, without transformation nor augmentation.
  const Hull& GetCollisionHull() const { return *collision_hull_; }

  // Returns the current danger margin.
  double GetDangerMargin() const { return danger_margin_; }

  // Returns the current reaction time.
  double GetReactionTime() const { return reaction_time_; }

  // Returns whether the danger hull is using velocity dependence.
  bool IsUsingVelocityDependence() const { return use_velocity_dependence_; }

  // Get a reference to the configured dynamic limits.
  const diff_drive::DynamicLimits& GetLimits() const;

 private:
  // The augmented hull has to be initialized first.
  const Hull* collision_hull_;
  LazyAugmentedHull augmented_hull_;
  LazyTransformedAugmentedHull transformed_hull_;

  double danger_margin_;
  double min_danger_margin_;
  double reaction_time_;
  bool use_velocity_dependence_;

  // Necessary for taking copies.
  diff_drive::State state_;
};

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_DANGER_HULL_H_

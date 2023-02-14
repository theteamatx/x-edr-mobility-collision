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

#include "collision/danger_hull.h"

#include <algorithm>
#include <utility>

#include "absl/log/check.h"
#include "diff_drive/dynamic_limits.h"

namespace mobility::collision {

DangerHull::DangerHull()
    : collision_hull_(nullptr),
      augmented_hull_(),
      transformed_hull_(augmented_hull_),
      danger_margin_(0.0),
      min_danger_margin_(0.0),
      reaction_time_(0.0),
      use_velocity_dependence_(false) {}

DangerHull::DangerHull(const Hull& collision_hull,
                       const diff_drive::DynamicLimits* limits,
                       double danger_margin, double reaction_time,
                       bool use_velocity_dependence)
    : collision_hull_(&collision_hull),
      augmented_hull_(limits, collision_hull),
      transformed_hull_(augmented_hull_),
      danger_margin_(danger_margin),
      min_danger_margin_(danger_margin_),
      reaction_time_(reaction_time),
      use_velocity_dependence_(use_velocity_dependence) {
  // Use with fixed margin when no velocity dependence is used.
  if (!use_velocity_dependence_) {
    augmented_hull_.SetDangerMargin(danger_margin_);
  }
  Clear();
}

DangerHull::DangerHull(const DangerHull& other)
    : collision_hull_(other.collision_hull_),
      augmented_hull_(other.augmented_hull_),
      transformed_hull_(augmented_hull_),
      danger_margin_(other.danger_margin_),
      min_danger_margin_(other.min_danger_margin_),
      reaction_time_(other.reaction_time_),
      use_velocity_dependence_(other.use_velocity_dependence_),
      state_(other.state_) {
  transformed_hull_.ApplyTransform(state_.GetPose());
}

DangerHull& DangerHull::operator=(const DangerHull& other) {
  danger_margin_ = other.danger_margin_;
  min_danger_margin_ = other.min_danger_margin_;
  reaction_time_ = other.reaction_time_;
  use_velocity_dependence_ = other.use_velocity_dependence_;
  state_ = other.state_;
  collision_hull_ = other.collision_hull_;
  augmented_hull_ = other.augmented_hull_;
  transformed_hull_.ResetHullBase(augmented_hull_);
  transformed_hull_.ApplyTransform(state_.GetPose());
  return *this;
}

DangerHull::DangerHull(DangerHull&& other)
    : collision_hull_(other.collision_hull_),
      augmented_hull_(std::move(other.augmented_hull_)),
      transformed_hull_(std::move(other.transformed_hull_)),
      danger_margin_(other.danger_margin_),
      min_danger_margin_(other.min_danger_margin_),
      reaction_time_(other.reaction_time_),
      use_velocity_dependence_(other.use_velocity_dependence_),
      state_(std::move(other.state_)) {
  // Nothing to do, see the assignment operator for explanations.
}

DangerHull& DangerHull::operator=(DangerHull&& other) {
  danger_margin_ = other.danger_margin_;
  min_danger_margin_ = other.min_danger_margin_;
  reaction_time_ = other.reaction_time_;
  use_velocity_dependence_ = other.use_velocity_dependence_;
  state_ = std::move(other.state_);
  collision_hull_ = other.collision_hull_;
  other.collision_hull_ = nullptr;
  // Move the augmented hull, which should just transfer ownership of the
  // heap-memory containing the augmented convex hulls.
  augmented_hull_ = std::move(other.augmented_hull_);
  // Now, other.transformed_hull_ is wired to augmented_hull_.
  // Move the transformed hull, which should remain valid because the move
  // does not rewire it, just transfers the pointers.
  transformed_hull_ =
      std::move(other.transformed_hull_);  // Now, transformed_hull_ is wired to
                                           // augmented_hull_.
  // Both augmented and transformed hulls already have the state applied to
  // them, so there is nothing more to do. In other words, this move does
  // not invalidate the cached hulls or allocate any memory.
  return *this;
}

void DangerHull::ApplyState(const diff_drive::State& state) {
  // The hull has to be augmented first.

  // Optionally use velocity dependent augmentation.
  if (use_velocity_dependence_) {
    danger_margin_ = std::max(
        min_danger_margin_,
        std::abs(state.GetArcVelocity().Translation()) * reaction_time_);
    augmented_hull_.SetDangerMargin(danger_margin_);
    augmented_hull_.ApplyMotion(state);
  }

  transformed_hull_.ApplyTransform(state.GetPose());
  state_ = state;
}

const diff_drive::DynamicLimits& DangerHull::GetLimits() const {
  CHECK(!augmented_hull_.GetConvexHulls().empty());
  return augmented_hull_.GetConvexHulls().front().GetLimits();
}

}  // namespace mobility::collision

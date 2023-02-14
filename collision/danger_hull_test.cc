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

#include <memory>
#include <utility>
#include <vector>

#include "diff_drive/test_trajectories.h"
#include "eigenmath/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mobility::collision {

namespace {
constexpr double kEpsilon = 1.0e-6;

using eigenmath::testing::UnorderedElementsAreApprox;

constexpr double kDangerMargin = 0.2;
constexpr double kReactionTime = 0.5;

std::vector<eigenmath::Vector2d> TransformPoints(
    const eigenmath::Pose2d& pose,
    const std::vector<eigenmath::Vector2d>& pts) {
  std::vector<eigenmath::Vector2d> result;
  for (auto& pt : pts) {
    result.emplace_back(pose * pt);
  }
  return result;
}

Hull CreatePotato() {
  const std::vector<eigenmath::Vector2d> potato_points = {
      {0.0, 0.0},   {1.0, 0.0},  {1.5, 0.25}, {1.75, 0.75},
      {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0},  {0.0, 2.0}};

  Hull hull;
  hull.Add(potato_points);
  return hull;
}

void ExpectDangerHullToMatchAugmentedHull(
    const DangerHull& danger_hull,
    const LazyAugmentedConvexHull& augmented_hull,
    const eigenmath::Pose2d& pose) {
  EXPECT_THAT(
      danger_hull.GetUntransformedHull().GetConvexHulls().front().GetPoints(),
      UnorderedElementsAreApprox(augmented_hull.GetPoints(), kEpsilon));
  EXPECT_THAT(danger_hull.GetHull().GetConvexHulls().front().GetPoints(),
              UnorderedElementsAreApprox(
                  TransformPoints(pose, augmented_hull.GetPoints()), kEpsilon));
}

TEST(DangerHull, BasicOperations) {
  const Hull hull = CreatePotato();
  LazyAugmentedConvexHull augmented_hull(
      &diff_drive::testing::kTestDynamicLimits, &hull.GetConvexHulls().front());
  augmented_hull.SetDangerMargin(kDangerMargin);
  DangerHull danger_hull(hull, &diff_drive::testing::kTestDynamicLimits,
                         kDangerMargin, kReactionTime,
                         /*use_velocity_dependence=*/true);

  // Check danger hull with no applied state:
  EXPECT_THAT(danger_hull.GetHull().GetConvexHulls().front().GetPoints(),
              UnorderedElementsAreApprox(augmented_hull.GetPoints(), kEpsilon));

  // Check danger hull with transform only.
  const diff_drive::State static_state{
      eigenmath::Pose2d{eigenmath::Vector2d{1.0, 2.0}, 0.5}};
  danger_hull.ApplyState(static_state);
  augmented_hull.ApplyMotion(static_state);
  ExpectDangerHullToMatchAugmentedHull(danger_hull, augmented_hull,
                                       static_state.GetPose());

  // Check danger hull with transform and velocity.
  const diff_drive::State moving_state{
      eigenmath::Pose2d{eigenmath::Vector2d{-2.0, 4.0}, -1.2},
      ArcVector{0.5, 1.2}};
  danger_hull.ApplyState(moving_state);
  augmented_hull.SetDangerMargin(
      std::abs(moving_state.GetArcVelocity().Translation()) * kReactionTime);
  augmented_hull.ApplyMotion(moving_state);
  ExpectDangerHullToMatchAugmentedHull(danger_hull, augmented_hull,
                                       moving_state.GetPose());

  danger_hull.Clear();
  augmented_hull.SetDangerMargin(kDangerMargin);
  augmented_hull.ApplyMotion(diff_drive::State{});
  EXPECT_THAT(danger_hull.GetHull().GetConvexHulls().front().GetPoints(),
              UnorderedElementsAreApprox(augmented_hull.GetPoints(), kEpsilon));
}

TEST(DangerHull, NoVelocityDependence) {
  const Hull hull = CreatePotato();
  LazyAugmentedConvexHull augmented_hull(
      &diff_drive::testing::kTestDynamicLimits, &hull.GetConvexHulls().front());
  augmented_hull.SetDangerMargin(kDangerMargin);
  DangerHull danger_hull(hull, &diff_drive::testing::kTestDynamicLimits,
                         kDangerMargin, kReactionTime,
                         /*use_velocity_dependence=*/false);

  // Check danger hull with no applied state:
  EXPECT_THAT(danger_hull.GetHull().GetConvexHulls().front().GetPoints(),
              UnorderedElementsAreApprox(augmented_hull.GetPoints(), kEpsilon));

  // Check danger hull with transform and velocity.
  const diff_drive::State moving_state{
      eigenmath::Pose2d{eigenmath::Vector2d{-2.0, 4.0}, -1.2},
      ArcVector{0.5, 1.2}};
  danger_hull.ApplyState(moving_state);
  ExpectDangerHullToMatchAugmentedHull(danger_hull, augmented_hull,
                                       moving_state.GetPose());
}

TEST(DangerHull, Copy) {
  const Hull hull = CreatePotato();
  LazyAugmentedConvexHull augmented_hull(
      &diff_drive::testing::kTestDynamicLimits, &hull.GetConvexHulls().front());
  const diff_drive::State moving_state{
      eigenmath::Pose2d{eigenmath::Vector2d{-2.0, 4.0}, -1.2},
      ArcVector{0.5, 1.2}};
  augmented_hull.SetDangerMargin(
      std::abs(moving_state.GetArcVelocity().Translation()) * kReactionTime);
  augmented_hull.ApplyMotion(moving_state);

  // Create a source danger hull.
  auto danger_hull = std::make_unique<DangerHull>(
      hull, &diff_drive::testing::kTestDynamicLimits, kDangerMargin,
      kReactionTime, true);
  danger_hull->ApplyState(moving_state);
  ExpectDangerHullToMatchAugmentedHull(*danger_hull, augmented_hull,
                                       moving_state.GetPose());

  auto copy_hull = std::make_unique<DangerHull>(*danger_hull);
  danger_hull.reset();
  ExpectDangerHullToMatchAugmentedHull(*copy_hull, augmented_hull,
                                       moving_state.GetPose());

  DangerHull assigned_hull;
  assigned_hull = *copy_hull;
  copy_hull.reset();
  ExpectDangerHullToMatchAugmentedHull(assigned_hull, augmented_hull,
                                       moving_state.GetPose());
}

TEST(DangerHull, Move) {
  const Hull hull = CreatePotato();
  LazyAugmentedConvexHull augmented_hull(
      &diff_drive::testing::kTestDynamicLimits, &hull.GetConvexHulls().front());
  const diff_drive::State moving_state{
      eigenmath::Pose2d{eigenmath::Vector2d{-2.0, 4.0}, -1.2},
      ArcVector{0.5, 1.2}};
  augmented_hull.SetDangerMargin(
      std::abs(moving_state.GetArcVelocity().Translation()) * kReactionTime);
  augmented_hull.ApplyMotion(moving_state);

  // Create a source danger hull.
  auto danger_hull = std::make_unique<DangerHull>(
      hull, &diff_drive::testing::kTestDynamicLimits, kDangerMargin,
      kReactionTime, true);
  danger_hull->ApplyState(moving_state);
  ExpectDangerHullToMatchAugmentedHull(*danger_hull, augmented_hull,
                                       moving_state.GetPose());

  auto move_hull = std::make_unique<DangerHull>(std::move(*danger_hull));
  danger_hull.reset();
  ExpectDangerHullToMatchAugmentedHull(*move_hull, augmented_hull,
                                       moving_state.GetPose());

  DangerHull assigned_hull;
  assigned_hull = std::move(*move_hull);
  move_hull.reset();
  ExpectDangerHullToMatchAugmentedHull(assigned_hull, augmented_hull,
                                       moving_state.GetPose());
}

}  // namespace
}  // namespace mobility::collision

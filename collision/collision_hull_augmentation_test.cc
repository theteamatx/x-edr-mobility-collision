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

#include <cmath>
#include <tuple>
#include <vector>

#include "diff_drive/state.h"
#include "diff_drive/test_trajectories.h"
#include "eigenmath/matchers.h"
#include "eigenmath/types.h"
#include "eigenmath/vector_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

using diff_drive::testing::kTestDynamicLimits;
using eigenmath::testing::IsApprox;
using ::testing::Combine;
using ::testing::Le;
using ::testing::UnorderedElementsAreArray;
using ::testing::ValuesIn;

// Test if two convex hulls are approximately equal.
MATCHER_P(IsApproxConvexHull, convex_hull,
          "is approximately equal to %(convex_hull)s") {
  return convex_hull.IsApprox(arg);
}

TEST(GetVelocityAtPoint, NotMovingAtOrigin) {
  diff_drive::State state;

  const eigenmath::Vector2d origin = eigenmath::Vector2d::Zero();
  const eigenmath::Vector2d velocity = GetVelocityAtPoint(state, origin);
  const eigenmath::Vector2d expected_velocity = eigenmath::Vector2d::Zero();
  EXPECT_THAT(velocity, IsApprox(expected_velocity));
}

TEST(GetVelocityAtPoint, NotMovingAtOffset) {
  diff_drive::State state;

  const eigenmath::Vector2d offset(2.0, 1.0);
  const eigenmath::Vector2d velocity = GetVelocityAtPoint(state, offset);
  const eigenmath::Vector2d expected_velocity = eigenmath::Vector2d::Zero();
  EXPECT_THAT(velocity, IsApprox(expected_velocity));
}

TEST(GetVelocityAtPoint, ForwardMovingAtOrigin) {
  diff_drive::State state;
  state.SetPose({{1.0, 5.0}, 0.4});
  state.SetArcVelocity({/*Translation=*/0.7, /*Rotation=*/0.0});

  const eigenmath::Vector2d origin = eigenmath::Vector2d::Zero();
  const eigenmath::Vector2d velocity = GetVelocityAtPoint(state, origin);
  const eigenmath::Vector2d expected_velocity =
      eigenmath::Vector2d(state.GetArcVelocity().Translation(), 0.0);
  EXPECT_THAT(velocity, IsApprox(expected_velocity));
}

TEST(GetVelocityAtPoint, ForwardMovingAtOffset) {
  diff_drive::State state;
  state.SetArcVelocity({/*Translation=*/0.7, /*Rotation=*/0.0});

  const eigenmath::Vector2d offset(2.0, 1.0);
  const eigenmath::Vector2d velocity = GetVelocityAtPoint(state, offset);
  const eigenmath::Vector2d expected_velocity(
      state.GetArcVelocity().Translation(), 0.0);
  EXPECT_THAT(velocity, IsApprox(expected_velocity));
}

TEST(GetVelocityAtPoint, RotationAtOrigin) {
  diff_drive::State state;
  state.SetPose({{5.0, 0.0}, 0.0});
  state.SetArcVelocity({/*Translation=*/0.0, /*Rotation=*/0.7});

  const eigenmath::Vector2d origin = eigenmath::Vector2d::Zero();
  const eigenmath::Vector2d velocity = GetVelocityAtPoint(state, origin);
  const eigenmath::Vector2d expected_velocity = eigenmath::Vector2d::Zero();
  EXPECT_THAT(velocity, IsApprox(expected_velocity));
}

TEST(GetVelocityAtPoint, LeftRotationAtOffset) {
  diff_drive::State state;
  state.SetArcVelocity({/*Translation=*/0.0, /*Rotation=*/0.7});

  const eigenmath::Vector2d offset(2.0, 1.0);
  const eigenmath::Vector2d velocity = GetVelocityAtPoint(state, offset);
  const eigenmath::Vector2d expected_velocity =
      eigenmath::Vector2d(-offset.y(), offset.x()) *
      state.GetArcVelocity().Rotation();
  EXPECT_THAT(velocity, IsApprox(expected_velocity));
}

TEST(GetVelocityAtPoint, GeneralMotionAtOrigin) {
  diff_drive::State state;
  state.SetPose({{5.0, 1.0}, 0.0});
  state.SetArcVelocity({/*Translation=*/0.3, /*Rotation=*/0.7});

  const eigenmath::Vector2d origin = eigenmath::Vector2d::Zero();
  const eigenmath::Vector2d velocity = GetVelocityAtPoint(state, origin);
  const eigenmath::Vector2d expected_velocity(
      state.GetArcVelocity().Translation(), 0.0);
  EXPECT_THAT(velocity, IsApprox(expected_velocity));
}

TEST(GetVelocityAtPoint, GeneralMotionAtOffset) {
  diff_drive::State state;
  state.SetArcVelocity({/*Translation=*/0.3, /*Rotation=*/0.7});

  const eigenmath::Vector2d offset(2.0, 1.0);
  const eigenmath::Vector2d velocity = GetVelocityAtPoint(state, offset);
  const eigenmath::Vector2d expected_velocity =
      eigenmath::Vector2d(state.GetArcVelocity().Translation(), 0.0) +
      state.GetArcVelocity().Rotation() *
          eigenmath::Vector2d(-offset.y(), offset.x());
  EXPECT_THAT(velocity, IsApprox(expected_velocity));
}

TEST(GetMomentaryStoppingPoint, NotMovingAtOrigin) {
  diff_drive::State state;
  state.SetPose({{4.0, 7.0}, 0.3});
  const eigenmath::Vector2d origin = eigenmath::Vector2d::Zero();

  const eigenmath::Vector2d expected_stopping_point = origin;
  EXPECT_THAT(GetMomentaryStoppingPoint(kTestDynamicLimits, state, origin),
              IsApprox(expected_stopping_point));
}

TEST(GetMomentaryStoppingPoint, ForwardMovingAtOffset) {
  diff_drive::State state;
  state.SetArcVelocity({/*Translation=*/1.0, /*Rotation=*/0.0});
  const eigenmath::Vector2d offset(2.0, 2.0);

  const eigenmath::Vector2d velocity = GetVelocityAtPoint(state, offset);
  const double max_acceleration =
      std::abs(kTestDynamicLimits.MinArcAcceleration().Translation());

  const double stopping_distance =
      0.5 * velocity.squaredNorm() / max_acceleration;
  const eigenmath::Vector2d expected_stopping_point =
      offset + stopping_distance * velocity.normalized();
  EXPECT_THAT(GetMomentaryStoppingPoint(kTestDynamicLimits, state, offset),
              IsApprox(expected_stopping_point));
}

TEST(GetMomentaryStoppingPoint, BackwardsMovingAtOffset) {
  diff_drive::State state;
  state.SetArcVelocity({/*Translation=*/-2.0, /*Rotation=*/0.0});
  const eigenmath::Vector2d offset(2.0, 2.0);

  const eigenmath::Vector2d velocity = GetVelocityAtPoint(state, offset);
  const double max_acceleration =
      std::abs(kTestDynamicLimits.MaxArcAcceleration().Translation());

  const double stopping_distance =
      0.5 * velocity.squaredNorm() / max_acceleration;
  const eigenmath::Vector2d expected_stopping_point =
      offset + stopping_distance * velocity.normalized();
  EXPECT_THAT(GetMomentaryStoppingPoint(kTestDynamicLimits, state, offset),
              IsApprox(expected_stopping_point));
}

TEST(GetMomentaryStoppingPoint, RotatingLeftAtOffset) {
  diff_drive::State state;
  state.SetArcVelocity({/*Translation=*/0.0, /*Rotation=*/1.0});
  const eigenmath::Vector2d offset(2.0, 2.0);

  const eigenmath::Vector2d velocity = GetVelocityAtPoint(state, offset);
  const double max_acceleration =
      std::abs(kTestDynamicLimits.MinArcAcceleration().Rotation()) *
      offset.norm();

  const double stopping_distance =
      0.5 * velocity.squaredNorm() / max_acceleration;
  const eigenmath::Vector2d expected_stopping_point =
      offset + stopping_distance * velocity.normalized();
  EXPECT_THAT(GetMomentaryStoppingPoint(kTestDynamicLimits, state, offset),
              IsApprox(expected_stopping_point));
}

TEST(AugmentConvexHull, NotMoving) {
  diff_drive::State state;
  const ConvexHull triangle({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});

  EXPECT_THAT(AugmentConvexHull(kTestDynamicLimits, state, triangle),
              IsApproxConvexHull(triangle));
}

TEST(AugmentConvexHull, ForwardMoving) {
  constexpr double velocity = 1.0;

  diff_drive::State state;
  state.SetArcVelocity({/*Translation=*/velocity, /*Rotation=*/0.0});
  const ConvexHull triangle({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});

  const double max_acceleration =
      std::abs(kTestDynamicLimits.MinArcAcceleration().Translation());
  const double stopping_distance = 0.5 * velocity * velocity / max_acceleration;

  const ConvexHull expected_hull({{0.0, 0.0},
                                  {1.0 + stopping_distance, 0.0},
                                  {stopping_distance, 1.0},
                                  {0.0, 1.0}});
  EXPECT_THAT(AugmentConvexHull(kTestDynamicLimits, state, triangle),
              IsApproxConvexHull(expected_hull));
}

TEST(AugmentConvexHull, RotatingLeft) {
  constexpr double velocity = 1.0;

  diff_drive::State state;
  state.SetArcVelocity({/*Translation=*/0.0, /*Rotation=*/velocity});
  const ConvexHull triangle({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});

  // Calculate the stopping distance for a point with distance 1.0 from the
  // center of the rotation.
  const double max_acceleration =
      std::abs(kTestDynamicLimits.MinArcAcceleration().Rotation());
  const double stopping_distance = 0.5 * velocity * velocity / max_acceleration;

  const ConvexHull expected_hull({{0.0, 0.0},
                                  {1.0, 0.0},
                                  {1.0, stopping_distance},
                                  {0.0, 1.0},
                                  {-stopping_distance, 1.0}});
  EXPECT_THAT(AugmentConvexHull(kTestDynamicLimits, state, triangle),
              IsApproxConvexHull(expected_hull));
}

TEST(ApproximatelyAugmentConvexHull, CompareToRegularAugmentation) {
  // Use forward motion on a square, where the approximation is exact.
  const ConvexHull square({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}});

  diff_drive::State state;
  state.SetArcVelocity({3.0, 0.0});
  const ConvexHull exact_augmentation =
      AugmentConvexHull(kTestDynamicLimits, state, square);
  std::vector<eigenmath::Vector2d> approximate_augmentation;
  ApproximatelyAugmentConvexHull(kTestDynamicLimits, state, square.GetPoints(),
                                 &approximate_augmentation);

  // Elements can be rotated.
  EXPECT_THAT(approximate_augmentation,
              UnorderedElementsAreArray(exact_augmentation.GetPoints()));
}

TEST(ApproximatelyAugmentConvexHull, PureRotationForAlignedSquare) {
  const std::vector<eigenmath::Vector2d> square_points = {
      {1.0, 1.0}, {-1.0, 1.0}, {-1.0, -1.0}, {1.0, -1.0}};

  // Get relative stopping point distance and manually specify expected shape.
  // Symmetry guarantess the same stopping distance at all points.
  diff_drive::State state;
  state.SetArcVelocity({0.0, 1.0});

  // Project each stopping point to the closest non-interfering ray at the base
  // point.
  const std::vector<eigenmath::Vector2d> directions = {
      {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}, {1.0, 0.0}};
  std::vector<eigenmath::Vector2d> expected_points;
  for (int i = 0; i < 4; ++i) {
    const eigenmath::Vector2d point = square_points[i];
    const eigenmath::Vector2d direction = directions[i];
    const eigenmath::Vector2d stopping_point =
        GetMomentaryStoppingPoint(kTestDynamicLimits, state, point);
    const eigenmath::Vector2d relative_stopping_point = stopping_point - point;
    const eigenmath::Vector2d projected_stopping_point =
        point + direction.dot(relative_stopping_point) * direction;

    expected_points.push_back(projected_stopping_point);
  }

  std::vector<eigenmath::Vector2d> augmented_points;
  ApproximatelyAugmentConvexHull(kTestDynamicLimits, state, square_points,
                                 &augmented_points);
  // Elements can be rotated.
  EXPECT_THAT(augmented_points, UnorderedElementsAreArray(expected_points));
}

// Compares the augmentation of a convex hull's bounding circle with the
// augmented convex hull.  The convex hull augmentation is explicitly checked
// above, so the comparison should be sufficient.
class AugmentBoundingCircleTest
    : public ::testing::TestWithParam<
          std::tuple<ConvexHull, diff_drive::State>> {};

TEST_P(AugmentBoundingCircleTest,
       AugmentedCircleContainedInBoundingCircleOfAugmentedHull) {
  const auto& convex_hull = std::get<0>(GetParam());
  const auto& state = std::get<1>(GetParam());

  eigenmath::Vector2d augmented_center = eigenmath::Vector2d::Zero();
  double augmented_radius;
  AugmentBoundingCircle(kTestDynamicLimits, state, convex_hull.GetCentroid(),
                        convex_hull.GetRadius(), &augmented_center,
                        &augmented_radius);

  // Ensure that each point in the augmented hull is contained in the
  // augmented bounding circle.
  const auto augmented_convex_hull =
      AugmentConvexHull(kTestDynamicLimits, state, convex_hull);
  for (const auto& point : augmented_convex_hull.GetPoints()) {
    const double distance_to_center = (point - augmented_center).norm();
    EXPECT_THAT(
        distance_to_center,
        Le(augmented_radius * (1.0 + std::numeric_limits<double>::epsilon())))
        << " point: \n"
        << point << ",\n center: \n"
        << augmented_center << ",\n radius: " << augmented_radius;
  }
}

std::vector<diff_drive::State> TestStates() {
  std::vector<diff_drive::State> states;
  // Use an arbitrary pose -- this should not have any effect.
  const double x = 50.0;
  const double y = -642.0;
  const double angle = 0.56;
  eigenmath::Pose2d pose({x, y}, angle);

  std::vector<double> values = {-1.0, -1e-3, 0.0, 1.0, 1e3};
  for (const double translation : values) {
    for (const double rotation : values) {
      states.emplace_back();
      states.back().SetPose(pose);
      states.back().SetArcVelocity({translation, rotation});
    }
  }
  return states;
}

std::vector<ConvexHull> TestHulls() {
  std::vector<ConvexHull> hulls;

  // triangle with origin
  hulls.emplace_back(ConvexHull({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}));

  // triangle around origin
  hulls.emplace_back(ConvexHull({{-1.0, -1.0}, {0.0, 1.0}, {1.0, 0.0}}));

  // offset shape
  hulls.emplace_back(
      ConvexHull({{-1.0, -8.0}, {-3.0, -2.0}, {-5.0, -6.0}, {-7.0, -4.0}}));

  return hulls;
}

const auto test_states = TestStates();
const auto test_hulls = TestHulls();

INSTANTIATE_TEST_SUITE_P(BoundingCircleAugmentation, AugmentBoundingCircleTest,
                         Combine(ValuesIn(test_hulls), ValuesIn(test_states)));

}  // namespace
}  // namespace mobility::collision

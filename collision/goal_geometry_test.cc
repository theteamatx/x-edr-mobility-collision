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

#include "collision/goal_geometry.h"

#include <cmath>
#include <vector>

#include "eigenmath/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

constexpr double kEpsilon = 1e-5;

using eigenmath::testing::IsApprox;
using eigenmath::testing::IsApproxTuple;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Pointwise;
using ::testing::UnorderedElementsAreArray;
using testing::UnorderedPointwise;

// A cost comparison function.  The cost should be comparable to the (matched
// against) L2 norm.
MATCHER(IsNormComparableTo, "") {
  const auto& actual = std::get<0>(arg);
  const auto& expected = std::get<1>(arg);
  return (actual >= expected * M_SQRT1_2) && (actual <= expected * M_SQRT2);
}

TEST(GoalGeometry, AttractionPoint) {
  GoalGeometry goal_geometry;

  goal_geometry.SetAttractionPoint(eigenmath::Vector2d(1.0, 2.0));
  goal_geometry.SetDistanceTolerance(0.1);

  EXPECT_FALSE(goal_geometry.IsAttractedToNothing());
  EXPECT_TRUE(goal_geometry.IsAttractedToSinglePoint());
  EXPECT_FALSE(goal_geometry.IsAttractedInDirection());

  EXPECT_TRUE(goal_geometry.IsOrientationArbitrary());
  EXPECT_FALSE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasInclusionHull());
  EXPECT_FALSE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasExclusionHull());

  const auto& attraction_point = goal_geometry.GetAttractionPoint();
  EXPECT_NEAR(attraction_point.x(), 1.0, kEpsilon);
  EXPECT_NEAR(attraction_point.y(), 2.0, kEpsilon);

  goal_geometry.ApplyTransform(eigenmath::Pose2d({-1.0, 3.0}, M_PI_2));
  EXPECT_NEAR(attraction_point.x(), -3.0, kEpsilon);
  EXPECT_NEAR(attraction_point.y(), 4.0, kEpsilon);

  goal_geometry.ApplyTransform(eigenmath::Pose2d({-3.0, -1.0}, -M_PI_2));
  EXPECT_NEAR(attraction_point.x(), 1.0, kEpsilon);
  EXPECT_NEAR(attraction_point.y(), 2.0, kEpsilon);

  eigenmath::Vector2d best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.0, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.0, kEpsilon);

  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.0, 2.0)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.05, 2.05)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.1, 2.1)));

  EXPECT_TRUE(goal_geometry.IsPointInRadiusOfGoalGeometry(
      eigenmath::Vector2d(1.1, 2.1), 0.1));
  EXPECT_FALSE(goal_geometry.IsPointInRadiusOfGoalGeometry(
      eigenmath::Vector2d(1.2, 2.2), 0.1));

  const GridFrame grid_frame("", eigenmath::Pose2d::Identity(), 0.05);
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(21, 41))));
  EXPECT_FALSE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(10, 20))));

  const GridRange goal_range = goal_geometry.GetGoalGridRange(grid_frame);
  EXPECT_EQ(goal_range.lower.x(), 18);
  EXPECT_EQ(goal_range.lower.y(), 38);
  EXPECT_EQ(goal_range.upper.x(), 23);
  EXPECT_EQ(goal_range.upper.y(), 43);

  std::vector<GridIndex> grid_goals;
  std::vector<double> grid_costs;
  goal_geometry.SampleGoalsOnGrid(grid_frame, &grid_goals, &grid_costs);
  ASSERT_EQ(grid_goals.size(), 1);
  ASSERT_EQ(grid_costs.size(), 1);
  EXPECT_EQ(grid_goals[0].x(), 20);
  EXPECT_EQ(grid_goals[0].y(), 40);
  EXPECT_NEAR(grid_costs[0], 0.0, kEpsilon);

  const LatticeFrame lattice_frame(grid_frame, 8);
  std::vector<LatticePose> lattice_goals;
  std::vector<double> lattice_costs;
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  ASSERT_EQ(lattice_goals.size(), 8);
  ASSERT_EQ(lattice_costs.size(), 8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(lattice_goals[i].position.x(), 20);
    EXPECT_EQ(lattice_goals[i].position.y(), 40);
    EXPECT_EQ(lattice_goals[i].angle, i);
    EXPECT_NEAR(lattice_costs[i], 0.0, kEpsilon);
  }

  goal_geometry.SetFixedOrientation(2.0, 0.5);
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  ASSERT_EQ(lattice_goals.size(), 2);
  ASSERT_EQ(lattice_costs.size(), 2);
  EXPECT_THAT(lattice_goals,
              UnorderedElementsAreArray({LatticePose(GridIndex(20, 40), 2),
                                         LatticePose(GridIndex(20, 40), 3)}));
  EXPECT_THAT(lattice_costs, UnorderedPointwise(DoubleNear(1e-12), {0.0, 0.0}));
}

TEST(GoalGeometry, AttractionDirection) {
  GoalGeometry goal_geometry;

  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));

  EXPECT_FALSE(goal_geometry.IsAttractedToNothing());
  EXPECT_FALSE(goal_geometry.IsAttractedToSinglePoint());
  EXPECT_TRUE(goal_geometry.IsAttractedInDirection());

  EXPECT_TRUE(goal_geometry.IsOrientationArbitrary());
  EXPECT_FALSE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasInclusionHull());
  EXPECT_FALSE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasExclusionHull());

  const auto& attraction_dir = goal_geometry.GetAttractionDirection();
  EXPECT_NEAR(attraction_dir.y() / attraction_dir.x(), 2.0, kEpsilon);
  EXPECT_NEAR(attraction_dir.norm(), 1.0, kEpsilon);
}

TEST(GoalGeometry, FixedOrientation) {
  GoalGeometry goal_geometry;

  goal_geometry.SetFixedOrientation(1.0, 0.5);

  EXPECT_FALSE(goal_geometry.IsOrientationArbitrary());
  EXPECT_TRUE(goal_geometry.IsOrientationFixed());
  EXPECT_FALSE(goal_geometry.IsToFaceTowardsPoint());

  EXPECT_TRUE(goal_geometry.IsAttractedToNothing());
  EXPECT_FALSE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasInclusionHull());
  EXPECT_FALSE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasExclusionHull());

  EXPECT_NEAR(goal_geometry.GetFixedOrientation(), 1.0, kEpsilon);
  EXPECT_NEAR(goal_geometry.GetOrientationRange(), 0.5, kEpsilon);

  goal_geometry.ApplyTransform(eigenmath::Pose2d({-1.0, 3.0}, M_PI_2));
  EXPECT_NEAR(goal_geometry.GetFixedOrientation(), 1.0 + M_PI_2, kEpsilon);

  goal_geometry.ApplyTransform(eigenmath::Pose2d({-3.0, -1.0}, -M_PI_2));
  EXPECT_NEAR(goal_geometry.GetFixedOrientation(), 1.0, kEpsilon);

  const auto goal_so2 =
      goal_geometry.ComputeOrientation(eigenmath::Vector2d(-2.0, -1.0));
  EXPECT_NEAR(goal_so2.angle(), 1.0, kEpsilon);
}

TEST(GoalGeometry, FixedOrientationToArbitrary) {
  GoalGeometry goal_geometry;

  goal_geometry.SetFixedOrientation(1.0, 8.0);

  EXPECT_TRUE(goal_geometry.IsOrientationArbitrary());
  EXPECT_FALSE(goal_geometry.IsOrientationFixed());
  EXPECT_FALSE(goal_geometry.IsToFaceTowardsPoint());

  EXPECT_TRUE(goal_geometry.IsAttractedToNothing());
  EXPECT_FALSE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasInclusionHull());
  EXPECT_FALSE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasExclusionHull());

  EXPECT_GT(goal_geometry.GetOrientationRange(), 2.0 * M_PI);
}

TEST(GoalGeometry, OrientationTarget) {
  GoalGeometry goal_geometry;

  goal_geometry.SetOrientationTarget(eigenmath::Vector2d(1.0, 2.0), 0.5);

  EXPECT_FALSE(goal_geometry.IsOrientationArbitrary());
  EXPECT_FALSE(goal_geometry.IsOrientationFixed());
  EXPECT_TRUE(goal_geometry.IsToFaceTowardsPoint());

  EXPECT_TRUE(goal_geometry.IsAttractedToNothing());
  EXPECT_FALSE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasInclusionHull());
  EXPECT_FALSE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasExclusionHull());

  EXPECT_NEAR(goal_geometry.GetOrientationTarget().x(), 1.0, kEpsilon);
  EXPECT_NEAR(goal_geometry.GetOrientationTarget().y(), 2.0, kEpsilon);
  EXPECT_NEAR(goal_geometry.GetOrientationRange(), 0.5, kEpsilon);

  goal_geometry.ApplyTransform(eigenmath::Pose2d({-1.0, 3.0}, M_PI_2));
  EXPECT_NEAR(goal_geometry.GetOrientationTarget().x(), -3.0, kEpsilon);
  EXPECT_NEAR(goal_geometry.GetOrientationTarget().y(), 4.0, kEpsilon);

  goal_geometry.ApplyTransform(eigenmath::Pose2d({-3.0, -1.0}, -M_PI_2));
  EXPECT_NEAR(goal_geometry.GetOrientationTarget().x(), 1.0, kEpsilon);
  EXPECT_NEAR(goal_geometry.GetOrientationTarget().y(), 2.0, kEpsilon);

  const auto goal_so2 =
      goal_geometry.ComputeOrientation(eigenmath::Vector2d(-2.0, -1.0));
  EXPECT_NEAR(goal_so2.angle(), 0.785398, kEpsilon);
}

TEST(GoalGeometry, OrientationTargetToArbitrary) {
  GoalGeometry goal_geometry;

  goal_geometry.SetOrientationTarget(eigenmath::Vector2d(1.0, 2.0), 8.0);

  EXPECT_TRUE(goal_geometry.IsOrientationArbitrary());
  EXPECT_FALSE(goal_geometry.IsOrientationFixed());
  EXPECT_FALSE(goal_geometry.IsToFaceTowardsPoint());

  EXPECT_TRUE(goal_geometry.IsAttractedToNothing());
  EXPECT_FALSE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasInclusionHull());
  EXPECT_FALSE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasExclusionHull());

  EXPECT_GT(goal_geometry.GetOrientationRange(), 2.0 * M_PI);
}

TEST(GoalGeometry, InclusionRadialSegment) {
  GoalGeometry goal_geometry;

  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));
  goal_geometry.SetInclusionRadialSegment(eigenmath::Vector2d(0.0, 0.0), 0.5,
                                          1.0, 0.0, M_PI);
  goal_geometry.SetDistanceTolerance(0.1);

  EXPECT_FALSE(goal_geometry.IsAttractedToNothing());
  EXPECT_FALSE(goal_geometry.IsAttractedToSinglePoint());
  EXPECT_TRUE(goal_geometry.IsAttractedInDirection());

  EXPECT_TRUE(goal_geometry.IsOrientationArbitrary());
  EXPECT_TRUE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasInclusionHull());
  EXPECT_FALSE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasExclusionHull());

  goal_geometry.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 2.0), 0.0));

  const auto& radial_segment = goal_geometry.GetInclusionRadialSegment();
  EXPECT_NEAR(radial_segment.center.x(), 1.0, kEpsilon);
  EXPECT_NEAR(radial_segment.center.y(), 2.0, kEpsilon);
  EXPECT_NEAR(radial_segment.inner_radius, 0.5, kEpsilon);
  EXPECT_NEAR(radial_segment.outer_radius, 1.0, kEpsilon);
  EXPECT_NEAR(radial_segment.start_angle, 0.0, kEpsilon);
  EXPECT_NEAR(radial_segment.end_angle, M_PI, kEpsilon);

  eigenmath::Vector2d best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.447214, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.894427, kEpsilon);

  goal_geometry.SetAttractionPoint(eigenmath::Vector2d(4.0, 4.0));
  best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.832050, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.554700, kEpsilon);

  goal_geometry.SetAttractionToNothing();
  best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.0, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.0, kEpsilon);
  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));

  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.0, 2.0)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.5, 2.05)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(2.0, 2.05)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.05, 2.5)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.05, 3.0)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(0.5, 2.05)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(0.0, 2.05)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.6, 1.95)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(0.6, 1.95)));

  EXPECT_TRUE(goal_geometry.IsPointInRadiusOfGoalGeometry(
      eigenmath::Vector2d(1.0, 2.45), 0.1));
  EXPECT_FALSE(goal_geometry.IsPointInRadiusOfGoalGeometry(
      eigenmath::Vector2d(1.0, 2.35), 0.1));

  const GridFrame grid_frame("", eigenmath::Pose2d::Identity(), 0.2);
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(8, 11))));
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(6, 11))));
  EXPECT_FALSE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(3, 5))));

  const GridRange goal_range = goal_geometry.GetGoalGridRange(grid_frame);
  EXPECT_EQ(goal_range.lower.x(), 0);
  EXPECT_EQ(goal_range.lower.y(), 5);
  EXPECT_EQ(goal_range.upper.x(), 11);
  EXPECT_EQ(goal_range.upper.y(), 16);

  std::vector<GridIndex> grid_goals;
  std::vector<double> grid_costs;
  goal_geometry.SampleGoalsOnGrid(grid_frame, &grid_goals, &grid_costs);
  EXPECT_THAT(grid_goals,
              UnorderedElementsAreArray(
                  {GridIndex{0, 10}, GridIndex{1, 10}, GridIndex{2, 10},
                   GridIndex{8, 10}, GridIndex{9, 10}, GridIndex{10, 10},
                   GridIndex{1, 11}, GridIndex{2, 11}, GridIndex{8, 11},
                   GridIndex{9, 11}, GridIndex{1, 12}, GridIndex{2, 12},
                   GridIndex{3, 12}, GridIndex{7, 12}, GridIndex{8, 12},
                   GridIndex{9, 12}, GridIndex{1, 13}, GridIndex{2, 13},
                   GridIndex{3, 13}, GridIndex{4, 13}, GridIndex{5, 13},
                   GridIndex{6, 13}, GridIndex{7, 13}, GridIndex{8, 13},
                   GridIndex{9, 13}, GridIndex{3, 14}, GridIndex{4, 14},
                   GridIndex{5, 14}, GridIndex{6, 14}, GridIndex{7, 14},
                   GridIndex{5, 15}}));
  EXPECT_THAT(
      grid_costs,
      ::testing::UnorderedPointwise(
          IsNormComparableTo(),
          {1.44721,  1.35777,  1.26833,  0.731672, 0.642229, 0.552786, 1.17889,
           1.08944,  0.552786, 0.463344, 1.0,      0.910557, 0.821115, 0.463344,
           0.373901, 0.284458, 0.821115, 0.731672, 0.642229, 0.552786, 0.463344,
           0.373901, 0.284458, 0.195016, 0.105573, 0.463344, 0.373901, 0.284458,
           0.195016, 0.105573, 0.105573}));

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.4), 8);
  std::vector<LatticePose> lattice_goals;
  std::vector<double> lattice_costs;
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  EXPECT_THAT(
      lattice_goals,
      UnorderedElementsAreArray(
          {LatticePose(GridIndex(0, 5), 0), LatticePose(GridIndex(0, 5), 1),
           LatticePose(GridIndex(0, 5), 2), LatticePose(GridIndex(0, 5), 3),
           LatticePose(GridIndex(0, 5), 4), LatticePose(GridIndex(0, 5), 5),
           LatticePose(GridIndex(0, 5), 6), LatticePose(GridIndex(0, 5), 7),
           LatticePose(GridIndex(1, 5), 0), LatticePose(GridIndex(1, 5), 1),
           LatticePose(GridIndex(1, 5), 2), LatticePose(GridIndex(1, 5), 3),
           LatticePose(GridIndex(1, 5), 4), LatticePose(GridIndex(1, 5), 5),
           LatticePose(GridIndex(1, 5), 6), LatticePose(GridIndex(1, 5), 7),
           LatticePose(GridIndex(4, 5), 0), LatticePose(GridIndex(4, 5), 1),
           LatticePose(GridIndex(4, 5), 2), LatticePose(GridIndex(4, 5), 3),
           LatticePose(GridIndex(4, 5), 4), LatticePose(GridIndex(4, 5), 5),
           LatticePose(GridIndex(4, 5), 6), LatticePose(GridIndex(4, 5), 7),
           LatticePose(GridIndex(5, 5), 0), LatticePose(GridIndex(5, 5), 1),
           LatticePose(GridIndex(5, 5), 2), LatticePose(GridIndex(5, 5), 3),
           LatticePose(GridIndex(5, 5), 4), LatticePose(GridIndex(5, 5), 5),
           LatticePose(GridIndex(5, 5), 6), LatticePose(GridIndex(5, 5), 7),
           LatticePose(GridIndex(1, 6), 0), LatticePose(GridIndex(1, 6), 1),
           LatticePose(GridIndex(1, 6), 2), LatticePose(GridIndex(1, 6), 3),
           LatticePose(GridIndex(1, 6), 4), LatticePose(GridIndex(1, 6), 5),
           LatticePose(GridIndex(1, 6), 6), LatticePose(GridIndex(1, 6), 7),
           LatticePose(GridIndex(4, 6), 0), LatticePose(GridIndex(4, 6), 1),
           LatticePose(GridIndex(4, 6), 2), LatticePose(GridIndex(4, 6), 3),
           LatticePose(GridIndex(4, 6), 4), LatticePose(GridIndex(4, 6), 5),
           LatticePose(GridIndex(4, 6), 6), LatticePose(GridIndex(4, 6), 7),
           LatticePose(GridIndex(2, 7), 0), LatticePose(GridIndex(2, 7), 1),
           LatticePose(GridIndex(2, 7), 2), LatticePose(GridIndex(2, 7), 3),
           LatticePose(GridIndex(2, 7), 4), LatticePose(GridIndex(2, 7), 5),
           LatticePose(GridIndex(2, 7), 6), LatticePose(GridIndex(2, 7), 7),
           LatticePose(GridIndex(3, 7), 0), LatticePose(GridIndex(3, 7), 1),
           LatticePose(GridIndex(3, 7), 2), LatticePose(GridIndex(3, 7), 3),
           LatticePose(GridIndex(3, 7), 4), LatticePose(GridIndex(3, 7), 5),
           LatticePose(GridIndex(3, 7), 6), LatticePose(GridIndex(3, 7), 7)}));
  EXPECT_THAT(
      lattice_costs,
      ::testing::UnorderedPointwise(
          IsNormComparableTo(),
          {1.44721,  1.44721,  1.44721,  1.44721,  1.44721,  1.44721,  1.44721,
           1.44721,  1.26833,  1.26833,  1.26833,  1.26833,  1.26833,  1.26833,
           1.26833,  1.26833,  0.731672, 0.731672, 0.731672, 0.731672, 0.731672,
           0.731672, 0.731672, 0.731672, 0.910557, 0.910557, 0.910557, 0.910557,
           0.910557, 0.910557, 0.910557, 0.910557, 0.552786, 0.552786, 0.552786,
           0.552786, 0.552786, 0.552786, 0.552786, 0.552786, 0.373901, 0.373901,
           0.373901, 0.373901, 0.373901, 0.373901, 0.373901, 0.373901, 0.373901,
           0.373901, 0.373901, 0.373901, 0.373901, 0.373901, 0.373901, 0.373901,
           0.195016, 0.195016, 0.195016, 0.195016, 0.195016, 0.195016, 0.195016,
           0.195016}));

  goal_geometry.SetFixedOrientation(2.0, 0.5);
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  EXPECT_THAT(
      lattice_goals,
      UnorderedElementsAreArray(
          {LatticePose(GridIndex(0, 5), 2), LatticePose(GridIndex(0, 5), 3),
           LatticePose(GridIndex(1, 5), 2), LatticePose(GridIndex(1, 5), 3),
           LatticePose(GridIndex(4, 5), 2), LatticePose(GridIndex(4, 5), 3),
           LatticePose(GridIndex(5, 5), 2), LatticePose(GridIndex(5, 5), 3),
           LatticePose(GridIndex(1, 6), 2), LatticePose(GridIndex(1, 6), 3),
           LatticePose(GridIndex(4, 6), 2), LatticePose(GridIndex(4, 6), 3),
           LatticePose(GridIndex(2, 7), 2), LatticePose(GridIndex(2, 7), 3),
           LatticePose(GridIndex(3, 7), 2), LatticePose(GridIndex(3, 7), 3)}));
  EXPECT_THAT(lattice_costs,
              ::testing::UnorderedPointwise(
                  IsNormComparableTo(),
                  {1.44721, 1.44721, 1.26833, 1.26833, 0.731672, 0.731672,
                   0.910557, 0.910557, 0.552786, 0.552786, 0.373901, 0.373901,
                   0.373901, 0.373901, 0.195016, 0.195016}));

  goal_geometry.SetOrientationTarget({-1.0, -1.0}, 0.5);
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  EXPECT_THAT(
      lattice_goals,
      UnorderedElementsAreArray(
          {LatticePose(GridIndex(0, 5), 5), LatticePose(GridIndex(0, 5), 6),
           LatticePose(GridIndex(1, 5), 5), LatticePose(GridIndex(1, 5), 6),
           LatticePose(GridIndex(4, 5), 5), LatticePose(GridIndex(5, 5), 5),
           LatticePose(GridIndex(1, 6), 5), LatticePose(GridIndex(1, 6), 6),
           LatticePose(GridIndex(4, 6), 5), LatticePose(GridIndex(2, 7), 5),
           LatticePose(GridIndex(2, 7), 6), LatticePose(GridIndex(3, 7), 5)}));
  EXPECT_THAT(lattice_costs, ::testing::UnorderedPointwise(
                                 IsNormComparableTo(),
                                 {1.44721, 1.44721, 1.26833, 1.26833, 0.731672,
                                  0.552786, 0.910557, 0.910557, 0.373901,
                                  0.373901, 0.373901, 0.195016}));
}

TEST(GoalGeometry, InclusionRadialSegmentSmall) {
  GoalGeometry goal_geometry;

  goal_geometry.SetAttractionToNothing();
  goal_geometry.SetInclusionRadialSegment(eigenmath::Vector2d(1.1, 2.1), 0.0,
                                          0.025, 0.0, M_PI);
  goal_geometry.SetDistanceTolerance(0.1);

  EXPECT_TRUE(goal_geometry.IsAttractedToNothing());
  EXPECT_FALSE(goal_geometry.IsAttractedToSinglePoint());
  EXPECT_FALSE(goal_geometry.IsAttractedInDirection());

  EXPECT_TRUE(goal_geometry.IsOrientationArbitrary());
  EXPECT_TRUE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasInclusionHull());
  EXPECT_FALSE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasExclusionHull());

  eigenmath::Vector2d best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.1, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.1, kEpsilon);

  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.0, 2.0)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.1, 2.1)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.2, 2.2)));

  EXPECT_TRUE(goal_geometry.IsPointInRadiusOfGoalGeometry(
      eigenmath::Vector2d(1.05, 2.05), 0.1));
  EXPECT_FALSE(goal_geometry.IsPointInRadiusOfGoalGeometry(
      eigenmath::Vector2d(1.2, 2.2), 0.1));

  const GridFrame grid_frame("", eigenmath::Pose2d::Identity(), 0.2);
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(8, 11))));
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(6, 11))));
  EXPECT_FALSE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(3, 5))));

  const GridRange goal_range = goal_geometry.GetGoalGridRange(grid_frame);
  EXPECT_EQ(goal_range.lower.x(), 5);
  EXPECT_EQ(goal_range.lower.y(), 9);
  EXPECT_EQ(goal_range.upper.x(), 8);
  EXPECT_EQ(goal_range.upper.y(), 12);

  std::vector<GridIndex> grid_goals;
  std::vector<double> grid_costs;
  goal_geometry.SampleGoalsOnGrid(grid_frame, &grid_goals, &grid_costs);
  EXPECT_THAT(grid_goals,
              UnorderedElementsAreArray({GridIndex{5, 10}, GridIndex{6, 10},
                                         GridIndex{5, 11}, GridIndex{6, 11}}));
}

TEST(GoalGeometry, InclusionHull) {
  GoalGeometry goal_geometry;

  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));
  Hull incl_hull;
  incl_hull.Add({eigenmath::Vector2d(0.0, 0.0), eigenmath::Vector2d(0.5, 0.0),
                 eigenmath::Vector2d(0.5, 0.5), eigenmath::Vector2d(0.0, 0.5)});
  goal_geometry.SetInclusionHull(incl_hull);
  goal_geometry.SetDistanceTolerance(0.1);

  EXPECT_FALSE(goal_geometry.IsAttractedToNothing());
  EXPECT_FALSE(goal_geometry.IsAttractedToSinglePoint());
  EXPECT_TRUE(goal_geometry.IsAttractedInDirection());

  EXPECT_TRUE(goal_geometry.IsOrientationArbitrary());
  EXPECT_FALSE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_TRUE(goal_geometry.HasInclusionHull());
  EXPECT_FALSE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasExclusionHull());

  goal_geometry.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 2.0), 0.0));

  eigenmath::Vector2d best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.408114, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.566228, kEpsilon);

  goal_geometry.SetAttractionPoint(eigenmath::Vector2d(4.0, 4.0));
  best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.548279, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.439814, kEpsilon);

  goal_geometry.SetAttractionToNothing();
  best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.25, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.25, kEpsilon);
  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));

  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(0.95, 2.0)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.45, 2.05)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.45, 2.55)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.45, 2.45)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.05, 1.95)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.05, 2.05)));

  const GridFrame grid_frame("", eigenmath::Pose2d::Identity(), 0.2);
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(8, 11))));
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(6, 11))));
  EXPECT_FALSE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(3, 5))));

  const GridRange goal_range = goal_geometry.GetGoalGridRange(grid_frame);
  EXPECT_EQ(goal_range.lower.x(), 4);
  EXPECT_EQ(goal_range.lower.y(), 9);
  EXPECT_EQ(goal_range.upper.x(), 9);
  EXPECT_EQ(goal_range.upper.y(), 14);

  std::vector<GridIndex> grid_goals;
  std::vector<double> grid_costs;
  goal_geometry.SampleGoalsOnGrid(grid_frame, &grid_goals, &grid_costs);
  EXPECT_THAT(grid_goals,
              UnorderedElementsAreArray(
                  {GridIndex{5, 10}, GridIndex{6, 10}, GridIndex{7, 10},
                   GridIndex{5, 11}, GridIndex{6, 11}, GridIndex{7, 11},
                   GridIndex{5, 12}, GridIndex{6, 12}, GridIndex{7, 12}}));
  EXPECT_THAT(grid_costs, ::testing::UnorderedPointwise(
                              IsNormComparableTo(),
                              {0.688964, 0.599521, 0.510078, 0.510078, 0.420635,
                               0.331193, 0.331193, 0.24175, 0.152307}));

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.3), 8);
  std::vector<LatticePose> lattice_goals;
  std::vector<double> lattice_costs;
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  EXPECT_THAT(
      lattice_goals,
      UnorderedElementsAreArray(
          {LatticePose(GridIndex(4, 7), 0), LatticePose(GridIndex(4, 7), 1),
           LatticePose(GridIndex(4, 7), 2), LatticePose(GridIndex(4, 7), 3),
           LatticePose(GridIndex(4, 7), 4), LatticePose(GridIndex(4, 7), 5),
           LatticePose(GridIndex(4, 7), 6), LatticePose(GridIndex(4, 7), 7),
           LatticePose(GridIndex(5, 7), 0), LatticePose(GridIndex(5, 7), 1),
           LatticePose(GridIndex(5, 7), 2), LatticePose(GridIndex(5, 7), 3),
           LatticePose(GridIndex(5, 7), 4), LatticePose(GridIndex(5, 7), 5),
           LatticePose(GridIndex(5, 7), 6), LatticePose(GridIndex(5, 7), 7),
           LatticePose(GridIndex(4, 8), 0), LatticePose(GridIndex(4, 8), 1),
           LatticePose(GridIndex(4, 8), 2), LatticePose(GridIndex(4, 8), 3),
           LatticePose(GridIndex(4, 8), 4), LatticePose(GridIndex(4, 8), 5),
           LatticePose(GridIndex(4, 8), 6), LatticePose(GridIndex(4, 8), 7),
           LatticePose(GridIndex(5, 8), 0), LatticePose(GridIndex(5, 8), 1),
           LatticePose(GridIndex(5, 8), 2), LatticePose(GridIndex(5, 8), 3),
           LatticePose(GridIndex(5, 8), 4), LatticePose(GridIndex(5, 8), 5),
           LatticePose(GridIndex(5, 8), 6), LatticePose(GridIndex(5, 8), 7)}));
  EXPECT_THAT(
      lattice_costs,
      ::testing::UnorderedPointwise(
          IsNormComparableTo(),
          {0.510078, 0.510078, 0.510078, 0.510078, 0.510078, 0.510078, 0.510078,
           0.510078, 0.375914, 0.375914, 0.375914, 0.375914, 0.375914, 0.375914,
           0.375914, 0.375914, 0.24175,  0.24175,  0.24175,  0.24175,  0.24175,
           0.24175,  0.24175,  0.24175,  0.107586, 0.107586, 0.107586, 0.107586,
           0.107586, 0.107586, 0.107586, 0.107586}));

  goal_geometry.SetFixedOrientation(2.0, 0.5);
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  EXPECT_THAT(
      lattice_goals,
      UnorderedElementsAreArray(
          {LatticePose(GridIndex(4, 7), 2), LatticePose(GridIndex(4, 7), 3),
           LatticePose(GridIndex(5, 7), 2), LatticePose(GridIndex(5, 7), 3),
           LatticePose(GridIndex(4, 8), 2), LatticePose(GridIndex(4, 8), 3),
           LatticePose(GridIndex(5, 8), 2), LatticePose(GridIndex(5, 8), 3)}));
  EXPECT_THAT(lattice_costs, ::testing::UnorderedPointwise(
                                 IsNormComparableTo(),
                                 {0.510078, 0.510078, 0.375914, 0.375914,
                                  0.24175, 0.24175, 0.107586, 0.107586}));

  goal_geometry.SetOrientationTarget({-1.0, -1.0}, 0.5);
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  EXPECT_THAT(
      lattice_goals,
      UnorderedElementsAreArray(
          {LatticePose(GridIndex(4, 7), 5), LatticePose(GridIndex(5, 7), 5),
           LatticePose(GridIndex(4, 8), 5), LatticePose(GridIndex(5, 8), 5)}));
  EXPECT_THAT(lattice_costs, ::testing::UnorderedPointwise(
                                 IsNormComparableTo(),
                                 {0.510078, 0.375914, 0.24175, 0.107586}));
}

TEST(GoalGeometry, InclusionHullSmall) {
  GoalGeometry goal_geometry;

  goal_geometry.SetAttractionToNothing();
  Hull incl_hull;
  incl_hull.Add(
      {eigenmath::Vector2d(1.075, 2.075), eigenmath::Vector2d(1.125, 2.075),
       eigenmath::Vector2d(1.125, 2.125), eigenmath::Vector2d(1.075, 2.125)});
  goal_geometry.SetInclusionHull(incl_hull);
  goal_geometry.SetDistanceTolerance(0.1);

  EXPECT_TRUE(goal_geometry.IsAttractedToNothing());
  EXPECT_FALSE(goal_geometry.IsAttractedToSinglePoint());
  EXPECT_FALSE(goal_geometry.IsAttractedInDirection());

  EXPECT_TRUE(goal_geometry.IsOrientationArbitrary());
  EXPECT_FALSE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_TRUE(goal_geometry.HasInclusionHull());
  EXPECT_FALSE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasExclusionHull());

  eigenmath::Vector2d best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.1, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.1, kEpsilon);

  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.0, 2.0)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.1, 2.1)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.2, 2.2)));

  EXPECT_TRUE(goal_geometry.IsPointInRadiusOfGoalGeometry(
      eigenmath::Vector2d(1.05, 2.05), 0.1));
  EXPECT_FALSE(goal_geometry.IsPointInRadiusOfGoalGeometry(
      eigenmath::Vector2d(1.2, 2.2), 0.1));

  const GridFrame grid_frame("", eigenmath::Pose2d::Identity(), 0.2);
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(8, 11))));
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(6, 11))));
  EXPECT_FALSE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(3, 5))));

  const GridRange goal_range = goal_geometry.GetGoalGridRange(grid_frame);
  EXPECT_EQ(goal_range.lower.x(), 5);
  EXPECT_EQ(goal_range.lower.y(), 9);
  EXPECT_EQ(goal_range.upper.x(), 8);
  EXPECT_EQ(goal_range.upper.y(), 12);

  std::vector<GridIndex> grid_goals;
  std::vector<double> grid_costs;
  goal_geometry.SampleGoalsOnGrid(grid_frame, &grid_goals, &grid_costs);
  EXPECT_THAT(grid_goals,
              UnorderedElementsAreArray({GridIndex{5, 10}, GridIndex{6, 10},
                                         GridIndex{5, 11}, GridIndex{6, 11}}));
}

TEST(GoalGeometry, ExclusionRadialSegment) {
  GoalGeometry goal_geometry;

  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));
  goal_geometry.SetInclusionRadialSegment(eigenmath::Vector2d(0.0, 0.0), 0.5,
                                          1.0, 0.0, M_PI);
  goal_geometry.SetExclusionRadialSegment(eigenmath::Vector2d(0.0, 0.0), 0.5,
                                          0.7, 0.0, M_PI);
  goal_geometry.SetDistanceTolerance(0.1);

  EXPECT_FALSE(goal_geometry.IsAttractedToNothing());
  EXPECT_FALSE(goal_geometry.IsAttractedToSinglePoint());
  EXPECT_TRUE(goal_geometry.IsAttractedInDirection());

  EXPECT_TRUE(goal_geometry.IsOrientationArbitrary());
  EXPECT_TRUE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasInclusionHull());
  EXPECT_TRUE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_FALSE(goal_geometry.HasExclusionHull());

  goal_geometry.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 2.0), 0.0));

  const auto& radial_segment = goal_geometry.GetExclusionRadialSegment();
  EXPECT_NEAR(radial_segment.center.x(), 1.0, kEpsilon);
  EXPECT_NEAR(radial_segment.center.y(), 2.0, kEpsilon);
  EXPECT_NEAR(radial_segment.inner_radius, 0.5, kEpsilon);
  EXPECT_NEAR(radial_segment.outer_radius, 0.7, kEpsilon);
  EXPECT_NEAR(radial_segment.start_angle, 0.0, kEpsilon);
  EXPECT_NEAR(radial_segment.end_angle, M_PI, kEpsilon);

  eigenmath::Vector2d best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.447214, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.894427, kEpsilon);

  goal_geometry.SetAttractionPoint(eigenmath::Vector2d(4.0, 4.0));
  best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.832050, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.554700, kEpsilon);

  goal_geometry.SetAttractionToNothing();
  best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.0, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.0, kEpsilon);
  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));

  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.0, 2.0)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.5, 2.05)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.95, 2.05)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(2.0, 2.05)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.05, 2.5)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.0, 2.95)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.05, 3.0)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(0.5, 2.05)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(0.05, 2.05)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(0.0, 2.05)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.6, 1.95)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(0.6, 1.95)));

  const GridFrame grid_frame("", eigenmath::Pose2d::Identity(), 0.2);
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(8, 11))));
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(6, 11))));
  EXPECT_FALSE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(3, 5))));

  const GridRange goal_range = goal_geometry.GetGoalGridRange(grid_frame);
  EXPECT_EQ(goal_range.lower.x(), 0);
  EXPECT_EQ(goal_range.lower.y(), 5);
  EXPECT_EQ(goal_range.upper.x(), 11);
  EXPECT_EQ(goal_range.upper.y(), 16);

  std::vector<GridIndex> grid_goals;
  std::vector<double> grid_costs;
  goal_geometry.SampleGoalsOnGrid(grid_frame, &grid_goals, &grid_costs);
  EXPECT_THAT(grid_goals,
              UnorderedElementsAreArray(
                  {GridIndex{0, 10},  GridIndex{1, 10}, GridIndex{9, 10},
                   GridIndex{10, 10}, GridIndex{1, 11}, GridIndex{9, 11},
                   GridIndex{1, 12},  GridIndex{2, 12}, GridIndex{8, 12},
                   GridIndex{9, 12},  GridIndex{1, 13}, GridIndex{2, 13},
                   GridIndex{3, 13},  GridIndex{7, 13}, GridIndex{8, 13},
                   GridIndex{9, 13},  GridIndex{3, 14}, GridIndex{4, 14},
                   GridIndex{5, 14},  GridIndex{6, 14}, GridIndex{7, 14},
                   GridIndex{5, 15}}));
  EXPECT_THAT(grid_costs,
              ::testing::UnorderedPointwise(
                  IsNormComparableTo(),
                  {1.44721,  1.35777,  0.642229, 0.552786, 1.17889,  0.463344,
                   1.0,      0.910557, 0.373901, 0.284458, 0.821115, 0.731672,
                   0.642229, 0.284458, 0.195016, 0.105573, 0.463344, 0.373901,
                   0.284458, 0.195016, 0.105573, 0.105573}));

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.4), 8);
  std::vector<LatticePose> lattice_goals;
  std::vector<double> lattice_costs;
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  EXPECT_THAT(
      lattice_goals,
      UnorderedElementsAreArray(
          {LatticePose(GridIndex(0, 5), 0), LatticePose(GridIndex(0, 5), 1),
           LatticePose(GridIndex(0, 5), 2), LatticePose(GridIndex(0, 5), 3),
           LatticePose(GridIndex(0, 5), 4), LatticePose(GridIndex(0, 5), 5),
           LatticePose(GridIndex(0, 5), 6), LatticePose(GridIndex(0, 5), 7),
           LatticePose(GridIndex(5, 5), 0), LatticePose(GridIndex(5, 5), 1),
           LatticePose(GridIndex(5, 5), 2), LatticePose(GridIndex(5, 5), 3),
           LatticePose(GridIndex(5, 5), 4), LatticePose(GridIndex(5, 5), 5),
           LatticePose(GridIndex(5, 5), 6), LatticePose(GridIndex(5, 5), 7),
           LatticePose(GridIndex(1, 6), 0), LatticePose(GridIndex(1, 6), 1),
           LatticePose(GridIndex(1, 6), 2), LatticePose(GridIndex(1, 6), 3),
           LatticePose(GridIndex(1, 6), 4), LatticePose(GridIndex(1, 6), 5),
           LatticePose(GridIndex(1, 6), 6), LatticePose(GridIndex(1, 6), 7),
           LatticePose(GridIndex(4, 6), 0), LatticePose(GridIndex(4, 6), 1),
           LatticePose(GridIndex(4, 6), 2), LatticePose(GridIndex(4, 6), 3),
           LatticePose(GridIndex(4, 6), 4), LatticePose(GridIndex(4, 6), 5),
           LatticePose(GridIndex(4, 6), 6), LatticePose(GridIndex(4, 6), 7),
           LatticePose(GridIndex(2, 7), 0), LatticePose(GridIndex(2, 7), 1),
           LatticePose(GridIndex(2, 7), 2), LatticePose(GridIndex(2, 7), 3),
           LatticePose(GridIndex(2, 7), 4), LatticePose(GridIndex(2, 7), 5),
           LatticePose(GridIndex(2, 7), 6), LatticePose(GridIndex(2, 7), 7),
           LatticePose(GridIndex(3, 7), 0), LatticePose(GridIndex(3, 7), 1),
           LatticePose(GridIndex(3, 7), 2), LatticePose(GridIndex(3, 7), 3),
           LatticePose(GridIndex(3, 7), 4), LatticePose(GridIndex(3, 7), 5),
           LatticePose(GridIndex(3, 7), 6), LatticePose(GridIndex(3, 7), 7)}));
  EXPECT_THAT(
      lattice_costs,
      ::testing::UnorderedPointwise(
          IsNormComparableTo(),
          {1.44721,  1.44721,  1.44721,  1.44721,  1.44721,  1.44721,  1.44721,
           1.44721,  0.910557, 0.910557, 0.910557, 0.910557, 0.910557, 0.910557,
           0.910557, 0.910557, 0.552786, 0.552786, 0.552786, 0.552786, 0.552786,
           0.552786, 0.552786, 0.552786, 0.373901, 0.373901, 0.373901, 0.373901,
           0.373901, 0.373901, 0.373901, 0.373901, 0.373901, 0.373901, 0.373901,
           0.373901, 0.373901, 0.373901, 0.373901, 0.373901, 0.195016, 0.195016,
           0.195016, 0.195016, 0.195016, 0.195016, 0.195016, 0.195016}));

  goal_geometry.SetFixedOrientation(2.0, 0.5);
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  EXPECT_THAT(
      lattice_goals,
      UnorderedElementsAreArray(
          {LatticePose(GridIndex(0, 5), 2), LatticePose(GridIndex(0, 5), 3),
           LatticePose(GridIndex(5, 5), 2), LatticePose(GridIndex(5, 5), 3),
           LatticePose(GridIndex(1, 6), 2), LatticePose(GridIndex(1, 6), 3),
           LatticePose(GridIndex(4, 6), 2), LatticePose(GridIndex(4, 6), 3),
           LatticePose(GridIndex(2, 7), 2), LatticePose(GridIndex(2, 7), 3),
           LatticePose(GridIndex(3, 7), 2), LatticePose(GridIndex(3, 7), 3)}));
  EXPECT_THAT(lattice_costs, ::testing::UnorderedPointwise(
                                 IsNormComparableTo(),
                                 {1.44721, 1.44721, 0.910557, 0.910557,
                                  0.552786, 0.552786, 0.373901, 0.373901,
                                  0.373901, 0.373901, 0.195016, 0.195016}));
}

TEST(GoalGeometry, ExclusionHull) {
  GoalGeometry goal_geometry;

  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));
  Hull incl_hull;
  incl_hull.Add({eigenmath::Vector2d(0.0, 0.0), eigenmath::Vector2d(0.5, 0.0),
                 eigenmath::Vector2d(0.5, 0.5), eigenmath::Vector2d(0.0, 0.5)});
  goal_geometry.SetInclusionHull(incl_hull);
  Hull excl_hull;
  excl_hull.Add(
      {eigenmath::Vector2d(-0.5, -0.5), eigenmath::Vector2d(0.25, -0.5),
       eigenmath::Vector2d(0.25, 0.25), eigenmath::Vector2d(-0.5, 0.25)});
  goal_geometry.SetExclusionHull(excl_hull);
  goal_geometry.SetDistanceTolerance(0.1);

  EXPECT_FALSE(goal_geometry.IsAttractedToNothing());
  EXPECT_FALSE(goal_geometry.IsAttractedToSinglePoint());
  EXPECT_TRUE(goal_geometry.IsAttractedInDirection());

  EXPECT_TRUE(goal_geometry.IsOrientationArbitrary());
  EXPECT_FALSE(goal_geometry.HasInclusionRadialSegment());
  EXPECT_TRUE(goal_geometry.HasInclusionHull());
  EXPECT_FALSE(goal_geometry.HasExclusionRadialSegment());
  EXPECT_TRUE(goal_geometry.HasExclusionHull());

  goal_geometry.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 2.0), 0.0));

  eigenmath::Vector2d best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.408114, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.566228, kEpsilon);

  goal_geometry.SetAttractionPoint(eigenmath::Vector2d(4.0, 4.0));
  best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.548279, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.439814, kEpsilon);

  goal_geometry.SetAttractionToNothing();
  best_goal_point = goal_geometry.ComputeBestPossibleGoal();
  EXPECT_NEAR(best_goal_point.x(), 1.25, kEpsilon);
  EXPECT_NEAR(best_goal_point.y(), 2.25, kEpsilon);
  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));

  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(0.95, 2.0)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.45, 2.05)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.45, 2.55)));
  EXPECT_TRUE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.45, 2.45)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.05, 1.95)));
  EXPECT_FALSE(
      goal_geometry.IsPointInGoalGeometry(eigenmath::Vector2d(1.05, 2.05)));

  const GridFrame grid_frame("", eigenmath::Pose2d::Identity(), 0.2);
  EXPECT_TRUE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(8, 11))));
  EXPECT_FALSE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(6, 11))));
  EXPECT_FALSE(goal_geometry.IntersectsGridRange(
      grid_frame, GridRange::OriginTo(GridIndex(3, 5))));

  const GridRange goal_range = goal_geometry.GetGoalGridRange(grid_frame);
  EXPECT_EQ(goal_range.lower.x(), 4);
  EXPECT_EQ(goal_range.lower.y(), 9);
  EXPECT_EQ(goal_range.upper.x(), 9);
  EXPECT_EQ(goal_range.upper.y(), 14);

  std::vector<GridIndex> grid_goals;
  std::vector<double> grid_costs;
  goal_geometry.SampleGoalsOnGrid(grid_frame, &grid_goals, &grid_costs);
  EXPECT_THAT(grid_goals,
              UnorderedElementsAreArray({GridIndex{7, 10}, GridIndex{7, 11},
                                         GridIndex{5, 12}, GridIndex{6, 12},
                                         GridIndex{7, 12}}));
  EXPECT_THAT(grid_costs,
              ::testing::UnorderedPointwise(
                  IsNormComparableTo(),
                  {0.510078, 0.331193, 0.331193, 0.24175, 0.152307}));

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.3), 8);
  std::vector<LatticePose> lattice_goals;
  std::vector<double> lattice_costs;
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  EXPECT_THAT(
      lattice_goals,
      UnorderedElementsAreArray(
          {LatticePose(GridIndex(5, 7), 0), LatticePose(GridIndex(5, 7), 1),
           LatticePose(GridIndex(5, 7), 2), LatticePose(GridIndex(5, 7), 3),
           LatticePose(GridIndex(5, 7), 4), LatticePose(GridIndex(5, 7), 5),
           LatticePose(GridIndex(5, 7), 6), LatticePose(GridIndex(5, 7), 7),
           LatticePose(GridIndex(4, 8), 0), LatticePose(GridIndex(4, 8), 1),
           LatticePose(GridIndex(4, 8), 2), LatticePose(GridIndex(4, 8), 3),
           LatticePose(GridIndex(4, 8), 4), LatticePose(GridIndex(4, 8), 5),
           LatticePose(GridIndex(4, 8), 6), LatticePose(GridIndex(4, 8), 7),
           LatticePose(GridIndex(5, 8), 0), LatticePose(GridIndex(5, 8), 1),
           LatticePose(GridIndex(5, 8), 2), LatticePose(GridIndex(5, 8), 3),
           LatticePose(GridIndex(5, 8), 4), LatticePose(GridIndex(5, 8), 5),
           LatticePose(GridIndex(5, 8), 6), LatticePose(GridIndex(5, 8), 7)}));
  EXPECT_THAT(
      lattice_costs,
      ::testing::UnorderedPointwise(
          IsNormComparableTo(),
          {0.375914, 0.375914, 0.375914, 0.375914, 0.375914, 0.375914,
           0.375914, 0.375914, 0.24175,  0.24175,  0.24175,  0.24175,
           0.24175,  0.24175,  0.24175,  0.24175,  0.107586, 0.107586,
           0.107586, 0.107586, 0.107586, 0.107586, 0.107586, 0.107586}));

  goal_geometry.SetFixedOrientation(2.0, 0.5);
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &lattice_goals,
                                     &lattice_costs);
  EXPECT_THAT(
      lattice_goals,
      UnorderedElementsAreArray(
          {LatticePose(GridIndex(5, 7), 2), LatticePose(GridIndex(5, 7), 3),
           LatticePose(GridIndex(4, 8), 2), LatticePose(GridIndex(4, 8), 3),
           LatticePose(GridIndex(5, 8), 2), LatticePose(GridIndex(5, 8), 3)}));
  EXPECT_THAT(lattice_costs,
              ::testing::UnorderedPointwise(
                  IsNormComparableTo(),
                  {0.375914, 0.375914, 0.24175, 0.24175, 0.107586, 0.107586}));
}

TEST(ComputeGoalCostForPose, AttractionPoint) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionPoint(eigenmath::Vector2d(1.0, 2.0));
  goal_geometry.SetDistanceTolerance(0.1);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.05), 8);
  std::vector<LatticePose> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals, Not(IsEmpty()));
  for (const auto& [lattice_pose, cost] : genit::ZipRange(goals, costs)) {
    const eigenmath::Pose2d goal_pose =
        lattice_frame.LatticeToFrame(lattice_pose);
    EXPECT_THAT(goal_geometry.ComputeGoalCostForPose(goal_pose),
                DoubleEq(cost));
  }
}

TEST(ComputeGoalCostForPose, InclusionRadialSegment) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));
  goal_geometry.SetInclusionRadialSegment(eigenmath::Vector2d(0.0, 0.0), 0.5,
                                          1.0, 0.0, M_PI);
  goal_geometry.SetDistanceTolerance(0.1);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.05), 8);
  std::vector<LatticePose> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals, Not(IsEmpty()));
  for (const auto& [lattice_pose, cost] : genit::ZipRange(goals, costs)) {
    const eigenmath::Pose2d goal_pose =
        lattice_frame.LatticeToFrame(lattice_pose);
    EXPECT_THAT(goal_geometry.ComputeGoalCostForPose(goal_pose),
                DoubleEq(cost));
  }
}

TEST(ComputeGoalCostForPose, InclusionHull) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));
  Hull incl_hull;
  incl_hull.Add({eigenmath::Vector2d(0.0, 0.0), eigenmath::Vector2d(0.5, 0.0),
                 eigenmath::Vector2d(0.5, 0.5), eigenmath::Vector2d(0.0, 0.5)});
  goal_geometry.SetInclusionHull(incl_hull);
  goal_geometry.SetDistanceTolerance(0.1);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.05), 8);
  std::vector<LatticePose> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals, Not(IsEmpty()));
  for (const auto& [lattice_pose, cost] : genit::ZipRange(goals, costs)) {
    const eigenmath::Pose2d goal_pose =
        lattice_frame.LatticeToFrame(lattice_pose);
    EXPECT_THAT(goal_geometry.ComputeGoalCostForPose(goal_pose),
                DoubleEq(cost));
  }
}

TEST(ComputeGoalCostForPose, ExclusionRadialSegment) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));
  goal_geometry.SetInclusionRadialSegment(eigenmath::Vector2d(0.0, 0.0), 0.5,
                                          1.0, 0.0, M_PI);
  goal_geometry.SetExclusionRadialSegment(eigenmath::Vector2d(0.0, 0.0), 0.5,
                                          0.7, 0.0, M_PI);
  goal_geometry.SetDistanceTolerance(0.1);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.05), 8);
  std::vector<LatticePose> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals, Not(IsEmpty()));
  for (const auto& [lattice_pose, cost] : genit::ZipRange(goals, costs)) {
    const eigenmath::Pose2d goal_pose =
        lattice_frame.LatticeToFrame(lattice_pose);
    EXPECT_THAT(goal_geometry.ComputeGoalCostForPose(goal_pose),
                DoubleEq(cost));
  }
}

TEST(ComputeGoalCostForPose, ExclusionHull) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));
  Hull incl_hull;
  incl_hull.Add({eigenmath::Vector2d(0.0, 0.0), eigenmath::Vector2d(0.5, 0.0),
                 eigenmath::Vector2d(0.5, 0.5), eigenmath::Vector2d(0.0, 0.5)});
  goal_geometry.SetInclusionHull(incl_hull);
  Hull excl_hull;
  excl_hull.Add(
      {eigenmath::Vector2d(-0.5, -0.5), eigenmath::Vector2d(0.25, -0.5),
       eigenmath::Vector2d(0.25, 0.25), eigenmath::Vector2d(-0.5, 0.25)});
  goal_geometry.SetExclusionHull(excl_hull);
  goal_geometry.SetDistanceTolerance(0.1);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.05), 8);
  std::vector<LatticePose> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsOnLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals, Not(IsEmpty()));
  for (const auto& [lattice_pose, cost] : genit::ZipRange(goals, costs)) {
    const eigenmath::Pose2d goal_pose =
        lattice_frame.LatticeToFrame(lattice_pose);
    EXPECT_THAT(goal_geometry.ComputeGoalCostForPose(goal_pose),
                DoubleEq(cost));
  }
}

TEST(SampleGoalsWithLattice, AttractionPoint) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionPoint(eigenmath::Vector2d(1.0, 2.0));
  goal_geometry.SetDistanceTolerance(0.1);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.07), 8);
  std::vector<eigenmath::Pose2d> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsWithLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals.size(), costs.size());

  // Goals should match attraction point exactly, and have orientation of
  // lattice.
  std::vector<eigenmath::Pose2d> poses;
  for (int angle_index = 0; angle_index < lattice_frame.num_angle_divisions;
       angle_index += 1) {
    poses.push_back(
        {{1.0, 2.0}, lattice_frame.LatticeAngleIndexToFrameSO2(angle_index)});
  }
  EXPECT_THAT(goals, Pointwise(IsApproxTuple(1e-12), poses));
}

TEST(SampleGoalsWithLattice, InclusionRadialSegment) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));
  goal_geometry.SetInclusionRadialSegment(eigenmath::Vector2d(0.3, 0.1), 0.5,
                                          1.0, 0.4, 0.8);
  goal_geometry.SetDistanceTolerance(0.1);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.07), 8);
  std::vector<eigenmath::Pose2d> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsWithLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals.size(), costs.size());

  // Goals should be the same as those for the lattice centered at the
  // attraction point.
  const LatticeFrame aligned_frame(
      GridFrame("", eigenmath::Pose2d({0.3, 0.1}, 0.0), 0.07), 8);
  std::vector<LatticePose> lattice_goals;
  goal_geometry.SampleGoalsOnLattice(aligned_frame, &lattice_goals, &costs);
  std::vector<eigenmath::Pose2d> expected_goals;
  for (const auto& pose : lattice_goals) {
    expected_goals.push_back(aligned_frame.LatticeToFrame(pose));
  }
  EXPECT_THAT(goals, UnorderedPointwise(IsApproxTuple(1e-8), expected_goals));
}

TEST(SampleGoalsWithLattice, InclusionHull) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));
  Hull incl_hull;
  incl_hull.Add({eigenmath::Vector2d(0.0, 0.0), eigenmath::Vector2d(0.5, 0.0),
                 eigenmath::Vector2d(0.5, 0.5), eigenmath::Vector2d(0.0, 0.5)});
  goal_geometry.SetInclusionHull(incl_hull);
  goal_geometry.SetDistanceTolerance(0.1);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.07), 8);
  std::vector<eigenmath::Pose2d> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsWithLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals.size(), costs.size());

  // Should not have to make adjustments.
  const LatticeFrame aligned_frame = lattice_frame;
  std::vector<LatticePose> lattice_goals;
  goal_geometry.SampleGoalsOnLattice(aligned_frame, &lattice_goals, &costs);
  std::vector<eigenmath::Pose2d> expected_goals;
  for (const auto& pose : lattice_goals) {
    expected_goals.push_back(aligned_frame.LatticeToFrame(pose));
  }
  EXPECT_THAT(goals, UnorderedPointwise(IsApproxTuple(1e-8), expected_goals));
}

TEST(SampleGoalsWithLattice, ExclusionRadialSegment) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionDirection(eigenmath::Vector2d(1.0, 2.0));
  goal_geometry.SetInclusionRadialSegment(eigenmath::Vector2d(0.5, 0.6), 0.5,
                                          1.0, 0.0, M_PI);
  goal_geometry.SetExclusionRadialSegment(eigenmath::Vector2d(0.0, 0.0), 0.5,
                                          0.7, 0.0, M_PI);
  goal_geometry.SetDistanceTolerance(0.1);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.07), 8);
  std::vector<eigenmath::Pose2d> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsWithLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals.size(), costs.size());

  // Goals should be the same as those for the lattice centered at the
  // center of the inclusion segment.
  const LatticeFrame aligned_frame(
      GridFrame("", eigenmath::Pose2d({0.5, 0.6}, 0.0), 0.07), 8);
  std::vector<LatticePose> lattice_goals;
  goal_geometry.SampleGoalsOnLattice(aligned_frame, &lattice_goals, &costs);
  std::vector<eigenmath::Pose2d> expected_goals;
  for (const auto& pose : lattice_goals) {
    expected_goals.push_back(aligned_frame.LatticeToFrame(pose));
  }
  EXPECT_THAT(goals, UnorderedPointwise(IsApproxTuple(1e-8), expected_goals));
}

TEST(SampleGoalsWithLattice, FixedOrientation) {
  GoalGeometry goal_geometry;
  goal_geometry.SetFixedOrientation(0.2, 0.05);
  goal_geometry.SetAttractionPoint({1.0, 2.0});

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.07), 8);
  std::vector<eigenmath::Pose2d> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsWithLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals.size(), costs.size());
  EXPECT_THAT(goals, Pointwise(IsApproxTuple(1e-12),
                               {eigenmath::Pose2d({1.0, 2.0}, 0.2)}));
}

TEST(SampleGoalsWithLattice, FixedOrientationRangeSmallTolerance) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionPoint({1.0, 2.0});
  goal_geometry.SetFixedOrientation(0.2, 1.0);
  goal_geometry.SetOrientationTolerance(0.1);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.07), 8);
  std::vector<eigenmath::Pose2d> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsWithLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals.size(), costs.size());
  EXPECT_THAT(goals, Pointwise(IsApproxTuple(1e-12),
                               {eigenmath::Pose2d({1.0, 2.0}, -0.7),
                                eigenmath::Pose2d({1.0, 2.0}, -0.4),
                                eigenmath::Pose2d({1.0, 2.0}, -0.1),
                                eigenmath::Pose2d({1.0, 2.0}, 0.2),
                                eigenmath::Pose2d({1.0, 2.0}, 0.5),
                                eigenmath::Pose2d({1.0, 2.0}, 0.8),
                                eigenmath::Pose2d({1.0, 2.0}, 1.1)}));
}

TEST(SampleGoalsWithLattice, FixedOrientationRangeLargeTolerance) {
  GoalGeometry goal_geometry;
  goal_geometry.SetFixedOrientation(0.2, 0.9);
  goal_geometry.SetAttractionPoint({1.0, 2.0});
  goal_geometry.SetOrientationTolerance(0.6);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.07), 8);
  std::vector<eigenmath::Pose2d> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsWithLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals.size(), costs.size());
  // With the tolerance added, we stay within the orientation bounds.
  EXPECT_THAT(goals, Pointwise(IsApproxTuple(1e-12),
                               {eigenmath::Pose2d({1.0, 2.0}, -0.1),
                                eigenmath::Pose2d({1.0, 2.0}, 0.2),
                                eigenmath::Pose2d({1.0, 2.0}, 0.5)}));
}

TEST(SampleGoalsWithLattice, OrientationTarget) {
  GoalGeometry goal_geometry;
  const eigenmath::Vector2d target_point(1.0, 2.0);
  goal_geometry.SetOrientationTarget(target_point, 0.01);
  goal_geometry.SetOrientationTolerance(0.05);
  Hull incl_hull;
  incl_hull.Add({eigenmath::Vector2d(0.0, 0.0), eigenmath::Vector2d(0.5, 0.0),
                 eigenmath::Vector2d(0.5, 0.5), eigenmath::Vector2d(0.0, 0.5)});
  goal_geometry.SetInclusionHull(incl_hull);
  goal_geometry.SetDistanceTolerance(0.1);

  const LatticeFrame lattice_frame(
      GridFrame("", eigenmath::Pose2d::Identity(), 0.07), 8);
  std::vector<eigenmath::Pose2d> goals;
  std::vector<double> costs;
  goal_geometry.SampleGoalsWithLattice(lattice_frame, &goals, &costs);
  EXPECT_THAT(goals.size(), costs.size());
  EXPECT_THAT(goals, Not(IsEmpty()));

  // Goals should have the target orientation.
  for (const auto& goal : goals) {
    const double distance = (target_point - goal.translation()).norm();
    if (distance > 1e-3) {
      const eigenmath::Vector2d reaches =
          goal.translation() + distance * goal.xAxis();
      EXPECT_THAT(reaches, IsApprox(target_point)) << "got goal " << goal;
    }
  }
}

TEST(DistanceToTargetOrientation, FixedOrientation) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionPoint({1.0, 2.0});
  goal_geometry.SetFixedOrientation(1.0, 0.1);

  // At target point.
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, 0.0}),
              DoubleEq(0.9));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, 0.9}),
              DoubleEq(0.0));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, 1.1}),
              DoubleEq(0.0));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, 1.5}),
              DoubleEq(0.4));
  // At other point.
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, 0.0}),
              DoubleEq(0.9));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, 0.9}),
              DoubleEq(0.0));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, 1.1}),
              DoubleEq(0.0));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, 1.5}),
              DoubleEq(0.4));
}

TEST(DistanceToTargetOrientation, FixedOrientationAtPi) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionPoint({1.0, 2.0});
  goal_geometry.SetFixedOrientation(M_PI, 0.1);

  // At target point.
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, 0.0}),
              DoubleEq(M_PI - 0.1));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, 0.1}),
              DoubleEq(M_PI - 0.2));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, -0.1}),
              DoubleEq(M_PI - 0.2));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, 3.0}),
              DoubleEq(M_PI - 3.1));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, -3.0}),
              DoubleNear(M_PI - 3.1, 1e-12));
  EXPECT_THAT(
      goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, M_PI - 0.01}),
      DoubleEq(0.0));
  EXPECT_THAT(
      goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, M_PI + 0.01}),
      DoubleEq(0.0));
  // At other point.
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, 0.0}),
              DoubleEq(M_PI - 0.1));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, 0.9}),
              DoubleEq(M_PI - 1.0));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, 1.1}),
              DoubleEq(M_PI - 1.2));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, M_PI}),
              DoubleEq(0.0));
}

TEST(DistanceToTargetOrientation, OrientationTarget) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionPoint({1.0, 2.0});
  goal_geometry.SetOrientationTarget({1.0, 0.0}, 0.1);

  // At the orientation target, the orientation is arbitrary.
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 0.0}, 0.0}),
              DoubleEq(0.0));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 0.0}, 1.0}),
              DoubleEq(0.0));

  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, -M_PI_2}),
              DoubleEq(0.0));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, 0.0}),
              DoubleEq(M_PI_2 - 0.1));

  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, 0.0}),
              DoubleEq(0.0));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, 1.0}),
              DoubleEq(0.9));
}

TEST(DistanceToTargetOrientation, ArbitraryOrientation) {
  GoalGeometry goal_geometry;
  goal_geometry.SetAttractionPoint({0.0, 0.0});
  goal_geometry.SetArbitraryOrientation();

  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, 0.0}),
              DoubleEq(0.0));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{1.0, 2.0}, 1.0}),
              DoubleEq(0.0));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, 2.0}),
              DoubleEq(0.0));
  EXPECT_THAT(goal_geometry.DistanceToTargetOrientation({{0.0, 0.0}, 3.0}),
              DoubleEq(0.0));
}

}  // namespace
}  // namespace mobility::collision

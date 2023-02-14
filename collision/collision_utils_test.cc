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

#include "collision/collision_utils.h"

#include <limits>
#include <vector>

#include "benchmark/benchmark.h"
#include "collision/convex_hull.h"
#include "collision/grid_test_utils.h"
#include "diff_drive/test_curves.h"
#include "diff_drive/test_trajectories.h"
#include "eigenmath/matchers.h"
#include "eigenmath/pose2.h"
#include "gtest/gtest.h"

namespace mobility::collision {

namespace {
constexpr double kGridResolution = 0.1;
constexpr double kEpsilon = 1.0e-6;

using diff_drive::testing::kTestCurve;
using diff_drive::testing::kTestTraj;
using testing::OccupancyGridPrintsTo;

TEST(CollisionUtils, GetWorstCaseReachableGridRangeTraj) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  Hull hull;
  hull.Add(points);
  const GridFrame grid_frame("dummy", eigenmath::Pose2d::Identity(),
                             kGridResolution);
  GridRange worst_case_range =
      GetWorstCaseReachableGridRange(grid_frame, kTestTraj, hull);
  EXPECT_EQ(worst_case_range.lower.x(), -22);
  EXPECT_EQ(worst_case_range.lower.y(), -25);
  EXPECT_EQ(worst_case_range.upper.x(), 29);
  EXPECT_EQ(worst_case_range.upper.y(), 26);
}

TEST(CollisionUtils, GetWorstCaseReachableGridRangeCurve) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  Hull hull;
  hull.Add(points);
  const GridFrame grid_frame("dummy", eigenmath::Pose2d::Identity(),
                             kGridResolution);
  GridRange worst_case_range =
      GetWorstCaseReachableGridRange(grid_frame, kTestCurve, hull);
  EXPECT_EQ(worst_case_range.lower.x(), -22);
  EXPECT_EQ(worst_case_range.lower.y(), -25);
  EXPECT_EQ(worst_case_range.upper.x(), 29);
  EXPECT_EQ(worst_case_range.upper.y(), 26);
}

TEST(CollisionUtils, ComputeReachableGridRangeTraj) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  Hull hull;
  hull.Add(points);
  const GridFrame grid_frame("dummy", eigenmath::Pose2d::Identity(),
                             kGridResolution);
  GridRange worst_case_range =
      ComputeReachableGridRange(grid_frame, kTestTraj, hull);
  EXPECT_EQ(worst_case_range.lower.x(), -10);
  EXPECT_EQ(worst_case_range.lower.y(), -10);
  EXPECT_EQ(worst_case_range.upper.x(), 16);
  EXPECT_EQ(worst_case_range.upper.y(), 24);
}

TEST(CollisionUtils, ComputeReachableGridRangeCurve) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  Hull hull;
  hull.Add(points);
  const GridFrame grid_frame("dummy", eigenmath::Pose2d::Identity(),
                             kGridResolution);
  GridRange worst_case_range =
      ComputeReachableGridRange(grid_frame, kTestCurve, hull);
  EXPECT_EQ(worst_case_range.lower.x(), -10);
  EXPECT_EQ(worst_case_range.lower.y(), -10);
  EXPECT_EQ(worst_case_range.upper.x(), 16);
  EXPECT_EQ(worst_case_range.upper.y(), 24);
}

TEST(CollisionUtils, CurveTraceContainsPoint) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  Hull hull;
  hull.Add(points);

  EXPECT_TRUE(CurveTraceContainsPoint(kTestCurve, hull, 0.5 * kGridResolution,
                                      eigenmath::Vector2d(0.1, 0.1)));
  EXPECT_FALSE(CurveTraceContainsPoint(kTestCurve, hull, 0.5 * kGridResolution,
                                       eigenmath::Vector2d(-0.1, 0.1)));
  EXPECT_FALSE(CurveTraceContainsPoint(kTestCurve, hull, 0.5 * kGridResolution,
                                       eigenmath::Vector2d(0.1, -0.1)));
  EXPECT_TRUE(CurveTraceContainsPoint(
      kTestCurve, hull, 0.5 * kGridResolution,
      eigenmath::Vector2d(1.5 / M_PI + 0.8, 0.95 + 0.5 / M_PI)));
  EXPECT_FALSE(CurveTraceContainsPoint(
      kTestCurve, hull, 0.5 * kGridResolution,
      eigenmath::Vector2d(1.5 / M_PI + 1.05, 1.0 + 0.5 / M_PI)));
  EXPECT_FALSE(CurveTraceContainsPoint(
      kTestCurve, hull, 0.5 * kGridResolution,
      eigenmath::Vector2d(1.5 / M_PI - 0.1, 1.0 + 0.5 / M_PI)));
}

TEST(CollisionUtils, DistanceIfPenetratingCurveTrace) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  Hull hull;
  hull.Add(points);

  EXPECT_NEAR(
      DistanceIfPenetratingCurveTrace(kTestCurve, hull, 0.5 * kGridResolution,
                                      eigenmath::Vector2d(0.1, 0.1)),
      -0.1, 0.05);
  EXPECT_NEAR(
      DistanceIfPenetratingCurveTrace(kTestCurve, hull, 0.5 * kGridResolution,
                                      eigenmath::Vector2d(-0.1, 0.1)),
      1.0, kEpsilon);
  EXPECT_NEAR(
      DistanceIfPenetratingCurveTrace(kTestCurve, hull, 0.5 * kGridResolution,
                                      eigenmath::Vector2d(0.1, -0.1)),
      1.0, kEpsilon);
  EXPECT_NEAR(DistanceIfPenetratingCurveTrace(
                  kTestCurve, hull, 0.5 * kGridResolution,
                  eigenmath::Vector2d(1.5 / M_PI + 0.8, 0.95 + 0.5 / M_PI)),
              -0.05, 0.02);
  EXPECT_NEAR(DistanceIfPenetratingCurveTrace(
                  kTestCurve, hull, 0.5 * kGridResolution,
                  eigenmath::Vector2d(1.5 / M_PI + 1.05, 1.0 + 0.5 / M_PI)),
              1.0, kEpsilon);
  EXPECT_NEAR(DistanceIfPenetratingCurveTrace(
                  kTestCurve, hull, 0.5 * kGridResolution,
                  eigenmath::Vector2d(1.5 / M_PI - 0.1, 1.0 + 0.5 / M_PI)),
              1.0, kEpsilon);
}

TEST(CollisionUtils, FillOccupancyGridWithCurveTrace) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  Hull hull;
  hull.Add(points);

  OccupancyGrid grid(
      GridFrame("dummy", eigenmath::Pose2d::Identity(), kGridResolution),
      GridRange(), OccupancyStatus::UNOCCUPIED);
  GridRange actual_range;
  FillOccupancyGridWithCurveTrace(kTestCurve, hull, &grid, &actual_range);
  grid.Reshape(actual_range);
  EXPECT_THAT(grid, OccupancyGridPrintsTo(R"""(
????????##?#?#?#?????????
??????###########????????
????#############?#??????
???#################?????
????################?????
??##################?#???
?######################??
?######################??
#######################??
?######################??
########################?
###########??###########?
############?###########?
########################?
########################?
########################?
########################?
########################?
########################?
########################?
########################?
########################?
????????################?
????????################?
?????????????###########?
?????????????###########?
?????????????###########?
?????????????########?##?
?????????????##########??
?????????????########?#??
?????????????##?##?##????
?????????????#####???????
??????????????#??????????)"""));

  // Also test IsPointInsideOccupiedRegion function.
  OccupancyGrid contour_grid(grid);
  GridRange::ShrinkBy(contour_grid.Range(), 1)
      .ForEachGridCoord([&contour_grid, &grid](GridIndex index) {
        if (IsPointInsideOccupiedRegion(grid, index)) {
          contour_grid.SetUnsafe(index, OccupancyStatus::UNOCCUPIED);
        }
      });
  EXPECT_THAT(contour_grid, OccupancyGridPrintsTo(R"""(
????????##?#?#?#?????????
??????##..#.#.#.#????????
????##..........#?#??????
???#.............#.#?????
????#..............#?????
??##...............#?#???
?#..................#.#??
?#....................#??
#.....................#??
?#....................#??
#..........##..........#?
#.........#??#.........#?
#..........#?#.........#?
#...........#..........#?
#......................#?
#......................#?
#......................#?
#......................#?
#......................#?
#......................#?
#......................#?
########...............#?
????????#..............#?
????????#####..........#?
?????????????#.........#?
?????????????#.........#?
?????????????#.......#.#?
?????????????#......#?##?
?????????????#.......##??
?????????????#.#..#.#?#??
?????????????##?##?##????
?????????????#.###???????
??????????????#??????????)"""));
}

TEST(CollisionUtils, FillCostGridWithCurveTrace) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  ConvexHull danger_hull(points);
  Hull hull;
  hull.Add(danger_hull.CreateBiggerHull(0.4));

  Grid<double> grid(
      GridFrame("dummy", eigenmath::Pose2d::Identity(), kGridResolution),
      GridRange(), 1.0);
  FillCostGridWithCurveTrace(
      kTestCurve, hull,
      [](double dist) -> double {
        if (dist > 0.0) {
          return 1.0;
        } else if (dist < -0.4) {
          return std::numeric_limits<double>::infinity();
        } else {
          return 0.4 / (0.4 + dist);
        }
      },
      &grid);
  EXPECT_EQ(DumpCostGrid(grid), R"""(
..................................
..............++++++++............
...........+++++++++++++..........
..........++++++++++++++++........
.......++++++++++oo++++++++.......
.......+++++++OOoo#ooo++++++......
......+++++oooO##O##o#o+o+++++....
.....++++++O###########o*+++++....
.....++++ooO#############o++++....
....++++o################oo++++...
....++++oo#################o++++..
...++++o###################+++++..
...++++O###################oo+++..
...++++o*###################o+++..
...+++o*#########oo#########+++++.
...+++o#########o++########oo++++.
...+++o#########o+o#########*++++.
..++++o#########oo##########*++++.
..++++o#####################*++++.
..++++o#####################*++++.
..++++o#####################*++++.
..++++o#####################*++++.
..++++o#####################*++++.
..++++o#####################*++++.
..++++o#####################*+++++
..++++o######################o++++
..+++++ooooooo################o+++
..++++++++++++o###############o+++
....++++++++++++++o##########o++++
.....+++++++++++++o##########o++++
............++++++o#########oo+++.
...............+++o#########o++++.
...............+++o#######oO+++++.
...............+++o#######++++++..
...............+++o##o#ooo+++++...
...............++++oo+o++++++++...
................+++++++++++++.....
................++++++++++++......
.................++++++++.+.......
......................+...........
..................................
..................................)""");

  // Also test IsPointInsideMaxCostRegion function.
  Grid<double> contour_grid(grid);
  GridRange::ShrinkBy(contour_grid.Range(), 1)
      .ForEachGridCoord([&contour_grid, &grid](GridIndex index) {
        if (IsPointInsideMaxCostRegion(grid, index, 10.0)) {
          contour_grid.SetUnsafe(index, 10.0);
        }
      });
  EXPECT_EQ(DumpCostGrid(contour_grid), R"""(
..................................
..............++++++++............
...........+++++++++++++..........
..........++++++++++++++++........
.......++++++++++oo++++++++.......
.......+++++++OOoo#ooo++++++......
......+++++oooOo#oo#ooo+o+++++....
.....++++++Oo#oooooooooo*+++++....
.....++++ooooooooooooooo#o++++....
....++++o#oooooooooooooo#oo++++...
....++++ooooooooooooooooo##o++++..
...++++o#ooooooooooooooooo#+++++..
...++++Oooooooooooooooooooooo+++..
...++++o*oooooooooooooooooooo+++..
...+++o*oooooooo#oooooooooo#+++++.
...+++ooooooooo#o++#ooooooooo++++.
...+++ooooooooo#o+o#oooooooo*++++.
..++++ooooooooo#oo#ooooooooo*++++.
..++++oooooooooo#ooooooooooo*++++.
..++++oooooooooooooooooooooo*++++.
..++++oooooooooooooooooooooo*++++.
..++++oooooooooooooooooooooo*++++.
..++++oooooooooooooooooooooo*++++.
..++++oooooooooooooooooooooo*++++.
..++++oooooooooooooooooooooo*+++++
..++++o#######oooooooooooooo#o++++
..+++++ooooooo#oooooooooooooo#o+++
..++++++++++++o####oooooooooo#o+++
....++++++++++++++o#oooooooo#o++++
.....+++++++++++++o#oooooooo#o++++
............++++++o#ooooooo#oo+++.
...............+++o#oooooo##o++++.
...............+++o#ooooo#oO+++++.
...............+++o#o#oo##++++++..
...............+++o##o#ooo+++++...
...............++++oo+o++++++++...
................+++++++++++++.....
................++++++++++++......
.................++++++++.+.......
......................+...........
..................................
..................................)""");
}

TEST(CollisionUtils, FillHullInOccupancyGrid) {
  OccupancyGrid grid(GridFrame("dummy", eigenmath::Pose2d::Identity(), 0.05),
                     GridRange::OriginTo(GridIndex(21, 21)),
                     OccupancyStatus::UNKNOWN);

  Hull hull;
  hull.Add(
      {{0.14, 0.14}, {0, 0.25}, {0.14, -0.14}, {-0.34, -0.14}, {-0.34, 0.14}});
  const eigenmath::Pose2d pose{grid.Frame().GridToFrame(10, 15), M_PI / 4};
  FillHullInOccupancyGrid(hull, pose, OccupancyStatus::UNOCCUPIED, &grid);
  EXPECT_THAT(grid, OccupancyGridPrintsTo(R"""(
?????????????????????
??????.....??????????
??????......?????????
?????........????????
?????.........???????
????...........??????
????..........???????
???..........????????
???.........?????????
????.......??????????
?????.....???????????
??????...????????????
???????.?????????????
?????????????????????
?????????????????????
?????????????????????
?????????????????????
?????????????????????
?????????????????????
?????????????????????
?????????????????????)"""));

  FillHullInOccupancyGrid(hull, eigenmath::Pose2d::Identity(),
                          OccupancyStatus::OCCUPIED, &grid);
  EXPECT_THAT(grid, OccupancyGridPrintsTo(R"""(
?????????????????????
??????.....??????????
??????......?????????
?????........????????
?????.........???????
????...........??????
????..........???????
???..........????????
???.........?????????
????.......??????????
?????.....???????????
??????...????????????
???????.?????????????
?????????????????????
?????????????????????
#????????????????????
###??????????????????
####?????????????????
####?????????????????
####?????????????????
####?????????????????)"""));
}

TEST(CollisionUtils, CollectOccupiedPointsOnHullContour) {
  const GridFrame grid_frame("dummy", eigenmath::Pose2d::Identity(), 0.05);
  const GridRange grid_range = GridRange::OriginTo(GridIndex(21, 21));

  Hull hull;
  hull.Add(
      {{0.14, 0.14}, {0, 0.25}, {0.14, -0.14}, {-0.34, -0.14}, {-0.34, 0.14}});
  const eigenmath::Pose2d pose{grid_frame.GridToFrame(10, 15), M_PI / 4};
  std::vector<eigenmath::Vector2d> grid_points;
  CollectOccupiedPointsOnHullContour(hull, pose, grid_frame, grid_range,
                                     &grid_points);

  OccupancyGrid grid(grid_frame, grid_range, OccupancyStatus::UNKNOWN);
  for (const eigenmath::Vector2d& pt : grid_points) {
    EXPECT_TRUE(
        grid.Set(grid_frame.FrameToGrid(pt), OccupancyStatus::OCCUPIED));
  }
  EXPECT_THAT(grid, OccupancyGridPrintsTo(R"""(
?????????????????????
??????#####??????????
??????#????#?????????
?????#??????#????????
?????#???????#???????
????#?????????#??????
????#????????#???????
???#????????#????????
???#???????#?????????
????#?????#??????????
?????#???#???????????
??????#?#????????????
???????#?????????????
?????????????????????
?????????????????????
?????????????????????
?????????????????????
?????????????????????
?????????????????????
?????????????????????
?????????????????????)"""));
}

void BM_FillOccupancyGridWithCurveTrace(benchmark::State& state) {
  Hull ego_danger_hull;
  ego_danger_hull.Add(
      {{0.14, 0.14}, {0.14, -0.14}, {-0.34, -0.14}, {-0.34, 0.14}});

  for (auto _ : state) {
    OccupancyGrid grid(
        GridFrame("dummy", eigenmath::Pose2d::Identity(), 0.1 / state.range(0)),
        GridRange(), OccupancyStatus::UNOCCUPIED);
    GridRange actual_range;
    FillOccupancyGridWithCurveTrace(kTestCurve, ego_danger_hull, &grid,
                                    &actual_range);
    grid.Reshape(actual_range);
    CHECK_GT(grid.Range().ComputeSize(), 8);
  }
}
BENCHMARK(BM_FillOccupancyGridWithCurveTrace)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

void BM_FillCostGridWithCurveTrace(benchmark::State& state) {
  Hull ego_danger_hull;
  ego_danger_hull.Add(
      {{0.14, 0.14}, {0.14, -0.14}, {-0.34, -0.14}, {-0.34, 0.14}});

  for (auto _ : state) {
    Grid<double> grid(
        GridFrame("dummy", eigenmath::Pose2d::Identity(), 0.1 / state.range(0)),
        GridRange(), 1.0);
    FillCostGridWithCurveTrace(
        kTestCurve, ego_danger_hull,
        [](double dist) -> double {
          if (dist > 0.0) {
            return 1.0;
          } else if (dist < -0.4) {
            return std::numeric_limits<double>::infinity();
          } else {
            return 0.16 / (0.16 - dist * dist);
          }
        },
        &grid, -0.4);
    CHECK_GT(grid.Range().ComputeSize(), 8);
  }
}
BENCHMARK(BM_FillCostGridWithCurveTrace)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

void BM_FillOccupancyGridWithHull(benchmark::State& state) {
  Hull ego_hull;
  ego_hull.Add({{0.14, 0.14}, {0.14, -0.14}, {-0.34, -0.14}, {-0.34, 0.14}});

  int kGridHalfLength = 10 * state.range(0);

  OccupancyGrid grid(
      GridFrame("dummy", eigenmath::Pose2d::Identity(), 0.1 / state.range(0)),
      GridRange(GridIndex(-kGridHalfLength, -kGridHalfLength),
                GridIndex(kGridHalfLength + 1, kGridHalfLength + 1)),
      OccupancyStatus::UNOCCUPIED);
  for (auto _ : state) {
    FillHullInOccupancyGrid(ego_hull, eigenmath::Pose2d::Identity(),
                            OccupancyStatus::OCCUPIED, &grid);
  }
}
BENCHMARK(BM_FillOccupancyGridWithHull)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

}  // namespace
}  // namespace mobility::collision

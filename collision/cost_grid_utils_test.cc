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

#include "collision/cost_grid_utils.h"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "collision/collision_utils.h"
#include "collision/grid_test_utils.h"
#include "collision/hull.h"
#include "diff_drive/test_curves.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

constexpr double kGridResolution = 0.1;
constexpr double kDangerMargin = 0.4;
using diff_drive::testing::kTestCurve;

template <int N>
char CostValueToCharImpl(double cost,
                         const std::pair<double, char> (&cost_table)[N]) {
  if (std::isfinite(cost)) {
    for (auto [threshold, result] : cost_table) {
      if (cost >= threshold) {
        return result;
      }
    }
    return '?';
  }
  if (std::isinf(cost)) {
    return '#';
  }
  return '!';
}

// Returns ascii art for cost values [1.0, inf].
char CostValueToChar(double cost) {
  static constexpr double kFudge = 1.0e-3;
  const std::pair<double, char> cost_table[] = {
      {2.0, '*'},          {1.5, 'O'},          {1.25, 'o'},
      {1.0 + kFudge, '+'}, {1.0 - kFudge, '.'},
  };
  return CostValueToCharImpl(cost, cost_table);
}

// Returns ascii art for "soft" cost values [1.0, inf].
char SoftCostToChar(double cost) {
  static constexpr double kFudge = 1.0e-3;
  const std::pair<double, char> cost_table[] = {
      {850.0, '9'},        {750.0, '8'},        {650.0, '7'}, {550.0, '6'},
      {450.0, '5'},        {350.0, '4'},        {250.0, '3'}, {150.0, '2'},
      {50.0, '1'},         {2.0, '*'},          {1.5, 'O'},   {1.25, 'o'},
      {1.0 + kFudge, '+'}, {1.0 - kFudge, '.'},
  };
  return CostValueToCharImpl(cost, cost_table);
}

TEST(CostGridUtils, CreateCostMaskAndConvolve) {
  const double danger_margin_sqr = kDangerMargin * kDangerMargin;
  Grid<double> mask =
      CreateCostMask(kGridResolution, kDangerMargin,
                     [danger_margin_sqr](double dist) -> double {
                       if (dist <= 0.0) {
                         return std::numeric_limits<double>::infinity();
                       } else if (dist >= kDangerMargin) {
                         return 1.0;
                       } else {
                         const double dist_left = dist - kDangerMargin;
                         return danger_margin_sqr /
                                (danger_margin_sqr - dist_left * dist_left);
                       }
                     });
  EXPECT_EQ(DumpGrid(mask, CostValueToChar), R"""(
.....
+++..
o+++.
*O++.
#*o+.)""");

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

  Grid<double> cost_trace(grid.Frame(), GridRange(), 1.0);
  ConvolveCostMaskOverOccupiedEdges(grid, mask, false, true, &cost_trace);
  EXPECT_EQ(DumpGrid(cost_trace, CostValueToChar), R"""(
...................................
...................................
...........++++++++++++............
.........++++oo+o+o+o+++...........
.......++++oO**O*O*O*O++++.........
......+++oO**##*#*#*#*Oo+++........
.....+++O**##**#*#*#*#**O+++.......
.....++O*##**OO*O*O**#*#*O+++......
....++o*#**Oo++o+o+oO*#*#*o+++.....
...+++O**#*o+++++++++O**#**O+++....
...++O*##*O++......+++o*#*#*O++....
..++o*#**O+++.......+++O*#*#*o+....
..++O*#*o+++..++++++.+++O**#*o+....
..+o*#*O+++..+++oo+++.+++o*#*o++...
..++O*#*o+..+++O**O+++.++o*#*O++...
..+o*#*O++..++O*##*O++..++O*#*o+...
..+o*#*o++..+o*#**#*o+..++o*#*o+...
..+o*#*o+...++O*#*#*o+...+o*#*o+...
..+o*#*o+...+++O*#*O++...+o*#*o+...
..+o*#*o+....+++O*O+++...+o*#*o+...
..+o*#*o+.....+++o+++....+o*#*o+...
..+o*#*o+......+++++.....+o*#*o+...
..+o*#*o+................+o*#*o+...
..+o*#*o+++++++..........+o*#*o+...
..+o*#*oooooo+++.........+o*#*o+...
..+o*#*******O++++++.....+o*#*o+...
..+o*########*Oooo+++....+o*#*o+...
..++O********#****O+++...+o*#*o+...
..+++ooooooo*#####*O++..++o*#*o+...
...+++++++++O*****#*o+.+++o*#*o+...
..........+++oooo*#*o++++O**#*o+...
...........+++++o*#*o+++O*#*#*o+...
...............+o*#*o++o*#*##*o+...
...............+o*#**OO*O*##*O++...
...............+o*#*#**#*#*#*o++...
...............+o*##*##*##**O++....
...............+o*#*###***Oo+++....
...............++O*#***Ooo++++.....
...............+++O*Ooo+++++.......
................+++o+++++..........
.................+++++.............
...................................
...................................)""");

  Grid<double> cost_full_trace(grid.Frame(), GridRange(), 1.0);
  ConvolveCostMaskOverOccupiedEdges(grid, mask, false, false, &cost_full_trace);
  EXPECT_EQ(DumpGrid(cost_full_trace, CostValueToChar), R"""(
...................................
...................................
...........++++++++++++............
.........++++oo+o+o+o+++...........
.......++++oO**O*O*O*O++++.........
......+++oO**##*#*#*#*Oo+++........
.....+++O**###########**O+++.......
.....++O*#############*#*O+++......
....++o*#################*o+++.....
...+++O**################**O+++....
...++O*##################*#*O++....
..++o*######################*o+....
..++O*######################*o+....
..+o*#######################*o++...
..++O*######################*O++...
..+o*########################*o+...
..+o*###########**###########*o+...
..+o*############*###########*o+...
..+o*########################*o+...
..+o*########################*o+...
..+o*########################*o+...
..+o*########################*o+...
..+o*########################*o+...
..+o*########################*o+...
..+o*########################*o+...
..+o*########################*o+...
..+o*########################*o+...
..++O********################*o+...
..+++ooooooo*################*o+...
...+++++++++O*****###########*o+...
..........+++oooo*###########*o+...
...........+++++o*###########*o+...
...............+o*########*##*o+...
...............+o*##########*O++...
...............+o*########*#*o++...
...............+o*##*##*##**O++....
...............+o*#####***Oo+++....
...............++O*#***Ooo++++.....
...............+++O*Ooo+++++.......
................+++o+++++..........
.................+++++.............
...................................
...................................)""");
}

TEST(CostGridUtils, CreateSoftCostTrace) {
  const double danger_margin_sqr = kDangerMargin * kDangerMargin;
  Grid<double> mask = CreateCostMask(
      kGridResolution, kDangerMargin,
      [danger_margin_sqr](double dist) -> double {
        if (dist >= kDangerMargin) {
          return 1.0;
        }
        const double dist_left = dist - kDangerMargin;
        const double denom = danger_margin_sqr - dist_left * dist_left;
        if (danger_margin_sqr > 100.0 * denom) {
          return 100.0;
        }
        return danger_margin_sqr / denom;
      });
  EXPECT_EQ(DumpGrid(mask, SoftCostToChar), R"""(
.....
+++..
o+++.
*O++.
1*o+.)""");

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

  Grid<double> cost_full_trace(grid.Frame(), GridRange(), 1.0);
  ConvolveCostMaskOverOccupiedEdges(grid, mask, false, false, &cost_full_trace);
  EXPECT_EQ(DumpGrid(cost_full_trace, SoftCostToChar), R"""(
...................................
...................................
...........++++++++++++............
.........++++oo+o+o+o+++...........
.......++++oO**O*O*O*O++++.........
......+++oO**11*1*1*1*Oo+++........
.....+++O**11111111111**O+++.......
.....++O*1111111111111*1*O+++......
....++o*11111111111111111*o+++.....
...+++O**1111111111111111**O+++....
...++O*111111111111111111*1*O++....
..++o*1111111111111111111111*o+....
..++O*1111111111111111111111*o+....
..+o*11111111111111111111111*o++...
..++O*1111111111111111111111*O++...
..+o*111111111111111111111111*o+...
..+o*11111111111**11111111111*o+...
..+o*111111111111*11111111111*o+...
..+o*111111111111111111111111*o+...
..+o*111111111111111111111111*o+...
..+o*111111111111111111111111*o+...
..+o*111111111111111111111111*o+...
..+o*111111111111111111111111*o+...
..+o*111111111111111111111111*o+...
..+o*111111111111111111111111*o+...
..+o*111111111111111111111111*o+...
..+o*111111111111111111111111*o+...
..++O********1111111111111111*o+...
..+++ooooooo*1111111111111111*o+...
...+++++++++O*****11111111111*o+...
..........+++oooo*11111111111*o+...
...........+++++o*11111111111*o+...
...............+o*11111111*11*o+...
...............+o*1111111111*O++...
...............+o*11111111*1*o++...
...............+o*11*11*11**O++....
...............+o*11111***Oo+++....
...............++O*1***Ooo++++.....
...............+++O*Ooo+++++.......
................+++o+++++..........
.................+++++.............
...................................
...................................)""");

  GraduateInteriorCosts(100.0, &cost_full_trace);
  EXPECT_EQ(DumpGrid(cost_full_trace, SoftCostToChar), R"""(
...................................
...................................
...........++++++++++++............
.........++++oo+o+o+o+++...........
.......++++oO**O*O*O*O++++.........
......+++oO**11*1*1*1*Oo+++........
.....+++O**11221212121**O+++.......
.....++O*1122332323221*1*O+++......
....++o*12233443434332121*o+++.....
...+++O**1234554545443221**O+++....
...++O*112345665656554321*1*O++....
..++o*1223456776556665432121*o+....
..++O*1234567765445676543221*o+....
..+o*12345677654334567654321*o++...
..++O*1234566543223456654321*O++...
..+o*123456654321123456654321*o+...
..+o*12345654321**12345654321*o+...
..+o*123456654321*12345654321*o+...
..+o*123456765432123456654321*o+...
..+o*123456776543234567654321*o+...
..+o*123456777654345677654321*o+...
..+o*123456667765456787654321*o+...
..+o*123455556776567887654321*o+...
..+o*123444445666678987654321*o+...
..+o*123333334555567887654321*o+...
..+o*122222223444456787654321*o+...
..+o*111111112333345677654321*o+...
..++O********1222234567654321*o+...
..+++ooooooo*1111123456654321*o+...
...+++++++++O*****12345654321*o+...
..........+++oooo*12345543221*o+...
...........+++++o*12345432121*o+...
...............+o*12344321*11*o+...
...............+o*1223323211*O++...
...............+o*12122121*1*o++...
...............+o*11*11*11**O++....
...............+o*12111***Oo+++....
...............++O*1***Ooo++++.....
...............+++O*Ooo+++++.......
................+++o+++++..........
.................+++++.............
...................................
...................................)""");
}

}  // namespace
}  // namespace mobility::collision

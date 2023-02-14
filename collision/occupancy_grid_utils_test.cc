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

#include "collision/occupancy_grid_utils.h"

#include <algorithm>
#include <tuple>
#include <vector>

#include "collision/grid_common.h"
#include "collision/grid_test_utils.h"
#include "collision/occupancy_grid.h"
#include "eigenmath/line_utils.h"
#include "eigenmath/matchers.h"
#include "genit/adjacent_circular_iterator.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

using eigenmath::testing::IsApproxUndirected;
using testing::OccupancyGridPrintsTo;
using ::testing::UnorderedPointwise;

using Boundary = std::vector<GridSegment>;
using BoundaryPoints = std::vector<GridIndex>;

constexpr double kResolution = 0.7;
constexpr double kTolerance = 1e-9;
constexpr bool kTreatUnknownAsObstacle = false;

// Compares two sets of undirected line segments.
template <typename T>
auto EqualsBoundarySet(T&& expected)
    -> decltype(UnorderedPointwise(IsApproxUndirected(kTolerance), expected)) {
  return UnorderedPointwise(IsApproxUndirected(kTolerance), expected);
}

TEST(MissingBoundarySegments, SmallestGrid) {
  const GridRange range = GridRange::OriginTo({2, 2});
  const auto missing = MissingBoundarySegments(range, {});
  const Boundary expected = {
      {{0, 0}, {1, 0}}, {{1, 0}, {1, 1}}, {{1, 1}, {0, 1}}, {{0, 0}, {0, 1}}};
  EXPECT_THAT(missing, EqualsBoundarySet(expected));
}

TEST(MissingBoundarySegments, VerticalLineInCenter) {
  const GridRange range = GridRange::OriginTo({3, 3});
  const auto missing = MissingBoundarySegments(range, {{{1, 0}, {1, 2}}});
  const Boundary expected = {{{0, 0}, {1, 0}}, {{1, 0}, {2, 0}},
                             {{2, 0}, {2, 2}}, {{2, 2}, {1, 2}},
                             {{1, 2}, {0, 2}}, {{0, 2}, {0, 0}}};
  EXPECT_THAT(missing, EqualsBoundarySet(expected));
}

TEST(MissingBoundarySegments, DiagonalLine) {
  const GridRange range = GridRange::OriginTo({3, 3});
  const auto missing = MissingBoundarySegments(range, {{{0, 0}, {2, 2}}});
  const Boundary expected = {
      {{0, 0}, {2, 0}}, {{2, 0}, {2, 2}}, {{2, 2}, {0, 2}}, {{0, 0}, {0, 2}}};
  EXPECT_THAT(missing, EqualsBoundarySet(expected));
}

TEST(MissingBoundarySegments, LineOnBoundary) {
  const GridRange range = GridRange::OriginTo({4, 3});
  const auto missing = MissingBoundarySegments(range, {{{1, 0}, {2, 0}}});
  const Boundary expected = {{{0, 0}, {1, 0}},
                             {{2, 0}, {3, 0}},
                             {{3, 0}, {3, 2}},
                             {{3, 2}, {0, 2}},
                             {{0, 0}, {0, 2}}};
  EXPECT_THAT(missing, EqualsBoundarySet(expected));
}

TEST(MissingBoundarySegments, LinesOnBoundary) {
  const GridRange range = GridRange::OriginTo({10, 3});
  const auto missing =
      MissingBoundarySegments(range, {{{1, 0}, {2, 0}}, {{5, 0}, {6, 0}}});
  const Boundary expected = {{{0, 0}, {1, 0}}, {{2, 0}, {5, 0}},
                             {{6, 0}, {9, 0}}, {{9, 0}, {9, 2}},
                             {{9, 2}, {0, 2}}, {{0, 0}, {0, 2}}};
  EXPECT_THAT(missing, EqualsBoundarySet(expected));
}

TEST(MissingBoundarySegments, MultipleSegments) {
  const GridRange range = GridRange::OriginTo({11, 11});
  const Boundary shape = {
      // Triangle with split line on boundary.
      {{2, 0}, {4, 0}},
      {{4, 0}, {6, 0}},
      {{6, 0}, {6, 2}},
      {{6, 2}, {2, 0}},

      // Single line.
      {{8, 0}, {10, 0}},

      // Triangle at corner.
      {{10, 5}, {10, 10}},
      {{10, 10}, {8, 10}},
      {{8, 10}, {10, 5}},

      // Trapezoid around corner.
      {{5, 10}, {2, 10}},
      {{2, 10}, {0, 8}},
      {{0, 8}, {0, 6}},
      {{0, 6}, {5, 10}},
  };
  const auto missing = MissingBoundarySegments(range, shape);
  const Boundary expected = {
      {{0, 0}, {2, 0}},   {{6, 0}, {8, 0}},   {{10, 0}, {10, 5}},
      {{8, 10}, {5, 10}}, {{2, 10}, {0, 10}}, {{0, 10}, {0, 8}},
      {{0, 6}, {0, 0}},
  };
  EXPECT_THAT(missing, EqualsBoundarySet(expected));
}

OccupancyGrid GridFromString(std::string_view drawn_grid) {
  OccupancyGrid grid;
  grid.Resolution() = kResolution;
  CreateOccupancyGrid(drawn_grid, &grid);
  return grid;
}

Boundary BoundaryFromPoints(const BoundaryPoints& points) {
  Boundary segments;
  for (const auto tuple : genit::AdjacentElementsCircularRange<2>(points)) {
    segments.emplace_back(tuple[0], tuple[1]);
  }
  return segments;
}

TEST(TestBoundary, HorizontalConnection) {
  OccupancyGrid grid = GridFromString(R"""(
.##...#..
.##..###.
.##...#..)""");
  EXPECT_TRUE(IsBoundary(grid, {1, 0}, {2, 0}, true));
  EXPECT_TRUE(IsBoundary(grid, {2, 0}, {1, 0}, true));
  EXPECT_TRUE(IsBoundary(grid, {1, 2}, {2, 2}, true));
  EXPECT_TRUE(IsBoundary(grid, {2, 2}, {1, 2}, true));

  EXPECT_FALSE(IsBoundary(grid, {1, 1}, {2, 1}, true));
  EXPECT_FALSE(IsBoundary(grid, {2, 1}, {1, 1}, true));
  EXPECT_FALSE(IsBoundary(grid, {5, 1}, {6, 1}, true));
  EXPECT_FALSE(IsBoundary(grid, {6, 1}, {5, 1}, true));
  EXPECT_FALSE(IsBoundary(grid, {6, 1}, {7, 1}, true));
  EXPECT_FALSE(IsBoundary(grid, {7, 1}, {6, 1}, true));
}

TEST(TestBoundary, VerticalConnection) {
  OccupancyGrid grid = GridFromString(R"""(
......#.
.###.###.
.###..#..)""");
  EXPECT_TRUE(IsBoundary(grid, {1, 0}, {1, 1}, true));
  EXPECT_TRUE(IsBoundary(grid, {1, 1}, {1, 0}, true));
  EXPECT_TRUE(IsBoundary(grid, {3, 0}, {3, 1}, true));
  EXPECT_TRUE(IsBoundary(grid, {3, 1}, {3, 0}, true));

  EXPECT_FALSE(IsBoundary(grid, {2, 0}, {2, 1}, true));
  EXPECT_FALSE(IsBoundary(grid, {2, 1}, {2, 0}, true));
  EXPECT_FALSE(IsBoundary(grid, {6, 0}, {6, 1}, true));
  EXPECT_FALSE(IsBoundary(grid, {6, 1}, {6, 0}, true));
  EXPECT_FALSE(IsBoundary(grid, {6, 1}, {6, 2}, true));
  EXPECT_FALSE(IsBoundary(grid, {6, 2}, {6, 1}, true));
}

TEST(TestBoundary, DiagonalConnection) {
  OccupancyGrid grid = GridFromString(R"""(
......#.
.###.###.
.###..#..)""");
  EXPECT_TRUE(IsBoundary(grid, {5, 1}, {6, 0}, true));
  EXPECT_TRUE(IsBoundary(grid, {6, 0}, {5, 1}, true));
  EXPECT_TRUE(IsBoundary(grid, {7, 1}, {6, 2}, true));
  EXPECT_TRUE(IsBoundary(grid, {6, 2}, {7, 1}, true));
  EXPECT_TRUE(IsBoundary(grid, {7, 1}, {6, 0}, true));
  EXPECT_TRUE(IsBoundary(grid, {6, 0}, {7, 1}, true));
  EXPECT_TRUE(IsBoundary(grid, {5, 1}, {6, 2}, true));
  EXPECT_TRUE(IsBoundary(grid, {6, 2}, {5, 1}, true));

  EXPECT_FALSE(IsBoundary(grid, {1, 1}, {2, 0}, true));
  EXPECT_FALSE(IsBoundary(grid, {2, 0}, {1, 1}, true));
  EXPECT_FALSE(IsBoundary(grid, {1, 0}, {2, 1}, true));
  EXPECT_FALSE(IsBoundary(grid, {2, 1}, {1, 0}, true));
}

class NextPointOnBoundaryTest : public ::testing::TestWithParam<
                                    std::tuple<OccupancyGrid, BoundaryPoints>> {
};

TEST_P(NextPointOnBoundaryTest, WalkAlongBoundary) {
  const auto& grid = std::get<0>(GetParam());
  const auto& boundary = std::get<1>(GetParam());
  for (const auto point_triple :
       genit::AdjacentElementsCircularRange<3>(boundary)) {
    EXPECT_EQ(NextPointOnBoundary(grid, point_triple[0], point_triple[1],
                                  kTreatUnknownAsObstacle),
              point_triple[2]);
  }
}

std::vector<std::tuple<OccupancyGrid, BoundaryPoints>>
ObstaclesAndStepwiseBoundaries() {
  return {
      // Square, outer boundary.
      std::make_tuple<OccupancyGrid, BoundaryPoints>(GridFromString(R"""(
.....
.###.
.#.#.
.###.
.....)"""),
                                                     {
                                                         {1, 1},
                                                         {2, 1},
                                                         {3, 1},
                                                         {3, 2},
                                                         {3, 3},
                                                         {2, 3},
                                                         {1, 3},
                                                         {1, 2},
                                                     }),

      // Square, inner boundary.
      std::make_tuple<OccupancyGrid, BoundaryPoints>(GridFromString(R"""(
.....
.###.
.#.#.
.###.
.....)"""),
                                                     {
                                                         {2, 1},
                                                         {1, 2},
                                                         {2, 3},
                                                         {3, 2},
                                                     }),

      // Regular Octagon.
      std::make_tuple<OccupancyGrid, BoundaryPoints>(GridFromString(R"""(
......
..##..
.####.
.####.
..##..
......)"""),
                                                     {
                                                         {2, 1},
                                                         {3, 1},
                                                         {4, 2},
                                                         {4, 3},
                                                         {3, 4},
                                                         {2, 4},
                                                         {1, 3},
                                                         {1, 2},
                                                     }),

      // Interior of larger square.
      std::make_tuple<OccupancyGrid, BoundaryPoints>(GridFromString(R"""(
......
.####.
.#..#.
.#..#.
.####.
......)"""),
                                                     {
                                                         {2, 1},
                                                         {1, 2},
                                                         {1, 3},
                                                         {2, 4},
                                                         {3, 4},
                                                         {4, 3},
                                                         {4, 2},
                                                         {3, 1},
                                                     }),

      // Crosses.
      std::make_tuple<OccupancyGrid, BoundaryPoints>(GridFromString(R"""(
.......
.#.#.#.
..#.#..
.#.#.#.
.......)"""),
                                                     {
                                                         {1, 1},
                                                         {2, 2},
                                                         {3, 1},
                                                         {4, 2},
                                                         {5, 1},
                                                         {4, 2},
                                                         {5, 3},
                                                         {4, 2},
                                                         {3, 3},
                                                         {2, 2},
                                                         {1, 3},
                                                         {2, 2},
                                                     }),
  };
}

INSTANTIATE_TEST_SUITE_P(TestNextPointOnBoundary2, NextPointOnBoundaryTest,
                         ::testing::ValuesIn(ObstaclesAndStepwiseBoundaries()));

TEST(ContourExtraction, Square) {
  OccupancyGrid grid = GridFromString(R"""(
.....
.###.
.###.
.###.
.....)""");
  const auto boundary = BoundaryFromPoints({
      {1, 1},
      {3, 1},
      {3, 3},
      {1, 3},
  });

  EXPECT_THAT(ExtractBoundaryLineSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(ContourExtraction, RegularOctagon) {
  OccupancyGrid grid = GridFromString(R"""(
......
..##..
.####.
.####.
..##..
......)""");
  const auto boundary = BoundaryFromPoints({
      {2, 1},
      {3, 1},
      {4, 2},
      {4, 3},
      {3, 4},
      {2, 4},
      {1, 3},
      {1, 2},
  });

  EXPECT_THAT(ExtractBoundaryLineSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(ContourExtraction, SingleCellObstacles) {
  OccupancyGrid grid = GridFromString(R"""(
..#...
#...#.
......
.....#)""");
  const auto boundary = BoundaryFromPoints({});

  EXPECT_THAT(ExtractBoundaryLineSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(ContourExtraction, ObstacleWithHole) {
  OccupancyGrid grid = GridFromString(R"""(
.###.
.#.#.
.###.
.###.)""");
  // Note that there are four vertical lines.  If there are two vertical lines,
  // the middle horizontal line would intersect with the interior of the left
  // and right line.
  const Boundary boundary = {
      {{1, 0}, {3, 0}}, {{1, 0}, {1, 2}}, {{3, 0}, {3, 2}}, {{1, 2}, {2, 1}},
      {{2, 1}, {3, 2}}, {{1, 2}, {2, 3}}, {{2, 3}, {3, 2}}, {{1, 3}, {1, 2}},
      {{1, 3}, {2, 3}}, {{2, 3}, {3, 3}}, {{3, 3}, {3, 2}},
  };

  EXPECT_THAT(ExtractBoundaryLineSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(ContourExtraction, ObstacleWithTwoHoles) {
  OccupancyGrid grid = GridFromString(R"""(
.#.##...
#.#..#..
#..##...
.##.....)""");
  const Boundary boundary = {
      {{1, 0}, {0, 1}}, {{1, 0}, {2, 0}}, {{2, 0}, {3, 1}}, {{0, 1}, {0, 2}},
      {{0, 2}, {1, 3}}, {{1, 3}, {2, 2}}, {{2, 2}, {3, 1}}, {{3, 1}, {4, 1}},
      {{4, 1}, {5, 2}}, {{5, 2}, {4, 3}}, {{4, 3}, {3, 3}}, {{3, 3}, {2, 2}},
  };
  EXPECT_THAT(ExtractBoundaryLineSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(ContourExtraction, DiagonalStructure) {
  OccupancyGrid grid = GridFromString(R"""(
.......
....#..
.....#.
....#..
..##...
...##..)""");
  const Boundary boundary = {
      {{4, 4}, {5, 3}}, {{5, 3}, {3, 1}}, {{3, 1}, {4, 0}},
      {{3, 0}, {4, 0}}, {{2, 1}, {3, 0}}, {{2, 1}, {3, 1}},
  };

  EXPECT_THAT(ExtractBoundaryLineSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(PolygonSegments, Square) {
  OccupancyGrid grid = GridFromString(R"""(
.....
.###.
.###.
.###.
.....)""");
  const auto boundary = BoundaryFromPoints({
      {1, 1},
      {3, 1},
      {3, 3},
      {1, 3},
  });

  EXPECT_THAT(ExtractBoundaryPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(PolygonSegments, RegularOctagon) {
  OccupancyGrid grid = GridFromString(R"""(
......
..##..
.####.
.####.
..##..
......)""");
  const auto boundary = BoundaryFromPoints({
      {2, 1},
      {3, 1},
      {4, 2},
      {4, 3},
      {3, 4},
      {2, 4},
      {1, 3},
      {1, 2},
  });

  EXPECT_THAT(ExtractBoundaryPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(PolygonSegments, SingleCellObstacles) {
  OccupancyGrid grid = GridFromString(R"""(
..#...
#...#.
......
.....#)""");
  const auto boundary = BoundaryFromPoints({});

  EXPECT_THAT(ExtractBoundaryPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(PolygonSegments, ObstacleWithLargeHole) {
  OccupancyGrid grid = GridFromString(R"""(
......
.####.
.#..#.
.#..#.
.####.
......)""");
  const auto outer_boundary = BoundaryFromPoints({
      {1, 1},
      {4, 1},
      {4, 4},
      {1, 4},
  });
  const auto inner_boundary = BoundaryFromPoints({
      {2, 1},
      {1, 2},
      {1, 3},
      {2, 4},
      {3, 4},
      {4, 3},
      {4, 2},
      {3, 1},
  });

  Boundary boundary{outer_boundary.cbegin(), outer_boundary.cend()};
  boundary.insert(boundary.end(), inner_boundary.cbegin(),
                  inner_boundary.cend());
  EXPECT_THAT(ExtractBoundaryPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(PolygonSegments, ObstacleWithHole) {
  OccupancyGrid grid = GridFromString(R"""(
.###.
.#.#.
.###.
.###.)""");
  const auto outer_boundary = BoundaryFromPoints({
      {1, 0},
      {3, 0},
      {3, 3},
      {1, 3},
  });
  const auto inner_boundary = BoundaryFromPoints({
      {2, 1},
      {1, 2},
      {2, 3},
      {3, 2},
  });
  Boundary boundary{outer_boundary.cbegin(), outer_boundary.cend()};
  boundary.insert(boundary.end(), inner_boundary.cbegin(),
                  inner_boundary.cend());
  EXPECT_THAT(ExtractBoundaryPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(PolygonSegments, ObstacleWithTwoHoles) {
  OccupancyGrid grid = GridFromString(R"""(
.#.##...
#.#..#..
#..##...
.##.....)""");
  const auto outer_boundary = BoundaryFromPoints({
      {1, 0},
      {2, 0},
      {3, 1},
      {4, 1},
      {5, 2},
      {4, 3},
      {3, 3},
      {2, 2},
      {1, 3},
      {0, 2},
      {0, 1},
  });
  const auto left_inner = BoundaryFromPoints({
      {0, 1},
      {0, 2},
      {1, 3},
      {3, 1},
      {2, 0},
      {1, 0},
  });
  const auto right_inner = BoundaryFromPoints({
      {2, 2},
      {3, 3},
      {4, 3},
      {5, 2},
      {4, 1},
      {3, 1},
  });

  Boundary boundary;
  boundary.insert(boundary.end(), outer_boundary.begin(), outer_boundary.end());
  boundary.insert(boundary.end(), left_inner.begin(), left_inner.end());
  boundary.insert(boundary.end(), right_inner.begin(), right_inner.end());
  EXPECT_THAT(ExtractBoundaryPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(PolygonSegments, DiagonalStructure) {
  OccupancyGrid grid = GridFromString(R"""(
.......
....#..
.....#.
....#..
..##...
...##..)""");
  const auto boundary = BoundaryFromPoints({
      {4, 4},
      {5, 3},
      {3, 1},
      {2, 1},
      {3, 0},
      {4, 0},
      {3, 1},
      {5, 3},
  });

  EXPECT_THAT(ExtractBoundaryPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(SimplifiedPolygonSegments, Square) {
  OccupancyGrid grid = GridFromString(R"""(
.....
.###.
.###.
.###.
.....)""");
  const auto boundary = BoundaryFromPoints({
      {1, 1},
      {3, 1},
      {3, 3},
      {1, 3},
  });

  EXPECT_THAT(ExtractSimplifiedPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(SimplifiedPolygonSegments, RegularOctagon) {
  OccupancyGrid grid = GridFromString(R"""(
......
..##..
.####.
.####.
..##..
......)""");
  const auto boundary = BoundaryFromPoints({
      {2, 1},
      {3, 1},
      {4, 2},
      {4, 3},
      {3, 4},
      {2, 4},
      {1, 3},
      {1, 2},
  });

  EXPECT_THAT(ExtractBoundaryPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(SimplifiedPolygonSegments, SingleCellObstacles) {
  OccupancyGrid grid = GridFromString(R"""(
..#...
#...#.
......
.....#)""");
  const auto boundary = BoundaryFromPoints({});

  EXPECT_THAT(ExtractSimplifiedPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(SimplifiedPolygonSegments, ObstacleWithTwoHoles) {
  OccupancyGrid grid = GridFromString(R"""(
.#.##...
#.#..#..
#..##...
.##.....)""");
  const auto outer_boundary = BoundaryFromPoints({
      {1, 0},
      {5, 2},
      {3, 3},
      {2, 2},
      {1, 3},
      {0, 1},
  });
  const auto left_inner = BoundaryFromPoints({
      {1, 0},
      {0, 2},
      {1, 3},
      {3, 1},
  });
  const auto right_inner = BoundaryFromPoints({
      {2, 2},
      {4, 3},
      {5, 2},
      {3, 1},
  });

  Boundary boundary;
  boundary.insert(boundary.end(), outer_boundary.begin(), outer_boundary.end());
  boundary.insert(boundary.end(), left_inner.begin(), left_inner.end());
  boundary.insert(boundary.end(), right_inner.begin(), right_inner.end());
  EXPECT_THAT(ExtractSimplifiedPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(SimplifiedPolygonSegments, DiagonalStructure) {
  OccupancyGrid grid = GridFromString(R"""(
.......
....#..
.....#.
....#..
..##...
...##..)""");
  const auto boundary = BoundaryFromPoints({
      {4, 4},
      {5, 3},
      {3, 1},
      {2, 1},
      {3, 0},
      {4, 0},
      {3, 1},
      {5, 3},
  });

  EXPECT_THAT(ExtractSimplifiedPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(SimplifiedPolygonSegments, TiltedLine) {
  OccupancyGrid grid = GridFromString(R"""(
......##.....#.
....####....#..
.#######...#...
.####.....#....
.........#.....)""");
  const auto left_boundary =
      BoundaryFromPoints({{1, 1}, {7, 2}, {7, 4}, {2, 2}, {1, 2}});
  const auto right_boundary = BoundaryFromPoints({{9, 0}, {13, 4}});

  Boundary boundary;
  boundary.insert(boundary.end(), left_boundary.begin(), left_boundary.end());
  boundary.insert(boundary.end(), right_boundary.begin(), right_boundary.end());
  EXPECT_THAT(ExtractSimplifiedPolygonSegments(grid, kTreatUnknownAsObstacle),
              EqualsBoundarySet(boundary))
      << DumpOccupancyGrid(grid);
}

TEST(LineOnBoundary, StraightLine) {
  // Following scenario.
  // .....
  // .###.
  // .###.
  // .###.
  // .....
  EXPECT_TRUE(ExtensionIsOnBoundary({{1, 1}, {2, 1}}, {3, 1}));

  // Walk around the corner.
  EXPECT_FALSE(ExtensionIsOnBoundary({{1, 1}, {3, 1}}, {3, 2}));
  EXPECT_FALSE(ExtensionIsOnBoundary({{1, 2}, {1, 1}}, {2, 1}));
}

TEST(LineOnBoundary, TiltedLine) {
  // Bottom row.  Cannot be simplified with single look-ahead.
  EXPECT_FALSE(ExtensionIsOnBoundary({{1, 1}, {4, 1}}, {5, 2}));
  // But we could do this step.
  EXPECT_TRUE(ExtensionIsOnBoundary({{1, 1}, {2, 2}}, {3, 2}));
  EXPECT_FALSE(ExtensionIsOnBoundary({{1, 1}, {7, 2}}, {7, 3}));

  // Top row.  Can be simplified somewhat.
  EXPECT_TRUE(ExtensionIsOnBoundary({{7, 4}, {6, 4}}, {5, 3}));
  EXPECT_TRUE(ExtensionIsOnBoundary({{7, 4}, {5, 3}}, {4, 3}));
  // Cannot go further with single step lookahead.
  EXPECT_FALSE(ExtensionIsOnBoundary({{7, 4}, {4, 3}}, {3, 2}));

  // Diagonal to the right.
  EXPECT_TRUE(ExtensionIsOnBoundary({{9, 0}, {9, 0}}, {10, 1}));
  EXPECT_TRUE(ExtensionIsOnBoundary({{9, 0}, {10, 1}}, {11, 2}));
  EXPECT_TRUE(ExtensionIsOnBoundary({{9, 0}, {11, 2}}, {12, 3}));
  EXPECT_TRUE(ExtensionIsOnBoundary({{9, 0}, {12, 3}}, {13, 4}));
}

TEST(GetAllOccupiedInFirstButNotInOther, BasicTest) {
  OccupancyGrid first_grid = GridFromString(R"""(
###
???
...)""");
  OccupancyGrid other_grid = GridFromString(R"""(
#?.
#?.
#?.)""");

  OccupancyGrid result_grid;
  GetAllOccupiedInFirstButNotInOther(
      first_grid, /*treat_unknown_in_first_as_occupied=*/false, other_grid,
      /*treat_unknown_in_other_as_occupied=*/false, &result_grid);
  EXPECT_THAT(result_grid, OccupancyGridPrintsTo(R"""(
.##
...
...)"""));

  GetAllOccupiedInFirstButNotInOther(
      first_grid, /*treat_unknown_in_first_as_occupied=*/true, other_grid,
      /*treat_unknown_in_other_as_occupied=*/false, &result_grid);
  EXPECT_THAT(result_grid, OccupancyGridPrintsTo(R"""(
.##
.##
...)"""));

  GetAllOccupiedInFirstButNotInOther(
      first_grid, /*treat_unknown_in_first_as_occupied=*/false, other_grid,
      /*treat_unknown_in_other_as_occupied=*/true, &result_grid);
  EXPECT_THAT(result_grid, OccupancyGridPrintsTo(R"""(
..#
...
...)"""));

  GetAllOccupiedInFirstButNotInOther(
      first_grid, /*treat_unknown_in_first_as_occupied=*/true, other_grid,
      /*treat_unknown_in_other_as_occupied=*/true, &result_grid);
  EXPECT_THAT(result_grid, OccupancyGridPrintsTo(R"""(
..#
..#
...)"""));
}

}  // namespace
}  // namespace mobility::collision

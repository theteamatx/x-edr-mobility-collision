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

#ifndef MOBILITY_COLLISION_COLLISION_GRID_TEST_UTILS_H_
#define MOBILITY_COLLISION_COLLISION_GRID_TEST_UTILS_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "collision/cost_grid_utils.h"
#include "collision/grid_common.h"
#include "collision/occupancy_grid.h"
#include "eigenmath/line_utils.h"
#include "gmock/gmock.h"

namespace mobility::collision {

char CostToChar(double cost);

// Generates an occupancy grid large enough to include all `lines`.  Marks
// `lines` and the parts of `fixed_lines` which fall into the grid as occupied,
// and the remaining space as unoccupied.
void CreateOccupancyGrid(
    const std::vector<eigenmath::LineSegment2d>& lines,
    const std::vector<eigenmath::LineSegment2d>& fixed_lines,
    OccupancyGrid* grid);

// Fills an occupancy grid from a string, and infers the grid range from the
// string dimensions.  The filled grid range starts at (0,0).
// Expects that the characters are from the following list.
// Characters mean:
//  '#' -> occupied
//  '?' -> unknown
//  '.' -> free
void CreateOccupancyGrid(std::string_view drawn_grid, OccupancyGrid* grid);

void CreateCostGrid(const DistanceToCost& distance_to_cost,
                    const std::vector<eigenmath::LineSegment2d>& lines,
                    const std::vector<eigenmath::LineSegment2d>& fixed_lines,
                    Grid<double>* cost_grid);

std::string DumpGridRange(
    const GridRange& grid_range,
    absl::FunctionRef<char(const GridIndex&)> index_to_char);

std::string DumpOccupancyGrid(const OccupancyGrid& grid);

std::string DumpCostGrid(const Grid<double>& cost_grid);

std::string DumpCostGridWithPath(const Grid<double>& cost_grid,
                                 const GridIndexSet& path_indices);

template <typename PointIter>
std::string DumpCostGridWithPath(const Grid<double>& cost_grid,
                                 PointIter pt_first, PointIter pt_last) {
  return DumpCostGridWithPath(cost_grid, GridIndexSet(pt_first, pt_last));
}

std::string DumpCostGridWithPath(
    const Grid<double>& cost_grid,
    const std::vector<eigenmath::Vector2d>& path_coords);

template <typename T, typename CellToCharFunc>
std::string DumpGrid(const Grid<T>& grid, CellToCharFunc cell_to_char) {
  return DumpGridRange(grid.Range(), [&](const GridIndex& index) {
    return cell_to_char(grid.GetUnsafe(index));
  });
}

namespace testing {

// Returns true if an occupancy grid matches the given expected string dump.
//
// Usage:
// EXPECT_THAT(grid, testing::OccupancyGridPrintsTo("\n..#.\n.#..\n.?.."));
MATCHER_P(OccupancyGridPrintsTo, expected,
          absl::StrCat("Expected grid prints to:", expected)) {
  const std::string dumped_grid = DumpOccupancyGrid(arg);
  *result_listener << "Actual grid prints to:" << dumped_grid;
  return expected == dumped_grid;
}

}  // namespace testing

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_GRID_TEST_UTILS_H_

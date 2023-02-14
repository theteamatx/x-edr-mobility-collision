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

#include "collision/grid_test_utils.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"
#include "collision/grid_common.h"
#include "collision/occupancy_grid.h"

namespace mobility::collision {

namespace {
OccupancyStatus CharToOccupancy(char c) {
  switch (c) {
    case '#':
      return OccupancyStatus::OCCUPIED;
    case '?':
      return OccupancyStatus::UNKNOWN;
    case '.':
      return OccupancyStatus::UNOCCUPIED;
    default:
      QCHECK(false) << "Unknown character " << c;
  }
}
}  // namespace

char CostToChar(double cost) {
  static constexpr double kFudge = 1.0e-3;
  if (std::isfinite(cost)) {
    const std::pair<double, char> cost_table[] = {
        {125.0, '*'},        {25.0, 'O'},         {5.0, 'o'},
        {1.0 + kFudge, '+'}, {1.0 - kFudge, '.'},
    };
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

void CreateOccupancyGrid(
    const std::vector<eigenmath::LineSegment2d>& lines,
    const std::vector<eigenmath::LineSegment2d>& fixed_lines,
    OccupancyGrid* grid) {
  GridRange lines_range;
  for (const auto& ln : lines) {
    lines_range.GrowToInclude(grid->Frame().FrameToGrid(ln.from));
    lines_range.GrowToInclude(grid->Frame().FrameToGrid(ln.to));
  }
  grid->Reshape(lines_range);
  grid->Fill(OccupancyStatus::UNOCCUPIED);
  for (const auto& ln : lines) {
    for (const GridIndex& cell : GridLine(grid->Frame().FrameToGrid(ln.from),
                                          grid->Frame().FrameToGrid(ln.to))) {
      grid->SetUnsafe(cell, OccupancyStatus::OCCUPIED);
    }
  }
  for (const auto& ln : fixed_lines) {
    for (const GridIndex& cell : GridLine(grid->Frame().FrameToGrid(ln.from),
                                          grid->Frame().FrameToGrid(ln.to))) {
      if (grid->IsInGridBounds(cell)) {
        grid->SetUnsafe(cell, OccupancyStatus::OCCUPIED);
      }
    }
  }
}

void CreateOccupancyGrid(std::string_view drawn_grid, OccupancyGrid* grid) {
  grid->SetDefaultValue(OccupancyStatus::UNKNOWN);
  int n_cols = 0;
  auto skip_empty_and_count_columns = [&n_cols](absl::string_view line) {
    n_cols = std::max<int>(n_cols, line.size());
    return absl::SkipEmpty()(line);
  };
  std::vector<absl::string_view> lines =
      absl::StrSplit(drawn_grid, '\n', skip_empty_and_count_columns);
  const int n_rows = lines.size();
  grid->GrowToInclude({{0, 0}, {n_cols, n_rows}});
  std::reverse(lines.begin(), lines.end());
  int row = 0;
  for (const auto& line : lines) {
    for (int col = 0; col < line.size(); ++col) {
      grid->SetUnsafe({col, row}, CharToOccupancy(line[col]));
    }
    ++row;
  }
}

void CreateCostGrid(const DistanceToCost& distance_to_cost,
                    const std::vector<eigenmath::LineSegment2d>& lines,
                    const std::vector<eigenmath::LineSegment2d>& fixed_lines,
                    Grid<double>* cost_grid) {
  const Grid<double> cost_mask =
      distance_to_cost.CreateCostMask(cost_grid->Resolution());
  cost_grid->SetDefaultValue(1.0);
  GridRange lines_range;
  for (const auto& ln : lines) {
    for (const GridIndex& cell :
         GridLine(cost_grid->Frame().FrameToGrid(ln.from),
                  cost_grid->Frame().FrameToGrid(ln.to))) {
      ApplyCostMaskAroundObstacle(cost_mask, cell, cost_grid);
      lines_range.GrowToInclude(cell);
    }
  }
  for (const auto& ln : fixed_lines) {
    for (const GridIndex& cell :
         GridLine(cost_grid->Frame().FrameToGrid(ln.from),
                  cost_grid->Frame().FrameToGrid(ln.to))) {
      if (lines_range.Contains(cell)) {
        ApplyCostMaskAroundObstacle(cost_mask, cell, cost_grid);
      }
    }
  }
}

std::string DumpGridRange(
    const GridRange& grid_range,
    absl::FunctionRef<char(const GridIndex&)> index_to_char) {
  std::ostringstream result;
  if (grid_range.Empty()) {
    result << "EMPTY";
  }
  for (GridIndex index(0, grid_range.upper.y() - 1);
       index.y() >= grid_range.lower.y(); --index.y()) {
    result << "\n";
    for (index.x() = grid_range.lower.x(); index.x() < grid_range.upper.x();
         ++index.x()) {
      result << index_to_char(index);
    }
  }
  return result.str();
}

std::string DumpOccupancyGrid(const OccupancyGrid& grid) {
  return DumpGridRange(grid.Range(), [&](const GridIndex& index) {
    const auto occ = grid.GetUnsafe(index);
    switch (occ) {
      case OccupancyStatus::UNOCCUPIED:
        return '.';
      case OccupancyStatus::UNKNOWN:
        return '?';
      case OccupancyStatus::OCCUPIED:
        return '#';
    }
  });
}

std::string DumpCostGrid(const Grid<double>& cost_grid) {
  return DumpGridRange(cost_grid.Range(), [&](const GridIndex& index) {
    return CostToChar(cost_grid.GetUnsafe(index));
  });
}

std::string DumpCostGridWithPath(const Grid<double>& cost_grid,
                                 const GridIndexSet& path_indices) {
  GridRange range = cost_grid.Range();
  for (const GridIndex& pp : path_indices) {
    range.GrowToInclude(pp);
  }
  return DumpGridRange(range, [&](const GridIndex& index) {
    if (0 != path_indices.count(index)) {
      return 'x';
    } else {
      double cost;
      if (cost_grid.Get(index, &cost)) {
        return CostToChar(cost);
      } else {
        return ':';
      }
    }
  });
}

std::string DumpCostGridWithPath(
    const Grid<double>& cost_grid,
    const std::vector<eigenmath::Vector2d>& path_coords) {
  GridIndexSet path_indices;
  for (auto& coord : path_coords) {
    path_indices.insert(cost_grid.Frame().FrameToGrid(coord));
  }
  return DumpCostGridWithPath(cost_grid, path_indices);
}

}  // namespace mobility::collision

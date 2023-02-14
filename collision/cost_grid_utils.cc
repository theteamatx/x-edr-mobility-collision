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

#include <algorithm>
#include <functional>
#include <limits>

#include "collision/collision_utils.h"
#include "collision/grid_common.h"

namespace mobility::collision {

DistanceToCost::DistanceToCost(double obstacle_radius, double buffer_zone)
    : obstacle_radius_(obstacle_radius), buffer_zone_(buffer_zone) {}

double DistanceToCost::operator()(double distance) const {
  distance -= obstacle_radius_;
  if (distance <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  if (distance >= buffer_zone_) {
    return 1.0;
  }
  return buffer_zone_ / distance;
}

double DistanceToCost::operator()(const Grid<double>& distance_grid,
                                  const GridIndex& index) const {
  double distance;
  if (distance_grid.Get(index, &distance)) {
    return (*this)(distance);
  }
  return 1.0;
}

Grid<double> DistanceToCost::CreateCostMask(double grid_resolution) const {
  return mobility::collision::CreateCostMask(
      grid_resolution, obstacle_radius_ + buffer_zone_, std::cref(*this));
}

void ApplyCostMaskAroundObstacle(const Grid<double>& cost_mask,
                                 const GridIndex& obstacle,
                                 Grid<double>* cost_grid) {
  const GridRange cost_mask_range(GridIndex(1, 1) - cost_mask.Range().upper,
                                  cost_mask.Range().upper);
  cost_grid->Reshape(GridRange::SpanningUnion(
      GridRange::ShiftBy(cost_mask_range, obstacle), cost_grid->Range()));
  cost_mask_range.ForEachGridCoord([&](const GridIndex& mask_index) {
    const GridIndex grid_index = mask_index + obstacle;
    const double grid_value = cost_grid->GetUnsafe(grid_index);
    const double mask_value = cost_mask.GetUnsafe(mask_index.cwiseAbs());
    if (grid_value < mask_value) {
      cost_grid->SetUnsafe(grid_index, mask_value);
    }
  });
}

namespace {
double GetCostFromMaskAndOccupancyGridImpl(const GridIndex& index,
                                           bool treat_unknown_as_obstacle,
                                           double cost_ceiling,
                                           const Grid<double>& cost_mask,
                                           const OccupancyGrid& top_grid,
                                           const OccupancyGrid* bottom_grid) {
  double cost_value = 1.0;
  const auto is_obstacle = [&](OccupancyStatus occupancy) -> bool {
    return (
        occupancy == OccupancyStatus::OCCUPIED ||
        (treat_unknown_as_obstacle && occupancy == OccupancyStatus::UNKNOWN));
  };

  const GridRange mask_range(GridIndex(1, 1) - cost_mask.Range().upper + index,
                             cost_mask.Range().upper + index);

  const GridRange top_grid_range =
      GridRange::Intersect(mask_range, top_grid.Range());
  top_grid.ForEachGridIndexWithCellValue(
      top_grid_range,
      [&](const GridIndex& global_index, const OccupancyStatus occupancy) {
        if (is_obstacle(occupancy)) {
          cost_value =
              std::max(cost_value,
                       cost_mask.GetUnsafe((global_index - index).cwiseAbs()));
        }
        return (cost_value < cost_ceiling);  // Bail out if cost is high.
      });

  if (bottom_grid == nullptr || top_grid_range == mask_range ||
      cost_value >= cost_ceiling) {
    return std::min(cost_value, cost_ceiling);
  }

  const GridRange::Quad bottom_grid_ranges =
      GridRange::Difference(mask_range, top_grid_range);
  for (const auto& bottom_grid_range : bottom_grid_ranges) {
    bottom_grid->ForEachGridIndexWithCellValue(
        bottom_grid_range,
        [&](const GridIndex& global_index, const OccupancyStatus occupancy) {
          if (is_obstacle(occupancy)) {
            cost_value = std::max(
                cost_value,
                cost_mask.GetUnsafe((global_index - index).cwiseAbs()));
          }
          return (cost_value < cost_ceiling);  // Bail out if cost is high.
        });
  }

  return std::min(cost_value, cost_ceiling);
}
}  // namespace

double GetCostFromMaskAndOccupancyGrid(const GridIndex& index,
                                       bool treat_unknown_as_obstacle,
                                       double cost_ceiling,
                                       const Grid<double>& cost_mask,
                                       const OccupancyGrid& top_grid) {
  return GetCostFromMaskAndOccupancyGridImpl(index, treat_unknown_as_obstacle,
                                             cost_ceiling, cost_mask, top_grid,
                                             nullptr);
}

double GetCostFromMaskAndOccupancyGrid(const GridIndex& index,
                                       bool treat_unknown_as_obstacle,
                                       double cost_ceiling,
                                       const Grid<double>& cost_mask,
                                       const OccupancyGrid& top_grid,
                                       const OccupancyGrid& bottom_grid) {
  return GetCostFromMaskAndOccupancyGridImpl(index, treat_unknown_as_obstacle,
                                             cost_ceiling, cost_mask, top_grid,
                                             &bottom_grid);
}

void ConvolveCostMaskOverOccupiedEdges(const OccupancyGrid& grid,
                                       const Grid<double>& mask,
                                       bool treat_unknown_as_obstacle,
                                       bool ignore_internal_points,
                                       Grid<double>* cost_grid) {
  cost_grid->GrowToInclude(
      GridRange::GrowBy(grid.Range(), mask.UpperBound().x()));
  grid.Range().ForEachGridCoord([&](GridIndex index) {
    // Apply whole cost mask only to exterior points, and later apply
    // only the maximum value of the cost mask to internal points.
    if (IsPointAnObstacle(grid, index, treat_unknown_as_obstacle, true)) {
      ApplyCostMaskAroundObstacle(mask, index, cost_grid);
    } else if (!ignore_internal_points &&
               IsPointInsideOccupiedRegion(grid, index,
                                           treat_unknown_as_obstacle)) {
      const double grid_value = cost_grid->GetUnsafe(index);
      const double mask_value = mask.GetUnsafe(0, 0);
      if (grid_value < mask_value) {
        cost_grid->SetUnsafe(index, mask_value);
      }
    }
  });
}

void GraduateInteriorCosts(double nominal_obstacle_cost,
                           Grid<double>* cost_grid) {
  GridRange range = cost_grid->Range();
  double cost_level = nominal_obstacle_cost;
  while (!range.Empty()) {
    GridRange next_range;
    range.ForEachGridCoord([&](GridIndex index) {
      if (IsPointInsideMaxCostRegion(
              *cost_grid, index,
              cost_level - std::numeric_limits<double>::epsilon())) {
        next_range.GrowToInclude(index);
        cost_grid->SetUnsafe(index, cost_level + nominal_obstacle_cost);
      }
    });
    range = next_range;
    cost_level += nominal_obstacle_cost;
  }
}

}  // namespace mobility::collision

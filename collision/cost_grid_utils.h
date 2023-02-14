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

#ifndef MOBILITY_COLLISION_COLLISION_COST_GRID_UTILS_H_
#define MOBILITY_COLLISION_COLLISION_COST_GRID_UTILS_H_

#include <cmath>

#include "collision/grid_common.h"
#include "collision/occupancy_grid.h"

namespace mobility::collision {

// This is mostly a deprecated distance-to-cost functor used for the global
// planner.
class DistanceToCost {
 public:
  DistanceToCost(double obstacle_radius, double buffer_zone);

  double operator()(double distance) const;

  double operator()(const Grid<double>& distance_grid,
                    const GridIndex& index) const;

  double MaxDistance() const { return obstacle_radius_ + buffer_zone_; }

  Grid<double> CreateCostMask(double grid_resolution) const;

 private:
  const double obstacle_radius_;
  const double buffer_zone_;
};

// Use a given cost mask (see CreateCostMask) and an occupied grid cell, this
// function sets all the cells of the output cost grid to the max between the
// existing value and that of the corresponding cost mask.
void ApplyCostMaskAroundObstacle(const Grid<double>& cost_mask,
                                 const GridIndex& obstacle,
                                 Grid<double>* cost_grid);

// Use a given cost mask (see CreateCostMask) at a given grid index to obtain
// the cost value, i.e., the maximum cost of any occupied cell within the
// footprint of the cost mask centered around the given index.
double GetCostFromMaskAndOccupancyGrid(const GridIndex& index,
                                       bool treat_unknown_as_obstacle,
                                       double cost_ceiling,
                                       const Grid<double>& cost_mask,
                                       const OccupancyGrid& top_grid);

// Use a given cost mask (see CreateCostMask) at a given grid index to obtain
// the cost value, i.e., the maximum cost of any occupied cell within the
// footprint of the cost mask centered around the given index.
double GetCostFromMaskAndOccupancyGrid(const GridIndex& index,
                                       bool treat_unknown_as_obstacle,
                                       double cost_ceiling,
                                       const Grid<double>& cost_mask,
                                       const OccupancyGrid& top_grid,
                                       const OccupancyGrid& bottom_grid);

// Creates a cost mask for a given grid resolution and distance range. A cost
// mask is a single-quadrant cost-map that goes from [0,0] to [dist, dist]
// where dist is the maximum distance range, discretized by the given
// resolution. Each cell of the mask will have a value which corresponds to the
// cost associated with the distance of that cell from the origin of the mask.
// The cost is computed from the distance using the dist_to_cost functor.
template <typename DistToCost>
Grid<double> CreateCostMask(double grid_resolution, double distance_range,
                            DistToCost dist_to_cost) {
  // Initialize the cost mask if the parameters have changed.
  const int distance_irange =
      std::lrint(std::ceil(distance_range / grid_resolution));
  Grid<double> cost_mask(
      GridFrame("", eigenmath::Pose2d::Identity(), grid_resolution),
      GridRange::OriginTo({distance_irange + 1, distance_irange + 1}), 1.0);
  for (int j = 0; j < distance_irange + 1; ++j) {
    for (int i = 0; i < distance_irange + 1; ++i) {
      const double distance = std::hypot(i, j) * grid_resolution;
      cost_mask.SetUnsafe(i, j, dist_to_cost(distance));
    }
  }
  return cost_mask;
}

// This function takes a cost mask (see CreateCostMask) and runs it around all
// the occupied cells of an occupancy grid (i.e., a convolution) and applies
// the maximum cost encountered for every cell to the output cost_grid.
// This function also grows the cost grid to a sufficient size to accommodate
// the convoluted occupancy grid.
void ConvolveCostMaskOverOccupiedEdges(const OccupancyGrid& grid,
                                       const Grid<double>& mask,
                                       bool treat_unknown_as_obstacle,
                                       bool ignore_internal_points,
                                       Grid<double>* cost_grid);

// This function creates a graduation of cost levels within the interior of
// a cost grid where there is a collision (cost above given nominal obstacle
// cost). Each interior "layer" gets an additional nominal obstacle cost.
// See cost_grid_utils_test.cc for an ascii illustration.
void GraduateInteriorCosts(double nominal_obstacle_cost,
                           Grid<double>* cost_grid);

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_COST_GRID_UTILS_H_

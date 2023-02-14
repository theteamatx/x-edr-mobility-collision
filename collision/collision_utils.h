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

#ifndef MOBILITY_COLLISION_COLLISION_COLLISION_UTILS_H_
#define MOBILITY_COLLISION_COLLISION_COLLISION_UTILS_H_

#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "collision/hull.h"
#include "collision/occupancy_grid.h"
#include "diff_drive/curve.h"
#include "diff_drive/trajectory.h"
#include "eigenmath/pose2.h"
#include "eigenmath/types.h"

namespace mobility::collision {

// This function performs a fast upper-bound (worst-case) calculation for
// the grid range that could be reachable by a point inside a given hull when
// swept across a given trajectory.
// This does not sweep over the trajectory, but makes a worst-case
// approximation instead.
GridRange GetWorstCaseReachableGridRange(
    const GridFrame& grid_frame, const diff_drive::Trajectory& trajectory,
    const Hull& hull);

// This function performs a fast upper-bound (worst-case) calculation for
// the grid range that could be reachable by a point inside a given hull when
// swept across a given curve.
// This does not sweep over the curve, but makes a worst-case
// approximation instead.
GridRange GetWorstCaseReachableGridRange(const GridFrame& grid_frame,
                                         const diff_drive::Curve& curve,
                                         const Hull& hull);

// This function calculates an upper-bound for the grid range that could be
// reachable by a point inside a given hull when swept across a given
// trajectory.
// This sweeps over the trajectory and grows a rectangular range that should
// contain the entire sweep of the given hull.
GridRange ComputeReachableGridRange(const GridFrame& grid_frame,
                                    const diff_drive::Trajectory& trajectory,
                                    const Hull& hull);

// This function calculates an upper-bound for the grid range that could be
// reachable by a point inside a given hull when swept across a given
// curve.
// This sweeps over the curve and grows a rectangular range that should
// contain the entire sweep of the given hull.
GridRange ComputeReachableGridRange(const GridFrame& grid_frame,
                                    const diff_drive::Curve& curve,
                                    const Hull& hull);

// This function calculates the grid range that could be reachable by a point
// inside a given hull.
GridRange ComputeReachableGridRange(const GridFrame& grid_frame,
                                    const Hull& hull);

// This function checks if a given grid cell (at "index") is entirely inside
// an occupied region (where occupied could be unknown if unknown is treated
// as obstacle points). Returns false if the point is not occupied or if the
// point is occupied but not inside an occupied region (i.e., on the edge).
bool IsPointInsideOccupiedRegion(const OccupancyGrid& grid, GridIndex index,
                                 bool treat_unknown_as_obstacle = false);

// This function checks if a given grid cell (at "index") is an obstacle point,
// considering whether unknown should be treated as an obstacle and whether
// internal points (entirely inside an occupied region) should be ignored.
bool IsPointAnObstacle(const OccupancyGrid& grid, GridIndex index,
                       OccupancyStatus occupancy,
                       bool treat_unknown_as_obstacle = false,
                       bool ignore_internal_points = false);
inline bool IsPointAnObstacle(const OccupancyGrid& grid, GridIndex index,
                              bool treat_unknown_as_obstacle = false,
                              bool ignore_internal_points = false) {
  const OccupancyStatus occupancy = grid.GetUnsafe(index);
  return IsPointAnObstacle(grid, index, occupancy, treat_unknown_as_obstacle,
                           ignore_internal_points);
}

// This function checks if a given grid cell (at `index`) is in the interior of
// a region where all costs are above a given threshold. Returns false if the
// point is not inside a max-cost region (e.g., on the boundary).
bool IsPointInsideMaxCostRegion(const Grid<double>& cost_grid,
                                const GridIndex& index,
                                double max_cost_threshold);

// Checks if a given query_point intersects (is contained by) the given hull
// when swept over the given curve.
// The max_resolution determines the cord-length steps taken when sweeping
// over the curve.
bool CurveTraceContainsPoint(const diff_drive::Curve& curve, const Hull& hull,
                             double max_resolution,
                             const eigenmath::Vector2d& query_point);

// Adds the contour of a given hull (expressed relative to a given pose) to
// a given occupancy grid as occupied cells. This function also optionally
// outputs the range of cells touched.
void AddHullContourToOccupancyGrid(const Hull& hull,
                                   const eigenmath::Pose2d& pose,
                                   OccupancyGrid* grid,
                                   GridRange* actual_range = nullptr);

// Collects all cells in a grid range which fall on the contour of a given hull
// (expressed relative to a given pose).
void CollectOccupiedPointsOnHullContour(
    const Hull& hull, const eigenmath::Pose2d& pose,
    const GridFrame& grid_frame, const GridRange& grid_range,
    std::vector<eigenmath::Vector2d>* grid_points);

// Collects all cells in a grid range which fall into the hull (expressed
// relative to a pose).
void CollectOccupiedGridCellsFromHull(const Hull& hull,
                                      const eigenmath::Pose2d& pose,
                                      const GridFrame& grid_frame,
                                      const GridRange& grid_range,
                                      std::vector<GridIndex>* grid_indices);

// Fills all cells in the occupancy grid which fall into the hull (expressed
// relative to a pose) with the provided occupancy value.
void FillHullInOccupancyGrid(const Hull& hull, const eigenmath::Pose2d& pose,
                             OccupancyStatus occupancy, OccupancyGrid* grid);

// Computes the signed distance (negative if penetrating) for a given
// query_point if it penetrates the given hull when swept over the given curve.
// The distance value will be positive if there is no penetration.
// Only negative distances are meaningful.
// The max_resolution determines the cord-length steps taken when sweeping
// over the curve.
// The min_distance (i.e., biggest negative distance) value puts a cap on
// the worst case we are interested in. If penetration distance goes below that,
// we stop the search early.
double DistanceIfPenetratingCurveTrace(
    const diff_drive::Curve& curve, const Hull& hull, double max_resolution,
    const eigenmath::Vector2d& query_point,
    double min_distance = -std::numeric_limits<double>::infinity());

// Fills the given occupancy grid with footprint of the hull swept over the
// given curve.
// The grid will be reshaped and grown as needed.
void FillOccupancyGridWithCurveTrace(const diff_drive::Curve& curve,
                                     const Hull& hull, OccupancyGrid* grid,
                                     GridRange* actual_range = nullptr);

// Fills the given cost grid with cost values that are mapped by
// distance_to_cost from a distance value computed for each grid cell to the
// hull swept over the given curve.
// Distance values will be positive if there is no penetration.
// Only negative distances are meaningful.
// The min_distance (i.e., biggest negative distance) value puts a cap on
// the worst case we are interested in. If penetration distance goes below that,
// we stop the search early.
// The grid will be reshaped and grown as needed.
void FillCostGridWithCurveTrace(
    const diff_drive::Curve& curve, const Hull& hull,
    std::function<double(double)> distance_to_cost, Grid<double>* grid,
    double min_distance = -std::numeric_limits<double>::infinity());

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_COLLISION_UTILS_H_

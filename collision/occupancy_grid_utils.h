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

#ifndef MOBILITY_COLLISION_COLLISION_OCCUPANCY_GRID_UTILS_H_
#define MOBILITY_COLLISION_COLLISION_OCCUPANCY_GRID_UTILS_H_

#include <vector>

#include "collision/grid_common.h"
#include "collision/occupancy_grid.h"
#include "eigenmath/line_utils.h"

namespace mobility::collision {

// Returns the parts of the bounding box boundary which are not covered by the
// line segments.  Assumes that the intersection of a segment A and a bounding
// box boudary segment B contains at least one endpoint of the segment A.
std::vector<GridSegment> MissingBoundarySegments(
    const GridRange& bounding_box,
    const std::vector<GridSegment>& line_segments);

// Checks whether the segment between two neighbouring occupied points is part
// of an obstacle boundary.  Assumes that the points are contained in the grid's
// range.
bool IsBoundary(const OccupancyGrid& grid, GridIndex from, GridIndex to,
                bool treat_unknown_as_obstacle);

// Same as above, but requires that the points from and to appear in
// counter-clockwise order when traversing along a boundary.
bool IsCounterClockwiseBoundary(const OccupancyGrid& grid, GridIndex from,
                                GridIndex to, bool treat_unknown_as_obstacle);

// Assumes a counter clockwise traversal along an obstacle boundary which
// visited the points `from` and `to`.  Returns the next point along the
// boundary to visit after `to`.
GridIndex NextPointOnBoundary(const OccupancyGrid& grid, GridIndex from,
                              GridIndex to, bool treat_unknown_as_obstacle);

// A single-step lookahead simplification strategy which checks whether the
// Bresenham line  [segment.from, segment.to]  is contained in the Bresenham
// line  [segment.from, next].
bool ExtensionIsOnBoundary(const GridSegment& segment, GridIndex next);

// Extracts line segments along the occupied cells which are on the boundary of
// occupied regions.  Skips obstacles which only consist of a single cell.
std::vector<GridSegment> ExtractBoundaryLineSegments(
    const OccupancyGrid& grid, bool treat_unknown_as_obstacle);

// Returns the segments of the boundary polygons as a list.  Skips obstacles
// which only consist of a single cell.
std::vector<GridSegment> ExtractBoundaryPolygonSegments(
    const OccupancyGrid& grid, bool treat_unknown_as_obstacle);

// Returns simplified segments of the boundary polygons as a list.  Skips
// obstacles which only consist of a single cell.
std::vector<GridSegment> ExtractSimplifiedPolygonSegments(
    const OccupancyGrid& grid, bool treat_unknown_as_obstacle);

// Creates a grid with the same resolution and frame as the first_grid, where
// all cells are either occupied if they are occupied only in first_grid or
// free otherwise.
void GetAllOccupiedInFirstButNotInOther(const OccupancyGrid& first_grid,
                                        bool treat_unknown_in_first_as_occupied,
                                        const OccupancyGrid& other_grid,
                                        bool treat_unknown_in_other_as_occupied,
                                        OccupancyGrid* result_grid);

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_OCCUPANCY_GRID_UTILS_H_

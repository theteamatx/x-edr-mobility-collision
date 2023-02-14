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

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "collision/grid_common.h"
#include "diff_drive/curve_trajectory_utils.h"
#include "genit/adjacent_circular_iterator.h"
#include "genit/cached_iterator.h"
#include "genit/transform_iterator.h"

namespace mobility::collision {

namespace {

GridRange GetWorstCaseReachableGridRangeImpl(
    const GridFrame& grid_frame, const eigenmath::Vector2d& start_pos,
    const eigenmath::Vector2d& finish_pos, double total_cord_length,
    const Hull& hull) {
  // Assuming the worst case, start and finish are the foci of an ellipse,
  // and then, the total_cord_length is its major axis length.
  // Compute the circle that contains the ellipse:
  const eigenmath::Vector2d ellipse_center = 0.5 * (start_pos + finish_pos);
  const double ellipse_max_radius = 0.5 * total_cord_length;
  const double hull_max_radius =
      hull.GetMaxRadiusAround(eigenmath::Vector2d::Zero());

  // Get the rectangular range around the ellipse (within circle) convolved
  // with the radius of the collision hull.
  return grid_frame.FrameCircleToGridRange(
      ellipse_center, ellipse_max_radius + hull_max_radius);
}

// This function visits the knot-points of a curve moving outwards from the
// closest knot point from a given query point.
// Stops when the predicate evaluates to true for a given 2d point.
template <typename Predicate>
bool VisitKnotPointsOutFromClosestPoint(const diff_drive::Curve& curve,
                                        const eigenmath::Vector2d& query_point,
                                        Predicate pred) {
  // Find the closest point on the curve and then, proceed outward from there.
  const auto curve_span = curve.GetCurvePointIteratorRange();
  const auto closest_pt_on_curve = curve.FindClosestKnotPointMatch(query_point);
  auto bwd_it = closest_pt_on_curve;
  auto fwd_it = closest_pt_on_curve;
  while (bwd_it > curve_span.begin() || fwd_it < curve_span.end()) {
    if (bwd_it > curve_span.begin()) {
      --bwd_it;
      if (pred(bwd_it->point.GetPose().inverse() * query_point)) {
        return true;
      }
    }
    if (fwd_it < curve_span.end()) {
      if (pred(fwd_it->point.GetPose().inverse() * query_point)) {
        return true;
      }
      ++fwd_it;
    }
  }
  return false;
}

template <typename Functor>
void ForAllGridCellsInHullImpl(const Hull& hull, const eigenmath::Pose2d& pose,
                               const GridFrame& grid_frame,
                               const GridRange& grid_range, Functor f) {
  auto hull_pt_to_grid = [&pose, grid_frame](const eigenmath::Vector2d& v) {
    return grid_frame.FrameToGrid(pose * v);
  };
  for (auto& hull : hull.GetConvexHulls()) {
    auto grid_vertices = genit::CachedRange(
        genit::TransformRange(hull.GetPoints(), std::cref(hull_pt_to_grid)));
    const int dist_to_lowest =
        std::min_element(begin(grid_vertices), end(grid_vertices),
                         [](const GridIndex& lhs, const GridIndex& rhs) {
                           return lhs.y() < rhs.y();
                         }) -
        begin(grid_vertices);
    // The forward (rhs) and back (lhs) iterators are at the same place on
    // different windings.
    using SegmentIter =
        genit::AdjacentCircularIterator<decltype(begin(grid_vertices)), 2>;
    SegmentIter rhs_segment_it(begin(grid_vertices), end(grid_vertices));
    rhs_segment_it += dist_to_lowest;
    SegmentIter lhs_segment_it = std::prev(rhs_segment_it);
    auto rhs_segment = *rhs_segment_it;
    auto lhs_segment = *lhs_segment_it;
    // Grid line iterators along the edges.
    GridLine rhs_grid_line(rhs_segment[0], rhs_segment[1]);
    auto rhs_grid_it = rhs_grid_line.begin();
    GridLine lhs_grid_line(lhs_segment[1], lhs_segment[0]);
    auto lhs_grid_it = lhs_grid_line.begin();

    // This helper function moves the `it` `GridLine` iterator until it has
    // increased the y coordinate.
    auto move_up_once = [](auto it, auto it_end) {
      const int start_y = (*it).y();
      return std::find_if(
          std::next(it), it_end,
          [start_y](const GridIndex& next) { return next.y() > start_y; });
    };
    auto move_outward = [](auto it, auto it_end, int exterior_move) {
      const GridIndex start = *it;
      return std::prev(std::find_if(
          std::next(it), it_end, [start, exterior_move](const GridIndex& next) {
            return (next.y() != start.y() ||
                    (next.x() - start.x()) * exterior_move < 0);
          }));
    };

    while (true) {
      // Fill in each horizontal scan-line until the end of one segment is
      // reached.
      while (rhs_grid_it != rhs_grid_line.end() &&
             lhs_grid_it != lhs_grid_line.end()) {
        // Move the iterators horizontally outwards first.
        rhs_grid_it = move_outward(rhs_grid_it, rhs_grid_line.end(), 1);
        lhs_grid_it = move_outward(lhs_grid_it, lhs_grid_line.end(), -1);
        const GridIndex lhs_start = *lhs_grid_it;
        const GridIndex rhs_start = *rhs_grid_it;
        if (lhs_start.y() >= grid_range.lower.y() &&
            lhs_start.y() < grid_range.upper.y()) {
          // Clamp the horizontal line to the grid range.
          const GridIndex lhs_start_clamped{
              std::max(lhs_start.x(), grid_range.lower.x()), lhs_start.y()};
          const GridIndex rhs_start_clamped{
              std::min(rhs_start.x(), grid_range.upper.x()), rhs_start.y()};
          if (lhs_start_clamped.x() <= rhs_start_clamped.x()) {
            for (const GridIndex& cell :
                 GridLine(lhs_start_clamped, rhs_start_clamped)) {
              f(cell);
            }
          }
        }
        rhs_grid_it = move_up_once(rhs_grid_it, rhs_grid_line.end());
        lhs_grid_it = move_up_once(lhs_grid_it, lhs_grid_line.end());
      }
      // Terminate if the segments meet up at the ends.
      if (rhs_segment[1] == lhs_segment[0]) {
        break;
      }
      // Move to the next segment.
      // Move one segment at a time, to check termination condition between
      // each, so that segments don't cross over passed the condition.
      if (rhs_grid_it == rhs_grid_line.end()) {
        ++rhs_segment_it;
        rhs_segment = *rhs_segment_it;
        rhs_grid_line = GridLine(rhs_segment[0], rhs_segment[1]);
        rhs_grid_it = move_up_once(rhs_grid_line.begin(), rhs_grid_line.end());
      } else {
        --lhs_segment_it;
        lhs_segment = *lhs_segment_it;
        lhs_grid_line = GridLine(lhs_segment[1], lhs_segment[0]);
        lhs_grid_it = move_up_once(lhs_grid_line.begin(), lhs_grid_line.end());
      }
    }
  }
}

// Calls the given functor for all cells along the contours of each convex hull
// of the given hull. It does not check for grid range.
template <typename Functor>
void ForAllGridCellsOnContourImpl(const Hull& hull,
                                  const eigenmath::Pose2d& pose,
                                  const GridFrame& grid_frame, Functor f) {
  auto hull_pt_to_grid = [&pose, grid_frame](const eigenmath::Vector2d& v) {
    return grid_frame.FrameToGrid(pose * v);
  };
  for (const auto& hull : hull.GetConvexHulls()) {
    for (const auto& segment : genit::AdjacentElementsCircularRange<2>(
             genit::CachedRange(genit::TransformRange(
                 hull.GetPoints(), std::cref(hull_pt_to_grid))))) {
      GridLine grid_line(segment[0], segment[1]);
      for (const GridIndex& cell : grid_line) {
        f(cell);
      }
    }
  }
}

void AddHullContourToOccupancyGridImpl(const Hull& hull,
                                       const eigenmath::Pose2d& pose,
                                       OccupancyGrid* grid,
                                       GridRange* actual_range) {
  if (actual_range != nullptr) {
    ForAllGridCellsOnContourImpl(
        hull, pose, grid->Frame(), [grid, actual_range](const GridIndex& cell) {
          grid->SetUnsafe(cell, OccupancyStatus::OCCUPIED);
          actual_range->GrowToInclude(cell);
        });
  } else {
    ForAllGridCellsOnContourImpl(
        hull, pose, grid->Frame(), [grid](const GridIndex& cell) {
          grid->SetUnsafe(cell, OccupancyStatus::OCCUPIED);
        });
  }
}

}  // namespace

GridRange GetWorstCaseReachableGridRange(
    const GridFrame& grid_frame, const diff_drive::Trajectory& trajectory,
    const Hull& hull) {
  // Get measurements of the size of the trajectory:
  const eigenmath::Vector2d start_pos =
      trajectory.GetStart().state.GetPose().translation();
  const eigenmath::Vector2d finish_pos =
      trajectory.GetFinish().state.GetPose().translation();
  const double total_cord_length = trajectory.ComputeTotalCordLength();

  return GetWorstCaseReachableGridRangeImpl(grid_frame, start_pos, finish_pos,
                                            total_cord_length, hull);
}

GridRange GetWorstCaseReachableGridRange(const GridFrame& grid_frame,
                                         const diff_drive::Curve& curve,
                                         const Hull& hull) {
  // Get measurements of the size of the trajectory:
  const eigenmath::Vector2d start_pos =
      curve.GetStart().point.GetPose().translation();
  const eigenmath::Vector2d finish_pos =
      curve.GetFinish().point.GetPose().translation();
  const double total_cord_length = curve.GetCordLengthSpan().Length();

  return GetWorstCaseReachableGridRangeImpl(grid_frame, start_pos, finish_pos,
                                            total_cord_length, hull);
}

GridRange ComputeReachableGridRange(const GridFrame& grid_frame,
                                    const diff_drive::Trajectory& trajectory,
                                    const Hull& hull) {
  const double hull_max_radius =
      hull.GetMaxRadiusAround(eigenmath::Vector2d::Zero());
  auto it = trajectory.BeginInCordLength();
  auto it_end = trajectory.EndInCordLength();
  GridRange result;
  for (; it < it_end; it += grid_frame.resolution) {
    result.SpanningUnion(grid_frame.FrameCircleToGridRange(
        it.GetState().GetPose().translation(), hull_max_radius));
  }
  return result;
}

GridRange ComputeReachableGridRange(const GridFrame& grid_frame,
                                    const diff_drive::Curve& curve,
                                    const Hull& hull) {
  const double hull_max_radius =
      hull.GetMaxRadiusAround(eigenmath::Vector2d::Zero());
  auto it = curve.BeginInCordLength();
  auto it_end = curve.EndInCordLength();
  GridRange result;
  for (; it < it_end; it += grid_frame.resolution) {
    result.SpanningUnion(grid_frame.FrameCircleToGridRange(
        it.GetPoint().GetPose().translation(), hull_max_radius));
  }
  return result;
}

GridRange ComputeReachableGridRange(const GridFrame& grid_frame,
                                    const Hull& hull) {
  GridRange result;
  for (auto& chull : hull.GetConvexHulls()) {
    for (auto& pt : chull.GetPoints()) {
      result.GrowToInclude(grid_frame.FrameToGrid(pt));
    }
  }
  return result;
}

bool IsPointInsideOccupiedRegion(const OccupancyGrid& grid, GridIndex index,
                                 bool treat_unknown_as_obstacle) {
  if (!grid.Range().Contains(
          GridRange(index - GridIndex(1, 1), index + GridIndex(2, 2)))) {
    return false;
  }
  const GridIndex neighbors[] = {
      index, index + GridIndex(-1, 0), index + GridIndex(1, 0),
      index + GridIndex(0, -1), index + GridIndex(0, 1)};
  if (treat_unknown_as_obstacle) {
    return absl::c_all_of(neighbors, [&](const GridIndex& i) {
      return grid.GetUnsafe(i) != OccupancyStatus::UNOCCUPIED;
    });
  } else {
    return absl::c_all_of(neighbors, [&](const GridIndex& i) {
      return grid.GetUnsafe(i) == OccupancyStatus::OCCUPIED;
    });
  }
}

bool IsPointAnObstacle(const OccupancyGrid& grid, GridIndex index,
                       OccupancyStatus occupancy,
                       bool treat_unknown_as_obstacle,
                       bool ignore_internal_points) {
  return (
      (occupancy == OccupancyStatus::OCCUPIED ||
       (treat_unknown_as_obstacle && occupancy == OccupancyStatus::UNKNOWN)) &&
      (!ignore_internal_points ||
       !IsPointInsideOccupiedRegion(grid, index, treat_unknown_as_obstacle)));
}

bool IsPointInsideMaxCostRegion(const Grid<double>& cost_grid,
                                const GridIndex& index,
                                double max_cost_threshold) {
  if (!cost_grid.Range().Contains(
          GridRange(index - GridIndex(1, 1), index + GridIndex(2, 2)))) {
    return false;
  }
  const GridIndex neighbors[] = {
      index, index + GridIndex(-1, 0), index + GridIndex(1, 0),
      index + GridIndex(0, -1), index + GridIndex(0, 1)};
  return absl::c_all_of(neighbors, [&](const GridIndex& i) {
    return cost_grid.GetUnsafe(i) >= max_cost_threshold;
  });
}

bool CurveTraceContainsPoint(const diff_drive::Curve& curve, const Hull& hull,
                             double max_resolution,
                             const eigenmath::Vector2d& query_point) {
  auto containment_test = [&hull](const eigenmath::Vector2d& relative_point) {
    return hull.Contains(relative_point);
  };
  if (max_resolution > 0.0) {
    diff_drive::Curve resampled_curve(curve.GetCordLengthSpan().Length() /
                                          max_resolution +
                                      2 * curve.GetSize());
    diff_drive::ResampleCurve(curve, max_resolution, &resampled_curve);
    return VisitKnotPointsOutFromClosestPoint(resampled_curve, query_point,
                                              containment_test);
  } else {
    return VisitKnotPointsOutFromClosestPoint(curve, query_point,
                                              containment_test);
  }
}

void AddHullContourToOccupancyGrid(const Hull& hull,
                                   const eigenmath::Pose2d& pose,
                                   OccupancyGrid* grid,
                                   GridRange* actual_range) {
  for (auto& chull : hull.GetConvexHulls()) {
    grid->GrowToInclude(grid->Frame().FrameCircleToGridRange(
        pose * chull.GetCentroid(), chull.GetRadius()));
  }
  AddHullContourToOccupancyGridImpl(hull, pose, grid, actual_range);
}

void CollectOccupiedPointsOnHullContour(
    const Hull& hull, const eigenmath::Pose2d& pose,
    const GridFrame& grid_frame, const GridRange& grid_range,
    std::vector<eigenmath::Vector2d>* grid_points) {
  ForAllGridCellsOnContourImpl(
      hull, pose, grid_frame,
      [grid_frame, grid_range, grid_points](const GridIndex& cell) {
        if (grid_range.Contains(cell)) {
          grid_points->push_back(grid_frame.GridToFrame(cell));
        }
      });
}

void CollectOccupiedGridCellsFromHull(const Hull& hull,
                                      const eigenmath::Pose2d& pose,
                                      const GridFrame& grid_frame,
                                      const GridRange& grid_range,
                                      std::vector<GridIndex>* grid_indices) {
  ForAllGridCellsInHullImpl(
      hull, pose, grid_frame, grid_range,
      [grid_indices](const GridIndex& cell) { grid_indices->push_back(cell); });
}

void FillHullInOccupancyGrid(const Hull& hull, const eigenmath::Pose2d& pose,
                             OccupancyStatus occupancy, OccupancyGrid* grid) {
  ForAllGridCellsInHullImpl(hull, pose, grid->Frame(), grid->Range(),
                            [occupancy, grid](const GridIndex& cell) {
                              grid->SetUnsafe(cell, occupancy);
                            });
}

double DistanceIfPenetratingCurveTrace(const diff_drive::Curve& curve,
                                       const Hull& hull, double max_resolution,
                                       const eigenmath::Vector2d& query_point,
                                       double min_distance) {
  double min_pen_dist = 1.0;
  auto distance_accum = [&hull, &min_pen_dist, min_distance](
                            const eigenmath::Vector2d& relative_point) {
    const double pen_dist =
        hull.DistanceIfLessThan(relative_point, std::min(0.0, min_pen_dist));
    min_pen_dist = std::min(pen_dist, min_pen_dist);
    return (min_pen_dist <= min_distance);
  };
  if (max_resolution > 0.0) {
    diff_drive::Curve resampled_curve(
        std::lrint(
            std::ceil(curve.GetCordLengthSpan().Length() / max_resolution)) +
        2 * curve.GetSize());
    diff_drive::ResampleCurve(curve, max_resolution, &resampled_curve);
    VisitKnotPointsOutFromClosestPoint(resampled_curve, query_point,
                                       distance_accum);
  } else {
    VisitKnotPointsOutFromClosestPoint(curve, query_point, distance_accum);
  }
  return min_pen_dist;
}

void FillOccupancyGridWithCurveTrace(const diff_drive::Curve& curve,
                                     const Hull& hull, OccupancyGrid* grid,
                                     GridRange* actual_range) {
  grid->SetDefaultValue(OccupancyStatus::UNKNOWN);
  GridRange curve_footprint =
      ComputeReachableGridRange(grid->Frame(), curve, hull);
  grid->GrowToInclude(curve_footprint);
  const double ds = grid->Frame().resolution * 0.25;
  auto curve_it = curve.BeginInCordLength();
  while (curve_it < curve.EndInCordLength()) {
    const diff_drive::CurvePoint curve_pt = curve_it.GetPoint();
    AddHullContourToOccupancyGridImpl(hull, curve_pt.GetPose(), grid,
                                      actual_range);
    curve_it += ds;
  }
  const diff_drive::CurvePoint curve_pt = curve.GetFinish().point;
  AddHullContourToOccupancyGridImpl(hull, curve_pt.GetPose(), grid,
                                    actual_range);
}

void FillCostGridWithCurveTrace(const diff_drive::Curve& curve,
                                const Hull& hull,
                                std::function<double(double)> distance_to_cost,
                                Grid<double>* grid, double min_distance) {
  grid->SetDefaultValue(1.0);
  GridRange curve_footprint =
      ComputeReachableGridRange(grid->Frame(), curve, hull);
  grid->GrowToInclude(curve_footprint);
  diff_drive::Curve resampled_curve(
      std::lrint(std::ceil(curve.GetCordLengthSpan().Length() /
                           grid->Frame().resolution)) *
          2 +
      2 * curve.GetSize());
  diff_drive::ResampleCurve(curve, 0.5 * grid->Frame().resolution,
                            &resampled_curve);
  curve_footprint.ForEachGridCoord([&](const GridIndex& index) {
    const double new_cost = distance_to_cost(DistanceIfPenetratingCurveTrace(
        resampled_curve, hull, -1.0, grid->Frame().GridToFrame(index),
        min_distance));
    if (new_cost > grid->GetUnsafe(index)) {
      grid->SetUnsafe(index, new_cost);
    }
  });
}

}  // namespace mobility::collision

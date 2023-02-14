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
#include <array>
#include <iterator>
#include <optional>
#include <queue>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "collision/collision_utils.h"
#include "collision/grid_common.h"
#include "collision/occupancy_grid.h"
#include "eigenmath/line_utils.h"
#include "eigenmath/vector_utils.h"
#include "genit/circular_iterator.h"

namespace mobility::collision {

namespace {

// Cell steps in counter-clockwise order.
const GridIndex kCellSteps[] = {{-1, -1}, {0, -1}, {1, -1}, {1, 0},
                                {1, 1},   {0, 1},  {-1, 1}, {-1, 0}};
constexpr int kSteps = sizeof(kCellSteps) / sizeof(kCellSteps[0]);

// Returns the index of a step in the array kCellSteps.
int IndexOfStep(GridIndex step) {
  QCHECK(step.lpNorm<Eigen::Infinity>() == 1);
  constexpr int index_map[] = {0, 1, 2, 7, -1, 3, 6, 5, 4};
  return index_map[(step.y() + 1) * 3 + (step.x() + 1)];
}

// For an occupied point, see if it lies inside a straight obstacle boundary,
// and not on any other boundaries.
bool HasOnlyTwoOppositeConnections(const OccupancyGrid& grid,
                                   const GridIndex point,
                                   bool treat_unknown_as_obstacle) {
  // Maps step index to boundary connection.
  static_assert(kSteps % 2 == 0,
                "Not all steps come with an opposite direction.");

  std::array<bool, kSteps> connections;
  for (int i = 0; i < kSteps; ++i) {
    const GridIndex other = point + kCellSteps[i];
    connections[i] =
        grid.Range().Contains(other) &&
        IsPointAnObstacle(grid, other, treat_unknown_as_obstacle) &&
        IsBoundary(grid, point, other, treat_unknown_as_obstacle);
  }
  std::array<bool, kSteps / 2> opposites;
  for (int i = 0; i < kSteps / 2; ++i) {
    opposites[i] = connections[i] && connections[i + kSteps / 2];
  }
  auto is_true = [](bool b) { return b; };
  return (std::count(connections.begin(), connections.end(), true) == 2) &&
         std::any_of(opposites.begin(), opposites.end(), is_true);
}

// Uses breadth first search, starting at an obstacle's extreme point
// boundary_cell.  Marks all boundary points of the corresponding obstacle as
// visited.
template <typename SegmentFunctor>
void AggregateObstacleContour(const OccupancyGrid& grid,
                              const GridIndex boundary_cell,
                              GridIndexSet* visited, SegmentFunctor add_segment,
                              bool treat_unknown_as_obstacle) {
  std::queue<GridIndex> unvisited{{boundary_cell}};

  while (!unvisited.empty()) {
    const GridIndex from = unvisited.front();
    unvisited.pop();
    if (!visited->insert(from).second) continue;

    for (const GridIndex& direction : kCellSteps) {
      GridIndex to = from + direction;
      if (!grid.Range().Contains(to) ||
          !IsPointAnObstacle(grid, to, treat_unknown_as_obstacle,
                             /*ignore_internal_points=*/false) ||
          !IsBoundary(grid, from, to, treat_unknown_as_obstacle)) {
        continue;
      }
      // Advance point while it does not participate in branching.
      while (
          HasOnlyTwoOppositeConnections(grid, to, treat_unknown_as_obstacle)) {
        // Do not add point to queue, since all cell segments through it are
        // covered by this enumeration.
        visited->insert(to);
        // Avoid garbage during possibly non-vectorized assignment.
        to = GridIndex{to + direction};
      }
      if (!visited->contains(to)) {
        add_segment({from, to});
        unvisited.push(to);
      }
    }
  }
}

// Checks whether the cell next can be used as the new endpoint of the segment.
// For a zero-length segment, any value for next is allowed.
bool IsColinearToSegment(const GridSegment& segment, GridIndex next) {
  // Check if next is on the same line as segment.
  const GridIndex direction = segment.to - segment.from;
  const GridIndex step = next - segment.to;
  QCHECK(step.lpNorm<Eigen::Infinity>() > 0);
  const int multiple =
      direction.lpNorm<Eigen::Infinity>() / step.lpNorm<Eigen::Infinity>();
  return (direction == multiple * step);
}

// Various simplification strategies.  A simplification strategy is used to
// traverse a single boundary, and is expected to write the simplified boundary
// to the output iterator.
using SimplificationOutputIt =
    std::back_insert_iterator<std::vector<GridSegment>>;

// Simplifies straight lines.
class ColinearSimplification {
 public:
  explicit ColinearSimplification(SimplificationOutputIt&& out) : out_(out) {}
  ~ColinearSimplification() {
    QCHECK(first_.has_value());
    // Write the last segment
    aggregate_.to = first_.value();
    *out_++ = aggregate_;
  }

  void Consume(GridIndex next) {
    if (!first_.has_value()) {
      first_ = next;
      aggregate_ = {first_.value(), first_.value()};
    } else if (IsColinearToSegment(aggregate_, next)) {
      aggregate_.to = next;
    } else {
      QCHECK(aggregate_.from != aggregate_.to) << aggregate_;
      *out_++ = aggregate_;
      aggregate_ = {aggregate_.to, next};
    }
  }

 private:
  SimplificationOutputIt out_;
  std::optional<GridIndex> first_;
  GridSegment aggregate_;
};

// Simplifies stepwise, assuming the Bresenham line at every step is on the
// boundary.
class StepwiseBresenhamLineSimplification {
 public:
  explicit StepwiseBresenhamLineSimplification(SimplificationOutputIt&& out)
      : out_(out) {}
  ~StepwiseBresenhamLineSimplification() {
    QCHECK(first_.has_value());
    // Write the last segment
    aggregate_.to = first_.value();
    *out_++ = aggregate_;
  }

  void Consume(GridIndex next) {
    if (!first_.has_value()) {
      first_ = next;
      aggregate_ = {first_.value(), first_.value()};
    } else if (ExtensionIsOnBoundary(aggregate_, next)) {
      aggregate_.to = next;
    } else {
      QCHECK(aggregate_.from != aggregate_.to) << aggregate_;
      *out_++ = aggregate_;
      aggregate_ = {aggregate_.to, next};
    }
  }

 private:
  SimplificationOutputIt out_;
  std::optional<GridIndex> first_;
  GridSegment aggregate_;
};

// Simplifies the boundary to use a small number of Bresenham lines.  Aggregates
// all points and simplifies on destruction.
//
// The strategy is a greedy simplification.
class BresenhamLineSimplification {
 public:
  explicit BresenhamLineSimplification(SimplificationOutputIt&& out)
      : out_(out) {}
  ~BresenhamLineSimplification() {
    if (ccw_points_.size() < 2) return;

    // Algorithm implementation.
    const auto circular_range = genit::CircularRange(ccw_points_);
    auto it = circular_range.begin();
    auto start = it++;
    auto end = it++;
    // Keep track of the possible Bresenham line slope range for points from
    // start to it - 1, and the longest valid Bresenham line so far.
    auto slope_range = SlopeRange(*end - *start);
    while (it != circular_range.end() + 1) {
      slope_range = Intersect(slope_range, SlopeRange(*it - *start));
      const bool changed_direction =
          (*end - *start).dot(*it - *std::prev(it)) < 0;
      if (changed_direction || slope_range.Empty()) {
        // Commit to last line segment, as it cannot be extended.
        *out_++ = GridSegment{*start, *end};
        start = end++;
        it = end;
        slope_range = SlopeRange(*end - *start);
      } else if (slope_range.Contains(*it - *start)) {
        // Extend current segment.
        end = it;
      }
      ++it;
    }
    *out_++ = GridSegment{*start, *end};
  }

  void Consume(GridIndex next) { ccw_points_.push_back(next); }

 private:
  // Uses a relative GridIndex as a slope of a line.

  // Expresses an R^2 convex cone which is assumed to be smaller than a
  // half-plane. See https://en.wikipedia.org/wiki/Convex_cone
  struct ConvexCone {
    GridIndex left;
    GridIndex right;

    bool Contains(const GridIndex& point) const {
      return eigenmath::RightOrthogonal(left).dot(point) >= 0 &&
             eigenmath::LeftOrthogonal(right).dot(point) >= 0;
    }
    bool Empty() const { return left == right; }
  };
  ConvexCone Intersect(const ConvexCone& lhs, const ConvexCone& rhs) {
    ConvexCone intersection{{0, 0}, {0, 0}};
    // Check for empty intersection.
    if (eigenmath::RightOrthogonal(lhs.left).dot(rhs.right) < 0 ||
        eigenmath::LeftOrthogonal(lhs.right).dot(rhs.left) < 0) {
      return intersection;
    }
    intersection.left = lhs.Contains(rhs.left) ? rhs.left : lhs.left;
    intersection.right = lhs.Contains(rhs.right) ? rhs.right : lhs.right;
    return intersection;  // NRVO
  }

  // A Bresenham from (0,0) goes through point if it passes a check against two
  // boundaries.  This function returns the direction towards the boundary of
  // point's grid cell along orth.
  GridIndex GetDirectionTowardBresenhamBoundary(const GridIndex& point,
                                                const GridIndex& orth) {
    if (orth.lpNorm<1>() != 2 * orth.lpNorm<Eigen::Infinity>()) {
      return orth / orth.lpNorm<Eigen::Infinity>();
    }
    QCHECK_NE(orth.x(), 0);
    QCHECK_NE(orth.y(), 0);
    // Signum can skip check for x == 0.
    const auto signum = [](int x) -> int { return (x > 0) ? 1 : -1; };
    const GridIndex x_axis = {signum(orth.x()), 0};
    const GridIndex y_axis = {0, signum(orth.y())};
    if (x_axis.dot(point) < 0) {
      return x_axis;
    } else {
      return y_axis;
    }
  }
  // Returns the slope range of Bresenham lines from (0,0) to point as a
  // convex cone.
  ConvexCone SlopeRange(const GridIndex& point) {
    const GridIndex kZero{0, 0};
    if (point == kZero) {
      return {kZero, kZero};
    }
    // Take a step half the grid size.  Calculate step using the orthogonals.
    return {2 * point + GetDirectionTowardBresenhamBoundary(
                            point, eigenmath::LeftOrthogonal(point)),
            2 * point + GetDirectionTowardBresenhamBoundary(
                            point, eigenmath::RightOrthogonal(point))};
  }

  SimplificationOutputIt out_;
  std::vector<GridIndex> ccw_points_;
};

// Walks along a boundary of an obstacle, and collects all segments which
// describe this boundary.  The boundary direction is defined by the
// predecessor point and the traversal is in counter-clockwise direction.
// Marks all segment parts as visited, and call sink on each element on the
// boundary once, starting at `cur`.
template <typename NextGridpointSink, typename VisitFunctor>
void AggregateContourAlongObstacleBoundary(const OccupancyGrid& grid,
                                           GridIndex cur, GridIndex prev,
                                           VisitFunctor visit_and_new,
                                           NextGridpointSink* sink,
                                           bool treat_unknown_as_obstacle) {
  sink->Consume(cur);
  while (visit_and_new(cur, prev)) {
    const GridIndex next =
        NextPointOnBoundary(grid, prev, cur, treat_unknown_as_obstacle);
    sink->Consume(next);

    prev = cur;
    cur = next;
  }
}

// Keeps a boundary point's predecessors for a boundary traversal.
class Predecessors {
 public:
  // Adds pred to the predecessors.
  // Returns whether pred is a new predecessor.
  bool AddPredecessor(GridIndex pred) {
    if (!Contains(pred)) {
      storage_.push_back(pred);
      return true;
    } else {
      return false;
    }
  }

  bool Contains(GridIndex point) {
    return std::any_of(storage_.cbegin(), storage_.cend(),
                       [point](GridIndex pred) { return pred == point; });
  }

 private:
  absl::InlinedVector<GridIndex, 4> storage_;
};

// For an obstacle boundary point, returns the set of predecessor points for
// counter-clockwise traversal along the boundaries.
absl::InlinedVector<GridIndex, 4> GetPredecessors(
    const OccupancyGrid& grid, GridIndex point,
    bool treat_unknown_as_obstacle) {
  absl::InlinedVector<GridIndex, 4> prevs;
  for (const GridIndex& step : kCellSteps) {
    GridIndex prev = point + step;
    if (grid.Range().Contains(prev) &&
        IsPointAnObstacle(grid, prev, treat_unknown_as_obstacle,
                          /*ignore_internal_points=*/false) &&
        IsCounterClockwiseBoundary(grid, prev, point,
                                   treat_unknown_as_obstacle)) {
      prevs.push_back(prev);
    }
  }

  return prevs;  // NRVO
}

// Iterates through all grid points and starts a counter-clockwise walk along
// the boundary at each unvisited obstacle boundary point.  Starts a search in
// both directions if the point has boundaries to two sides.
//
// Creates a new simplification object per boundary which receives the
// boundary points in counter-clockwise order.  The simplification strategy is
// assumed to receive a sink on construction.
template <typename SimplificationStrategy>
std::vector<GridSegment> ExtractSimplifiedBoundaryLineSegments(
    const OccupancyGrid& grid, bool treat_unknown_as_obstacle) {
  // Visited boundary points, and the predecessors indication the traversal
  // direction for which the point has been visited.
  absl::flat_hash_map<GridIndex, Predecessors, GridIndexHash> visited;

  // Marks the point as being visited from point prev.  Returns whether the
  // pair was visited for the first time.
  auto visit_and_new = [&](GridIndex point, GridIndex prev) {
    auto& preds = visited[point];
    return preds.AddPredecessor(prev);
  };

  std::vector<GridSegment> segments;
  grid.Range().ForEachGridCoord([&](GridIndex point) {
    if (IsPointAnObstacle(grid, point, treat_unknown_as_obstacle,
                          /*ignore_internal_points=*/true)) {
      for (const GridIndex& pred :
           GetPredecessors(grid, point, treat_unknown_as_obstacle)) {
        if (!visited[point].Contains(pred)) {
          SimplificationStrategy boundary_eater{std::back_inserter(segments)};
          AggregateContourAlongObstacleBoundary(grid, point, pred,
                                                visit_and_new, &boundary_eater,
                                                treat_unknown_as_obstacle);
        }
      }
    }
  });

  return segments;  // NRVO
}

// Calculates the intersection between segment and bounding box boundaries,
// and adds these to the intersections array.  By convention, the ordering of
// intersections is:  lower, right, upper, left.
//
// An intersection is ordered, with the lower coordinate point as its `from`
// point.  This aids sorting the segments.
void AppendIntersections(const GridRange& bbox, const GridSegment& segment,
                         const int boundary_index,
                         std::vector<Interval<int>>* intersections) {
  const GridIndex bottom_left = bbox.lower;
  const GridIndex top_right = bbox.upper - GridIndex{1, 1};
  // Find bounding box enclosure of the segment.
  const int min_x = std::min(segment.from.x(), segment.to.x());
  const int min_y = std::min(segment.from.y(), segment.to.y());
  const int max_x = std::max(segment.from.x(), segment.to.x());
  const int max_y = std::max(segment.from.y(), segment.to.y());
  // Note -- if intersection is a segment, the enclosure is degenerate.

  // Prepare test values for boundary order lower, right, upper, left.
  const int bbox_coords[] = {bottom_left.y(), top_right.x(), top_right.y(),
                             bottom_left.x()};
  const int enclosure_coords[] = {min_y, max_x, max_y, min_x};
  const int i = boundary_index;
  // Test if an endpoint lies on the boundary.
  if (enclosure_coords[i] == bbox_coords[i]) {
    // Test if the segment is colinear with the boundary.
    if (enclosure_coords[i] == enclosure_coords[(i + 2) % 4]) {
      const int values[] = {enclosure_coords[(i + 1) % 4],
                            enclosure_coords[(i + 3) % 4]};
      (*intersections)
          .emplace_back(std::min(values[0], values[1]),
                        std::max(values[0], values[1]));
    } else {
      const int test_axis = (i + 1) % 2;
      const int value_axis = i % 2;
      const int value = (segment.from[test_axis] == bbox_coords[i])
                            ? segment.from[value_axis]
                            : segment.to[value_axis];
      (*intersections).emplace_back(value, value);
    }
  }
}

}  // namespace

std::vector<GridSegment> MissingBoundarySegments(
    const GridRange& bounding_box,
    const std::vector<GridSegment>& line_segments) {
  // Get intersections of segments with bounding box.  Boundaries are ordered
  // lower, right, upper, left.

  const GridIndex bottom_left = bounding_box.lower;
  const GridIndex top_right = bounding_box.upper - GridIndex{1, 1};
  const int bbox_coords[] = {bottom_left.y(), top_right.x(), top_right.y(),
                             bottom_left.x()};

  // Add intervals on the boundary which are not part of the intersections.
  std::vector<eigenmath::LineSegment2<int>> missing;
  for (int i = 0; i < 4; ++i) {
    std::vector<Interval<int>> complement;
    for (const auto& segment : line_segments) {
      AppendIntersections(bounding_box, segment, i, &complement);
    }

    // Sort intersections.  The default interval comparison has reverse ordering
    // in the end point.
    std::sort(complement.begin(), complement.end());
    complement.emplace_back(top_right[i % 2], top_right[i % 2]);

    // Find the intervals which are not excluded by the complement.
    const int other_coord = bbox_coords[i];
    int next_min = bottom_left[i % 2];
    for (const auto& exclusive : complement) {
      const Interval<int> inclusive{next_min, exclusive.min()};
      next_min = std::max(next_min, exclusive.max());
      if (!inclusive.Empty()) {
        if (i % 2 == 0) {
          missing.emplace_back(GridIndex{inclusive.min(), other_coord},
                               GridIndex{inclusive.max(), other_coord});
        } else {
          missing.emplace_back(GridIndex{other_coord, inclusive.min()},
                               GridIndex{other_coord, inclusive.max()});
        }
      }
    }
  }
  return missing;  // NRVO
}

bool IsBoundary(const OccupancyGrid& grid, GridIndex from, GridIndex to,
                bool treat_unknown_as_obstacle) {
  QCHECK(grid.Range().Contains(from));
  QCHECK(grid.Range().Contains(to));
  QCHECK(IsPointAnObstacle(grid, from, treat_unknown_as_obstacle));
  QCHECK(IsPointAnObstacle(grid, to, treat_unknown_as_obstacle));
  QCHECK((from - to).lpNorm<Eigen::Infinity>() == 1);

  auto is_free_space = [&grid,
                        treat_unknown_as_obstacle](const GridIndex point) {
    return !grid.Range().Contains(point) ||
           !IsPointAnObstacle(grid, point, treat_unknown_as_obstacle,
                              /*ignore_internal_points=*/false);
  };
  const GridIndex delta = to - from;
  if (delta.lpNorm<1>() == 1) {
    // Horizontal or vertical.
    const GridIndex offset = eigenmath::LeftOrthogonal(delta);
    return (is_free_space(from + offset) && is_free_space(to + offset)) ||
           (is_free_space(from - offset) && is_free_space(to - offset));
  } else if (delta.lpNorm<1>() == 2) {
    // Diagonal.
    const GridIndex neighbour_cell = from + GridIndex{delta.x(), 0};
    const GridIndex opposite_neighbour = from + GridIndex{0, delta.y()};
    return is_free_space(neighbour_cell) || is_free_space(opposite_neighbour);
  } else {
    QCHECK(false) << "Had difference\n" << delta;
    return false;
  }
}

bool IsCounterClockwiseBoundary(const OccupancyGrid& grid, GridIndex from,
                                GridIndex to, bool treat_unknown_as_obstacle) {
  QCHECK(grid.Range().Contains(from));
  QCHECK(grid.Range().Contains(to));
  QCHECK(IsPointAnObstacle(grid, from, treat_unknown_as_obstacle));
  QCHECK(IsPointAnObstacle(grid, to, treat_unknown_as_obstacle));
  QCHECK((from - to).lpNorm<Eigen::Infinity>() == 1);

  auto is_free_space = [&grid,
                        treat_unknown_as_obstacle](const GridIndex point) {
    return !grid.Range().Contains(point) ||
           !IsPointAnObstacle(grid, point, treat_unknown_as_obstacle,
                              /*ignore_internal_points=*/false);
  };
  const GridIndex delta = to - from;
  const GridIndex turned_delta = eigenmath::LeftOrthogonal(delta);
  if (delta.lpNorm<1>() == 1) {
    // Horizontal or vertical.
    const GridIndex offset = turned_delta;
    return is_free_space(from + offset) && is_free_space(to + offset);
  } else if (delta.lpNorm<1>() == 2) {
    // Diagonal.
    const GridIndex offset = (delta + turned_delta) / 2;
    return is_free_space(from + offset);
  } else {
    QCHECK(false) << "Had difference\n" << delta;
    return false;
  }
}

GridIndex NextPointOnBoundary(const OccupancyGrid& grid, GridIndex from,
                              GridIndex to, bool treat_unknown_as_obstacle) {
  QCHECK(grid.Range().Contains(from));
  QCHECK(grid.Range().Contains(to));
  QCHECK(IsPointAnObstacle(grid, from, treat_unknown_as_obstacle));
  QCHECK(IsPointAnObstacle(grid, to, treat_unknown_as_obstacle));
  QCHECK((from - to).lpNorm<Eigen::Infinity>() == 1);

  // From cell `to`, step two cells further than `from` in counter-clockwise
  // order and find the next occupied cell.
  auto step_to_from =
      genit::CircularRange(std::begin(kCellSteps), std::end(kCellSteps))
          .begin() +
      IndexOfStep(from - to);
  auto next_step = std::find_if(
      step_to_from + 2, step_to_from + kSteps, [&](GridIndex step) {
        const GridIndex next = to + step;
        return grid.Range().Contains(next) &&
               IsPointAnObstacle(grid, next, treat_unknown_as_obstacle,
                                 /*ignore_internal_points=*/false);
      });
  return to + *next_step;
}

// Checks whether the segment  [segment.from, next]  is part of the obstacle
// boundary.  Uses GridLine, i.e. Bresenham lines.
bool ExtensionIsOnBoundary(const GridSegment& segment, GridIndex next) {
  // Ensure that all points occupied by segment are also occupied by the
  // extended segment.
  GridLine prev_points{segment.from, segment.to};
  GridLine extended_points{segment.from, next};
  if (extended_points.end() - extended_points.begin() !=
      1 + (prev_points.end() - prev_points.begin())) {
    return false;
  }
  for (auto prev_it = prev_points.begin(), ext_it = extended_points.begin();
       prev_it != prev_points.end(); ++prev_it, ++ext_it) {
    if (*prev_it != *ext_it) {
      return false;
    }
  }
  return true;
}

// Iterates through all grid points and starts a breadth first search at each
// unvisited obstacle boundary point.
std::vector<GridSegment> ExtractBoundaryLineSegments(
    const OccupancyGrid& grid, bool treat_unknown_as_obstacle) {
  GridIndexSet visited;
  std::vector<GridSegment> cell_segments;
  auto is_unvisited_obstacle_boundary = [&](GridIndex index) {
    return IsPointAnObstacle(grid, index, treat_unknown_as_obstacle,
                             /*ignore_internal_points=*/true) &&
           (visited.count(index) == 0);
  };

  auto add_segment = [&](const GridSegment& segment) {
    cell_segments.push_back(segment);
  };

  // Iterate over unvisited obstacle boundary points, and aggregate the line
  // segments for the corresponding obstacle.
  for (const GridIndex& index : grid.Range()) {
    if (is_unvisited_obstacle_boundary(index)) {
      // Walk along contour and aggregate line segments.
      AggregateObstacleContour(grid, index, &visited, add_segment,
                               treat_unknown_as_obstacle);
    }
  }

  return cell_segments;  // NRVO
}

std::vector<GridSegment> ExtractBoundaryPolygonSegments(
    const OccupancyGrid& grid, bool treat_unknown_as_obstacle) {
  return ExtractSimplifiedBoundaryLineSegments<ColinearSimplification>(
      grid, treat_unknown_as_obstacle);
}

std::vector<GridSegment> ExtractSimplifiedPolygonSegments(
    const OccupancyGrid& grid, bool treat_unknown_as_obstacle) {
  return ExtractSimplifiedBoundaryLineSegments<BresenhamLineSimplification>(
      grid, treat_unknown_as_obstacle);
}

void GetAllOccupiedInFirstButNotInOther(const OccupancyGrid& first_grid,
                                        bool treat_unknown_in_first_as_occupied,
                                        const OccupancyGrid& other_grid,
                                        bool treat_unknown_in_other_as_occupied,
                                        OccupancyGrid* result_grid) {
  CHECK_EQ(first_grid.FrameId(), other_grid.FrameId());
  result_grid->Frame() = first_grid.Frame();
  result_grid->SetDefaultValue(OccupancyStatus::UNOCCUPIED);
  result_grid->Reshape(first_grid.Range());
  result_grid->Fill(OccupancyStatus::UNOCCUPIED);
  first_grid.Range().ForEachGridCoord([&](const GridIndex index_in_first) {
    if (!IsPointAnObstacle(first_grid, index_in_first,
                           treat_unknown_in_first_as_occupied)) {
      return;
    }
    result_grid->SetUnsafe(index_in_first, OccupancyStatus::OCCUPIED);
    const GridRange range_in_other = GridFrame::GridToGridInclusive(
        first_grid.Frame(), index_in_first, other_grid.Frame());
    range_in_other.ForEachGridCoord([&](const GridIndex& index_in_other) {
      if ((GridFrame::GridToGrid(other_grid.Frame(), index_in_other,
                                 first_grid.Frame()) == index_in_first) &&
          other_grid.Range().Contains(index_in_other) &&
          IsPointAnObstacle(other_grid, index_in_other,
                            treat_unknown_in_other_as_occupied)) {
        result_grid->SetUnsafe(index_in_first, OccupancyStatus::UNOCCUPIED);
        return false;  // Break.
      }
      return true;  // Continue.
    });
  });
}

}  // namespace mobility::collision

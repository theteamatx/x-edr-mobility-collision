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

#ifndef MOBILITY_COLLISION_COLLISION_GRID_COMMON_H_
#define MOBILITY_COLLISION_COLLISION_GRID_COMMON_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <iosfwd>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "eigenmath/line_utils.h"
#include "eigenmath/types.h"
#include "genit/iterator_facade.h"

namespace mobility::collision {

// Notice also the GridIndexLessThan and GridIndexHash functions, to use
// GridIndex as key in standard containers.
using GridIndex = eigenmath::Vector2<int>;

// Hash function for GridIndex. We do not specialize std::hash
// (or GoodFastHash) because GridIndex is really Eigen::Matrix which we
// should not interfere with (in case the Eigen library adds std::hash in
// the future).
struct GridIndexHash {
  size_t operator()(const GridIndex &index) const {
    return absl::HashOf(0, index.x(), index.y());
  }
};

// Less-than function for GridIndex. We do not specialize std::less
// because GridIndex is really Eigen::Matrix which we should not interfere
// with (in case the Eigen library adds std::less in the future).
struct GridIndexLessThan {
  bool operator()(const GridIndex &lhs, const GridIndex &rhs) const {
    return lhs.x() < rhs.x() || (lhs.x() == rhs.x() && lhs.y() < rhs.y());
  }
};

// Compute the actual modulus function, where negative values become positive
// values in the range [0,rhs).
inline int StrictModulus(int lhs, int rhs) {
  DCHECK_GT(rhs, 0);

  // According to the benchmark, the implementation is faster than the
  // non-branching
  //   return (lhs % rhs + rhs) % rhs;
  int out = lhs % rhs;
  if (out < 0) {
    out += rhs;
  }
  return out;
}

// Grid move increments in x and y can be mapped into a flat index.
// The following functions can be used to obtain the index of a
// move, the move from an index, and the compile-time array of move
// lengths by index.

inline std::array<GridIndex, 8> GridEightNeighborhood() {
  return {GridIndex{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0},
          {-1, 1},           {0, 1},  {1, 1}};
}

inline constexpr std::array<double, 8> GridEightNeighborNorms() {
  return {M_SQRT2, 1.0, M_SQRT2, 1.0, 1.0, M_SQRT2, 1.0, M_SQRT2};
}

inline int EightNeighborMoveAsFlatIndex(GridIndex move_xy) {
  // Grid move increments in x and y into a flat index:
  // 0: (-1,-1), 1: (0,-1), 2: (1,-1), 3: (-1,0), 4: (1,0), 5: (-1,1),
  // 6: (0,1), 7: (1,1)
  int move_index = move_xy.x() + 3 * move_xy.y() + 4;
  return move_index - move_index / 5;
}

using GridIndexSet = ::absl::flat_hash_set<GridIndex, GridIndexHash>;

template <typename T>
using GridIndexMap = ::absl::flat_hash_map<GridIndex, T, GridIndexHash>;

template <typename T>
using GridIndexNodeMap = ::absl::node_hash_map<GridIndex, T, GridIndexHash>;

// A range in a grid is a bounding box over indices. The lower bounds
// are inclusive, and upper bounds exclusive. E.g. the range {{1, 2},
// {3, 4}} denotes X-indices 1 and 2, and Y-indices 2 and 3.
struct GridRange {
  GridRange() : lower({0, 0}), upper({0, 0}) {}

  GridRange(const GridIndex &ll, const GridIndex &uu) : lower(ll), upper(uu) {}

  explicit GridRange(const GridIndex &pt)
      : lower(pt), upper(pt + GridIndex{1, 1}) {}

  // Returns the "practically" unlimited range. Indices containing a
  // std::numeric_limits<int>::max() will still be considered outside
  // this range.
  static GridRange Unlimited() {
    return GridRange{
        {std::numeric_limits<int>::min(), std::numeric_limits<int>::min()},
        {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()}};
  }

  // Returns a range that goes from the origin (0,0) to the given upper limit.
  // The upper bound is exclusive (one past last).
  static GridRange OriginTo(const GridIndex &upper) {
    return GridRange(GridIndex(0, 0), upper);
  }
  static GridRange OriginTo(int upper_x, int upper_y) {
    return GridRange(GridIndex(0, 0), GridIndex(upper_x, upper_y));
  }

  GridIndex::Scalar XSpan() const { return std::max(0, upper.x() - lower.x()); }

  GridIndex::Scalar YSpan() const { return std::max(0, upper.y() - lower.y()); }

  constexpr bool Contains(int x, int y) const {
    return x >= lower.x() && x < upper.x() && y >= lower.y() && y < upper.y();
  }

  constexpr bool Contains(const GridIndex &index) const {
    return Contains(index.x(), index.y());
  }

  bool Contains(const GridRange &other) const {
    return other.lower.x() >= lower.x() && other.upper.x() <= upper.x() &&
           other.lower.y() >= lower.y() && other.upper.y() <= upper.y();
  }

  GridIndex Clamp(const GridIndex &index) const {
    GridIndex result = index;
    if (result.x() >= upper.x()) {
      result.x() = upper.x() - 1;
    }
    if (result.x() < lower.x()) {
      result.x() = lower.x();
    }
    if (result.y() >= upper.y()) {
      result.y() = upper.y() - 1;
    }
    if (result.y() < lower.y()) {
      result.y() = lower.y();
    }
    return result;
  }

  bool Empty() const {
    return lower.x() >= upper.x() || lower.y() >= upper.y();
  }

  bool operator==(const GridRange &other) const {
    return lower == other.lower && upper == other.upper;
  }

  bool operator!=(const GridRange &other) const {
    return lower != other.lower || upper != other.upper;
  }

  // Grows the range sufficiently to contain the given grid-index.
  static GridRange GrowToInclude(const GridRange &range,
                                 const GridIndex &index);

  // Grows this range sufficiently to contain the given grid-index.
  void GrowToInclude(const GridIndex &index) {
    *this = GrowToInclude(*this, index);
  }

  // Grows the boundaries of the range by a given amount on each side.
  static GridRange GrowBy(const GridRange &range, int amount) {
    if (range.Empty()) {
      return range;  // Nothing to grow onto.
    }
    return GridRange({range.lower.x() - amount, range.lower.y() - amount},
                     {range.upper.x() + amount, range.upper.y() + amount});
  }

  // Grows the boundaries of this range by a given amount on each side.
  void GrowBy(int amount) { *this = GrowBy(*this, amount); }

  // Shrinks the boundaries of the range by a given amount on each side.
  static GridRange ShrinkBy(const GridRange &range, int amount) {
    if (range.Empty()) {
      return range;  // Nothing to shrink.
    }
    GridRange result;
    result.lower.x() = range.lower.x() + amount;
    result.upper.x() = std::max(result.lower.x(), range.upper.x() - amount);
    result.lower.y() = range.lower.y() + amount;
    result.upper.y() = std::max(result.lower.y(), range.upper.y() - amount);
    return result;
  }

  // Shrinks the boundaries of this range by a given amount on each side.
  void ShrinkBy(int amount) { *this = ShrinkBy(*this, amount); }

  // Translates the range by a given offset.
  static GridRange ShiftBy(const GridRange &range, GridIndex offset) {
    return GridRange{range.lower + offset, range.upper + offset};
  }

  // Translates this range by a given offset.
  void ShiftBy(GridIndex offset) { *this = ShiftBy(*this, offset); }

  // This class is used to put together at most 4 grid-ranges that represent
  // range decompositions that arise from some set operations below.
  class Quad;

  // Returns the spanning union of the given ranges. A spanning union is the
  // range that entirely contains all the given ranges, but might also contain
  // more space to span all the space in-between the given ranges.
  static GridRange SpanningUnion(const GridRange &aa, const GridRange &bb);

  // Same as above, for a vector of ranges to span.
  static GridRange SpanningUnion(const std::vector<GridRange> &ranges);

  // Same as above, for a quad of ranges to span.
  static GridRange SpanningUnion(const Quad &ranges);

  // Same as above, for any number of ranges to span.
  template <typename... Args>
  static GridRange SpanningUnion(const GridRange &aa, const GridRange &bb,
                                 const Args &...others) {
    return SpanningUnion(SpanningUnion(aa, bb), others...);
  }

  // Same as above, sets this range to the union of this range and the other.
  void SpanningUnion(const GridRange &other) {
    *this = SpanningUnion(*this, other);
  }

  // Returns the intersection of the given ranges.
  static GridRange Intersect(const GridRange &aa, const GridRange &bb);

  // Sets this range to the intersection of this range and the other.
  void Intersect(const GridRange &other) { *this = Intersect(*this, other); }

  // Checks if this range intersects the other.
  bool Intersects(const GridRange &other) const {
    return !Intersect(*this, other).Empty();
  }

  // Returns the complement of A, as up to four sub-ranges. The
  // returned ranges all have at least two "infinite" boundaries.
  // std::numeric_limits<int>::min() and max() are used as "infinity" here.
  static Quad Complement(const GridRange &aa);

  // Returns A\B, the set of indices that are in A but not in B, as
  // up to four sub-ranges. The returned quad is either empty, or contains
  // only non-empty ranges.
  static Quad Difference(const GridRange &aa, const GridRange &bb);

  // Returns the non-spanning union of the given ranges. A non-spanning union
  // is set of disjoint ranges that contain all the given ranges, but does not
  // span any space not contained in the given ranges.
  static Quad NonSpanningUnion(const GridRange &aa, const GridRange &bb);

  // Returns a vector of grid ranges that come from computing the intersection
  // of a given range with all the ranges in the given vector.
  // Equivalent to {Intersect(aa, bb[0]), ..., Intersect(aa, bb[N-1])}.
  static std::vector<GridRange> Intersect(const GridRange &aa,
                                          const std::vector<GridRange> &bb);

  // Returns a quad of grid ranges that come from computing the intersection
  // of a given range with all the ranges in the given quad.
  // Equivalent to {Intersect(aa, bb[0]), ..., Intersect(aa, bb[N-1])}.
  static Quad Intersect(const GridRange &aa, const Quad &bb);

  // Removes any self-intersections in the given vector of ranges such that
  // the resulting vector spans the same space as before, but without having
  // any overlap between ranges.
  static void RemoveSelfIntersections(std::vector<GridRange> *ranges);

  // Returns true of the given grid index is contained in one of the aa ranges.
  static bool Contains(const std::vector<GridRange> &aa,
                       const GridIndex &index);
  static bool Contains(const Quad &aa, const GridIndex &index);

  int ComputeSize() const {
    if (Empty()) {
      return 0;
    }
    return (upper.x() - lower.x()) * (upper.y() - lower.y());
  }

  class Iterator
      : public genit::IteratorFacade<Iterator, const GridIndex &,
                                     std::bidirectional_iterator_tag> {
   public:
    Iterator(const GridRange &range, GridIndex current)
        : lower_x_(range.lower.x()),
          upper_x_(range.upper.x() - 1),
          current_(current) {}

   private:
    friend class genit::IteratorFacadePrivateAccess<Iterator>;

    void Increment() {
      ++current_.x();
      if (current_.x() > upper_x_) {
        current_.x() = lower_x_;
        ++current_.y();
      }
    }
    void Decrement() {
      --current_.x();
      if (current_.x() < lower_x_) {
        current_.x() = upper_x_;
        --current_.y();
      }
    }
    const GridIndex &Dereference() const { return current_; }
    bool IsEqual(const Iterator &other) const {
      return current_ == other.current_;
    }

    int lower_x_;
    int upper_x_;
    GridIndex current_;
  };

  Iterator begin() const { return Iterator(*this, lower); }
  Iterator end() const {
    if (Empty()) {
      // Account for non-trivial empty ranges.  This avoids calling Increment()
      // on empty ranges.
      return begin();
    } else {
      return Iterator(*this, GridIndex{lower.x(), upper.y()});
    }
  }

  // Iterates through a range and calls a given functor for each coordinate.
  // If the functor has a `bool` return type and its call returns false, the
  // iteration terminates (otherwise, it continues).
  template <typename Functor>
  void ForEachGridCoord(Functor f) const {
    GridIndex current_pt = lower;
    for (; current_pt.y() < upper.y(); ++current_pt.y()) {
      for (; current_pt.x() < upper.x(); ++current_pt.x()) {
        if constexpr (std::is_same_v<bool, decltype(f(current_pt))>) {
          if (!f(current_pt)) {
            return;
          }
        } else {
          f(current_pt);
        }
      }
      current_pt.x() = lower.x();
    }
  }

  GridIndex lower, upper;
};

class GridRange::Quad {
 public:
  GridRange *begin() { return ranges_; }
  const GridRange *begin() const { return ranges_; }
  GridRange *end() { return ranges_ + size_; }
  const GridRange *end() const { return ranges_ + size_; }

  int size() const { return size_; }
  void resize(int size) { size_ = size; }

  GridRange &operator[](int i) { return ranges_[i]; }
  const GridRange &operator[](int i) const { return ranges_[i]; }

 private:
  GridRange ranges_[4] = {GridRange{}, GridRange{}, GridRange{}, GridRange{}};
  int size_ = 0;
};

// Specifies the reference frame and resolution of a grid so that conversions
// between continuous world coordinate points and the discrete grid cell on
// which those points fall can be done.
// This class also stores a string to identify the frame of reference.
struct GridFrame {
  GridFrame() = default;
  GridFrame(absl::string_view frame_id_, const eigenmath::Pose2d &origin_,
            double resolution_)
      : frame_id(frame_id_), origin(origin_), resolution(resolution_) {}

  bool operator==(const GridFrame &other) const {
    return frame_id == other.frame_id &&
           origin.translation() == other.origin.translation() &&
           origin.angle() == other.origin.angle() &&
           resolution == other.resolution;
  }

  // Transform world coordinate (real-valued 2d vector) into grid cell
  // indices.
  inline GridIndex FrameToGrid(const eigenmath::Vector2d &world_coords) const {
    const eigenmath::Vector2d relative = origin.inverse() * world_coords;
    return {std::lrint(relative.x() / resolution),
            std::lrint(relative.y() / resolution)};
  }

  // Transform world coordinate (real-valued 2d vector) into grid cell
  // range enclosing the point (square range of the cells within one grid
  // resolution distance of the world coordinates given).
  GridRange FrameToGridRange(const eigenmath::Vector2d &world_coords) const {
    DCHECK_GT(resolution, 0.0) << "Grid resolution must be positive!";
    const eigenmath::Vector2d relative = origin.inverse() * world_coords;
    const GridIndex lb = {std::lrint(std::floor(relative.x() / resolution)),
                          std::lrint(std::floor(relative.y() / resolution))};
    return {lb, {lb.x() + 2, lb.y() + 2}};
  }

  // Transform relative coordinate (real-valued 2d vector) w.r.t. the origin
  // into grid cell indices.
  GridIndex RelativeToOriginToGrid(
      const eigenmath::Vector2d &local_coords) const {
    DCHECK_GT(resolution, 0.0) << "Grid resolution must be positive!";
    return {std::lrint(local_coords.x() / resolution),
            std::lrint(local_coords.y() / resolution)};
  }

  // Transform grid cell indices into world coordinate (real-valued 2d
  // vector).
  eigenmath::Vector2d GridToFrame(const GridIndex &data_coords) const {
    return GridToFrame(data_coords.x(), data_coords.y());
  }

  // Transform grid cell indices into world coordinate (real-valued 2d
  // vector).
  // This function does not truncate the decimals in the coordinates
  // so that it can be used to convert sub-pixel coordinates into world frame.
  eigenmath::Vector2d GridToFrame(double coord_x, double coord_y) const {
    return origin * (eigenmath::Vector2d{coord_x, coord_y} * resolution);
  }

  // Resets the grid origin to the coordinates of the grid cell that is
  // nearest to the given origin.
  // The new orientation is used as is, but it is recommended to use a
  // consistent orientation with the previous origin.
  eigenmath::Pose2d SnapPoseToGridCell(const eigenmath::Pose2d &pose) const {
    return {GridToFrame(FrameToGrid(pose.translation())), pose.angle()};
  }

  // Gets the range of grid coordinates that covers a given circle in world
  // coordinates.
  GridRange FrameCircleToGridRange(
      const eigenmath::Vector2d &world_coords_center, double radius) const {
    const GridIndex grid_center = FrameToGrid(world_coords_center);
    const int footprint_halfwidth = std::lrint(std::ceil(radius / resolution));
    return {{grid_center.x() - footprint_halfwidth,
             grid_center.y() - footprint_halfwidth},
            {grid_center.x() + footprint_halfwidth + 1,
             grid_center.y() + footprint_halfwidth + 1}};
  }

  // Computes the grid range in dst_frame that overlaps with the cell at
  // src_index in src_frame. This computation is inclusive in the sense that
  // no cell in dst_frame can overlap with the src_frame cell if it is not in
  // the range returned by this function. All cells in the output range are not
  // guaranteed to overlap with the source cell, however.
  // There is no "exclusive" version of this function because that cannot be
  // represented as a GridRange, and the best way to get the cells in dst_frame
  // that map to the src_index cell is to test each index in the inclusive
  // range that is returned from this function.
  // The frame_id of both grid frames must match.
  static GridRange GridToGridInclusive(const GridFrame &src_frame,
                                       const GridIndex &src_index,
                                       const GridFrame &dst_frame);

  // Computes the grid index in dst_frame that contains the center of the cell
  // at src_index in src_frame. This computation does not consider cell-to-cell
  // overlaps between src_frame and dst_frame, for that, use should use
  // GridToGridInclusive to get a grid range.
  // The frame_id of both grid frames must match.
  static GridIndex GridToGrid(const GridFrame &src_frame,
                              const GridIndex &src_index,
                              const GridFrame &dst_frame) {
    DCHECK_EQ(src_frame.frame_id, dst_frame.frame_id);
    return dst_frame.FrameToGrid(src_frame.GridToFrame(src_index));
  }

  // Converts between grid indices of different grid frames, using a cached
  // transform.
  class GridToGridFunctor {
   public:
    explicit GridToGridFunctor(
        const Eigen::Affine2d &dst_affine_src = Eigen::Affine2d::Identity())
        : dst_affine_src_(dst_affine_src) {}

    GridIndex operator()(const GridIndex &src_index) const {
      const eigenmath::Vector2d exact_dst_index =
          dst_affine_src_ * src_index.cast<double>();
      return {static_cast<int>(std::rint(exact_dst_index.x())),
              static_cast<int>(std::rint(exact_dst_index.y()))};
    }

    GridToGridFunctor Inverse() const {
      return GridToGridFunctor(dst_affine_src_.inverse());
    }

   private:
    Eigen::Affine2d dst_affine_src_;
  };

  // Similar to the function above, this overload returns a functor for repeated
  // conversions between grid frames.
  // Due to rounding, the result of the overloads may differ slightly.  Thus, in
  // some cases it might be that GridToGrid(src, index, dst) != GridToGrid(src,
  // dst)(index).  Use compatible resolutions and origins to avoid this.
  static GridToGridFunctor GridToGrid(const GridFrame &src_frame,
                                      const GridFrame &dst_frame);

  std::string frame_id = "";
  eigenmath::Pose2d origin = eigenmath::Pose2d::Identity();
  double resolution = 1.0;
};

// Returns the smallest range in the target frame (indicated by the
// transformation `target_from_source`) covering `source_range`.
//
// Note: Some indices outside this range might also project back into
// `source_range`.
GridRange SmallestTargetRangeCoveringSourceRange(
    const GridRange &source_range,
    const GridFrame::GridToGridFunctor &target_from_source);

// Returns a range in the target frame which includes all indices that map back
// into the source range.
GridRange FullTargetRangeProjectingOntoSourceRange(
    const GridRange &source_range,
    const GridFrame::GridToGridFunctor &target_from_source);

template <typename Sink>
void AbslStringify(Sink &sink, const GridFrame &frame) {
  absl::Format(&sink, "frame = \"%s\", %v, resolution = %f", frame.frame_id,
               frame.origin, frame.resolution);
}

// Print to ostream for testing and debugging.
inline std::ostream &operator<<(std::ostream &stream, const GridFrame &frame) {
  return stream << absl::StrCat(frame);
}

// Line segment of grid cells.
using GridSegment = eigenmath::LineSegment2<int>;

// Utility for iterating over a line in a grid.  Uses the digital
// differential analyzer algorithm
// https://en.wikipedia.org/wiki/Digital_differential_analyzer_(graphics_algorithm)
class GridLine {
 public:
  GridLine(const GridIndex &from, const GridIndex &to);

  explicit GridLine(const GridSegment &segment)
      : GridLine(segment.from, segment.to) {}

  class Iterator
      : public genit::IteratorFacade<Iterator, GridIndex,
                                     std::random_access_iterator_tag> {
   public:
    Iterator(const GridLine &range, int index);

    GridIndex RightOrthogonalPoint() const {
      return {std::rint(current_.x() - delta_.y()),
              std::rint(current_.y() + delta_.x())};
    }

   private:
    friend class genit::IteratorFacadePrivateAccess<Iterator>;

    GridIndex Dereference() const {
      return GridIndex{std::rint(current_.x()), std::rint(current_.y())};
    }
    void Increment() {
      current_ += delta_;
      ++index_;
    }
    void Decrement() {
      current_ -= delta_;
      --index_;
    }
    void Advance(int i) {
      current_ += i * delta_;
      index_ += i;
    }
    int DistanceTo(const Iterator &other) const {
      return other.index_ - index_;
    }
    bool IsEqual(const Iterator &other) const { return index_ == other.index_; }

    eigenmath::Vector2d current_;
    eigenmath::Vector2d delta_;
    int index_;
  };

  Iterator begin() const { return Iterator{*this, 0}; }
  Iterator end() const { return Iterator{*this, length_}; }

  void LengthenOnBothEnds(int increments) {
    const eigenmath::Vector2d new_from =
        from_.cast<double>() - increments * delta_;
    from_ = GridIndex{std::rint(new_from.x()), std::rint(new_from.y())};
    length_ += 2 * increments;
  }

 private:
  GridIndex from_;
  eigenmath::Vector2d delta_;
  int length_;
};

template <typename Sink>
void AbslStringify(Sink &sink, const GridRange &range) {
  absl::Format(&sink, "(%v, %v)", eigenmath::AbslStringified(range.lower),
               eigenmath::AbslStringified(range.upper));
}

inline std::ostream &operator<<(std::ostream &os, const GridRange &range) {
  os << absl::StrCat(range);
  return os;
}

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_GRID_COMMON_H_

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

// The Grid type defined in this header is meant to represent any type of 2d
// grid that stores some value-type object in each of its cells. It also pairs
// that grid with a transform (frame id, pose and resolution (scale)) between
// some world coordinates and the grid indices.
//
// Grid orientation:
//
//  world (0,0)----------> +y
//  |
//  | grid (0,0)----->+col
//  |    |xxxxxxxxxxxx
//  |    |xxxxxxxxxxxx
//  |    |xxxxxxxxxxxx
//  |    vxxxxxxxxxxxx
//  |   +row
//  |
// +x
//
// Currently, a OccupancyGrid and a Costmap are implemented based on a templated
// Grid class.
//
// The grid stores its values using a modulus on the x-y indices of the cells,
// which means that the physical storage might look like this:
//
// XXXX......XXX
// XXXX......XXX
// .............
// XXXX......XXX
// XXXX......XXX
// XXXX......XXX
//
// where '.' are unused cells and 'X' are used / active cells.
//
// There are also conversion functions provided to allow conversion between the
// different grids.

#ifndef MOBILITY_COLLISION_COLLISION_OCCUPANCY_GRID_H_
#define MOBILITY_COLLISION_COLLISION_OCCUPANCY_GRID_H_

#include <cmath>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "collision/grid_common.h"
#include "eigenmath/pose2.h"
#include "eigenmath/types.h"

namespace mobility::collision {

// Templated grid which is based on a Eigen::Matrix. Any type supported by Eigen
// can be used template parameter.
// For alignment and properties see class level comment.
// For implementation details and an explanation of the storage strategy, see
// the comments that follow the class declaration.
template <typename T>
class Grid {
 public:
  using Scalar = T;
  using StorageMatrix = eigenmath::MatrixX<Scalar>;

  // Create an empty grid.
  explicit Grid(const Scalar& default_value = Scalar())
      : Grid(GridFrame{}, GridRange{}, default_value) {}

  // Create a grid with the given size. Keep in mind the upper limit
  // of a range is one-past the end.
  explicit Grid(const GridRange& initial_bounds,
                const Scalar& default_value = Scalar())
      : Grid(GridFrame{}, initial_bounds, default_value) {}

  // Create a grid with the given lower and upper bounds. Keep in mind
  // the upper limit of a range is one-past the end.
  Grid(GridIndex lower, GridIndex upper, const Scalar& default_value = Scalar())
      : Grid(GridFrame{}, GridRange{lower, upper}, default_value) {}

  // Create a grid with the given size. Keep in mind the upper limit
  // of a range is one-past the end.
  Grid(const GridFrame& grid_frame, const GridRange& initial_bounds,
       const Scalar& default_value = Scalar())
      : default_value_(default_value),
        data_(StorageMatrix::Constant(initial_bounds.XSpan(),
                                      initial_bounds.YSpan(), default_value)),
        bounds_(initial_bounds),
        frame_(grid_frame) {
    CHECK_GT(frame_.resolution, 0.0) << "Grid resolution must be positive!";
  }

  // Create a grid with the given lower and upper bounds. Keep in mind
  // the upper limit of a range is one-past the end.
  Grid(const GridFrame& grid_frame, GridIndex lower, GridIndex upper,
       const Scalar& default_value = Scalar())
      : Grid(grid_frame, GridRange{lower, upper}, default_value) {}

  // Copy constructor uses size-optimized allocation.
  Grid(const Grid& /*other*/);
  Grid& operator=(const Grid& /*other*/);
  Grid(Grid&&) = default;
  Grid& operator=(Grid&&) = default;

  // ======== PARAMETER GETTERS =======================================

  // Retrieve the parameters describing the structure of this grid.
  GridFrame& Frame() { return frame_; }
  const GridFrame& Frame() const { return frame_; }

  std::string& FrameId() { return frame_.frame_id; }
  const std::string& FrameId() const { return frame_.frame_id; }

  eigenmath::Pose2d& Origin() { return frame_.origin; }
  const eigenmath::Pose2d& Origin() const { return frame_.origin; }

  double& Resolution() { return frame_.resolution; }
  const double& Resolution() const { return frame_.resolution; }

  int Rows() const { return bounds_.XSpan(); }
  int Cols() const { return bounds_.YSpan(); }
  GridIndex SpanSize() const { return GridIndex(Rows(), Cols()); }
  const GridIndex& LowerBound() const { return bounds_.lower; }
  const GridIndex& UpperBound() const { return bounds_.upper; }
  const GridRange& Range() const { return bounds_; }

  bool Empty() const { return bounds_.Empty(); }

  int RowCapacity() const { return data_.rows(); }
  int ColCapacity() const { return data_.cols(); }

  // Returns false if world coordinate falls outside of the grid.
  bool IsInFrameBounds(const eigenmath::Vector2d& world_coords) const {
    return bounds_.Contains(frame_.FrameToGrid(world_coords));
  }

  // Returns false if grid coordinate falls outside of the grid.
  bool IsInGridBounds(const GridIndex& data_coords) const {
    return bounds_.Contains(data_coords);
  }

  // Returns false if grid coordinate falls outside of the grid.
  bool IsInGridBounds(int x, int y) const { return bounds_.Contains(x, y); }

  // Coordinate will be placed on the nearest edge cell if not in the grid.
  GridIndex ClampedFrameToGrid(const eigenmath::Vector2d& world_coords) const {
    return bounds_.Clamp(frame_.FrameToGrid(world_coords));
  }

  // Clamp a world coordinate to the nearest world coordinate still in the
  // grid.
  eigenmath::Vector2d Clamp(const eigenmath::Vector2d& world_coords) const {
    eigenmath::Vector2d relative = frame_.origin.inverse() * world_coords;
    const eigenmath::Vector2d relative_lower{
        frame_.resolution * bounds_.lower.x(),
        frame_.resolution * bounds_.lower.y()};
    const eigenmath::Vector2d relative_upper{
        frame_.resolution * (bounds_.upper.x() - 1),
        frame_.resolution * (bounds_.upper.y() - 1)};
    if (relative.x() > relative_upper.x()) {
      relative.x() = relative_upper.x();
    }
    if (relative.x() < relative_lower.x()) {
      relative.x() = relative_lower.x();
    }
    if (relative.y() > relative_upper.y()) {
      relative.y() = relative_upper.y();
    }
    if (relative.y() < relative_lower.y()) {
      relative.y() = relative_lower.y();
    }
    return frame_.origin * relative;
  }

  // Resets the grid origin to the coordinates of the grid cell that is
  // nearest to the given origin.
  // The new orientation is used as is, but it is recommended to use a
  // consistent orientation with the previous origin.
  void SetOriginRoundedToGridCell(const eigenmath::Pose2d& new_origin) {
    frame_.origin = frame_.SnapPoseToGridCell(new_origin);
  }

  // Get the range of grid coordinates that cover a given circle in world
  // coord.
  GridRange FrameCircleToGridRange(
      const eigenmath::Vector2d& world_coords_center, double radius) const {
    GridRange result =
        frame_.FrameCircleToGridRange(world_coords_center, radius);
    result.Intersect(bounds_);
    return result;  // NRVO
  }

  // ======== DATA ACCESSORS ==========================================

  // Indexes the underlying data directly without bound checks. You'd
  // better be sure the given index is valid before calling this
  // method.
  const Scalar& GetUnsafe(int x, int y) const {
    return data_(LogicalToStorageRow(x), LogicalToStorageCol(y));
  }

  const Scalar& GetUnsafe(const GridIndex& index) const {
    return GetUnsafe(index.x(), index.y());
  }

  // Unguarded write access to the underlying data.
  void SetUnsafe(int x, int y, const Scalar& value) {
    data_(LogicalToStorageRow(x), LogicalToStorageCol(y)) = value;
  }

  void SetUnsafe(const GridIndex& index, const Scalar& value) {
    SetUnsafe(index.x(), index.y(), value);
  }

  // Guarded access, defaulting to provided value.
  Scalar GetOrDefaultTo(const GridIndex& index, const Scalar& value) const {
    if (!IsInGridBounds(index)) {
      return value;
    }
    return GetUnsafe(index);
  }

  // Guarded read access to the underlying data.
  bool Get(int x, int y, Scalar* value) const {
    if (!IsInGridBounds(x, y)) {
      return false;
    }
    *value = data_(LogicalToStorageRow(x), LogicalToStorageCol(y));
    return true;
  }

  bool Get(const GridIndex& index, Scalar* value) const {
    return Get(index.x(), index.y(), value);
  }

  // Guarded write access to the underlying data.
  bool Set(int x, int y, const Scalar& value) {
    if (!IsInGridBounds(x, y)) {
      return false;
    }
    data_(LogicalToStorageRow(x), LogicalToStorageCol(y)) = value;
    return true;
  }

  // Write access to fill the complete internal buffer.
  // Returns false if the operation failed because the dimensions of the
  // supplied buffer don't match the internal dimensions and true on successful
  // assignment.
  bool Set(const Eigen::Ref<const StorageMatrix, Eigen::Unaligned,
                            Eigen::OuterStride<>>& data) {
    if (data.rows() != RowCapacity() || data.cols() != ColCapacity()) {
      return false;
    }

    data_ = data;

    return true;
  }

  bool Set(const GridIndex& index, const Scalar& value) {
    return Set(index.x(), index.y(), value);
  }

  // Fills the cells of the grid within a given range with a given value.
  void Fill(const GridRange& range, const Scalar& value) {
    FillLogicalRangeNoBoundsCheck(GridRange::Intersect(bounds_, range), value);
  }

  // Fills the cells of the grid within a given range with a given value.
  void Fill(const GridIndex& lower, const GridIndex& upper,
            const Scalar& value) {
    Fill(GridRange{lower, upper}, value);
  }

  // Fills all the cells of the grid with a given value.
  void Fill(const Scalar& value) {
    FillLogicalRangeNoBoundsCheck(bounds_, value);
  }

  // Calls a map functor on all the values in the grid and assigns
  // the result to the corresponding grid cell.
  // Traversal order depends on the implementation.
  template <typename Map>
  void ApplyInPlace(const GridRange& range, const Map& map);

  template <typename Map>
  void ApplyInPlace(const Map& map) {
    ApplyInPlace(bounds_, map);
  }

  // Creates a copy of this grid and calls a map functor on all the values in
  // that grid and assigns the result to the corresponding grid cell.
  // Traversal order depends on the implementation.
  template <typename Map>
  Grid Apply(const Map& map) const {
    Grid grid(*this);
    grid.ApplyInPlace(map);
    return grid;
  }

  // Calls a functor on all the values in the grid.
  // Traversal order depends on the implementation.
  // If the functor has a `bool` return type and its call returns false, the
  // iteration terminates (otherwise, it continues).
  template <typename Func>
  void ForEachCellValue(const GridRange& range, const Func& f);

  template <typename Func>
  void ForEachCellValue(const Func& f) {
    ForEachCellValue(bounds_, f);
  }

  // Calls a functor on all (index, value) pairs in the grid.
  // Traversal order depends on the implementation.
  // If the functor has a `bool` return type and its call returns false, the
  // iteration terminates (otherwise, it continues).
  template <typename Func>
  void ForEachGridIndexWithCellValue(const GridRange& range,
                                     const Func& f) const;
  template <typename Func>
  void ForEachGridIndexWithCellValue(const GridRange& range, const Func& f);

  template <typename Func>
  void ForEachGridIndexWithCellValue(const Func& f) const {
    ForEachGridIndexWithCellValue(bounds_, f);
  }
  template <typename Func>
  void ForEachGridIndexWithCellValue(const Func& f) {
    ForEachGridIndexWithCellValue(bounds_, f);
  }

  // ======== STORAGE MUTATORS ========================================

  // Sets and gets the default value that is used to fill in new cells
  // when reshaping the grid.
  void SetDefaultValue(const Scalar& value) { default_value_ = value; }
  const Scalar& GetDefaultValue() const { return default_value_; }

  // Reshapes the grid to the given range. Any cells the lie both in
  // the old range and the new range will be retained, and still
  // accessible at the same indices.
  void Reshape(const GridRange& next_range);

  // Moves the boundaries of the grid by a translation.
  void SlideBoundsBy(const GridIndex& translation) {
    if (translation == GridIndex(0, 0)) {
      return;
    }
    Reshape(GridRange::ShiftBy(bounds_, translation));
  }

  // Moves the boundaries of the grid to a destination.
  void SlideBoundsTo(const GridIndex& target_lower_bound) {
    if (target_lower_bound == bounds_.lower) {
      return;
    }
    Reshape(GridRange::ShiftBy(bounds_, target_lower_bound - bounds_.lower));
  }

  // Reshape the grid as needed such that the given index becomes
  // valid. If the index is already valid, this method does nothing.
  void GrowToInclude(const GridIndex& index) {
    const GridRange next_range = GridRange::GrowToInclude(bounds_, index);
    if (next_range != bounds_) {
      Reshape(next_range);
    }
  }

  // Reshape the grid as needed such that the given range becomes
  // valid. If the range is already valid, this method does nothing.
  void GrowToInclude(const GridRange& range) {
    const GridRange next_range = GridRange::SpanningUnion(bounds_, range);
    if (next_range != bounds_) {
      Reshape(next_range);
    }
  }

  // Allocates enough storage, if needed, so that this grid could grow
  // without needing further allocations up to the given range.
  void Reserve(int row_capacity, int col_capacity) {
    if (row_capacity > RowCapacity() && col_capacity > ColCapacity()) {
      GrowCapacityTo(row_capacity, col_capacity);
    }
  }

  const StorageMatrix& RawStorage() const { return data_; }

 private:
  int LogicalToStorageRow(int logical_row) const {
    return StrictModulus(logical_row, RowCapacity());
  }

  int LogicalToStorageCol(int logical_col) const {
    return StrictModulus(logical_col, ColCapacity());
  }

  // Provides the inverse mapping to the LogicalToStorage variants for a
  // contiguous storage block.
  struct StorageToLogicalFunctor {
    GridIndex operator()(const GridIndex& storage_index) const {
      return storage_index + offset;
    }
    GridIndex offset;
  };

  // Returns a functor to convert storage indices to logical indices for the
  // contiguous block with lower index `storage_index`.
  StorageToLogicalFunctor StorageToLogicalForBlockStartingWithStorageIndex(
      const GridIndex& storage_index) const;

  template <typename GridType, typename Func>
  static void ForEachGridIndexWithCellValue(GridType& grid,
                                            const GridRange& range,
                                            const Func& f);

  GridRange::Quad LogicalToStorageRanges(const GridRange& logical_range) const;

  void FillLogicalRangeNoBoundsCheck(GridRange range, const Scalar& value);

  void GrowCapacityTo(int req_row_capacity, int req_col_capacity);

  void CopyDataFrom(const StorageMatrix& data,
                    const GridRange::Quad& orig_ranges);

  static constexpr double kGrowthFactor = 1.5;

  Scalar default_value_;
  StorageMatrix data_;
  GridRange bounds_;
  GridFrame frame_;
};

// Implementation details about the storage strategy.
//
// The "bounds_" field contains the logical range of active (or allocated)
// cells of the grid. This logical range is mapped onto a storage matrix as
// so:
//               0           ColCapacity
//            0_ |           |
//               XXXX......XXX
//               XXXX......XXX _ bounds_.upper.x() % RowCapacity
//               .............
//               ............. _ bounds_.lower.x() % RowCapacity
//               XXXX......XXX
//               XXXX......XXX
// RowCapacity _ XXXX......XXX
//                  |      |
//  bounds_.upper.y()      bounds_.lower.y()
//      % ColCapacity          % ColCapacity
//
// where '.' are unused cells and 'X' are used / active cells.
// The LogicalToStorageRanges function can be used to compute the 4 storage
// ranges (shown by 'X') for a corresponding logical range (e.g., bounds_).
//
// When bounds_ is moved (e.g., SlideBoundsBy or Reshape), we get:
//
//               0           ColCapacity
//            0_ |           |
//               XXXXN.....OOX
//               XXXXN.....OOX
//               NNNNN.......N _ bounds_.upper.x() % RowCapacity
//               .............
//               OOOO......OOO
//               OOOO......OOO
// RowCapacity _ XXXXN.....OOX _ bounds_.lower.x() % RowCapacity
//                   |       |
//   bounds_.upper.y()       bounds_.lower.y()
//       % ColCapacity           % ColCapacity
//
// where 'O' are old cells (removed) and 'N' are new cells (activated).
//
// The aim of this is, of course, to avoid copying all the data that is
// retained when the bounds are changed. This is because most reshaping
// operations on the grids involved shifting, increasing or decreasing the
// bounds only slightly compared to its existing size.
//
// FillLogicalRangeNoBoundsCheck:
// This function provides an efficient (block-wise) filling of data over a
// given logical range (subset of bounds_). It works by applying the modulus
// operator on the lower and upper bounds of the logical range to compute the
// (up-to) 4 storage ranges in the matrix (as depicted in the first picture
// above), and then, filling each storage block-matrix with the given value.
//
// Reshape (without reallocation):
// When the bounds_ are changed and capacity is sufficient to accommodate the
// new bounds, the existing data in the overlapping region between the old and
// new bounds does not need to be touched at all. However, all the new cells
// must be filled with the default_value_. To do this, the (up-to) 4 ranges
// that make up the difference between the new and old bounds (i.e., covered
// by the new range but not by the old one) are computed and each are filled
// using the FillLogicalRangeNoBoundsCheck function.
//
// GrowCapacityTo (also called by Reshape if capacity must be increased):
// Growing the capacity of the grid is a tricky operation. Obviously, a new
// storage matrix is created for the new capacity (next power-of-two after
// required capacity in rows and columns). Then, all the elements in bounds_
// must be copied from the old storage matrix to the new one. It works as
// follows:
//   1) Compute the (up-to) 4 storage ranges for bounds_ in the old matrix.
//   2) For each old storage ranges:
//     a) Map the old storage range to its logical range.
//     b) Compute its (up-to) 4 storage ranges in the new matrix.
//     c) For each new storage ranges:
//       i) Map the new storage range to its old storage range.
//       ii) Block-assign the old storage block to the new storage block.
// This leads to up to 16 block-assignments, although, in practice it is
// probably rare.

template <typename T>
Grid<T>::Grid(const Grid<T>& other)
    : default_value_(other.default_value_), frame_(other.frame_) {
  // Allocate and copy data_
  GrowCapacityTo(other.bounds_.XSpan(), other.bounds_.YSpan());
  bounds_ = other.bounds_;
  const GridRange::Quad orig_ranges =
      other.LogicalToStorageRanges(other.bounds_);
  CopyDataFrom(other.data_, orig_ranges);
}

template <typename T>
Grid<T>& Grid<T>::operator=(const Grid<T>& other) {
  if (this != &other) {
    default_value_ = other.default_value_;
    bounds_ = GridRange{};
    frame_ = other.frame_;
    GrowCapacityTo(other.bounds_.XSpan(), other.bounds_.YSpan());
    bounds_ = other.bounds_;
    const GridRange::Quad orig_ranges =
        other.LogicalToStorageRanges(other.bounds_);
    CopyDataFrom(other.data_, orig_ranges);
  }
  return *this;
}

template <typename T>
void Grid<T>::Reshape(const GridRange& next_range) {
  const GridRange overlap = GridRange::Intersect(bounds_, next_range);
  bounds_ = overlap;
  if (next_range.XSpan() > RowCapacity() ||
      next_range.YSpan() > ColCapacity()) {
    GrowCapacityTo(next_range.XSpan(), next_range.YSpan());
  }
  bounds_ = next_range;
  if (overlap.Empty()) {
    FillLogicalRangeNoBoundsCheck(next_range, default_value_);
  } else {
    // Compute the difference between the next range and the overlap:
    GridRange new_ranges[4] = {
        GridRange{{next_range.lower.x(), overlap.lower.y()},
                  {overlap.lower.x(), next_range.upper.y()}},
        GridRange{{overlap.lower.x(), overlap.upper.y()},
                  {next_range.upper.x(), next_range.upper.y()}},
        GridRange{{overlap.upper.x(), next_range.lower.y()},
                  {next_range.upper.x(), overlap.upper.y()}},
        GridRange{{next_range.lower.x(), next_range.lower.y()},
                  {overlap.upper.x(), overlap.lower.y()}}};
    // Fill what is in next_range but not in overlap with default value:
    for (const auto& new_range : new_ranges) {
      if (new_range.Empty()) {
        continue;
      }
      FillLogicalRangeNoBoundsCheck(new_range, default_value_);
    }
  }
}

template <typename T>
GridRange::Quad Grid<T>::LogicalToStorageRanges(
    const GridRange& logical_range) const {
  GridRange::Quad result;
  if (RowCapacity() == 0 || ColCapacity() == 0 || logical_range.Empty()) {
    return result;  // NRVO
  }
  const int storage_lower_x =
      StrictModulus(logical_range.lower.x(), RowCapacity());
  const int storage_lower_y =
      StrictModulus(logical_range.lower.y(), ColCapacity());
  const int storage_upper_x =
      StrictModulus(logical_range.upper.x() - 1, RowCapacity()) + 1;
  const int storage_upper_y =
      StrictModulus(logical_range.upper.y() - 1, ColCapacity()) + 1;
  if (storage_lower_x >= storage_upper_x &&
      storage_lower_y >= storage_upper_y) {
    result.resize(4);
    result[0] = GridRange({storage_lower_x, storage_lower_y},
                          {RowCapacity(), ColCapacity()});
    result[1] =
        GridRange({0, storage_lower_y}, {storage_upper_x, ColCapacity()});
    result[2] =
        GridRange({storage_lower_x, 0}, {RowCapacity(), storage_upper_y});
    result[3] = GridRange({0, 0}, {storage_upper_x, storage_upper_y});
  } else if (storage_lower_x >= storage_upper_x) {
    result.resize(2);
    result[0] = GridRange({storage_lower_x, storage_lower_y},
                          {RowCapacity(), storage_upper_y});
    result[1] =
        GridRange({0, storage_lower_y}, {storage_upper_x, storage_upper_y});
  } else if (storage_lower_y >= storage_upper_y) {
    result.resize(2);
    result[0] = GridRange({storage_lower_x, storage_lower_y},
                          {storage_upper_x, ColCapacity()});
    result[1] =
        GridRange({storage_lower_x, 0}, {storage_upper_x, storage_upper_y});
  } else {
    result.resize(1);
    result[0] = GridRange({storage_lower_x, storage_lower_y},
                          {storage_upper_x, storage_upper_y});
  }
  return result;  // NRVO
}

template <typename T>
typename Grid<T>::StorageToLogicalFunctor
Grid<T>::StorageToLogicalForBlockStartingWithStorageIndex(
    const GridIndex& storage_index) const {
  // Calculate the difference to bounds_.lower to find the offset between the
  // storage range and the logical range.
  const GridIndex lower_as_storage(LogicalToStorageRow(bounds_.lower.x()),
                                   LogicalToStorageCol(bounds_.lower.y()));
  const GridIndex delta_to_lower(
      StrictModulus(storage_index.x() - lower_as_storage.x(), RowCapacity()),
      StrictModulus(storage_index.y() - lower_as_storage.y(), ColCapacity()));
  const GridIndex offset = bounds_.lower + delta_to_lower - storage_index;
  return StorageToLogicalFunctor{offset};
}

template <typename T>
void Grid<T>::FillLogicalRangeNoBoundsCheck(GridRange range,
                                            const Scalar& value) {
  // Calculate the (up to) 4 storage ranges to cover the given range.
  const GridRange::Quad storage_ranges = LogicalToStorageRanges(range);
  // Fill each of the storages ranges, using Eigen's block views:
  const GridRange total_storage_range{{0, 0}, {RowCapacity(), ColCapacity()}};
  for (const auto& storage_range : storage_ranges) {
    CHECK(total_storage_range.Contains(storage_range))
        << "total_storage_range (" << total_storage_range.lower.transpose()
        << ") -> (" << total_storage_range.upper.transpose()
        << "), storage_range: (" << storage_range.lower.transpose() << ") -> ("
        << storage_range.upper.transpose() << "), range: ("
        << range.lower.transpose() << ") -> (" << range.upper.transpose()
        << ")";
    data_.block(storage_range.lower.x(), storage_range.lower.y(),
                storage_range.XSpan(), storage_range.YSpan()) =
        StorageMatrix::Constant(storage_range.XSpan(), storage_range.YSpan(),
                                value);
  }
}

template <typename T>
template <typename Map>
void Grid<T>::ApplyInPlace(const GridRange& range, const Map& map) {
  // Calculate the (up to) 4 storage ranges to cover the given range.
  const GridRange valid_range = GridRange::Intersect(bounds_, range);
  const GridRange::Quad storage_ranges = LogicalToStorageRanges(valid_range);
  // Fill each of the storages ranges, using Eigen's block views:
  for (const auto& storage_range : storage_ranges) {
    storage_range.ForEachGridCoord([&](const GridIndex& index) {
      data_(index.x(), index.y()) = map(data_(index.x(), index.y()));
    });
  }
}

template <typename T>
template <typename Func>
void Grid<T>::ForEachCellValue(const GridRange& range, const Func& f) {
  // Calculate the (up to) 4 storage ranges to cover the given range.
  const GridRange valid_range = GridRange::Intersect(bounds_, range);
  const GridRange::Quad storage_ranges = LogicalToStorageRanges(valid_range);
  // Fill each of the storages ranges, using Eigen's block views:
  for (const auto& storage_range : storage_ranges) {
    storage_range.ForEachGridCoord(
        [&](const GridIndex& index) { return f(data_(index.x(), index.y())); });
  }
}

template <typename T>
template <typename GridType, typename Func>
void Grid<T>::ForEachGridIndexWithCellValue(GridType& grid,
                                            const GridRange& range,
                                            const Func& f) {
  static_assert(std::is_same_v<std::remove_const_t<GridType>, Grid<T>>);
  // Calculate the (up to) 4 storage ranges to cover the given range.
  const GridRange valid_range = GridRange::Intersect(grid.bounds_, range);
  const GridRange::Quad storage_ranges =
      grid.LogicalToStorageRanges(valid_range);
  // Fill each of the storages ranges, using Eigen's block views:
  for (const auto& storage_range : storage_ranges) {
    auto to_logical = grid.StorageToLogicalForBlockStartingWithStorageIndex(
        storage_range.lower);
    storage_range.ForEachGridCoord([&](const GridIndex& index) {
      return f(to_logical(index), grid.data_(index.x(), index.y()));
    });
  }
}

template <typename T>
template <typename Func>
void Grid<T>::ForEachGridIndexWithCellValue(const GridRange& range,
                                            const Func& f) const {
  ForEachGridIndexWithCellValue(*this, range, f);
}

// Same implementation as above, allows modifying the cell value.
template <typename T>
template <typename Func>
void Grid<T>::ForEachGridIndexWithCellValue(const GridRange& range,
                                            const Func& f) {
  ForEachGridIndexWithCellValue(*this, range, f);
}

template <typename T>
void Grid<T>::GrowCapacityTo(int req_row_capacity, int req_col_capacity) {
  // Grows the storage matrix to contain at least the requested capacity.
  // Preserve the values within bounds_.
  // The (1 << lrint(ceil(log2(n + 1)))) expressions round up to the
  // next power-of-two.
  const int next_row_capacity =
      (1 << std::lrint(std::ceil(std::log2(req_row_capacity + 1))));
  const int next_col_capacity =
      (1 << std::lrint(std::ceil(std::log2(req_col_capacity + 1))));
  if (bounds_.Empty()) {
    data_.resize(next_row_capacity, next_col_capacity);
    return;
  }

  const GridRange::Quad orig_ranges = LogicalToStorageRanges(bounds_);
  StorageMatrix tmp_data(next_row_capacity, next_col_capacity);
  data_.swap(tmp_data);
  CopyDataFrom(tmp_data, orig_ranges);
}

template <typename T>
void Grid<T>::CopyDataFrom(const StorageMatrix& tmp_data,
                           const GridRange::Quad& orig_ranges) {
  // The code below works as follows:
  // For each sub-range of bounds_ that is stored contiguously in the
  // original storage matrix (at most 4 sub-ranges), compute the sub-ranges
  // that it covers in the new storage matrix (at most 4). Then, simply copy
  // over the elements in each sub-sub-range from the original storage to
  // the new one, using Eigen's block operations.
  // At worst, 16 sub-sub-blocks have to be copied individually.
  const GridIndex orig_range_offset = orig_ranges[0].lower;
  for (const auto& orig_range : orig_ranges) {
    // Since orig_range is a storage range, we must find the logical
    // range that it corresponds to to obtain its sub-ranges in the new
    // storage matrix.
    const GridRange base_orig_range =
        GridRange::ShiftBy(orig_range, -orig_range_offset);
    const GridIndex base_to_logical{
        bounds_.lower.x() +
            (base_orig_range.lower.x() < 0 ? tmp_data.rows() : 0),
        bounds_.lower.y() +
            (base_orig_range.lower.y() < 0 ? tmp_data.cols() : 0)};
    const GridRange logical_orig_range =
        GridRange::ShiftBy(base_orig_range, base_to_logical);
    const GridRange::Quad next_ranges =
        LogicalToStorageRanges(logical_orig_range);
    const GridIndex next_range_offset = next_ranges[0].lower;
    for (const auto& next_range : next_ranges) {
      // Since next_range is a storage range in the new storage matrix,
      // we must find the storage range in the old storage matrix that it
      // corresponds to.
      const GridRange base_next_range =
          GridRange::ShiftBy(next_range, -next_range_offset);
      const GridIndex base_to_orig_storage{
          orig_range.lower.x() +
              (base_next_range.lower.x() < 0 ? data_.rows() : 0),
          orig_range.lower.y() +
              (base_next_range.lower.y() < 0 ? data_.cols() : 0)};
      const GridRange orig_storage_next_range =
          GridRange::ShiftBy(base_next_range, base_to_orig_storage);
      data_.block(next_range.lower.x(), next_range.lower.y(),
                  next_range.XSpan(), next_range.YSpan()) =
          tmp_data.block(orig_storage_next_range.lower.x(),
                         orig_storage_next_range.lower.y(),
                         orig_storage_next_range.XSpan(),
                         orig_storage_next_range.YSpan());
    }
  }
}

// Helper function to allow easier printing of a grid for debugging.
// Note: If there are further overloads with types which shouldn't be cast
// to int, this function has to become an independent template function.
template <typename Sink, typename T>
void AbslStringify(Sink& sink, const Grid<T>& grid) {
  absl::Format(&sink, "frame: %v, range: %v, data:\n", grid.Frame(),
               grid.Range());
  for (int row = grid.LowerBound().x(); row < grid.UpperBound().x(); ++row) {
    for (int col = grid.LowerBound().y(); col < grid.UpperBound().y(); ++col) {
      absl::Format(&sink, " %v", grid.GetUnsafe(row, col));
    }
    absl::Format(&sink, "\n");
  }
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Grid<T>& grid) {
  return stream << absl::StrCat(grid);
}

// Comparison functions:
template <typename T>
bool operator==(const Grid<T>& lhs, const Grid<T>& rhs) {
  if (!lhs.Origin().isApprox(rhs.Origin(),
                             std::numeric_limits<double>::epsilon(),
                             std::numeric_limits<double>::epsilon()) ||
      lhs.Resolution() != rhs.Resolution() || lhs.FrameId() != rhs.FrameId() ||
      lhs.Range() != rhs.Range()) {
    return false;
  }
  for (int col = lhs.LowerBound().y(); col < lhs.UpperBound().y(); ++col) {
    for (int row = lhs.LowerBound().x(); row < lhs.UpperBound().x(); ++row) {
      if (lhs.GetUnsafe(row, col) != rhs.GetUnsafe(row, col)) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
bool operator!=(const Grid<T>& lhs, const Grid<T>& rhs) {
  return !(lhs == rhs);
}

// State of a grid cell.
enum class OccupancyStatus : uint8_t {
  UNOCCUPIED = 0,
  OCCUPIED = 1,
  UNKNOWN = 2
};

template <typename Sink>
void AbslStringify(Sink& sink, OccupancyStatus occupancy) {
  absl::Format(&sink, "%d", static_cast<int>(occupancy));
}

inline std::ostream& operator<<(std::ostream& stream,
                                OccupancyStatus occupancy) {
  return stream << absl::StrCat(occupancy);
}

// Returns a string name for the given OccupancyStatus.
inline std::string GetOccupancyStatusName(OccupancyStatus occupancy) {
  switch (occupancy) {
    case OccupancyStatus::UNOCCUPIED:
      return "OccupancyStatus::UNOCCUPIED";
    case OccupancyStatus::OCCUPIED:
      return "OccupancyStatus::OCCUPIED";
    case OccupancyStatus::UNKNOWN:
      return "OccupancyStatus::UNKNOWN";
  }
}

// Returns the corresponding OccupancyStatus enum given a string name or fail if
// the name is invalid.
inline absl::StatusOr<OccupancyStatus> GetOccupancyStatus(
    absl::string_view occupancy_status_name) {
  if (occupancy_status_name == "OccupancyStatus::UNOCCUPIED") {
    return OccupancyStatus::UNOCCUPIED;
  }
  if (occupancy_status_name == "OccupancyStatus::OCCUPIED") {
    return OccupancyStatus::OCCUPIED;
  }
  if (occupancy_status_name == "OccupancyStatus::UNKNOWN") {
    return OccupancyStatus::UNKNOWN;
  } else {
    return absl::Status(absl::StatusCode::kUnavailable, "");
  }
}

// Adds special meanings to the costmap.
enum class CostmapStatus : int {
  FREE = 0,
  MAX_COST =
      std::numeric_limits<std::underlying_type<CostmapStatus>::type>::max()
};

// Occupancy grid as used for localization.
using OccupancyGrid = Grid<OccupancyStatus>;

// Occupancy grid as used path planning.
using Costmap = Grid<int>;

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_OCCUPANCY_GRID_H_

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

#include "collision/occupancy_grid.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "absl/random/distributions.h"
#include "benchmark/benchmark.h"
#include "eigenmath/sampling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

using ::testing::Eq;
using ::testing::Message;
using ::testing::StrEq;
using ::testing::UnorderedElementsAreArray;

constexpr const char kDefaultFrame[] = "map";

std::string DumpCharGrid(const Grid<char> &grid) {
  std::ostringstream ss;
  ss << "\n" << grid.Range();
  for (int row = grid.LowerBound().x(); row < grid.UpperBound().x(); ++row) {
    ss << "\n";
    for (int col = grid.LowerBound().y(); col < grid.UpperBound().y(); ++col) {
      ss << grid.GetUnsafe(row, col);
    }
  }
  return ss.str();
}

void FillIntGrid(const GridRange &range, int seed, Grid<int> *grid) {
  grid->Reshape(range);
  grid->Range().ForEachGridCoord(
      [&](const GridIndex &index) { grid->SetUnsafe(index, seed++); });
}

class OccupancyGridFixture : public ::testing::Test {
 protected:
  OccupancyGridFixture()
      : frame_(kDefaultFrame),
        origin_({1.0, 2.0}, M_PI),
        rows_(3),
        cols_(4),
        resolution_(0.1),
        grid_(GridFrame(frame_, origin_, resolution_),
              GridRange::OriginTo(rows_, cols_)) {
    grid_.SetUnsafe(0, 0, OccupancyStatus::OCCUPIED);
    grid_.SetUnsafe(0, 1, OccupancyStatus::UNOCCUPIED);
    grid_.SetUnsafe(0, 2, OccupancyStatus::OCCUPIED);
    grid_.SetUnsafe(0, 3, OccupancyStatus::OCCUPIED);
    grid_.SetUnsafe(1, 0, OccupancyStatus::UNOCCUPIED);
    grid_.SetUnsafe(1, 1, OccupancyStatus::UNKNOWN);
    grid_.SetUnsafe(1, 2, OccupancyStatus::UNKNOWN);
    grid_.SetUnsafe(1, 3, OccupancyStatus::OCCUPIED);
    grid_.SetUnsafe(2, 0, OccupancyStatus::OCCUPIED);
    grid_.SetUnsafe(2, 1, OccupancyStatus::OCCUPIED);
    grid_.SetUnsafe(2, 2, OccupancyStatus::OCCUPIED);
    grid_.SetUnsafe(2, 3, OccupancyStatus::UNKNOWN);
  }

  std::string frame_;
  eigenmath::Pose2d origin_;
  int rows_;
  int cols_;
  double resolution_;
  OccupancyGrid grid_;
};

TEST(OccupancyGridTest, ConstructorZeroResolution) {
  EXPECT_DEATH(OccupancyGrid(
                   GridFrame(kDefaultFrame, eigenmath::Pose2d::Identity(), 0.0),
                   GridRange::OriginTo(2, 3)),
               "");
}

TEST_F(OccupancyGridFixture, ParamsConstructors) {
  const GridFrame grid_frame(frame_, origin_, resolution_);
  const GridRange grid_range = GridRange::OriginTo(rows_, cols_);
  const OccupancyGrid grid_from_params(grid_frame, grid_range);
  for (const GridFrame &frames : {grid_from_params.Frame(), grid_.Frame()}) {
    EXPECT_EQ(frames.frame_id, grid_frame.frame_id);
    EXPECT_EQ(frames.origin.translation(), grid_frame.origin.translation());
    EXPECT_EQ(frames.origin.angle(), grid_frame.origin.angle());
    EXPECT_EQ(frames.resolution, grid_frame.resolution);
  }
  for (const GridRange &ranges : {grid_from_params.Range(), grid_.Range()}) {
    EXPECT_EQ(ranges.lower.x(), grid_range.lower.x());
    EXPECT_EQ(ranges.lower.y(), grid_range.lower.y());
    EXPECT_EQ(ranges.upper.x(), grid_range.upper.x());
    EXPECT_EQ(ranges.upper.y(), grid_range.upper.y());
  }
}

TEST_F(OccupancyGridFixture, IsInFrameBounds) {
  EXPECT_TRUE(grid_.IsInFrameBounds({0.8, 1.7}));
  EXPECT_FALSE(grid_.IsInFrameBounds({-1.2, 1.3}));
}

TEST_F(OccupancyGridFixture, IsInGridBounds) {
  EXPECT_TRUE(grid_.IsInGridBounds({2, 3}));
  EXPECT_FALSE(grid_.IsInGridBounds({12, 13}));
}

TEST_F(OccupancyGridFixture, WorldToGrid) {
  const eigenmath::Vector2d world_coords{0.8, 1.7};
  const GridIndex grid_coords = grid_.Frame().FrameToGrid(world_coords);
  const eigenmath::Vector2d local_coords = origin_.inverse() * world_coords;
  const GridIndex grid_coords_expected = {
      static_cast<int>(std::rint(local_coords.x() / resolution_)),
      static_cast<int>(std::rint(local_coords.y() / resolution_))};
  EXPECT_EQ(grid_coords, grid_coords_expected)
      << "\nExpected:\n"
      << grid_coords_expected << "\nGot:\n"
      << grid_coords;
}

TEST_F(OccupancyGridFixture, ClampedWorldToGrid) {
  const eigenmath::Vector2d world_coords{2.2, -0.3};
  const GridIndex grid_coords = grid_.ClampedFrameToGrid(world_coords);
  const GridIndex grid_coords_expected{0, 3};
  EXPECT_EQ(grid_coords, grid_coords_expected)
      << "\nExpected:\n"
      << grid_coords_expected << "\nGot:\n"
      << grid_coords;
}

TEST_F(OccupancyGridFixture, GridToWorld) {
  const GridIndex grid_coords{2, 3};
  const eigenmath::Vector2d world_coords =
      grid_.Frame().GridToFrame(grid_coords);
  const eigenmath::Vector2d world_coords_expected =
      origin_ * (grid_coords.cast<double>() * resolution_);
  EXPECT_EQ(world_coords, world_coords_expected)
      << "\nExpected:\n"
      << world_coords_expected << "\nGot:\n"
      << world_coords;
}

TEST_F(OccupancyGridFixture, Clamp) {
  {
    const eigenmath::Vector2d world_coords{2.2, -0.3};
    const eigenmath::Vector2d clamped_world_coords = grid_.Clamp(world_coords);
    EXPECT_EQ(clamped_world_coords, eigenmath::Vector2d(1.0, 1.7))
        << "Expected (1.0 1.7), but got (" << clamped_world_coords.transpose()
        << ")";
  }

  {
    const eigenmath::Vector2d world_coords{0.3, 2.2};
    const eigenmath::Vector2d clamped_world_coords = grid_.Clamp(world_coords);
    EXPECT_EQ(clamped_world_coords, eigenmath::Vector2d(0.8, 2.0))
        << "Expected (0.8 2.0), but got (" << clamped_world_coords.transpose()
        << ")";
  }
}

TEST_F(OccupancyGridFixture, WorldGridRoundTrip) {
  Eigen::IOFormat f(2, Eigen::DontAlignCols, ", ", ", ", "", "", "(", ")");
  const GridIndex grid_coords{2, 3};
  // Perturb the world coordinates to within the same cell and make sure we go
  // back to the center of the cell when rounding.
  for (const double x_perturbation : {-1.0, 0.0, 1.0}) {
    for (const double y_perturbation : {-1.0, 0.0, 1.0}) {
      const eigenmath::Vector2d perturbation =
          eigenmath::Vector2d(x_perturbation, y_perturbation) * resolution_ / 3;
      const eigenmath::Vector2d perturbed_coords =
          grid_.Frame().GridToFrame(grid_coords) + perturbation;
      const GridIndex grid_coords_roundtrip =
          grid_.Frame().FrameToGrid(perturbed_coords);

      EXPECT_EQ(grid_coords, grid_coords_roundtrip)
          << "\nGridToWorld: "
          << grid_.Frame().GridToFrame(grid_coords).format(f)
          << "\nPerturbation: " << perturbation.format(f)
          << "\nFinal world coords perturbation: " << perturbed_coords.format(f)
          << "\nExpected:" << grid_coords.format(f)
          << "\nActual:" << grid_coords_roundtrip.format(f);
    }
  }
}

TEST_F(OccupancyGridFixture, WorldGridAverageRoundTrip) {
  const GridIndex grid_coords_1{2, 3};
  const GridIndex grid_coords_2{5, 8};
  const eigenmath::Vector2d grid_coords_avg =
      (grid_coords_1 + grid_coords_2).cast<double>() * 0.5;

  const eigenmath::Vector2d world_coords_1 =
      grid_.Frame().GridToFrame(grid_coords_1);
  const eigenmath::Vector2d world_coords_2 =
      grid_.Frame().GridToFrame(grid_coords_2);
  const eigenmath::Vector2d world_coords_avg =
      (world_coords_1 + world_coords_2) * 0.5;

  const eigenmath::Vector2d world_coords_of_grid_coords_avg =
      grid_.Frame().GridToFrame(grid_coords_avg.x(), grid_coords_avg.y());

  EXPECT_DOUBLE_EQ(world_coords_avg.x(), world_coords_of_grid_coords_avg.x());
  EXPECT_DOUBLE_EQ(world_coords_avg.y(), world_coords_of_grid_coords_avg.y());
}

TEST_F(OccupancyGridFixture, ForEachGridIndexWithCellValue) {
  std::vector<GridIndex> indices;
  grid_.ForEachGridIndexWithCellValue(
      [&](const GridIndex &index, OccupancyStatus occupancy) {
        indices.push_back(index);
        OccupancyStatus expected_occupancy;
        EXPECT_TRUE(grid_.Get(index, &expected_occupancy));
        EXPECT_EQ(occupancy, expected_occupancy);
      });
  EXPECT_THAT(indices, UnorderedElementsAreArray(grid_.Range().begin(),
                                                 grid_.Range().end()));
}

TEST_F(OccupancyGridFixture, ForEachGridIndexWithCellValueWithPredicate) {
  int indices_count = 0;
  const int max_indices_count = grid_.Range().ComputeSize() / 2 - 4;
  grid_.ForEachGridIndexWithCellValue(
      [&](const GridIndex &index, OccupancyStatus occupancy) {
        ++indices_count;
        OccupancyStatus expected_occupancy;
        EXPECT_TRUE(grid_.Get(index, &expected_occupancy));
        EXPECT_EQ(occupancy, expected_occupancy);
        return indices_count < max_indices_count;
      });
  // Check that we stopped early.
  EXPECT_LT(indices_count, grid_.Range().ComputeSize());
}

TEST_F(OccupancyGridFixture,
       SetFromMappedBufferWithCorrectDimensionsFillsInternalStorage) {
  std::vector<OccupancyStatus> buffer(grid_.Cols() * grid_.Rows(),
                                      OccupancyStatus::UNKNOWN);
  buffer[0] = OccupancyStatus::UNOCCUPIED;
  buffer[1] = OccupancyStatus::UNOCCUPIED;
  buffer[2] = OccupancyStatus::UNOCCUPIED;
  buffer[3] = OccupancyStatus::UNOCCUPIED;
  buffer[4] = OccupancyStatus::OCCUPIED;
  buffer[5] = OccupancyStatus::OCCUPIED;
  Eigen::Map<const OccupancyGrid::StorageMatrix, 0, Eigen::OuterStride<>>
      mapped_buffer(buffer.data(), grid_.Rows(), grid_.Cols(), grid_.Rows());
  ASSERT_TRUE(grid_.Set(mapped_buffer));

  for (int row = 0; row < grid_.Rows(); ++row) {
    for (int col = 0; col < grid_.Cols(); ++col) {
      SCOPED_TRACE(Message() << "Row: " << row << " Col: " << col);
      EXPECT_THAT(grid_.RawStorage()(row, col), Eq(mapped_buffer(row, col)));
    }
  }
}

TEST_F(OccupancyGridFixture,
       SetWithMappedBufferAndWrongDimensionsLeavesInternalStorageUntouched) {
  const OccupancyGrid::StorageMatrix original_data = grid_.RawStorage();
  std::vector<OccupancyStatus> buffer((grid_.Cols() + 1) * grid_.Rows(),
                                      OccupancyStatus::UNKNOWN);
  Eigen::Map<const OccupancyGrid::StorageMatrix, 0, Eigen::OuterStride<>>
      mapped_buffer(buffer.data(), grid_.Rows(), grid_.Cols() + 1,
                    grid_.Rows());
  EXPECT_FALSE(grid_.Set(mapped_buffer));
  for (int row = 0; row < grid_.Rows(); ++row) {
    for (int col = 0; col < grid_.Cols(); ++col) {
      SCOPED_TRACE(Message() << "Row: " << row << " Col: " << col);
      EXPECT_THAT(grid_.RawStorage()(row, col), Eq(original_data(row, col)));
    }
  }
}

TEST(GridTest, Access) {
  Grid<char> grid(GridFrame(), GridRange({0, 0}, {5, 5}), '.');
  for (const GridIndex &index : GridRange({3, 0}, {5, 5})) {
    grid.SetUnsafe(index, 'X');
  }

  EXPECT_EQ(grid.GetOrDefaultTo({0, 0}, 'o'), '.');
  EXPECT_EQ(grid.GetOrDefaultTo({3, 1}, 'o'), 'X');
  EXPECT_EQ(grid.GetOrDefaultTo({5, 3}, 'o'), 'o');
  EXPECT_EQ(grid.GetOrDefaultTo({3, -1}, 'o'), 'o');

  EXPECT_EQ(DumpCharGrid(grid), R"""(
([0; 0], [5; 5])
.....
.....
.....
XXXXX
XXXXX)""");
}

TEST(GridTest, Fill) {
  Grid<char> grid(GridFrame(), GridRange({0, 0}, {5, 5}), '.');

  EXPECT_EQ(grid.Rows(), 5);
  EXPECT_EQ(grid.Cols(), 5);
  EXPECT_EQ(grid.Range(), GridRange({0, 0}, {5, 5}));

  EXPECT_EQ(DumpCharGrid(grid), R"""(
([0; 0], [5; 5])
.....
.....
.....
.....
.....)""");

  grid.Fill('X');

  EXPECT_EQ(DumpCharGrid(grid), R"""(
([0; 0], [5; 5])
XXXXX
XXXXX
XXXXX
XXXXX
XXXXX)""");
}

TEST(GridTest, Copy) {
  Grid<char> grid(GridFrame(), GridRange({-2, -2}, {3, 3}), '.');

  EXPECT_EQ(grid.Rows(), 5);
  EXPECT_EQ(grid.Cols(), 5);
  EXPECT_EQ(grid.Range(), GridRange({-2, -2}, {3, 3}));
  // Get a large buffer.
  grid.Reserve(1000, 1000);
  EXPECT_GE(grid.RowCapacity(), 1000);
  EXPECT_GE(grid.ColCapacity(), 1000);

  int i = 0;
  for (int row = grid.LowerBound().x(); row < grid.UpperBound().x(); ++row) {
    for (int col = grid.LowerBound().y(); col < grid.UpperBound().y(); ++col) {
      grid.SetUnsafe(row, col, 'a' + i);
      ++i;
    }
  }

  Grid<char> copy1 = grid;
  Grid<char> copy2;
  copy2 = grid;
  EXPECT_LT(copy1.RowCapacity(), 1000);
  EXPECT_LT(copy1.ColCapacity(), 1000);
  EXPECT_LT(copy2.RowCapacity(), 1000);
  EXPECT_LT(copy2.ColCapacity(), 1000);

  EXPECT_THAT(DumpCharGrid(copy1), StrEq(DumpCharGrid(grid)));
  EXPECT_THAT(DumpCharGrid(copy2), StrEq(DumpCharGrid(grid)));

  EXPECT_EQ(DumpCharGrid(grid), R"""(
([-2; -2], [3; 3])
abcde
fghij
klmno
pqrst
uvwxy)""");
}

TEST(GridTest, ApplyInPlace) {
  Grid<char> grid(GridFrame(), GridRange({0, 0}, {5, 5}), '.');

  int i = 0;
  for (int row = grid.LowerBound().x(); row < grid.UpperBound().x(); ++row) {
    for (int col = grid.LowerBound().y(); col < grid.UpperBound().y(); ++col) {
      grid.SetUnsafe(row, col, 'a' + i);
      ++i;
    }
  }

  EXPECT_EQ(DumpCharGrid(grid), R"""(
([0; 0], [5; 5])
abcde
fghij
klmno
pqrst
uvwxy)""");

  grid.ApplyInPlace([](char c) { return c + ('A' - 'a'); });
  EXPECT_EQ(DumpCharGrid(grid), R"""(
([0; 0], [5; 5])
ABCDE
FGHIJ
KLMNO
PQRST
UVWXY)""");

  Grid<char> grid_copy = grid.Apply([](char c) { return c + ('a' - 'A'); });
  EXPECT_EQ(DumpCharGrid(grid_copy), R"""(
([0; 0], [5; 5])
abcde
fghij
klmno
pqrst
uvwxy)""");
}

TEST(GridTest, ForEachCellValue) {
  Grid<char> grid(GridFrame(), GridRange({0, 0}, {5, 5}), '.');

  int i = 0;
  for (int row = grid.LowerBound().x(); row < grid.UpperBound().x(); ++row) {
    for (int col = grid.LowerBound().y(); col < grid.UpperBound().y(); ++col) {
      grid.SetUnsafe(row, col, 'a' + i);
      ++i;
    }
  }

  EXPECT_EQ(DumpCharGrid(grid), R"""(
([0; 0], [5; 5])
abcde
fghij
klmno
pqrst
uvwxy)""");

  int char_sum = 0;
  grid.ForEachCellValue([&](char c) { char_sum += c; });
  EXPECT_EQ(char_sum, (('a' + 'y') / 2) * 25);
}

TEST(GridTest, ForEachCellValueWithPredicate) {
  Grid<char> grid(GridFrame(), GridRange({0, 0}, {5, 5}), '.');

  int i = 0;
  for (int row = grid.LowerBound().x(); row < grid.UpperBound().x(); ++row) {
    for (int col = grid.LowerBound().y(); col < grid.UpperBound().y(); ++col) {
      grid.SetUnsafe(row, col, 'a' + i);
      ++i;
    }
  }

  EXPECT_EQ(DumpCharGrid(grid), R"""(
([0; 0], [5; 5])
abcde
fghij
klmno
pqrst
uvwxy)""");

  int visit_count = 0;
  bool found_even_char = false;
  grid.ForEachCellValue([&](char c) {
    ++visit_count;
    if (c % 2 == 0) {
      found_even_char = true;
    }
    return !found_even_char;
  });
  // Check that we stopped early.
  EXPECT_LT(visit_count, grid.Range().ComputeSize());
}

TEST(GridTest, Slide) {
  Grid<char> grid(GridFrame(), GridRange({0, 0}, {5, 5}), '.');

  int i = 0;
  for (int row = grid.LowerBound().x(); row < grid.UpperBound().x(); ++row) {
    for (int col = grid.LowerBound().y(); col < grid.UpperBound().y(); ++col) {
      grid.SetUnsafe(row, col, 'a' + i);
      ++i;
    }
  }

  EXPECT_EQ(DumpCharGrid(grid), R"""(
([0; 0], [5; 5])
abcde
fghij
klmno
pqrst
uvwxy)""");

  grid.SlideBoundsBy({1, 0});
  EXPECT_EQ(DumpCharGrid(grid), R"""(
([1; 0], [6; 5])
fghij
klmno
pqrst
uvwxy
.....)""");

  grid.SlideBoundsBy({0, 1});
  EXPECT_EQ(DumpCharGrid(grid), R"""(
([1; 1], [6; 6])
ghij.
lmno.
qrst.
vwxy.
.....)""");

  grid.SlideBoundsBy({-1, -1});
  EXPECT_EQ(DumpCharGrid(grid), R"""(
([0; 0], [5; 5])
.....
.ghij
.lmno
.qrst
.vwxy)""");

  grid.SlideBoundsBy({-1, -1});
  EXPECT_EQ(DumpCharGrid(grid), R"""(
([-1; -1], [4; 4])
.....
.....
..ghi
..lmn
..qrs)""");

  grid.SlideBoundsTo({17, 0});
  EXPECT_EQ(DumpCharGrid(grid), R"""(
([17;  0], [22;  5])
.....
.....
.....
.....
.....)""");

  grid.SetUnsafe(18, 0, 'X');
  grid.SetUnsafe({19, 0}, 'Y');
  grid.SetUnsafe(20, 0, 'Z');
  EXPECT_TRUE(grid.Set({18, 1}, 'A'));
  EXPECT_TRUE(grid.Set(18, 2, 'B'));
  EXPECT_EQ(DumpCharGrid(grid), R"""(
([17;  0], [22;  5])
.....
XAB..
Y....
Z....
.....)""");
}

TEST(GridTest, BoundsCheck) {
  {
    Grid<char> grid(GridFrame(), GridRange({0, 0}, {5, 5}), '.');
    char c;
    EXPECT_FALSE(grid.Get(5, 5, &c)) << "Check failed.*Bounds.*Contains";
  }

  {
    Grid<char> grid(GridFrame(), GridRange({5, 5}, {100, 100}), '.');
    char c;
    EXPECT_FALSE(grid.Get(0, 0, &c)) << "Check failed.*Bounds.*Contains";
  }

  {
    Grid<char> grid(GridFrame(), GridRange({0, 0}, {5, 5}), '.');
    grid.SlideBoundsTo({100, 100});
    char c;
    EXPECT_FALSE(grid.Get(0, 0, &c)) << "Check failed.*Bounds.*Contains";
  }
}

TEST(GridTest, Assignment) {
  Grid<int> in, out;
  FillIntGrid(GridRange({0, 0}, {5, 5}), 22, &in);
  FillIntGrid(GridRange({1, 1}, {6, 6}), 22, &out);
  EXPECT_NE(in, out);
  FillIntGrid(GridRange({0, 0}, {5, 5}), 99, &out);
  EXPECT_NE(in, out);
  FillIntGrid(GridRange({0, 0}, {5, 6}), 22, &out);
  EXPECT_NE(in, out);
  out = in;
  EXPECT_EQ(in, out);
}

TEST(GridTest, GrowToInclude) {
  Grid<int> grid;
  FillIntGrid(GridRange({0, 0}, {5, 5}), 22, &grid);
  grid.SetDefaultValue(-222);
  grid.GrowToInclude({6, -1});
  for (int ii = 0; ii < 6; ++ii) {
    for (int jj = -1; jj < 5; ++jj) {
      int vv;
      EXPECT_TRUE(grid.Get({ii, jj}, &vv));
      if (ii < 5 && jj > -1) {
        EXPECT_NE(vv, -222);
      } else {
        EXPECT_EQ(vv, -222);
      }
    }
  }
}

TEST(GridTest, ReshapeBeyondMargin) {
  constexpr int kIncrease = 6;
  const GridIndex offsets[] = {
      GridIndex{-kIncrease, -kIncrease},
      GridIndex{-kIncrease, 0},
      GridIndex{-kIncrease, kIncrease},
      GridIndex{0, -kIncrease},
      GridIndex{0, 0},
      GridIndex{0, kIncrease},
      GridIndex{kIncrease, -kIncrease},
      GridIndex{kIncrease, 0},
      GridIndex{kIncrease, kIncrease},
  };
  const GridRange initial_ranges[] = {
      {{0, 0}, {0, 0}},
      {{0, 0}, {kIncrease / 2, kIncrease / 2}},
      {{0, 0}, {kIncrease + 5, kIncrease + 5}},
  };
  const char kInitialValue = 'i';
  const int kCheckValue = 'c';
  for (const auto &lower_offset : offsets) {
    for (const auto &upper_offset : offsets) {
      for (const auto &initial_range : initial_ranges) {
        Grid<char> grid(GridFrame(), initial_range);
        grid.Fill(kInitialValue);
        std::ostringstream oss;
        oss << "lower_offset " << lower_offset << "   upper_offset "
            << upper_offset << "\n"
            << "initial grid\n  " << initial_range << "\n"
            << DumpCharGrid(grid);
        grid.SetDefaultValue(kCheckValue);
        const GridRange range(initial_range.lower + lower_offset,
                              initial_range.upper + upper_offset);
        grid.Reshape(range);
        oss << "reshaped grid\n  " << range << "\n" << DumpCharGrid(grid);
        if (range.Empty()) {
          EXPECT_TRUE(grid.Empty());
        } else {
          range.ForEachGridCoord([&](const GridIndex &index) {
            char vv;
            EXPECT_TRUE(grid.Get(index, &vv)) << oss.str();
            if (initial_range.Contains(index)) {
              EXPECT_EQ(kInitialValue, vv) << oss.str();
            } else {
              EXPECT_EQ(kCheckValue, vv) << oss.str();
            }
          });
        }
      }
    }
  }
}

TEST(GridTest, ReshapeStressTest) {
  Grid<double> grid;
  EXPECT_FALSE(grid.IsInGridBounds({0, 0}));
  EXPECT_FALSE(grid.IsInGridBounds({-1, 0}));
  EXPECT_FALSE(grid.IsInGridBounds({0, -1}));
  EXPECT_FALSE(grid.IsInGridBounds({1, 0}));
  EXPECT_FALSE(grid.IsInGridBounds({0, 1}));
  EXPECT_FALSE(grid.IsInGridBounds({1, 1}));

  grid.Reshape({{0, 0}, {5, 5}});
  EXPECT_TRUE(grid.IsInGridBounds({0, 0}));
  EXPECT_FALSE(grid.IsInGridBounds({-1, 0}));
  EXPECT_FALSE(grid.IsInGridBounds({0, -1}));
  EXPECT_TRUE(grid.IsInGridBounds({1, 0}));
  EXPECT_TRUE(grid.IsInGridBounds({0, 1}));
  EXPECT_TRUE(grid.IsInGridBounds({1, 1}));
  EXPECT_FALSE(grid.IsInGridBounds({5, 5}));

  grid.Reshape({{-5, -5}, {0, 0}});
  EXPECT_FALSE(grid.IsInGridBounds({0, 0}));
  EXPECT_FALSE(grid.IsInGridBounds({-1, 0}));
  EXPECT_FALSE(grid.IsInGridBounds({0, -1}));
  EXPECT_FALSE(grid.IsInGridBounds({1, 0}));
  EXPECT_FALSE(grid.IsInGridBounds({0, 1}));
  EXPECT_FALSE(grid.IsInGridBounds({1, 1}));
  EXPECT_TRUE(grid.IsInGridBounds({-5, -5}));

  const std::function<double(const GridIndex &)> IndexToValue =
      [](const GridIndex &index) -> double {
    return 1000.0 + 17.0 * (index.x() + 7) + 42.0 * (index.y() + 13);
  };
  constexpr int kExtent = 16;
  const double default_value = double();
  for (GridIndex delta_upper = {-kExtent, -kExtent}; delta_upper.x() <= kExtent;
       delta_upper.x() += 4) {
    for (delta_upper.y() = -kExtent; delta_upper.y() <= kExtent;
         delta_upper.y() += 4) {
      for (GridIndex delta_lower{-kExtent, -kExtent};
           delta_lower.x() <= kExtent; delta_lower.x() += 4) {
        for (delta_lower.y() = -kExtent; delta_lower.y() <= kExtent;
             delta_lower.y() += 4) {
          grid.Reshape(GridRange());
          const GridRange old_range{{3, 3}, {9, 9}};
          grid.Reshape(old_range);
          old_range.ForEachGridCoord([&](const GridIndex &index) {
            const double value = IndexToValue(index);
            EXPECT_TRUE(grid.Set(index, value));
            double check;
            EXPECT_TRUE(grid.Get(index, &check));
            EXPECT_EQ(value, check);
          });

          const GridRange new_range{old_range.lower + delta_lower,
                                    old_range.upper + delta_upper};
          const GridRange check_range{
              {std::min(old_range.lower.x(), new_range.lower.x()),
               std::min(old_range.lower.y(), new_range.lower.y())},
              {std::max(old_range.upper.x(), new_range.upper.x()),
               std::max(old_range.upper.y(), new_range.upper.y())}};

          grid.Reshape(new_range);
          if (new_range.Empty()) {
            EXPECT_TRUE(grid.Empty())
                << "empty new range " << new_range
                << " but grid reports non-empty " << grid.Range();
          } else {
            EXPECT_TRUE(grid.Range() == new_range);
          }
          check_range.ForEachGridCoord([&](const GridIndex &index) {
            double check;
            if (old_range.Contains(index)) {
              if (new_range.Contains(index)) {
                EXPECT_TRUE(grid.Get(index, &check));
                EXPECT_EQ(check, IndexToValue(index));
              } else {
                EXPECT_FALSE(grid.Get(index, &check));
              }
            } else {
              if (new_range.Contains(index)) {
                EXPECT_TRUE(grid.Get(index, &check));
                EXPECT_EQ(check, default_value);
              } else {
                EXPECT_FALSE(grid.Get(index, &check))
                    << "old range " << old_range << " new range " << new_range
                    << " expected fail to get at " << index;
              }
            }
          });
        }
      }
    }
  }
}

// Disabled because randomized tests should not run as presubmits and
// this test can also be quite long to run.
TEST(GridTest, DISABLED_ReshapeRandomizedTest) {
  const std::function<double(const GridIndex &)> IndexToValue =
      [](const GridIndex &index) -> double {
    return 1000.0 + 17.0 * (index.x() + 7) + 42.0 * (index.y() + 13);
  };
  constexpr int kExtent = 128;
  const double default_value = double();
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(-kExtent, kExtent);
  for (int i = 0; i < 1e6; ++i) {
    Grid<double> grid;
    const GridRange old_range{{dis(gen), dis(gen)}, {dis(gen), dis(gen)}};
    grid.Reshape(old_range);
    old_range.ForEachGridCoord([&](const GridIndex &index) {
      const double value = IndexToValue(index);
      EXPECT_TRUE(grid.Set(index, value));
      double check;
      EXPECT_TRUE(grid.Get(index, &check));
      EXPECT_EQ(value, check);
    });
    const GridRange new_range{{dis(gen), dis(gen)}, {dis(gen), dis(gen)}};
    grid.Reshape(new_range);
    if (new_range.Empty()) {
      EXPECT_TRUE(grid.Empty())
          << "empty new range " << new_range << " but grid reports non-empty "
          << grid.Range();
    } else {
      EXPECT_TRUE(grid.Range() == new_range);
    }
    const GridRange check_range =
        GridRange::SpanningUnion(new_range, old_range);
    check_range.ForEachGridCoord([&](const GridIndex &index) {
      double check;
      if (old_range.Contains(index)) {
        if (new_range.Contains(index)) {
          EXPECT_TRUE(grid.Get(index, &check));
          EXPECT_EQ(check, IndexToValue(index));
        } else {
          EXPECT_FALSE(grid.Get(index, &check));
        }
      } else {
        if (new_range.Contains(index)) {
          EXPECT_TRUE(grid.Get(index, &check));
          EXPECT_EQ(check, default_value);
        } else {
          EXPECT_FALSE(grid.Get(index, &check))
              << "old range " << old_range << " new range " << new_range
              << " expected fail to get at " << index;
        }
      }
    });
  }
}

TEST(GridTest, GridLine) {
  struct {
    GridIndex from, to;
    const char *expected;
  } checks[] = {
      {{1, 3}, {8, 5}, R"""(
([0; 0], [10; 10])
..........
...*......
...*......
....*.....
....*.....
....*.....
....*.....
.....*....
.....*....
..........)"""},
      {{3, 7}, {6, 2}, R"""(
([0; 0], [10; 10])
..........
..........
..........
.......*..
.....**...
...**.....
..*.......
..........
..........
..........)"""},
  };

  const GridRange range{{0, 0}, {10, 10}};
  Grid<char> grid(range);
  for (int ii = 0; ii < sizeof(checks) / sizeof(*checks); ++ii) {
    for (int jj = 0; jj < 2; ++jj) {
      grid.Fill('.');
      GridIndex from = checks[ii].from;
      GridIndex to = checks[ii].to;
      if (0 != jj) {
        std::swap(from, to);
      }
      for (const GridIndex cell : GridLine(from, to)) {
        EXPECT_TRUE(grid.Set(cell, '*'));
      }
      const std::string result = DumpCharGrid(grid);
      if (result != checks[ii].expected) {
        ADD_FAILURE() << "line from " << from << " to " << to
                      << " should have yielded\n"
                      << checks[ii].expected << "\nbut instead resulted in\n"
                      << result;
      }
    }
  }
}

// Move to test utils.  See if similar function exists elsewhere.
template <typename Generator>
void FillOccupanciesRandomly(OccupancyGrid &grid, Generator &generator) {
  OccupancyStatus statuses[] = {OccupancyStatus::OCCUPIED,
                                OccupancyStatus::UNKNOWN,
                                OccupancyStatus::UNOCCUPIED};
  grid.ForEachCellValue([&](OccupancyStatus &status) {
    status = statuses[absl::Uniform(generator, 0, 3)];
  });
}

void BM_GridIterationByLogicalOrder(benchmark::State &state) {
  const int size = state.range(0);
  const GridIndex offset(12, 23);
  OccupancyGrid grid(offset, offset + GridIndex(size, size));
  eigenmath::TestGenerator generator(eigenmath::kGeneratorTestSeed);
  FillOccupanciesRandomly(grid, generator);
  for (auto _ : state) {
    grid.Range().ForEachGridCoord([&](const GridIndex &index) {
      benchmark::DoNotOptimize(grid.GetUnsafe(index));
    });
  }
}
BENCHMARK(BM_GridIterationByLogicalOrder)->Arg(1)->Arg(4)->Arg(10)->Arg(100);

void BM_GridIterationByStorageOrder(benchmark::State &state) {
  const int size = state.range(0);
  const GridIndex offset(12, 23);
  OccupancyGrid grid(offset, offset + GridIndex(size, size));
  eigenmath::TestGenerator generator(eigenmath::kGeneratorTestSeed);
  FillOccupanciesRandomly(grid, generator);
  for (auto _ : state) {
    grid.ForEachGridIndexWithCellValue(
        [](const GridIndex &index, OccupancyStatus status) {
          benchmark::DoNotOptimize(index);
          benchmark::DoNotOptimize(status);
        });
  }
}
BENCHMARK(BM_GridIterationByStorageOrder)->Arg(1)->Arg(4)->Arg(10)->Arg(100);

}  // namespace
}  // namespace mobility::collision

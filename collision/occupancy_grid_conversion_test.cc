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

#include "collision/occupancy_grid_conversion.h"

#include <string>

#include "collision/grid_common.h"
#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

TEST(OccupancyGridTest, ConvertToAndFromProto) {
  const std::string frame = "map";
  const eigenmath::Pose2d origin{{1.2, 4.5}, 5.5};
  constexpr int kRows = 3;
  constexpr int kCols = 4;
  constexpr double kResolution = 0.1;

  OccupancyGrid grid = {GridFrame(frame, origin, kResolution),
                        GridRange::OriginTo(kRows, kCols)};

  grid.SetUnsafe(0, 0, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(0, 1, OccupancyStatus::UNOCCUPIED);
  grid.SetUnsafe(0, 2, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(0, 3, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(1, 0, OccupancyStatus::UNOCCUPIED);
  grid.SetUnsafe(1, 1, OccupancyStatus::UNKNOWN);
  grid.SetUnsafe(1, 2, OccupancyStatus::UNKNOWN);
  grid.SetUnsafe(1, 3, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(2, 0, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(2, 1, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(2, 2, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(2, 3, OccupancyStatus::UNKNOWN);

  OccupancyGridProto proto;
  ASSERT_TRUE(ToProto(grid, &proto).ok());

  OccupancyGrid grid_back;
  ASSERT_TRUE(FromProto(proto, &grid_back).ok());

  EXPECT_EQ(grid_back, grid) << grid_back << "\n" << grid;
}

TEST(OccupancyGridTest, WithRangeOffsetConvertToAndFromProto) {
  const std::string frame = "map";
  const eigenmath::Pose2d origin{{1.2, 4.5}, 5.5};
  constexpr int kRows = 3;
  constexpr int kCols = 4;
  constexpr double kResolution = 0.1;
  const GridIndex grid_offset(7, 11);

  OccupancyGrid grid = {
      GridFrame(frame, origin, kResolution),
      GridRange::ShiftBy(GridRange::OriginTo(kRows, kCols), grid_offset)};

  grid.SetUnsafe(7, 11, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(7, 12, OccupancyStatus::UNOCCUPIED);
  grid.SetUnsafe(7, 13, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(7, 14, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(8, 11, OccupancyStatus::UNOCCUPIED);
  grid.SetUnsafe(8, 12, OccupancyStatus::UNKNOWN);
  grid.SetUnsafe(8, 13, OccupancyStatus::UNKNOWN);
  grid.SetUnsafe(8, 14, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(9, 11, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(9, 12, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(9, 13, OccupancyStatus::OCCUPIED);
  grid.SetUnsafe(9, 14, OccupancyStatus::UNKNOWN);

  OccupancyGridProto proto;
  ASSERT_TRUE(ToProto(grid, &proto).ok());

  OccupancyGrid converted_grid;
  ASSERT_TRUE(FromProto(proto, &converted_grid).ok());

  // The grids are not identical, but equivalent.
  for (const GridIndex& orig : grid.Range()) {
    OccupancyStatus status;
    ASSERT_TRUE(converted_grid.Get(orig, &status));
    EXPECT_EQ(status, grid.GetUnsafe(orig));
  }
}

TEST(OccupancyGridTest, ConvertFromFillProto) {
  std::string proto_str = R"""(
    grid_range {
      upper_x: 300
      upper_y: 400
    }

    grid_frame {
      frame_id: "robot"
      origin {
        tx: -15
        ty: -20
      }
      resolution: 0.1
    }
  
    occupancy_fill: UNKNOWN
    
  )""";
  OccupancyGridProto proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_str, &proto));

  OccupancyGrid grid;
  ASSERT_TRUE(FromProto(proto, &grid).ok());

  EXPECT_EQ(grid.Rows(), 300);
  EXPECT_EQ(grid.Cols(), 400);
  grid.Range().ForEachGridCoord([grid](const GridIndex& index) {
    EXPECT_EQ(grid.GetUnsafe(index), OccupancyStatus::UNKNOWN);
  });
}

}  // namespace
}  // namespace mobility::collision

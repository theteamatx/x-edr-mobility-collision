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

#include "collision/grid_common_conversion.h"

#include <string>

#include "collision/grid_common.h"
#include "collision/grid_common.pb.h"
#include "eigenmath/matchers.h"
#include "eigenmath/pose2.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

using eigenmath::testing::IsApprox;

TEST(GridRangeConversionTest, GridRange) {
  GridRange grid_range(GridIndex(1, 4), GridIndex(3, 7));
  GridRangeProto proto;
  ToProto(grid_range, proto);

  GridRange converted_grid_range;
  FromProto(proto, converted_grid_range);
  EXPECT_EQ(converted_grid_range.lower, grid_range.lower);
  EXPECT_EQ(converted_grid_range.upper, grid_range.upper);
}

TEST(GridFrameConversionTest, GridFrame) {
  const std::string frame = "map";
  const eigenmath::Pose2d origin{{1.2, 4.5}, 5.5};
  constexpr double kResolution = 0.1;
  GridFrame grid_frame(frame, origin, kResolution);
  GridFrameProto proto;
  ToProto(grid_frame, proto);
  GridFrame converted_grid_frame;
  FromProto(proto, converted_grid_frame);
  EXPECT_EQ(grid_frame.frame_id, converted_grid_frame.frame_id);
  EXPECT_EQ(grid_frame.resolution, converted_grid_frame.resolution);
  EXPECT_THAT(grid_frame.origin, IsApprox(converted_grid_frame.origin));
}

}  // namespace
}  // namespace mobility::collision

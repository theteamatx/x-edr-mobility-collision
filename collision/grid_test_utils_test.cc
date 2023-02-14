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

#include "collision/grid_test_utils.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

TEST(OccupancyGrid, DumpIsLeftInverseOfCreate) {
  const char* drawn_grids[] = {R"""(
...
...)""",
                               R"""(
..##
#??.)"""};
  for (const auto& drawing : drawn_grids) {
    OccupancyGrid grid;
    CreateOccupancyGrid(drawing, &grid);
    EXPECT_THAT(DumpOccupancyGrid(grid), ::testing::StrEq(drawing));
  }
}

}  // namespace
}  // namespace mobility::collision

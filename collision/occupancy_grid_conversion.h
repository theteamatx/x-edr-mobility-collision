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

// Helper functions for the conversion between mobility:OccupancyGrid and
// mobility::OccupancyGridProto messages.

#ifndef MOBILITY_COLLISION_COLLISION_OCCUPANCY_GRID_CONVERSION_H_
#define MOBILITY_COLLISION_COLLISION_OCCUPANCY_GRID_CONVERSION_H_

#include "absl/status/status.h"
#include "collision/occupancy_grid.h"
#include "collision/occupancy_grid.pb.h"

namespace mobility::collision {

// Returns an occupancy status corresponding to a pixel value.
OccupancyStatus PixelToOccupancy(uint8_t pixel);

// Returns a uint8 pixel value corresponding to an occupancy status.
uint8_t OccupancyToPixel(OccupancyStatus pixel);

absl::Status ToProto(const OccupancyGrid& data_in,
                     OccupancyGridProto* proto_out);

absl::Status FromProto(const OccupancyGridProto& proto_in,
                       OccupancyGrid* data_out);

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_OCCUPANCY_GRID_CONVERSION_H_

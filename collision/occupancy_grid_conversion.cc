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

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "collision/grid_common.h"
#include "collision/grid_common_conversion.h"
#include "collision/occupancy_grid.h"
#include "eigenmath/eigenmath.pb.h"

namespace mobility::collision {

OccupancyStatus PixelToOccupancy(const uint8_t pixel) {
  if (pixel == 0) {
    return OccupancyStatus::OCCUPIED;
  } else if (pixel == 255) {
    return OccupancyStatus::UNOCCUPIED;
  } else {
    return OccupancyStatus::UNKNOWN;
  }
}

uint8_t OccupancyToPixel(const OccupancyStatus pixel) {
  switch (pixel) {
    case OccupancyStatus::OCCUPIED: {
      return 0;
    }
    case OccupancyStatus::UNOCCUPIED: {
      return 255;
    }
    case OccupancyStatus::UNKNOWN: {
      return 100;
    }
    default: {
      CHECK(false) << "Bad occupancy status: (int) " << static_cast<int>(pixel);
    }
  }
}

absl::Status ToProto(const OccupancyGrid& data_in,
                     OccupancyGridProto* proto_out) {
  CHECK_NE(proto_out, nullptr);
  ToProto(data_in.Frame(), *proto_out->mutable_grid_frame());
  ToProto(data_in.Range(), *proto_out->mutable_grid_range());
  std::vector<uint8_t> buffer(data_in.Range().XSpan() *
                              data_in.Range().YSpan());
  int i = 0;
  data_in.Range().ForEachGridCoord([&](const GridIndex& index) {
    buffer[i] = OccupancyToPixel(data_in.GetUnsafe(index));
    ++i;
  });
  proto_out->set_occupancy_raw(reinterpret_cast<char*>(buffer.data()),
                               buffer.size());
  return absl::OkStatus();
}

absl::Status FromProto(const OccupancyGridProto& proto_in,
                       OccupancyGrid* data_out) {
  CHECK_NE(data_out, nullptr);
  GridFrame frame_out;
  FromProto(proto_in.grid_frame(), frame_out);
  GridRange range_out;
  FromProto(proto_in.grid_range(), range_out);

  switch (proto_in.occupancy_data_case()) {
    case OccupancyGridProto::kOccupancyPng:
      return absl::InvalidArgumentError(
          "OccupancyGridProto::occupancy_png not supported!");
    case OccupancyGridProto::kOccupancyFill:
      *data_out = OccupancyGrid(
          frame_out, range_out,
          static_cast<OccupancyStatus>(proto_in.occupancy_fill()));
      break;
    case OccupancyGridProto::kOccupancyRaw: {
      *data_out = OccupancyGrid(frame_out, range_out);
      const std::vector<uint8_t> buffer(
          reinterpret_cast<const uint8_t*>(proto_in.occupancy_raw().data()),
          reinterpret_cast<const uint8_t*>(proto_in.occupancy_raw().data()) +
              proto_in.occupancy_raw().size());
      int i = 0;
      data_out->Range().ForEachGridCoord([&](const GridIndex& index) {
        data_out->SetUnsafe(index, PixelToOccupancy(buffer[i]));
        ++i;
      });
      break;
    }
    case OccupancyGridProto::OCCUPANCY_DATA_NOT_SET: {
      return absl::InvalidArgumentError(
          "OccupancyGridProto::occupancy_data not set.");
    } break;
  }
  return absl::OkStatus();
}

}  // namespace mobility::collision

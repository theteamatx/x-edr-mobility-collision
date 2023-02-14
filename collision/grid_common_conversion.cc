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

#include "eigenmath/conversions.h"
#include "eigenmath/eigenmath.pb.h"

namespace mobility::collision {

void ToProto(const GridRange& data_in, GridRangeProto& proto_out) {
  proto_out.set_lower_x(data_in.lower.x());
  proto_out.set_lower_y(data_in.lower.y());
  proto_out.set_upper_x(data_in.upper.x());
  proto_out.set_upper_y(data_in.upper.y());
}

void FromProto(const GridRangeProto& proto, GridRange& data_out) {
  data_out.lower.x() = proto.lower_x();
  data_out.lower.y() = proto.lower_y();
  data_out.upper.x() = proto.upper_x();
  data_out.upper.y() = proto.upper_y();
}

void ToProto(const GridFrame& grid_frame, GridFrameProto& proto) {
  proto.set_frame_id(grid_frame.frame_id);
  *proto.mutable_origin() =
      eigenmath::conversions::ProtoFromPose(grid_frame.origin);
  proto.set_resolution(grid_frame.resolution);
}

void FromProto(const GridFrameProto& proto, GridFrame& grid_frame) {
  grid_frame = GridFrame(proto.frame_id(),
                         eigenmath::conversions::PoseFromProto(proto.origin()),
                         proto.resolution());
}

}  // namespace mobility::collision

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

#ifndef MOBILITY_COLLISION_COLLISION_GRID_COMMON_CONVERSION_H_
#define MOBILITY_COLLISION_COLLISION_GRID_COMMON_CONVERSION_H_

#include "collision/grid_common.h"
#include "collision/grid_common.pb.h"
#include "eigenmath/eigenmath.pb.h"

namespace mobility::collision {

void FromProto(const GridRangeProto& proto, GridRange& data_out);

void ToProto(const GridRange& data_in, GridRangeProto& proto_out);

void ToProto(const GridFrame& grid_frame, GridFrameProto& proto);

void FromProto(const GridFrameProto& proto, GridFrame& grid_frame);

}  // namespace mobility::collision

#endif  // MOBILITY_COLLISION_COLLISION_GRID_COMMON_CONVERSION_H_

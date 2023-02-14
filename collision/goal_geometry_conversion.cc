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

#include "collision/goal_geometry_conversion.h"

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "collision/goal_geometry.h"
#include "collision/goal_geometry.pb.h"
#include "collision/hull_conversion.h"
#include "eigenmath/conversions.h"
#include "eigenmath/eigenmath.pb.h"

namespace mobility::collision {

absl::Status FromProto(const GoalGeometryProto& proto, GoalGeometry* data_out) {
  data_out->Clear();

  // Attraction field:
  switch (proto.attraction_spec_case()) {
    case GoalGeometryProto::kUniform:
      data_out->SetAttractionToNothing();
      break;
    case GoalGeometryProto::kAttractionPoint:
      data_out->SetAttractionPoint(eigenmath::conversions::EigenVectorFromProto(
          proto.attraction_point()));
      break;
    case GoalGeometryProto::kAttractionDirection:
      data_out->SetAttractionDirection(
          eigenmath::conversions::EigenVectorFromProto(
              proto.attraction_direction()));
      break;
    case GoalGeometryProto::ATTRACTION_SPEC_NOT_SET:
      data_out->SetAttractionToNothing();
      break;
    default:
      data_out->SetAttractionToNothing();
      break;
  }

  // Orientation:
  switch (proto.orientation_spec_case()) {
    case GoalGeometryProto::kNoOrientation:
      data_out->SetArbitraryOrientation();
      break;
    case GoalGeometryProto::kOrientation:
      data_out->SetFixedOrientation(proto.orientation(),
                                    proto.orientation_radius());
      break;
    case GoalGeometryProto::kFaceTowardsPoint:
      data_out->SetOrientationTarget(
          eigenmath::conversions::EigenVectorFromProto(
              proto.face_towards_point()),
          proto.orientation_radius());
      break;
    case GoalGeometryProto::ORIENTATION_SPEC_NOT_SET:
      data_out->SetArbitraryOrientation();
      break;
    default:
      data_out->SetArbitraryOrientation();
      break;
  }

  // Inclusion zone:
  switch (proto.inclusion_zone().geometry_case()) {
    case GoalZoneProto::kRadialSegment: {
      auto& radial_segment = proto.inclusion_zone().radial_segment();
      data_out->SetInclusionRadialSegment(
          eigenmath::conversions::EigenVectorFromProto(radial_segment.center()),
          radial_segment.inner_radius(), radial_segment.outer_radius(),
          radial_segment.start_angle(), radial_segment.end_angle());
      break;
    }
    case GoalZoneProto::kHull: {
      auto& hull = proto.inclusion_zone().hull();
      Hull hull_out;
      auto hull_status = FromProto(hull, &hull_out);
      if (!hull_status.ok()) {
        return hull_status;
      }
      data_out->SetInclusionHull(hull_out);
      break;
    }
    case GoalZoneProto::GEOMETRY_NOT_SET:
      data_out->SetNoInclusionZone();
      break;
  }

  // Exclusion zone:
  switch (proto.exclusion_zone().geometry_case()) {
    case GoalZoneProto::kRadialSegment: {
      auto& radial_segment = proto.exclusion_zone().radial_segment();
      data_out->SetExclusionRadialSegment(
          eigenmath::conversions::EigenVectorFromProto(radial_segment.center()),
          radial_segment.inner_radius(), radial_segment.outer_radius(),
          radial_segment.start_angle(), radial_segment.end_angle());
      break;
    }
    case GoalZoneProto::kHull: {
      auto& hull = proto.exclusion_zone().hull();
      Hull hull_out;
      auto hull_status = FromProto(hull, &hull_out);
      if (!hull_status.ok()) {
        return hull_status;
      }
      data_out->SetExclusionHull(hull_out);
      break;
    }
    case GoalZoneProto::GEOMETRY_NOT_SET:
      data_out->SetNoExclusionZone();
      break;
  }

  data_out->SetDistanceTolerance(proto.distance_tolerance());
  data_out->SetOrientationTolerance(proto.orientation_tolerance());

  return absl::OkStatus();
}

absl::Status ToProto(const GoalGeometry& data_in,
                     GoalGeometryProto* proto_out) {
  // Attraction field:
  if (data_in.IsAttractedToSinglePoint()) {
    *proto_out->mutable_attraction_point() =
        eigenmath::conversions::ProtoFromVector2d(data_in.GetAttractionPoint());
  } else if (data_in.IsAttractedInDirection()) {
    *proto_out->mutable_attraction_direction() =
        eigenmath::conversions::ProtoFromVector2d(
            data_in.GetAttractionDirection());
  } else {
    proto_out->mutable_uniform();
  }

  // Orientation:
  if (data_in.IsOrientationFixed()) {
    proto_out->set_orientation(data_in.GetFixedOrientation());
  } else if (data_in.IsToFaceTowardsPoint()) {
    *proto_out->mutable_face_towards_point() =
        eigenmath::conversions::ProtoFromVector2d(
            data_in.GetOrientationTarget());
  } else {
    proto_out->mutable_no_orientation();
  }
  proto_out->set_orientation_radius(data_in.GetOrientationRange());

  // Inclusion zone:
  if (data_in.HasInclusionRadialSegment()) {
    auto& radial_segment_in = data_in.GetInclusionRadialSegment();
    auto* radial_goal =
        proto_out->mutable_inclusion_zone()->mutable_radial_segment();
    *radial_goal->mutable_center() =
        eigenmath::conversions::ProtoFromVector2d(radial_segment_in.center);
    radial_goal->set_inner_radius(radial_segment_in.inner_radius);
    radial_goal->set_outer_radius(radial_segment_in.outer_radius);
    radial_goal->set_start_angle(radial_segment_in.start_angle);
    radial_goal->set_end_angle(radial_segment_in.end_angle);
  } else if (data_in.HasInclusionHull()) {
    auto& hull_in = data_in.GetInclusionHull();
    auto hull_status =
        ToProto(hull_in, proto_out->mutable_inclusion_zone()->mutable_hull());
    if (!hull_status.ok()) {
      return hull_status;
    }
  }

  // Exclusion zone:
  if (data_in.HasExclusionRadialSegment()) {
    auto& radial_segment_in = data_in.GetExclusionRadialSegment();
    auto* radial_goal =
        proto_out->mutable_exclusion_zone()->mutable_radial_segment();
    *radial_goal->mutable_center() =
        eigenmath::conversions::ProtoFromVector2d(radial_segment_in.center);
    radial_goal->set_inner_radius(radial_segment_in.inner_radius);
    radial_goal->set_outer_radius(radial_segment_in.outer_radius);
    radial_goal->set_start_angle(radial_segment_in.start_angle);
    radial_goal->set_end_angle(radial_segment_in.end_angle);
  } else if (data_in.HasExclusionHull()) {
    auto& hull_in = data_in.GetExclusionHull();
    auto hull_status =
        ToProto(hull_in, proto_out->mutable_exclusion_zone()->mutable_hull());
    if (!hull_status.ok()) {
      return hull_status;
    }
  }

  // Distance / orientation tolerances:
  proto_out->set_distance_tolerance(data_in.GetDistanceTolerance());
  proto_out->set_orientation_tolerance(data_in.GetOrientationTolerance());

  return absl::OkStatus();
}

}  // namespace mobility::collision

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

#include <cmath>
#include <string>

#include "collision/goal_geometry.h"
#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

TEST(GoalGeometryConversion, LegacyGoalGeometry) {
  std::string proto_str = R"""(
    attraction_point {
      vec: 1.0
      vec: 2.0
    }
    inclusion_zone {
      radial_segment {
        center {
          vec: 1.0
          vec: 2.0
        }
        inner_radius: 0.0
        outer_radius: 1.0
        start_angle: 0.0
        end_angle: 0.0
      }
    }
    distance_tolerance: 0.5
    orientation: 1.57
    orientation_tolerance: 0.1
    orientation_radius: 0.05
  )""";
  GoalGeometryProto proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_str, &proto));

  GoalGeometry data_out;
  ASSERT_TRUE(FromProto(proto, &data_out).ok());

  GoalGeometryProto proto_out;
  ASSERT_TRUE(ToProto(data_out, &proto_out).ok());

  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equivalent(proto_out, proto));
}

TEST(GoalGeometryConversion, GoalGeometryUniformInclZoneNoOrientation) {
  std::string proto_str = R"""(
    uniform {}
    inclusion_zone {
      radial_segment {
        center {
          vec: 1.0
          vec: 2.0
        }
        inner_radius: 0.1
        outer_radius: 1.0
        start_angle: 0.5
        end_angle: 3.0
      }
    }
    exclusion_zone {
      radial_segment {
        center {
          vec: 1.2
          vec: 2.3
        }
        inner_radius: 0.05
        outer_radius: 0.4
        start_angle: 0.2
        end_angle: 3.1
      }
    }
    distance_tolerance: 0.5
    no_orientation {}
    orientation_radius: 10
  )""";
  GoalGeometryProto proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_str, &proto));

  GoalGeometry data_out;
  ASSERT_TRUE(FromProto(proto, &data_out).ok());

  GoalGeometryProto proto_out;
  ASSERT_TRUE(ToProto(data_out, &proto_out).ok());

  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equivalent(proto_out, proto));
}

TEST(GoalGeometryConversion, GoalGeometryAttractionDirHullsFaceTowards) {
  std::string proto_str = R"""(
    attraction_direction {
      vec: 0.54030230586596639
      vec: 0.84147098480929194
    }
    inclusion_zone {
      hull {
        convex_hulls {
          points {
            vec: 0.0
            vec: 0.0
          }
          points {
            vec: 1.0
            vec: 0.0
          }
          points {
            vec: 1.0
            vec: 1.0
          }
          points {
            vec: 0.0
            vec: 1.0
          }
        }
      }
    }
    exclusion_zone {
      hull {
        convex_hulls {
          points {
            vec: 0.4
            vec: 0.4
          }
          points {
            vec: 0.6
            vec: 0.4
          }
          points {
            vec: 0.6
            vec: 0.6
          }
          points {
            vec: 0.4
            vec: 0.6
          }
        }
      }
    }
    distance_tolerance: 0.5
    face_towards_point {
      vec: 4.0
      vec: 5.0
    }
    orientation_tolerance: 0.2
    orientation_radius: 0.05
  )""";
  GoalGeometryProto proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_str, &proto));

  GoalGeometry data_out;
  ASSERT_TRUE(FromProto(proto, &data_out).ok());

  GoalGeometryProto proto_out;
  ASSERT_TRUE(ToProto(data_out, &proto_out).ok());

  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equivalent(proto_out, proto));
}

}  // namespace
}  // namespace mobility::collision

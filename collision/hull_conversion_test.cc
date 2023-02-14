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

#include "collision/hull_conversion.h"

#include <cmath>
#include <string>

#include "collision/hull.h"
#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

constexpr double kEpsilon = 1e-6;

TEST(HullConversion, Hull) {
  std::string proto_str = R"""(
    convex_hulls {
      points {
        vec: 0.1
        vec: 0.1
      }
      points {
        vec: -0.1
        vec: 0.1
      }
      points {
        vec: 0.1
        vec: -0.1
      }
      points {
        vec: -0.1
        vec: -0.1
      }
    }
    convex_hulls {
      points {
        vec: 0.5
        vec: 0.5
      }
      points {
        vec: 1.0
        vec: 0.5
      }
      points {
        vec: 1.0
        vec: 1.0
      }
      points {
        vec: 0.5
        vec: 1.0
      }
    }
    )""";
  HullProto proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_str, &proto));

  Hull data_out;
  ASSERT_TRUE(FromProto(proto, &data_out).ok());

  ASSERT_EQ(data_out.GetConvexHulls().size(), 2);

  auto& hull0 = data_out.GetConvexHulls()[0];
  ASSERT_EQ(hull0.GetPoints().size(), 4);
  EXPECT_NEAR(hull0.GetPoints()[0].x(), -0.1, kEpsilon);
  EXPECT_NEAR(hull0.GetPoints()[0].y(), -0.1, kEpsilon);
  EXPECT_NEAR(hull0.GetPoints()[1].x(), 0.1, kEpsilon);
  EXPECT_NEAR(hull0.GetPoints()[1].y(), -0.1, kEpsilon);
  EXPECT_NEAR(hull0.GetPoints()[2].x(), 0.1, kEpsilon);
  EXPECT_NEAR(hull0.GetPoints()[2].y(), 0.1, kEpsilon);
  EXPECT_NEAR(hull0.GetPoints()[3].x(), -0.1, kEpsilon);
  EXPECT_NEAR(hull0.GetPoints()[3].y(), 0.1, kEpsilon);

  auto& hull1 = data_out.GetConvexHulls()[1];
  ASSERT_EQ(hull1.GetPoints().size(), 4);
  EXPECT_NEAR(hull1.GetPoints()[0].x(), 0.5, kEpsilon);
  EXPECT_NEAR(hull1.GetPoints()[0].y(), 0.5, kEpsilon);
  EXPECT_NEAR(hull1.GetPoints()[1].x(), 1.0, kEpsilon);
  EXPECT_NEAR(hull1.GetPoints()[1].y(), 0.5, kEpsilon);
  EXPECT_NEAR(hull1.GetPoints()[2].x(), 1.0, kEpsilon);
  EXPECT_NEAR(hull1.GetPoints()[2].y(), 1.0, kEpsilon);
  EXPECT_NEAR(hull1.GetPoints()[3].x(), 0.5, kEpsilon);
  EXPECT_NEAR(hull1.GetPoints()[3].y(), 1.0, kEpsilon);

  EXPECT_TRUE(data_out.Contains(0.09, 0.09));
  EXPECT_TRUE(data_out.Contains(0.09, -0.09));
  EXPECT_TRUE(data_out.Contains(-0.09, -0.09));
  EXPECT_TRUE(data_out.Contains(-0.09, 0.09));
  EXPECT_FALSE(data_out.Contains(0.09, 0.11));
  EXPECT_FALSE(data_out.Contains(0.11, -0.09));
  EXPECT_FALSE(data_out.Contains(-0.09, -0.11));
  EXPECT_FALSE(data_out.Contains(-0.11, 0.09));

  EXPECT_TRUE(data_out.Contains(0.99, 0.99));
  EXPECT_TRUE(data_out.Contains(0.99, 0.51));
  EXPECT_TRUE(data_out.Contains(0.51, 0.51));
  EXPECT_TRUE(data_out.Contains(0.51, 0.99));
  EXPECT_FALSE(data_out.Contains(0.99, 1.01));
  EXPECT_FALSE(data_out.Contains(1.01, 0.51));
  EXPECT_FALSE(data_out.Contains(0.51, 0.49));
  EXPECT_FALSE(data_out.Contains(0.49, 0.99));

  HullProto proto_out;
  ASSERT_TRUE(ToProto(data_out, &proto_out).ok());
  ASSERT_EQ(proto_out.convex_hulls_size(), 2);

  auto& proto_hull0 = proto_out.convex_hulls(0).points();
  ASSERT_EQ(proto_hull0.size(), 4);
  EXPECT_NEAR(proto_hull0[0].vec(0), -0.1, kEpsilon);
  EXPECT_NEAR(proto_hull0[0].vec(1), -0.1, kEpsilon);
  EXPECT_NEAR(proto_hull0[1].vec(0), 0.1, kEpsilon);
  EXPECT_NEAR(proto_hull0[1].vec(1), -0.1, kEpsilon);
  EXPECT_NEAR(proto_hull0[2].vec(0), 0.1, kEpsilon);
  EXPECT_NEAR(proto_hull0[2].vec(1), 0.1, kEpsilon);
  EXPECT_NEAR(proto_hull0[3].vec(0), -0.1, kEpsilon);
  EXPECT_NEAR(proto_hull0[3].vec(1), 0.1, kEpsilon);

  auto& proto_hull1 = proto_out.convex_hulls(1).points();
  ASSERT_EQ(proto_hull1.size(), 4);
  EXPECT_NEAR(proto_hull1[0].vec(0), 0.5, kEpsilon);
  EXPECT_NEAR(proto_hull1[0].vec(1), 0.5, kEpsilon);
  EXPECT_NEAR(proto_hull1[1].vec(0), 1.0, kEpsilon);
  EXPECT_NEAR(proto_hull1[1].vec(1), 0.5, kEpsilon);
  EXPECT_NEAR(proto_hull1[2].vec(0), 1.0, kEpsilon);
  EXPECT_NEAR(proto_hull1[2].vec(1), 1.0, kEpsilon);
  EXPECT_NEAR(proto_hull1[3].vec(0), 0.5, kEpsilon);
  EXPECT_NEAR(proto_hull1[3].vec(1), 1.0, kEpsilon);
}

}  // namespace
}  // namespace mobility::collision

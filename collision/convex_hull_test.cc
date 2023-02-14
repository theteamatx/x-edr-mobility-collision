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

#include "collision/convex_hull.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"
#include "collision/collision_hull_augmentation.h"
#include "collision/hull.h"
#include "collision/oriented_box_2d.h"
#include "diff_drive/test_trajectories.h"
#include "eigenmath/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mobility::collision {

namespace {
constexpr double kEpsilon = 1.0e-6;

using ::testing::ContainerEq;
using ::testing::UnorderedElementsAreArray;

// Creates a convex hull from the unit square.
ConvexHull Square() {
  std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}};
  return ConvexHull{points};
}

diff_drive::State ForwardMotion() {
  diff_drive::State state;
  state.SetArcVelocity({/*Translation=*/1.0, /*Rotation=*/0.0});
  return state;
}

template <typename Generator>
ConvexHull SampleConvexHull(int n_points, Generator generator) {
  std::vector<eigenmath::Vector2d> points;
  for (int i = 0; i < n_points; ++i) {
    points.emplace_back(generator(), generator());
  }
  ConvexHull hull(points);
  while (hull.GetPoints().size() != n_points) {
    points = hull.GetPoints();
    points.emplace_back(generator(), generator());
    hull = ConvexHull(points);
  }
  return hull;
}

double BruteForceDistanceBetween(const ConvexHull& lhs, const ConvexHull& rhs) {
  double min_distance = std::numeric_limits<double>::max();
  for (auto& lhs_point : lhs.GetPoints()) {
    const double dist = rhs.Distance(lhs_point);
    if (dist < min_distance) {
      min_distance = dist;
    }
  }
  for (auto& rhs_point : rhs.GetPoints()) {
    const double dist = lhs.Distance(rhs_point);
    if (dist < min_distance) {
      min_distance = dist;
    }
  }
  return min_distance;
}

TEST(ConvexHull, Create) {
  constexpr double kPerturbation = std::numeric_limits<double>::epsilon();
  const struct {
    std::vector<eigenmath::Vector2d> in;
    std::vector<eigenmath::Vector2d> out;
    eigenmath::Vector2d centroid;
    double radius;
  } checks[] = {
      {{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}},
       {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}},
       {0.33333333, 0.33333333},
       0.745356},
      {{{1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}},
       {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}},
       {0.33333333, 0.33333333},
       0.745356},
      {{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {0.1, 0.1}},
       {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}},
       {0.33333333, 0.33333333},
       0.745356},
      {{{1.0, 0.0},
        {1.0, 0.0},
        {1.0, 0.0 + kPerturbation},
        {0.0, 0.0},
        {0.0 + kPerturbation, 0.0},
        {0.0, 0.0},
        {0.0, 1.0},
        {0.0, 1.0 + kPerturbation},
        {0.0, 1.0}},
       {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}},
       {0.33333333, 0.33333333},
       0.745356},
      {{{0.0, 0.0},
        {0.0, 0.0 + kPerturbation},
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 0.0},
        {1.0 + kPerturbation, 0.0},
        {0.0, 1.0 + kPerturbation},
        {0.0, 1.0},
        {0.0, 1.0}},
       {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}},
       {0.33333333, 0.33333333},
       0.745356},
  };
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    const ConvexHull hull(checks[ii].in);
    ASSERT_EQ(checks[ii].out.size(), hull.GetPoints().size())
        << "check[" << ii << "] should have resulted in "
        << checks[ii].out.size() << " hull points, but there are "
        << hull.GetPoints().size();
    EXPECT_THAT(hull.GetCentroid(),
                eigenmath::testing::IsApprox(checks[ii].centroid, kEpsilon))
        << "Failed on check[" << ii << "].centroid";
    EXPECT_NEAR(checks[ii].radius, hull.GetRadius(), kEpsilon)
        << "check[" << ii << "].radius should be " << checks[ii].radius
        << " but is " << hull.GetRadius();
    for (int jj = 0; jj < checks[ii].out.size(); ++jj) {
      EXPECT_THAT(hull.GetPoints()[jj],
                  eigenmath::testing::IsApprox(checks[ii].out[jj], kEpsilon))
          << "Failed on check[" << ii << "].out[" << jj << "]";
    }
    for (int jj = 0; jj < checks[ii].in.size(); ++jj) {
      EXPECT_TRUE(hull.Contains(checks[ii].in[jj]))
          << "check[" << ii << "] pt[" << jj
          << "] = " << checks[ii].in[jj].transpose()
          << " should be contained in hull.";
    }
  }
}

TEST(ConvexHull, Randomized) {
  constexpr double pad = 0.1;
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(-1.0, 1.0);
  std::vector<eigenmath::Vector2d> points;
  for (int ii = 0; ii < 1000; ++ii) {
    points.emplace_back(uni_dist(rand_eng), uni_dist(rand_eng));
  }
  for (int ii = 0; ii < 1000 - 6; ++ii) {
    const std::vector<eigenmath::Vector2d> hull_pts(points.begin() + ii,
                                                    points.begin() + ii + 6);
    const ConvexHull hull(hull_pts);
    const eigenmath::Vector2d centroid = hull.GetCentroid();
    const double radius = hull.GetRadius();
    for (auto& pt : hull_pts) {
      const eigenmath::Vector2d dp = pt - centroid;
      const double sqr_dist = dp.squaredNorm();
      EXPECT_LE(sqr_dist - kEpsilon, radius * radius);
      EXPECT_TRUE(hull.Contains(pt));
    }
    // This is just to check that we don't have completely crazy growth:
    const auto grown = hull.CreateBiggerHull(pad);
    EXPECT_LT(grown.GetRadius(), radius + 1.5 * pad);
    EXPECT_LT((grown.GetCentroid() - centroid).norm(), 2.0 * pad);
  }
}

TEST(ConvexHull, RandomizedIsApprox) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(-1.0, 1.0);
  std::vector<eigenmath::Vector2d> points;
  for (int ii = 0; ii < 100; ++ii) {
    points.emplace_back(uni_dist(rand_eng), uni_dist(rand_eng));
  }
  for (int ii = 0; ii < 100 - 6; ++ii) {
    const std::vector<eigenmath::Vector2d> hull_pts(points.begin() + ii,
                                                    points.begin() + ii + 6);
    std::vector<eigenmath::Vector2d> hull_pts_shuffled = hull_pts;
    std::shuffle(hull_pts_shuffled.begin(), hull_pts_shuffled.end(), rand_eng);
    const ConvexHull hull(hull_pts);
    const ConvexHull hull_shuffled(hull_pts_shuffled);
    EXPECT_THAT(hull_shuffled.GetCentroid(),
                eigenmath::testing::IsApprox(hull.GetCentroid(), kEpsilon));
    EXPECT_NEAR(hull.GetRadius(), hull_shuffled.GetRadius(), kEpsilon);
    EXPECT_TRUE(hull.IsApprox(hull_shuffled, kEpsilon));
  }
}

TEST(ConvexHull, Grow) {
  constexpr double pad = 0.1;
  const double spi4 = sin(M_PI / 4.0);
  const double spi8 = sin(M_PI / 8.0);
  const double cpi8 = cos(M_PI / 8.0);
  const struct {
    std::vector<eigenmath::Vector2d> in;
    std::vector<eigenmath::Vector2d> out;
  } checks[] = {
      {{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}},
       {{1.0, -pad},
        {1.0 + pad * cpi8, -pad * spi8},
        {1.0 + pad * spi4, pad * spi4},
        {pad * spi4, 1.0 + pad * spi4},
        {-pad * spi8, 1.0 + pad * cpi8},
        {-pad, 1.0},
        {-pad, 0.0},
        {-pad * spi4, -pad * spi4},
        {0.0, -pad}}},
  };
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    const ConvexHull hull(checks[ii].in);
    const auto grown = hull.CreateBiggerHull(pad);
    ASSERT_EQ(checks[ii].out.size(), grown.GetPoints().size())
        << "check[" << ii << "] should have resulted in "
        << checks[ii].out.size() << " grown points, but there are "
        << grown.GetPoints().size();
    for (int jj = 0; jj < checks[ii].out.size(); ++jj) {
      EXPECT_THAT(grown.GetPoints()[jj],
                  eigenmath::testing::IsApprox(checks[ii].out[jj], kEpsilon))
          << "Failed on check[" << ii << "].out[" << jj << "]";
    }
  }
}

TEST(ConvexHull, GrowCornerCases) {
  constexpr double pad = 0.1;
  const double spi4 = sin(M_PI / 4.0);
  const double spi8 = sin(M_PI / 8.0);
  const double cpi8 = cos(M_PI / 8.0);
  constexpr double kPerturbation = std::numeric_limits<double>::epsilon();
  const struct {
    std::vector<eigenmath::Vector2d> in;
    std::vector<eigenmath::Vector2d> out;
  } checks[] = {
      {{{0.0, 0.0}, {0.5, -kPerturbation}, {1.0, 0.0}, {0.0, 1.0}},
       {{0.5, -pad - kPerturbation},
        {1.0, -pad},
        {1.0 + pad * cpi8, -pad * spi8},
        {1.0 + pad * spi4, pad * spi4},
        {pad * spi4, 1.0 + pad * spi4},
        {-pad * spi8, 1.0 + pad * cpi8},
        {-pad, 1.0},
        {-pad, 0.0},
        {-pad * spi4, -pad * spi4},
        {0.0, -pad}}},
      {{{1.0, 0.0},
        {0.0, 1.0},
        {0.0, 0.0},
        {0.5, -kPerturbation},
        {0.5, -kPerturbation}},
       {{0.5, -pad - kPerturbation},
        {1.0, -pad},
        {1.0 + pad * cpi8, -pad * spi8},
        {1.0 + pad * spi4, pad * spi4},
        {pad * spi4, 1.0 + pad * spi4},
        {-pad * spi8, 1.0 + pad * cpi8},
        {-pad, 1.0},
        {-pad, 0.0},
        {-pad * spi4, -pad * spi4},
        {0.0, -pad}}},
      {{{kPerturbation, 0.0},
        {0.0, 0.0},
        {0.5, -5.0 * kPerturbation},
        {1.0, 0.0}},
       {{0.5, -pad - 5.0 * kPerturbation},
        {1.0, -pad},
        {1.0 + pad, 0.0},
        {1.0, pad},
        {0.0, pad},
        {-pad, 0.0},
        {0.0, -pad}}},
  };
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    const ConvexHull hull(checks[ii].in);
    const auto grown = hull.CreateBiggerHull(pad);
    EXPECT_EQ(checks[ii].out.size(), grown.GetPoints().size())
        << "check[" << ii << "] should have resulted in "
        << checks[ii].out.size() << " grown points, but there are "
        << grown.GetPoints().size();
    for (int jj = 0;
         jj < checks[ii].out.size() && jj < grown.GetPoints().size(); ++jj) {
      EXPECT_THAT(grown.GetPoints()[jj],
                  eigenmath::testing::IsApprox(checks[ii].out[jj], kEpsilon))
          << "Failed on check[" << ii << "].out[" << jj << "]";
    }
  }
}

TEST(ConvexHull, Contains) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  const ConvexHull hull(points);

  static const struct {
    eigenmath::Vector2d point;
    bool inside;
  } checks[] = {
      {{0.1, 0.1}, true},    {{-0.1, 0.1}, false}, {{0.1, -0.1}, false},
      {{-0.1, -0.1}, false}, {{0.8, 0.1}, true},   {{1.1, 0.1}, false},
      {{0.8, -0.1}, false},  {{1.1, -0.1}, false}, {{-0.1, 0.0}, false}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    ASSERT_EQ(checks[ii].inside, hull.Contains(checks[ii].point))
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be "
        << (checks[ii].inside ? "inside" : "outside");
  }
}

TEST(ConvexHull, Distance) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
  const ConvexHull hull(points);

  static const struct {
    eigenmath::Vector2d point;
    double distance;
  } checks[] = {{{0.1, 0.1}, -0.1}, {{-0.1, 0.1}, 0.1},
                {{0.1, -0.1}, 0.1}, {{-0.1, -0.1}, 0.141421356},
                {{0.8, 0.1}, -0.1}, {{1.1, 0.1}, 0.1},
                {{0.8, -0.1}, 0.1}, {{1.1, -0.1}, 0.141421356},
                {{-0.1, 0.0}, 0.1}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    EXPECT_NEAR(checks[ii].distance, hull.Distance(checks[ii].point), kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be at distance "
        << checks[ii].distance;
  }
}

TEST(ConvexHull, DistancePotato) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0},   {1.0, 0.0},  {1.5, 0.25}, {1.75, 0.75},
      {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0},  {0.0, 2.0}};
  const ConvexHull hull(points);

  static const struct {
    eigenmath::Vector2d point;
    double distance;
  } checks[] = {{{1.25, 0.1}, 0.022360679},   {{1.25, 0.15}, -0.022360679},
                {{1.76, 1.0}, 0.01},          {{1.74, 1.0}, -0.01},
                {{1.51, 1.76}, 0.0141421356}, {{1.49, 1.74}, -0.0134164079}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    EXPECT_NEAR(checks[ii].distance, hull.Distance(checks[ii].point), kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be at distance "
        << checks[ii].distance;
  }
}

TEST(ConvexHull, DistancePenetrating) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
  const ConvexHull hull(points);

  static const struct {
    eigenmath::Vector2d point;
    double distance;
    eigenmath::Vector2d closest_point;
  } checks[] = {{{0.1, 0.1}, -0.1, {0.1, 0.0}}, {{-0.1, 0.1}, 1.0, {0.0, 0.0}},
                {{0.1, -0.1}, 1.0, {0.0, 0.0}}, {{-0.1, -0.1}, 1.0, {0.0, 0.0}},
                {{0.8, 0.1}, -0.1, {0.8, 0.0}}, {{1.1, 0.1}, 1.0, {0.0, 0.0}},
                {{0.8, -0.1}, 1.0, {0.0, 0.0}}, {{1.1, -0.1}, 1.0, {0.0, 0.0}},
                {{-0.1, 0.0}, 1.0, {0.0, 0.0}}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    EXPECT_NEAR(checks[ii].distance,
                hull.DistanceIfPenetrating(checks[ii].point), kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].point.transpose() << ")";
    if (checks[ii].distance < 0.0) {
      eigenmath::Vector2d closest_point = eigenmath::Vector2d::Zero();
      EXPECT_NEAR(
          hull.ClosestPointIfPenetrating(checks[ii].point, &closest_point),
          checks[ii].distance, kEpsilon);
      EXPECT_THAT(closest_point, eigenmath::testing::IsApprox(
                                     checks[ii].closest_point, kEpsilon))
          << "check[" << ii << "] point (" << checks[ii].point.transpose()
          << ")";
    }
  }
}

TEST(ConvexHull, DistanceLessThan) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
  const ConvexHull hull(points);

  static const struct {
    eigenmath::Vector2d point;
    double distance;
  } checks[] = {{{0.1, 0.1}, 1.0},   {{-0.1, 0.1}, 1.0}, {{0.1, -0.1}, 1.0},
                {{-0.1, -0.1}, 1.0}, {{0.8, 0.1}, 1.0},  {{1.1, 0.1}, 1.0},
                {{0.8, -0.1}, 1.0},  {{1.1, -0.1}, 1.0}, {{-0.1, 0.0}, 1.0}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    EXPECT_NEAR(checks[ii].distance,
                hull.DistanceIfLessThan(checks[ii].point, -0.15), kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be at distance "
        << checks[ii].distance;
  }
}

TEST(ConvexHull, DistancePenetratingPotato) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0},   {1.0, 0.0},  {1.5, 0.25}, {1.75, 0.75},
      {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0},  {0.0, 2.0}};
  const ConvexHull hull(points);

  static const struct {
    eigenmath::Vector2d point;
    double distance;
  } checks[] = {{{1.25, 0.1}, 1.0},  {{1.25, 0.15}, -0.022360679},
                {{1.76, 1.0}, 1.0},  {{1.74, 1.0}, -0.01},
                {{1.51, 1.76}, 1.0}, {{1.49, 1.74}, -0.0134164079}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    EXPECT_NEAR(checks[ii].distance,
                hull.DistanceIfPenetrating(checks[ii].point), kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be at distance "
        << checks[ii].distance;
  }
}

TEST(ConvexHull, DistanceLessThanPotato) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0},   {1.0, 0.0},  {1.5, 0.25}, {1.75, 0.75},
      {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0},  {0.0, 2.0}};
  const ConvexHull hull(points);

  static const struct {
    eigenmath::Vector2d point;
    double distance;
  } checks[] = {{{1.25, 0.1}, 1.0},  {{1.25, 0.15}, -0.022360679},
                {{1.76, 1.0}, 1.0},  {{1.74, 1.0}, 1.0},
                {{1.51, 1.76}, 1.0}, {{1.49, 1.74}, 1.0}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    EXPECT_NEAR(checks[ii].distance,
                hull.DistanceIfLessThan(checks[ii].point, -0.014), kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be at distance "
        << checks[ii].distance;
  }
}

TEST(ConvexHull, ContainsOtherHullsPotato) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0},   {1.0, 0.0},  {1.5, 0.25}, {1.75, 0.75},
      {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0},  {0.0, 2.0}};
  const ConvexHull hull(points);

  const std::vector<eigenmath::Vector2d> inner_points = {
      {0.5, 0.5}, {1.0, 0.5}, {1.0, 1.0}};
  const ConvexHull inner_hull(inner_points);
  EXPECT_TRUE(hull.Contains(inner_hull));

  const std::vector<eigenmath::Vector2d> straddle_points = {
      {0.5, 0.5}, {1.0, -0.5}, {1.0, 1.0}};
  const ConvexHull straddle_hull(straddle_points);
  EXPECT_FALSE(hull.Contains(straddle_hull));

  const std::vector<eigenmath::Vector2d> outer_points = {
      {-1.0, 0.5}, {-1.5, 0.5}, {-1.5, 1.0}};
  const ConvexHull outer_hull(outer_points);
  EXPECT_FALSE(hull.Contains(outer_hull));
}

TEST(ConvexHull, AreOverlappingPotato) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0},   {1.0, 0.0},  {1.5, 0.25}, {1.75, 0.75},
      {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0},  {0.0, 2.0}};
  const ConvexHull hull1(points);
  ConvexHull hull2(points);
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_FALSE(AreOverlapping(hull1, hull2));
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(-1.0, 0.5), 0.0));
  EXPECT_TRUE(AreOverlapping(hull1, hull2));
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(3.0, 0.5), 0.0));
  EXPECT_FALSE(AreOverlapping(hull1, hull2));
}

TEST(ConvexHull, DistanceBetweenPotato) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0},   {1.0, 0.0},  {1.5, 0.25}, {1.75, 0.75},
      {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0},  {0.0, 2.0}};
  const ConvexHull hull1(points);
  ConvexHull hull2(points);
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), 0.25, kEpsilon);
  EXPECT_GT(
      DistanceBetween(hull1, hull2, std::numeric_limits<double>::lowest(), 0.0),
      0.25 - kEpsilon);
  EXPECT_NEAR(DistanceBetween(hull1, hull2, 0.0), 0.25, kEpsilon);
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(-1.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -0.75, kEpsilon);
  EXPECT_NEAR(
      DistanceBetween(hull1, hull2, std::numeric_limits<double>::lowest(), 0.0),
      -0.75, kEpsilon);
  EXPECT_LT(DistanceBetween(hull1, hull2, 0.0), 0.0 + kEpsilon);
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(3.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), 2.26384628, kEpsilon);
  EXPECT_GT(
      DistanceBetween(hull1, hull2, std::numeric_limits<double>::lowest(), 0.0),
      0.0);
  EXPECT_NEAR(DistanceBetween(hull1, hull2, 0.0), 2.26384628, kEpsilon);
}

// Regression test for b/232981330.
TEST(ConvexHull, FindPenetrationDepth_b232981330) {
  const std::vector<eigenmath::Vector2d> lhs_points = {
      {-14.3522245292267296, -10.2268929052049806},
      {-14.4196172863719756, -10.1241277638332701},
      {-14.5183782726323951, -10.0509927818502618},
      {-14.5500438255053144, -10.0352672369875151},
      {-14.6555184927167019, -10.0023590235000128},
      {-14.7659959178191489, -10.0039697627790183},
      {-14.9838667649920811, -10.0420607056303091},
      {-14.9947150532050575, -10.0441342483378602},
      {-15.005492515043219, -10.0465490470538423},
      {-15.1017132567586749, -10.0697085273210245},
      {-15.1166461264865504, -10.0736532569714221},
      {-15.1313904222148565, -10.0782529562877663},
      {-15.1408795746819251, -10.0814446351803628},
      {-15.1476102851082963, -10.0837850477436604},
      {-15.1542919521061723, -10.086262005542979},
      {-15.3544623941902927, -10.1627937767897958},
      {-15.4290036728729447, -10.2021618384527208},
      {-15.4919699658807115, -10.2582108567456487},
      {-15.5268328039044832, -10.297741601135975},
      {-15.57780577904723, -10.3735698832228174},
      {-15.6074160886189635, -10.4600071560060943},
      {-15.6178428198004653, -10.511673203890501},
      {-15.6240725701372138, -10.6028289100746829},
      {-15.6064972140485541, -10.6924909389249816},
      {-15.5068059625347257, -10.988882122724597},
      {-15.4666232314934504, -11.0709401551483602},
      {-15.4065715122149278, -11.1398020718432118},
      {-15.3670407678246015, -11.1746649098669835},
      {-15.2912124857377592, -11.2256378850097303},
      {-15.2047752129544804, -11.2552481945814637},
      {-15.1531091650700738, -11.2656749257629638},
      {-15.0619534588903807, -11.2719046760999948},
      {-14.9722914300441179, -11.2543293200139196},
      {-14.962802277576154, -11.251137641121419},
      {-14.9628022775670466, -11.2511376411183566},
      {-14.9628022775673468, -11.2511376411184578},
      {-14.6759002462356971, -11.15463806849713},
      {-14.6759002462353969, -11.1546380684970288},
      {-14.6759002462262877, -11.1546380684939646},
      {-14.6664110937584997, -11.1514463896009435},
      {-14.6611005356825892, -11.1496126070984687},
      {-14.6558200970119401, -11.1476938176911791},
      {-14.5765572737999314, -11.1181692380668444},
      {-14.5456019329333603, -11.1049098914614444},
      {-14.5160641592196207, -11.0887369719784434},
      {-14.3987225473623077, -11.0169436258135853},
      {-14.3540479364706641, -10.9845055667018539},
      {-14.3150299756252117, -10.9454460317747895},
      {-14.2961164291604721, -10.923258698448878},
      {-14.2219753716072965, -10.7772049922691391},
      {-14.2224063376511509, -10.6134112323995708},
      {-14.2624002120762015, -10.4491374501847734},
      {-14.2703877577949907, -10.4213704107026697},
      {-14.2806383475795204, -10.3943568071429855},
  };
  const ConvexHull lhs_hull(lhs_points, ConvexHull::kTestOnlyClockwisePoints);
  EXPECT_THAT(lhs_points, testing::ElementsAreArray(lhs_hull.GetPoints()));
  const std::vector<eigenmath::Vector2d> rhs_points = {
      {-14.4758433388137409, -12.0204727480410458},
      {-14.2312987138090321, -12.1298197415552789},
      {-13.9810595505492046, -12.0342205795280019},
      {-13.543130945157829, -11.6199702212166418},
      {-13.433783951643596, -11.3754255962119331},
      {-13.5293831136708711, -11.1251864329521055},
      {-13.8769496541163111, -10.7577532436159782},
      {-14.1214942791210198, -10.6484062501017451},
      {-14.3717334423808474, -10.744005412129022},
      {-14.8096620477722229, -11.1582557704403822},
      {-14.919009041286456, -11.4028003954450909},
      {-14.8234098792591809, -11.6530395587049185},
  };
  const ConvexHull rhs_hull(rhs_points, ConvexHull::kTestOnlyClockwisePoints);
  EXPECT_NEAR(DistanceBetween(lhs_hull, rhs_hull), 0.0094920049626, 1e-3);
}

TEST(ConvexHull, FindPenetrationDepth_b232981330_2) {
  const std::vector<eigenmath::Vector2d> lhs_points = {
      {81.62386129013315, -31.18673858346783},
      {81.79763332490973, -31.27751170488882},
      {81.99295490803806, -31.26060138829975},
      {82.02845366227567, -31.2468232009167},
      {82.06868553107212, -31.22819588223248},
      {82.10623982395619, -31.20463202528593},
      {82.28576730105381, -31.0754480056004},
      {82.29462933104911, -31.06885639268138},
      {82.30327896812609, -31.06198844430154},
      {82.37980556743435, -30.9992311956299},
      {82.39152935386642, -30.98917613113635},
      {82.40279811425597, -30.97861362529933},
      {82.40672524172109, -30.97476636755919},
      {82.41176455856113, -30.96972798690434},
      {82.41670025469213, -30.96458805523748},
      {82.56355121401506, -30.80851078706477},
      {82.61350301715385, -30.74060597228757},
      {82.64576991264079, -30.66272725913451},
      {82.65993394941827, -30.61195839772035},
      {82.67278986900499, -30.52149902622836},
      {82.66179512560439, -30.43079462624302},
      {82.64867711056681, -30.37974548414839},
      {82.61458097777187, -30.29497741063232},
      {82.55970705068601, -30.22192246771008},
      {82.34087253384168, -29.99854490907888},
      {82.26896037991239, -29.94218167038162},
      {82.18491069358949, -29.9063511974819},
      {82.13414183217532, -29.89218716070441},
      {82.04368246068333, -29.87933124111771},
      {81.95297806069799, -29.8903259845183},
      {81.90192891860335, -29.90344399955589},
      {81.81716084508763, -29.93754013235064},
      {81.7441059021656, -29.99241405943613},
      {81.74017877470044, -29.99626131717629},
      {81.74017877469987, -29.99626131717685},
      {81.74017877469987, -29.99626131717684},
      {81.52072834353385, -30.21124857628103},
      {81.52072834353383, -30.21124857628104},
      {81.52072834353325, -30.2112485762816},
      {81.5168012160681, -30.21509583402178},
      {81.51281959378446, -30.21905960361445},
      {81.50890210947564, -30.22308677409553},
      {81.45041272526892, -30.2841874114134},
      {81.42832325297761, -30.30960588136064},
      {81.40877885623658, -30.33702959293652},
      {81.33442314466815, -30.45276449080217},
      {81.31858861943087, -30.48003984623978},
      {81.30527366758545, -30.50862986654569},
      {81.29076907621636, -30.54383805918888},
      {81.26971061879826, -30.73798928971193},
      {81.35595980563153, -30.91320126622336},
      {81.46899246403419, -31.03695017883622},
      {81.47684439207883, -31.04526932726432},
      {81.48496399411351, -31.05332743295893}};
  const ConvexHull lhs_hull(lhs_points, ConvexHull::kTestOnlyClockwisePoints);
  EXPECT_THAT(lhs_points, testing::ElementsAreArray(lhs_hull.GetPoints()));

  const std::vector<eigenmath::Vector2d> rhs_points = {
      {80.48958444913217, -30.05000599963096},
      {80.32861876924009, -30.48009106963754},
      {80.51891491493886, -30.89802706291628},
      {80.73947737482879, -31.10384241295679},
      {81.16956244483538, -31.26480809284887},
      {81.58749843811411, -31.07451194715011},
      {81.9089280456401, -30.73005122105057},
      {82.06989372553217, -30.29996615104399},
      {81.8795975798334, -29.88203015776525},
      {81.65903511994347, -29.67621480772474},
      {81.22895004993688, -29.51524912783266},
      {80.81101405665815, -29.70554527353142}};
  const ConvexHull rhs_hull(rhs_points, ConvexHull::kTestOnlyClockwisePoints);
  EXPECT_THAT(rhs_points, testing::ElementsAreArray(rhs_hull.GetPoints()));

  EXPECT_NEAR(DistanceBetween(lhs_hull, rhs_hull), -0.561724, 1e-3);
}

ConvexHull Diamond() {
  ConvexHull hull({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}});
  return hull;
}

TEST(ConvexHull, DistanceBetweenSquares) {
  const std::vector<eigenmath::Vector2d> points = {
      {-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};
  const ConvexHull hull1(points);

  ConvexHull hull2 = hull1;
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -2, kEpsilon);

  hull2 = hull1;
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(1.0, 0.0), 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -1, kEpsilon);

  hull2 = hull1;
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(1.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -1, kEpsilon);

  hull2 = hull1;
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(1.8, -1.8), 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -0.2, kEpsilon);

  hull2 = Diamond();
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -2, kEpsilon);

  hull2.ApplyTransform(eigenmath::Pose2d({3.0, 0.0}, 0.0));
  EXPECT_NEAR(DistanceBetween(Diamond(), hull2), 1, kEpsilon);
}

TEST(ConvexHull, DistanceBetweenTriangles) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, -1.0}, {0.0, 1.0}, {1.0, 0.0}};
  const ConvexHull hull1(points);

  ConvexHull hull2 = hull1;
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -1.0, kEpsilon);

  hull2.ApplyTransform(
      eigenmath::Pose2d({std::numeric_limits<double>::epsilon(), 0.0}, 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -1.0, kEpsilon);

  hull2.ApplyTransform(
      eigenmath::Pose2d({0.0, std::numeric_limits<double>::epsilon()}, 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -1.0, kEpsilon);

  const std::vector<eigenmath::Vector2d> mirrored_points = {
      {0.0, 0.0}, {1.0, 1.0}, {1.0, -1.0}};
  hull2 = ConvexHull{mirrored_points};
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -1.0 / M_SQRT2, kEpsilon);

  hull2 = ConvexHull{mirrored_points};
  hull2.ApplyTransform(eigenmath::Pose2d({0.5, 0.0}, 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -0.5 / M_SQRT2, kEpsilon);

  hull2 = ConvexHull{mirrored_points};
  hull2.ApplyTransform(eigenmath::Pose2d({1.5, 0.0}, 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), 0.5, kEpsilon);

  hull2 = ConvexHull{mirrored_points};
  hull2.ApplyTransform(eigenmath::Pose2d({-1.5, 0.0}, 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), 0.5, kEpsilon);

  hull2 = ConvexHull{points};
  hull2.ApplyTransform(eigenmath::Pose2d({1.5, 0.0}, 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), 0.5, kEpsilon);

  hull2 = ConvexHull{points};
  hull2.ApplyTransform(eigenmath::Pose2d({-1.5, 0.0}, 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), 0.5, kEpsilon);

  hull2 = ConvexHull{points};
  hull2.ApplyTransform(eigenmath::Pose2d({0.0, 2.5}, 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), 0.5, kEpsilon);
}

TEST(ConvexHull, DistanceBetweenReferenceCases) {
  // There is nothing particularly special about these cases, they are just
  // random cases that were verified by hand (visually and hand-calculated).
  {
    const std::vector<eigenmath::Vector2d> points1 = {
        {1.00019, 1.19202},  {1.06178, 0.222309}, {2.12976, 0.0346281},
        {2.86265, 0.711132}, {2.7673, 1.28905},   {2.62775, 1.65665},
        {1.94741, 1.77519},  {1.38316, 1.82184}};
    const std::vector<eigenmath::Vector2d> points2 = {
        {0.0643063, 0.835551}, {0.18319, 0.411576}, {0.997263, 0.428123},
        {1.88845, 1.03793},    {1.88726, 1.58889},  {1.5843, 1.84942},
        {0.914173, 1.94936},   {0.114555, 1.77511}};
    const ConvexHull hull1(points1);
    const ConvexHull hull2(points2);
    EXPECT_LE(DistanceBetween(hull1, hull2),
              BruteForceDistanceBetween(hull1, hull2) + kEpsilon);
    EXPECT_NEAR(DistanceBetween(hull1, hull2), -0.83902152, kEpsilon);
    EXPECT_NEAR(DistanceBetween(hull1, hull2,
                                std::numeric_limits<double>::lowest(), 0.0),
                -0.83902152, kEpsilon);
    EXPECT_LT(DistanceBetween(hull1, hull2, 0.0), 0.0 + kEpsilon);
  }

  {
    const std::vector<eigenmath::Vector2d> points1 = {
        {0.764754, 1.78064}, {0.466188, 0.855983}, {1.39079, 0.289475},
        {2.3208, 0.649832},  {2.44418, 1.22241},   {2.44927, 1.61558},
        {1.8599, 1.97552},   {1.35214, 2.22598}};
    const std::vector<eigenmath::Vector2d> points2 = {
        {0.0643063, 0.835551}, {0.18319, 0.411576}, {0.997263, 0.428123},
        {1.88845, 1.03793},    {1.88726, 1.58889},  {1.5843, 1.84942},
        {0.914173, 1.94936},   {0.114555, 1.77511}};
    const ConvexHull hull1(points1);
    const ConvexHull hull2(points2);
    EXPECT_LE(DistanceBetween(hull1, hull2),
              BruteForceDistanceBetween(hull1, hull2) + kEpsilon);
    EXPECT_NEAR(DistanceBetween(hull1, hull2), -1.27073023, kEpsilon);
    EXPECT_NEAR(DistanceBetween(hull1, hull2,
                                std::numeric_limits<double>::lowest(), 0.0),
                -1.27073023, kEpsilon);
    EXPECT_LT(DistanceBetween(hull1, hull2, 0.0), 0.0 + kEpsilon);
  }

  {
    const std::vector<eigenmath::Vector2d> points1 = {
        {1.39796, -0.897625},   {1.05631, 0.0119963},  {-0.0199923, -0.119872},
        {-0.523783, -0.980669}, {-0.264078, -1.50567}, {-0.0234065, -1.81661},
        {0.661933, -1.73163},   {1.21527, -1.61174}};
    const std::vector<eigenmath::Vector2d> points2 = {
        {0.0643063, 0.835551}, {0.18319, 0.411576}, {0.997263, 0.428123},
        {1.88845, 1.03793},    {1.88726, 1.58889},  {1.5843, 1.84942},
        {0.914173, 1.94936},   {0.114555, 1.77511}};
    const ConvexHull hull1(points1);
    const ConvexHull hull2(points2);
    EXPECT_NEAR(DistanceBetween(hull1, hull2),
                BruteForceDistanceBetween(hull1, hull2), kEpsilon);
    EXPECT_NEAR(DistanceBetween(hull1, hull2), 0.42021849, kEpsilon);
    EXPECT_GT(DistanceBetween(hull1, hull2,
                              std::numeric_limits<double>::lowest(), 0.0),
              0.42021849 - kEpsilon);
    EXPECT_NEAR(DistanceBetween(hull1, hull2, 0.0), 0.42021849, kEpsilon);
  }
}

TEST(ConvexHull, DistanceBetweenAgainstBruteForce) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  std::vector<eigenmath::Vector2d> points_test1, points_test2;
  for (int i = 0; i < 8; ++i) {
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    points_test2.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
  }
  ConvexHull hull_test1(points_test1);
  while (hull_test1.GetPoints().size() != 8) {
    points_test1 = hull_test1.GetPoints();
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test1 = ConvexHull(points_test1);
  }
  ConvexHull hull_test2(points_test2);
  while (hull_test2.GetPoints().size() != 8) {
    points_test2 = hull_test2.GetPoints();
    points_test2.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test2 = ConvexHull(points_test2);
  }

  for (int i = 0; i < 5e4; ++i) {
    const eigenmath::Pose2d pose_test(
        eigenmath::Vector2d(uni_dist(rand_eng) * 2.0, uni_dist(rand_eng) * 2.0),
        uni_dist(rand_eng) * 2.0 * M_PI);
    hull_test1.ApplyTransform(pose_test);
    const double actual_result = DistanceBetween(hull_test2, hull_test1);
    const double brute_force_result =
        BruteForceDistanceBetween(hull_test2, hull_test1);
    if (actual_result > 0.0) {
      // Brute-force can accurately compute disjoint case only.
      EXPECT_NEAR(actual_result, brute_force_result, 3e-3);
    } else {
      EXPECT_LE(actual_result, brute_force_result + 3e-3);
    }
    hull_test1.ApplyTransform(pose_test.inverse());
  }
}

TEST(ConvexHull, MaxRadiusAround) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
  const ConvexHull hull(points);

  static const struct {
    eigenmath::Vector2d point;
    double radius;
  } checks[] = {{{0.0, 0.0}, M_SQRT2},         {{1.0, 0.0}, M_SQRT2},
                {{1.0, 1.0}, M_SQRT2},         {{0.0, 1.0}, M_SQRT2},
                {{-1.0, -1.0}, 2.0 * M_SQRT2}, {{2.0, -1.0}, 2.0 * M_SQRT2}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    EXPECT_NEAR(checks[ii].radius, hull.GetMaxRadiusAround(checks[ii].point),
                kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be at distance "
        << checks[ii].radius;
  }
}

TEST(ConvexHull, GetMinAreaBoundingBox) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {0.5, 0.5}, {0.0, 1.0}, {-0.5, 0.5}};
  const ConvexHull hull(points);

  OrientedBox2d box = hull.GetMinAreaBoundingBox();
  EXPECT_NEAR(box.Width(), 0.707107, kEpsilon);
  EXPECT_NEAR(box.Height(), 0.707107, kEpsilon);
  std::vector<eigenmath::Vector2d> four_corners = box.GetPoints();
  EXPECT_THAT(four_corners,
              UnorderedElementsAreArray(
                  {eigenmath::testing::IsApprox(eigenmath::Vector2d{0.0, 1.0},
                                                kEpsilon),
                   eigenmath::testing::IsApprox(eigenmath::Vector2d{0.0, 0.0},
                                                kEpsilon),
                   eigenmath::testing::IsApprox(eigenmath::Vector2d{0.5, 0.5},
                                                kEpsilon),
                   eigenmath::testing::IsApprox(eigenmath::Vector2d{-0.5, 0.5},
                                                kEpsilon)}));
}

TEST(ConvexHull, TransformAndCopy) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  const ConvexHull orig_hull(points);
  ConvexHull trans_hull(points);

  eigenmath::Pose2d rel_pose(eigenmath::Vector2d(0.5, 1.0), 0.5 * M_PI);

  // Check that assigning a bigger hull to a smaller one is not going to work:
  const std::vector<eigenmath::Vector2d> more_points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
  const ConvexHull bigger_hull(more_points);
  ASSERT_FALSE(trans_hull.TransformAndCopy(rel_pose, bigger_hull));

  ASSERT_TRUE(trans_hull.TransformAndCopy(rel_pose, orig_hull));

  static const struct {
    eigenmath::Vector2d point;
    bool inside;
  } checks[] = {{{0.4, 1.1}, true},   {{0.4, 0.9}, false}, {{0.6, 1.1}, false},
                {{0.4, 2.0}, false},  {{0.45, 1.9}, true}, {{0.55, 1.9}, false},
                {{0.0, 1.55}, false}, {{0.0, 1.45}, true}, {{-0.4, 1.05}, true},
                {{-0.4, 0.95}, false}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    ASSERT_EQ(checks[ii].inside, trans_hull.Contains(checks[ii].point))
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be "
        << (checks[ii].inside ? "inside" : "outside");
  }
}

TEST(AlignedPaddingBox, CallOperator) {
  AlignedPaddingBox box1(1.0, 2.0, 3.0, 4.0);

  EXPECT_THAT(
      box1(eigenmath::Vector2d(0.5, 0.0), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(1.0, 0.0), kEpsilon));
  EXPECT_THAT(
      box1(eigenmath::Vector2d(-0.5, 0.0), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(-2.0, 0.0), kEpsilon));
  EXPECT_THAT(
      box1(eigenmath::Vector2d(0.0, 0.5), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(0.0, 3.0), kEpsilon));
  EXPECT_THAT(
      box1(eigenmath::Vector2d(0.0, -0.5), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(0.0, -4.0), kEpsilon));

  EXPECT_THAT(
      box1(eigenmath::Vector2d(0.5, 0.5), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(1.0, 1.0), kEpsilon));
  EXPECT_THAT(
      box1(eigenmath::Vector2d(0.5, -0.5), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(1.0, -1.0), kEpsilon));
  EXPECT_THAT(
      box1(eigenmath::Vector2d(-0.5, 0.5), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(-2.0, 2.0), kEpsilon));
  EXPECT_THAT(
      box1(eigenmath::Vector2d(-0.5, -0.5), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(-2.0, -2.0), kEpsilon));

  EXPECT_THAT(
      box1(eigenmath::Vector2d(0.1, 0.5), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(0.6, 3.0), kEpsilon));
  EXPECT_THAT(
      box1(eigenmath::Vector2d(0.1, -0.5), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(0.8, -4.0), kEpsilon));
  EXPECT_THAT(
      box1(eigenmath::Vector2d(-0.1, 0.5), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(-0.6, 3.0), kEpsilon));
  EXPECT_THAT(
      box1(eigenmath::Vector2d(-0.1, -0.5), 0.5),
      eigenmath::testing::IsApprox(eigenmath::Vector2d(-0.8, -4.0), kEpsilon));
}

TEST(AlignedPaddingBox, ConvexHullGrowth) {
  constexpr double pad_fwd = 0.2;
  constexpr double pad_side = 0.1;
  constexpr double pad_back = 0.15;
  const AlignedPaddingBox padding(pad_fwd, pad_back, pad_side, pad_side);
  const double tpi8 = tan(M_PI / 8.0);
  const struct {
    std::vector<eigenmath::Vector2d> in;
    std::vector<eigenmath::Vector2d> out;
  } checks[] = {
      {{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}},
       {{1.0, -pad_side},
        {1.0 + pad_fwd, -pad_fwd * tpi8},
        {1.0 + pad_side, pad_side},
        {pad_side, 1.0 + pad_side},
        {-pad_side * tpi8, 1.0 + pad_side},
        {-pad_back, 1.0},
        {-pad_back, 0.0},
        {-pad_side, -pad_side},
        {0.0, -pad_side}}},
  };
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    const ConvexHull hull(checks[ii].in);
    const auto grown = hull.CreateBiggerHull(padding);
    ASSERT_EQ(checks[ii].out.size(), grown.GetPoints().size())
        << "check[" << ii << "] should have resulted in "
        << checks[ii].out.size() << " grown points, but there are "
        << grown.GetPoints().size();
    for (int jj = 0; jj < checks[ii].out.size(); ++jj) {
      EXPECT_THAT(grown.GetPoints()[jj],
                  eigenmath::testing::IsApprox(checks[ii].out[jj], kEpsilon))
          << "Failed on check[" << ii << "].out[" << jj << "]";
    }
  }
}

TEST(LazyTransformedConvexHull, BasicOperations) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  const ConvexHull orig_hull(points);
  LazyTransformedConvexHull lazy_hull(&orig_hull);

  eigenmath::Pose2d rel_pose(eigenmath::Vector2d(0.5, 1.0), 0.5 * M_PI);

  lazy_hull.ApplyTransform(rel_pose);
  EXPECT_FALSE(lazy_hull.HasTransformedCentroid());
  EXPECT_FALSE(lazy_hull.HasTransformedPoints());

  const eigenmath::Vector2d centroid = lazy_hull.GetCentroid();
  EXPECT_TRUE(lazy_hull.HasTransformedCentroid());
  EXPECT_FALSE(lazy_hull.HasTransformedPoints());
  EXPECT_THAT(centroid,
              eigenmath::testing::IsApprox(
                  eigenmath::Vector2d(0.1666666666, 1.3333333333), kEpsilon));
}

TEST(LazyTransformedConvexHull, Contains) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  const ConvexHull orig_hull(points);
  LazyTransformedConvexHull lazy_hull(&orig_hull);

  eigenmath::Pose2d rel_pose(eigenmath::Vector2d(0.5, 1.0), 0.5 * M_PI);

  lazy_hull.ApplyTransform(rel_pose);
  EXPECT_FALSE(lazy_hull.HasTransformedCentroid());
  EXPECT_FALSE(lazy_hull.HasTransformedPoints());

  EXPECT_FALSE(lazy_hull.Contains(0.0, 0.0));
  EXPECT_TRUE(lazy_hull.HasTransformedCentroid());
  EXPECT_FALSE(lazy_hull.HasTransformedPoints());

  EXPECT_FALSE(lazy_hull.Contains(0.45, 0.95));
  EXPECT_TRUE(lazy_hull.HasTransformedCentroid());
  EXPECT_TRUE(lazy_hull.HasTransformedPoints());

  static const struct {
    eigenmath::Vector2d point;
    bool inside;
  } checks[] = {{{0.4, 1.1}, true},   {{0.4, 0.9}, false}, {{0.6, 1.1}, false},
                {{0.4, 2.0}, false},  {{0.45, 1.9}, true}, {{0.55, 1.9}, false},
                {{0.0, 1.55}, false}, {{0.0, 1.45}, true}, {{-0.4, 1.05}, true},
                {{-0.4, 0.95}, false}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    lazy_hull.ApplyTransform(rel_pose);
    ASSERT_EQ(checks[ii].inside, lazy_hull.Contains(checks[ii].point))
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be "
        << (checks[ii].inside ? "inside" : "outside");
  }
}

TEST(LazyTransformedConvexHull, Distance) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  const ConvexHull orig_hull(points);
  LazyTransformedConvexHull lazy_hull(&orig_hull);

  eigenmath::Pose2d rel_pose(eigenmath::Vector2d(0.5, 1.0), 0.5 * M_PI);

  ConvexHull trans_hull(points);
  ASSERT_TRUE(trans_hull.TransformAndCopy(rel_pose, orig_hull));

  lazy_hull.ApplyTransform(rel_pose);
  EXPECT_FALSE(lazy_hull.HasTransformedCentroid());
  EXPECT_FALSE(lazy_hull.HasTransformedPoints());

  EXPECT_NEAR(lazy_hull.Distance(0.0, 0.0), trans_hull.Distance(0.0, 0.0),
              kEpsilon);
  EXPECT_FALSE(lazy_hull.HasTransformedCentroid());
  EXPECT_TRUE(lazy_hull.HasTransformedPoints());

  static const eigenmath::Vector2d checks[] = {
      {0.4, 1.1},  {0.4, 0.9},  {0.6, 1.1},  {0.4, 2.0},   {0.45, 1.9},
      {0.55, 1.9}, {0.0, 1.55}, {0.0, 1.45}, {-0.4, 1.05}, {-0.4, 0.95}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    lazy_hull.ApplyTransform(rel_pose);
    EXPECT_NEAR(lazy_hull.Distance(checks[ii]), trans_hull.Distance(checks[ii]),
                kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].x() << ", "
        << checks[ii].y() << ")";
  }
}

TEST(LazyTransformedConvexHull, DistancePenetrating) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  const ConvexHull orig_hull(points);
  LazyTransformedConvexHull lazy_hull(&orig_hull);

  eigenmath::Pose2d rel_pose(eigenmath::Vector2d(0.5, 1.0), 0.5 * M_PI);

  ConvexHull trans_hull(points);
  ASSERT_TRUE(trans_hull.TransformAndCopy(rel_pose, orig_hull));

  lazy_hull.ApplyTransform(rel_pose);
  EXPECT_FALSE(lazy_hull.HasTransformedCentroid());
  EXPECT_FALSE(lazy_hull.HasTransformedPoints());

  EXPECT_NEAR(lazy_hull.DistanceIfPenetrating(0.0, 0.0),
              trans_hull.DistanceIfPenetrating(0.0, 0.0), kEpsilon);
  EXPECT_TRUE(lazy_hull.HasTransformedCentroid());
  EXPECT_FALSE(lazy_hull.HasTransformedPoints());

  EXPECT_NEAR(lazy_hull.DistanceIfPenetrating(0.45, 0.95),
              trans_hull.DistanceIfPenetrating(0.45, 0.95), kEpsilon);
  EXPECT_TRUE(lazy_hull.HasTransformedCentroid());
  EXPECT_TRUE(lazy_hull.HasTransformedPoints());

  static const eigenmath::Vector2d checks[] = {
      {0.4, 1.1},  {0.4, 0.9},  {0.6, 1.1},  {0.4, 2.0},   {0.45, 1.9},
      {0.55, 1.9}, {0.0, 1.55}, {0.0, 1.45}, {-0.4, 1.05}, {-0.4, 0.95}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    lazy_hull.ApplyTransform(rel_pose);
    EXPECT_NEAR(lazy_hull.DistanceIfPenetrating(checks[ii]),
                trans_hull.DistanceIfPenetrating(checks[ii]), kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].x() << ", "
        << checks[ii].y() << ")";
    if (trans_hull.DistanceIfPenetrating(checks[ii]) < 0.0) {
      eigenmath::Vector2d lazy_closest_point = eigenmath::Vector2d::Zero();
      EXPECT_NEAR(
          lazy_hull.ClosestPointIfPenetrating(checks[ii], &lazy_closest_point),
          trans_hull.DistanceIfPenetrating(checks[ii]), kEpsilon);
      eigenmath::Vector2d trans_closest_point = eigenmath::Vector2d::Zero();
      EXPECT_NEAR(trans_hull.ClosestPointIfPenetrating(checks[ii],
                                                       &trans_closest_point),
                  trans_hull.DistanceIfPenetrating(checks[ii]), kEpsilon);
      EXPECT_THAT(lazy_closest_point,
                  eigenmath::testing::IsApprox(trans_closest_point, kEpsilon))
          << "check[" << ii << "] point (" << checks[ii].x() << ", "
          << checks[ii].y() << ")";
    }
  }
}

TEST(LazyTransformedConvexHull, AreOverlappingPotato) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0},   {1.0, 0.0},  {1.5, 0.25}, {1.75, 0.75},
      {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0},  {0.0, 2.0}};
  const ConvexHull orig_hull1(points);
  LazyTransformedConvexHull lazy_hull1(&orig_hull1);
  const ConvexHull orig_hull2(points);
  ConvexHull trans_hull2(orig_hull2);
  LazyTransformedConvexHull lazy_hull2(&orig_hull2);
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_FALSE(AreOverlapping(lazy_hull1, lazy_hull2));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_FALSE(AreOverlapping(orig_hull1, lazy_hull2));
  EXPECT_FALSE(AreOverlapping(lazy_hull1, trans_hull2));
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(-1.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_TRUE(AreOverlapping(lazy_hull1, lazy_hull2));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_TRUE(AreOverlapping(orig_hull1, lazy_hull2));
  EXPECT_TRUE(AreOverlapping(lazy_hull1, trans_hull2));
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(3.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_FALSE(AreOverlapping(lazy_hull1, lazy_hull2));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_FALSE(AreOverlapping(orig_hull1, lazy_hull2));
  EXPECT_FALSE(AreOverlapping(lazy_hull1, trans_hull2));
}

TEST(LazyTransformedConvexHull, DistanceBetweenPotato) {
  const std::vector<eigenmath::Vector2d> points = {
      {0.0, 0.0},   {1.0, 0.0},  {1.5, 0.25}, {1.75, 0.75},
      {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0},  {0.0, 2.0}};
  const ConvexHull orig_hull1(points);
  LazyTransformedConvexHull lazy_hull1(&orig_hull1);
  const ConvexHull orig_hull2(points);
  ConvexHull trans_hull2(orig_hull2);
  LazyTransformedConvexHull lazy_hull2(&orig_hull2);
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2), 0.25, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_GT(DistanceBetween(orig_hull1, lazy_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            0.25 - kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2, 0.0), 0.25, kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2), 0.25, kEpsilon);
  EXPECT_GT(DistanceBetween(lazy_hull1, trans_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            0.25 - kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2, 0.0), 0.25, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2), 0.25, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_GT(DistanceBetween(lazy_hull1, lazy_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            0.25 - kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2, 0.0), 0.25, kEpsilon);
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(-1.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2), -0.75, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2,
                              std::numeric_limits<double>::lowest(), 0.0),
              -0.75, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_LT(DistanceBetween(orig_hull1, lazy_hull2, 0.0), 0.0 + kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2), -0.75, kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2,
                              std::numeric_limits<double>::lowest(), 0.0),
              -0.75, kEpsilon);
  EXPECT_LT(DistanceBetween(lazy_hull1, trans_hull2, 0.0), 0.0 + kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2), -0.75, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2,
                              std::numeric_limits<double>::lowest(), 0.0),
              -0.75, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_LT(DistanceBetween(lazy_hull1, lazy_hull2, 0.0), 0.0 + kEpsilon);
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(3.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2), 2.26384628, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_GT(DistanceBetween(orig_hull1, lazy_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            0.0);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2, 0.0), 2.26384628,
              kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2), 2.26384628, kEpsilon);
  EXPECT_GT(DistanceBetween(lazy_hull1, trans_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            0.0);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2, 0.0), 2.26384628,
              kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2), 2.26384628, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_GT(DistanceBetween(lazy_hull1, lazy_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            0.0);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2, 0.0), 2.26384628,
              kEpsilon);
}

TEST(Hull, RemoveInternalHulls) {
  const std::vector<eigenmath::Vector2d> potato_points = {
      {0.0, 0.0},   {1.0, 0.0},  {1.5, 0.25}, {1.75, 0.75},
      {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0},  {0.0, 2.0}};
  const std::vector<eigenmath::Vector2d> inner_points = {
      {0.5, 0.5}, {1.0, 0.5}, {1.0, 1.0}};
  const std::vector<eigenmath::Vector2d> straddle_points = {
      {0.5, 0.5}, {1.0, -0.5}, {1.0, 1.0}};
  const std::vector<eigenmath::Vector2d> outer_points = {
      {-1.0, 0.5}, {-1.5, 0.5}, {-1.5, 1.0}};

  Hull hull;
  hull.Add(potato_points);
  hull.Add(inner_points);
  hull.Add(straddle_points);
  hull.Add(outer_points);

  hull.RemoveInternalHulls();
  EXPECT_EQ(hull.GetConvexHulls().size(), 3);

  EXPECT_TRUE(hull.IsApprox(hull, kEpsilon));

  for (auto& pt : potato_points) {
    EXPECT_TRUE(hull.Contains(pt)) << " Point: " << pt.transpose();
  }
  for (auto& pt : inner_points) {
    EXPECT_TRUE(hull.Contains(pt)) << " Point: " << pt.transpose();
  }
  for (auto& pt : straddle_points) {
    EXPECT_TRUE(hull.Contains(pt)) << " Point: " << pt.transpose();
  }
  for (auto& pt : outer_points) {
    EXPECT_TRUE(hull.Contains(pt)) << " Point: " << pt.transpose();
  }
}

TEST(Hull, SimplifyHulls) {
  const std::vector<eigenmath::Vector2d> potato_points = {
      {0.0, 0.0},   {1.0, 0.0},  {1.5, 0.25}, {1.75, 0.75},
      {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0},  {0.0, 2.0}};
  const std::vector<eigenmath::Vector2d> inner_points = {
      {0.5, 0.5}, {1.0, 0.5}, {1.0, 1.0}};
  const std::vector<eigenmath::Vector2d> straddle_points = {
      {0.5, 0.5}, {1.0, -0.5}, {1.0, 1.0}};
  const std::vector<eigenmath::Vector2d> small_straddle_points = {
      {0.5, 1.5}, {1.0, 2.2}, {1.0, 1.5}};
  const std::vector<eigenmath::Vector2d> outer_points = {
      {-1.0, 0.5}, {-1.5, 0.5}, {-1.5, 1.0}};

  Hull hull;
  hull.Add(potato_points);
  hull.Add(inner_points);
  hull.Add(straddle_points);
  hull.Add(small_straddle_points);
  hull.Add(outer_points);

  hull.SimplifyHulls();
  EXPECT_EQ(hull.GetConvexHulls().size(), 3);

  Hull hull_shuffled;
  hull_shuffled.Add(straddle_points);
  hull_shuffled.Add(inner_points);
  hull_shuffled.Add(outer_points);
  hull_shuffled.Add(small_straddle_points);
  hull_shuffled.Add(potato_points);
  hull_shuffled.SimplifyHulls();
  EXPECT_TRUE(hull.IsApprox(hull_shuffled, kEpsilon));

  for (auto& pt : potato_points) {
    EXPECT_TRUE(hull.Contains(pt)) << " Point: " << pt.transpose();
  }
  for (auto& pt : inner_points) {
    EXPECT_TRUE(hull.Contains(pt)) << " Point: " << pt.transpose();
  }
  for (auto& pt : straddle_points) {
    EXPECT_TRUE(hull.Contains(pt)) << " Point: " << pt.transpose();
  }
  for (auto& pt : small_straddle_points) {
    EXPECT_TRUE(hull.Contains(pt)) << " Point: " << pt.transpose();
  }
  for (auto& pt : outer_points) {
    EXPECT_TRUE(hull.Contains(pt)) << " Point: " << pt.transpose();
  }
}

TEST(Hull, Contains) {
  Hull hull;

  const std::vector<eigenmath::Vector2d> points1 = {
      {0.5, 0.0}, {1.0, 0.0}, {0.5, 0.5}};
  const std::vector<eigenmath::Vector2d> points2 = {
      {0.0, 0.5}, {0.5, 0.5}, {0.0, 1.0}};
  const std::vector<eigenmath::Vector2d> points3 = {
      {0.0, 0.0}, {0.5, 0.0}, {0.0, 0.5}};
  const std::vector<eigenmath::Vector2d> points4 = {
      {0.5, 0.5}, {0.5, 0.0}, {0.0, 0.5}};

  hull.Add(points1);
  hull.Add(points2);
  hull.Add(points3);
  hull.Add(points4);

  static const struct {
    eigenmath::Vector2d point;
    bool inside;
  } checks[] = {
      {{0.1, 0.1}, true},    {{-0.1, 0.1}, false}, {{0.1, -0.1}, false},
      {{-0.1, -0.1}, false}, {{0.8, 0.1}, true},   {{1.1, 0.1}, false},
      {{0.8, -0.1}, false},  {{1.1, -0.1}, false}, {{-0.1, 0.0}, false}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    EXPECT_EQ(checks[ii].inside, hull.Contains(checks[ii].point))
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be "
        << (checks[ii].inside ? "inside" : "outside");
  }
}

TEST(Hull, DistancePenetrating) {
  Hull hull;

  const std::vector<eigenmath::Vector2d> points1 = {
      {0.5, 0.0}, {1.0, 0.0}, {0.5, 0.5}};
  const std::vector<eigenmath::Vector2d> points2 = {
      {0.0, 0.5}, {0.5, 0.5}, {0.0, 1.0}};
  const std::vector<eigenmath::Vector2d> points3 = {
      {0.0, 0.0}, {0.5, 0.0}, {0.0, 0.5}};
  const std::vector<eigenmath::Vector2d> points4 = {
      {0.5, 0.5}, {0.5, 0.0}, {0.0, 0.5}};

  hull.Add(points1);
  hull.Add(points2);
  hull.Add(points3);
  hull.Add(points4);

  static const struct {
    eigenmath::Vector2d point;
    double distance;
    eigenmath::Vector2d closest_point;
  } checks[] = {{{0.1, 0.1}, -0.1, {0.1, 0.0}},
                {{-0.1, 0.1}, 1.0, {0.0, 0.0}},
                {{0.1, -0.1}, 1.0, {0.0, 0.0}},
                {{-0.1, -0.1}, 1.0, {0.0, 0.0}},
                {{0.8, 0.1}, -0.05 * M_SQRT2, {0.85, 0.15}},
                {{1.1, 0.1}, 1.0, {0.0, 0.0}},
                {{0.8, -0.1}, 1.0, {0.0, 0.0}},
                {{1.1, -0.1}, 1.0, {0.0, 0.0}},
                {{-0.1, 0.0}, 1.0, {0.0, 0.0}}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    EXPECT_NEAR(checks[ii].distance,
                hull.DistanceIfPenetrating(checks[ii].point), kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be at " << checks[ii].distance
        << " from hull.";
    EXPECT_NEAR((checks[ii].distance < -0.08 ? checks[ii].distance : 1.0),
                hull.DistanceIfLessThan(checks[ii].point, -0.08), kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].point.transpose()
        << ") should be at " << checks[ii].distance << " from hull.";
    if (checks[ii].distance < 0.0) {
      eigenmath::Vector2d closest_point = eigenmath::Vector2d::Zero();
      EXPECT_NEAR(
          hull.ClosestPointIfPenetrating(checks[ii].point, &closest_point),
          checks[ii].distance, kEpsilon);
      EXPECT_THAT(closest_point, eigenmath::testing::IsApprox(
                                     checks[ii].closest_point, kEpsilon))
          << "check[" << ii << "] point (" << checks[ii].point.transpose()
          << ")";
    }
  }
}

TEST(Hull, AreOverlappingPotato) {
  const std::vector<eigenmath::Vector2d> points1 = {
      {0.0, 0.0}, {1.0, 0.0}, {1.5, 0.25}, {1.75, 0.75}, {1.75, 1.25}};
  const std::vector<eigenmath::Vector2d> points2 = {
      {0.0, 0.0}, {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0}, {0.0, 2.0}};
  Hull hull1;
  hull1.Add(points1);
  hull1.Add(points2);
  Hull hull2;
  hull2.Add(points1);
  hull2.Add(points2);
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_FALSE(AreOverlapping(hull1, hull2));
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(-1.0, 0.5), 0.0));
  EXPECT_TRUE(AreOverlapping(hull1, hull2));
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(3.0, 0.5), 0.0));
  EXPECT_FALSE(AreOverlapping(hull1, hull2));
}

TEST(Hull, DistanceBetweenPotato) {
  const std::vector<eigenmath::Vector2d> points1 = {
      {0.0, 0.0}, {1.0, 0.0}, {1.5, 0.25}, {1.75, 0.75}, {1.75, 1.25}};
  const std::vector<eigenmath::Vector2d> points2 = {
      {0.0, 0.0}, {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0}, {0.0, 2.0}};
  Hull hull1;
  hull1.Add(points1);
  hull1.Add(points2);
  Hull hull2;
  hull2.Add(points1);
  hull2.Add(points2);
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), 0.25, kEpsilon);
  EXPECT_GT(
      DistanceBetween(hull1, hull2, std::numeric_limits<double>::lowest(), 0.0),
      -kEpsilon);
  EXPECT_NEAR(DistanceBetween(hull1, hull2, 0.0), 0.25, kEpsilon);
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(-1.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), -0.75, kEpsilon);
  EXPECT_NEAR(
      DistanceBetween(hull1, hull2, std::numeric_limits<double>::lowest(), 0.0),
      -0.75, kEpsilon);
  EXPECT_LT(DistanceBetween(hull1, hull2, 0.0), 0.0 + kEpsilon);
  hull2.ApplyTransform(eigenmath::Pose2d(eigenmath::Vector2d(3.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(hull1, hull2), 2.26384628, kEpsilon);
  EXPECT_GT(
      DistanceBetween(hull1, hull2, std::numeric_limits<double>::lowest(), 0.0),
      -kEpsilon);
  EXPECT_NEAR(DistanceBetween(hull1, hull2, 0.0), 2.26384628, kEpsilon);
}

TEST(Hull, MaxRadiusAround) {
  Hull hull;

  const std::vector<eigenmath::Vector2d> points1 = {
      {0.5, 0.0}, {1.0, 0.0}, {0.5, 0.5}};
  const std::vector<eigenmath::Vector2d> points2 = {
      {0.0, 0.5}, {0.5, 0.5}, {0.0, 1.0}};
  const std::vector<eigenmath::Vector2d> points3 = {
      {0.0, 0.0}, {0.5, 0.0}, {0.0, 0.5}};
  const std::vector<eigenmath::Vector2d> points4 = {
      {0.5, 0.5}, {1.0, 1.0}, {1.0, 0.5}};

  hull.Add(points1);
  hull.Add(points2);
  hull.Add(points3);
  hull.Add(points4);

  static const struct {
    eigenmath::Vector2d point;
    double radius;
  } checks[] = {{{0.0, 0.0}, M_SQRT2},         {{1.0, 0.0}, M_SQRT2},
                {{1.0, 1.0}, M_SQRT2},         {{0.0, 1.0}, M_SQRT2},
                {{-1.0, -1.0}, 2.0 * M_SQRT2}, {{2.0, -1.0}, 2.0 * M_SQRT2}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    EXPECT_NEAR(checks[ii].radius, hull.GetMaxRadiusAround(checks[ii].point),
                kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be at distance "
        << checks[ii].radius;
  }
}

TEST(Hull, TransformAndCopy) {
  Hull orig_hull;

  const std::vector<eigenmath::Vector2d> points1 = {
      {0.5, 0.0}, {1.0, 0.0}, {0.5, 0.5}};
  const std::vector<eigenmath::Vector2d> points2 = {
      {0.0, 0.5}, {0.5, 0.5}, {0.0, 1.0}};
  const std::vector<eigenmath::Vector2d> points3 = {
      {0.0, 0.0}, {0.5, 0.0}, {0.0, 0.5}};
  const std::vector<eigenmath::Vector2d> points4 = {
      {0.5, 0.5}, {0.5, 0.0}, {0.0, 0.5}};

  orig_hull.Add(points1);
  orig_hull.Add(points2);
  orig_hull.Add(points3);
  orig_hull.Add(points4);

  Hull trans_hull;
  trans_hull = orig_hull;

  eigenmath::Pose2d rel_pose(eigenmath::Vector2d(0.5, 1.0), 0.5 * M_PI);

  ASSERT_TRUE(trans_hull.TransformAndCopy(rel_pose, orig_hull));

  static const struct {
    eigenmath::Vector2d point;
    bool inside;
  } checks[] = {{{0.4, 1.1}, true},   {{0.4, 0.9}, false}, {{0.6, 1.1}, false},
                {{0.4, 2.0}, false},  {{0.45, 1.9}, true}, {{0.55, 1.9}, false},
                {{0.0, 1.55}, false}, {{0.0, 1.45}, true}, {{-0.4, 1.05}, true},
                {{-0.4, 0.95}, false}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    ASSERT_EQ(checks[ii].inside, trans_hull.Contains(checks[ii].point))
        << "check[" << ii << "] point (" << checks[ii].point.x() << " "
        << checks[ii].point.y() << ") should be "
        << (checks[ii].inside ? "inside" : "outside");
  }
}

TEST(LazyTransformedHull, AgainstNormalHull) {
  Hull orig_hull;

  const std::vector<eigenmath::Vector2d> points1 = {
      {0.5, 0.0}, {1.0, 0.0}, {0.5, 0.5}};
  const std::vector<eigenmath::Vector2d> points2 = {
      {0.0, 0.5}, {0.5, 0.5}, {0.0, 1.0}};
  const std::vector<eigenmath::Vector2d> points3 = {
      {0.0, 0.0}, {0.5, 0.0}, {0.0, 0.5}};
  const std::vector<eigenmath::Vector2d> points4 = {
      {0.5, 0.5}, {0.5, 0.0}, {0.0, 0.5}};

  orig_hull.Add(points1);
  orig_hull.Add(points2);
  orig_hull.Add(points3);
  orig_hull.Add(points4);

  Hull trans_hull;
  trans_hull = orig_hull;

  eigenmath::Pose2d rel_pose(eigenmath::Vector2d(0.5, 1.0), 0.5 * M_PI);

  ASSERT_TRUE(trans_hull.TransformAndCopy(rel_pose, orig_hull));

  LazyTransformedHull lazy_hull(orig_hull);

  static const eigenmath::Vector2d checks[] = {
      {0.4, 1.1},  {0.4, 0.9},  {0.6, 1.1},  {0.4, 2.0},   {0.45, 1.9},
      {0.55, 1.9}, {0.0, 1.55}, {0.0, 1.45}, {-0.4, 1.05}, {-0.4, 0.95}};
  constexpr int nchecks = sizeof(checks) / sizeof(*checks);
  for (int ii = 0; ii < nchecks; ++ii) {
    lazy_hull.ApplyTransform(rel_pose);
    EXPECT_EQ(trans_hull.Contains(checks[ii]), lazy_hull.Contains(checks[ii]))
        << "check[" << ii << "] point (" << checks[ii].x() << ", "
        << checks[ii].y() << ")";
    lazy_hull.ApplyTransform(rel_pose);
    EXPECT_NEAR(trans_hull.Distance(checks[ii]), lazy_hull.Distance(checks[ii]),
                kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].x() << ", "
        << checks[ii].y() << ")";
    lazy_hull.ApplyTransform(rel_pose);
    EXPECT_NEAR(trans_hull.DistanceIfPenetrating(checks[ii]),
                lazy_hull.DistanceIfPenetrating(checks[ii]), kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].x() << ", "
        << checks[ii].y() << ")";
    EXPECT_NEAR(trans_hull.DistanceIfLessThan(checks[ii], -0.05),
                lazy_hull.DistanceIfLessThan(checks[ii], -0.05), kEpsilon)
        << "check[" << ii << "] point (" << checks[ii].x() << ", "
        << checks[ii].y() << ")";
    if (trans_hull.DistanceIfPenetrating(checks[ii]) < 0.0) {
      eigenmath::Vector2d lazy_closest_point = eigenmath::Vector2d::Zero();
      EXPECT_NEAR(
          lazy_hull.ClosestPointIfPenetrating(checks[ii], &lazy_closest_point),
          trans_hull.DistanceIfPenetrating(checks[ii]), kEpsilon);
      eigenmath::Vector2d trans_closest_point = eigenmath::Vector2d::Zero();
      EXPECT_NEAR(trans_hull.ClosestPointIfPenetrating(checks[ii],
                                                       &trans_closest_point),
                  trans_hull.DistanceIfPenetrating(checks[ii]), kEpsilon);
      EXPECT_THAT(lazy_closest_point,
                  eigenmath::testing::IsApprox(trans_closest_point, kEpsilon))
          << "check[" << ii << "] point (" << checks[ii].x() << ", "
          << checks[ii].y() << ")";
    }
  }
}

TEST(LazyTransformedHull, AreOverlappingPotato) {
  const std::vector<eigenmath::Vector2d> points1 = {
      {0.0, 0.0}, {1.0, 0.0}, {1.5, 0.25}, {1.75, 0.75}, {1.75, 1.25}};
  const std::vector<eigenmath::Vector2d> points2 = {
      {0.0, 0.0}, {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0}, {0.0, 2.0}};
  Hull orig_hull1;
  orig_hull1.Add(points1);
  orig_hull1.Add(points2);
  LazyTransformedHull lazy_hull1(orig_hull1);
  Hull orig_hull2;
  orig_hull2.Add(points1);
  orig_hull2.Add(points2);
  Hull trans_hull2(orig_hull2);
  LazyTransformedHull lazy_hull2(orig_hull2);
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_FALSE(AreOverlapping(lazy_hull1, lazy_hull2));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_FALSE(AreOverlapping(orig_hull1, lazy_hull2));
  EXPECT_FALSE(AreOverlapping(lazy_hull1, trans_hull2));
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(-1.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_TRUE(AreOverlapping(lazy_hull1, lazy_hull2));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_TRUE(AreOverlapping(orig_hull1, lazy_hull2));
  EXPECT_TRUE(AreOverlapping(lazy_hull1, trans_hull2));
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(3.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_FALSE(AreOverlapping(lazy_hull1, lazy_hull2));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_FALSE(AreOverlapping(orig_hull1, lazy_hull2));
  EXPECT_FALSE(AreOverlapping(lazy_hull1, trans_hull2));
}

TEST(LazyTransformedHull, DistanceBetweenPotato) {
  const std::vector<eigenmath::Vector2d> points1 = {
      {0.0, 0.0}, {1.0, 0.0}, {1.5, 0.25}, {1.75, 0.75}, {1.75, 1.25}};
  const std::vector<eigenmath::Vector2d> points2 = {
      {0.0, 0.0}, {1.75, 1.25}, {1.5, 1.75}, {1.0, 2.0}, {0.0, 2.0}};
  Hull orig_hull1;
  orig_hull1.Add(points1);
  orig_hull1.Add(points2);
  LazyTransformedHull lazy_hull1(orig_hull1);
  Hull orig_hull2;
  orig_hull2.Add(points1);
  orig_hull2.Add(points2);
  Hull trans_hull2(orig_hull2);
  LazyTransformedHull lazy_hull2(orig_hull2);
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2), 0.25, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_GT(DistanceBetween(orig_hull1, lazy_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            -kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2, 0.0), 0.25, kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2), 0.25, kEpsilon);
  EXPECT_GT(DistanceBetween(lazy_hull1, trans_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            -kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2, 0.0), 0.25, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2), 0.25, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_GT(DistanceBetween(lazy_hull1, lazy_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            -kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(2.0, 0.5), 0.0));
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2, 0.0), 0.25, kEpsilon);
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(-1.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2), -0.75, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2,
                              std::numeric_limits<double>::lowest(), 0.0),
              -0.75, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_LT(DistanceBetween(orig_hull1, lazy_hull2, 0.0), 0.0 + kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2), -0.75, kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2,
                              std::numeric_limits<double>::lowest(), 0.0),
              -0.75, kEpsilon);
  EXPECT_LT(DistanceBetween(lazy_hull1, trans_hull2, 0.0), 0.0 + kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2), -0.75, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2,
                              std::numeric_limits<double>::lowest(), 0.0),
              -0.75, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(1.0, 1.0), 0.0));
  EXPECT_LT(DistanceBetween(lazy_hull1, lazy_hull2, 0.0), 0.0 + kEpsilon);
  trans_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(3.0, 0.5), 0.0));
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2), 2.26384628, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_GT(DistanceBetween(orig_hull1, lazy_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            -kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_NEAR(DistanceBetween(orig_hull1, lazy_hull2, 0.0), 2.26384628,
              kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2), 2.26384628, kEpsilon);
  EXPECT_GT(DistanceBetween(lazy_hull1, trans_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            -kEpsilon);
  EXPECT_NEAR(DistanceBetween(lazy_hull1, trans_hull2, 0.0), 2.26384628,
              kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2), 2.26384628, kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_GT(DistanceBetween(lazy_hull1, lazy_hull2,
                            std::numeric_limits<double>::lowest(), 0.0),
            -kEpsilon);
  lazy_hull2.ApplyTransform(
      eigenmath::Pose2d(eigenmath::Vector2d(4.0, 1.5), 0.0));
  EXPECT_NEAR(DistanceBetween(lazy_hull1, lazy_hull2, 0.0), 2.26384628,
              kEpsilon);
}

// Sets up a lazy augmented hull containing Square() and a forward translated
// copy of it.
class LazyAugmentedConvexHullTest : public ::testing::Test {
 protected:
  LazyAugmentedConvexHullTest()
      : orig_(Square()),
        eager_augmented_(orig_),
        lazy_augmented_(&diff_drive::testing::kTestDynamicLimits, &orig_) {}

  void SetUp() override {
    const auto motion = ForwardMotion();
    lazy_augmented_.ApplyMotion(motion);
    eager_augmented_ = AugmentConvexHull(
        diff_drive::testing::kTestDynamicLimits, motion, orig_);
    EXPECT_FALSE(lazy_augmented_.HasAugmentedBoundingCircle());
    EXPECT_FALSE(lazy_augmented_.HasAugmentedPoints());
  }

  ConvexHull orig_;
  ConvexHull eager_augmented_;
  LazyAugmentedConvexHull lazy_augmented_;
};

TEST_F(LazyAugmentedConvexHullTest, ComparisonToEagerHull) {
  EXPECT_THAT(lazy_augmented_.GetPoints(),
              ContainerEq(eager_augmented_.GetPoints()));
}

TEST_F(LazyAugmentedConvexHullTest, Contains) {
  const eigenmath::Vector2d point_outside_circle{{-0.1, -0.1}};
  EXPECT_EQ(eager_augmented_.Contains(point_outside_circle),
            lazy_augmented_.Contains(point_outside_circle));
  EXPECT_TRUE(lazy_augmented_.HasAugmentedBoundingCircle());
  EXPECT_FALSE(lazy_augmented_.HasAugmentedPoints());

  // Line reaching into the object.
  std::vector<eigenmath::Vector2d> points;
  for (double x = 0.0; x < 2.0; x += 0.3) {
    points.push_back({x, 0.5});
  }
  for (const eigenmath::Vector2d& point : points) {
    EXPECT_EQ(eager_augmented_.Contains(point),
              lazy_augmented_.Contains(point));
  }
  EXPECT_TRUE(lazy_augmented_.HasAugmentedBoundingCircle());
  EXPECT_TRUE(lazy_augmented_.HasAugmentedPoints());
}

TEST_F(LazyAugmentedConvexHullTest, Distance) {
  const eigenmath::Vector2d point_outside{{-1.0, -1.0}};
  EXPECT_EQ(eager_augmented_.Distance(point_outside),
            lazy_augmented_.Distance(point_outside));
  EXPECT_TRUE(lazy_augmented_.HasAugmentedPoints());

  const eigenmath::Vector2d point_inside{{0.5, 0.5}};
  EXPECT_EQ(eager_augmented_.Distance(point_inside),
            lazy_augmented_.Distance(point_inside));
}

TEST_F(LazyAugmentedConvexHullTest, DistanceIfPenetrating) {
  const eigenmath::Vector2d far_point{{-1.0, -1.0}};
  EXPECT_EQ(eager_augmented_.DistanceIfPenetrating(far_point),
            lazy_augmented_.DistanceIfPenetrating(far_point));
  EXPECT_TRUE(lazy_augmented_.HasAugmentedBoundingCircle());
  EXPECT_FALSE(lazy_augmented_.HasAugmentedPoints());

  const eigenmath::Vector2d close_point{{-0.1, 0.5}};
  EXPECT_EQ(eager_augmented_.DistanceIfPenetrating(close_point),
            lazy_augmented_.DistanceIfPenetrating(close_point));
  EXPECT_TRUE(lazy_augmented_.HasAugmentedBoundingCircle());
  EXPECT_TRUE(lazy_augmented_.HasAugmentedPoints());
}

TEST(LazyAugmentedConvexHullTestCases, ComparisonToEagerHull) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  auto generator = [&]() { return uni_dist(rand_eng); };

  for (int hull_number = 0; hull_number < 10; ++hull_number) {
    ConvexHull orig = SampleConvexHull(4, generator);
    for (int motion_number = 0; motion_number < 5; ++motion_number) {
      LazyAugmentedConvexHull lazy_augmented(
          &diff_drive::testing::kTestDynamicLimits, &orig);

      constexpr double kDangerMargin = 0.5;
      diff_drive::State state;
      state.SetArcVelocity({generator(), generator()});

      // Lazy evaluation.
      lazy_augmented.ApplyMotion(state);
      lazy_augmented.SetDangerMargin(kDangerMargin);

      // Eager evaluation.
      ConvexHull eager_augmented = AugmentConvexHull(
          diff_drive::testing::kTestDynamicLimits, state, orig);
      eager_augmented = eager_augmented.CreateBiggerHull(kDangerMargin);

      EXPECT_THAT(lazy_augmented.GetPoints(),
                  ContainerEq(eager_augmented.GetPoints()))
          << "state: " << state;

      // The constructor requires points to be far enough apart.  Check that no
      // additional points are present.  Note that the augmented hull can hold
      // the in a different order (rotated).
      eager_augmented = ConvexHull(eager_augmented.GetPoints());
      EXPECT_THAT(lazy_augmented.GetPoints(),
                  UnorderedElementsAreArray(eager_augmented.GetPoints()));
    }
  }
}

TEST(LazyTransformedAugmentedHullTestCases, ComparisonToEagerHull) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  auto generator = [&]() { return uni_dist(rand_eng); };

  for (int hull_number = 0; hull_number < 5; ++hull_number) {
    ConvexHull orig = SampleConvexHull(4, generator);
    for (int motion_number = 0; motion_number < 5; ++motion_number) {
      LazyAugmentedConvexHull lazy_augmented(
          &diff_drive::testing::kTestDynamicLimits, &orig);
      LazyTransformedAugmentedConvexHull lazy_transformed(&lazy_augmented);

      constexpr double kDangerMargin = 0.5;
      diff_drive::State state;
      state.SetArcVelocity({generator(), generator()});
      state.SetPose({{generator(), generator()}, generator()});
      lazy_augmented.ApplyMotion(state);
      lazy_augmented.SetDangerMargin(kDangerMargin);
      lazy_transformed.ApplyTransform(state.GetPose());

      // std::vector<eigenmath::Vector2d> augmented_points;
      ConvexHull eager_augmented = AugmentConvexHull(
          diff_drive::testing::kTestDynamicLimits, state, orig);
      // ConvexHull eager_augmented(augmented_points);
      eager_augmented = eager_augmented.CreateBiggerHull(kDangerMargin);
      ConvexHull eager_transformed = eager_augmented;
      eager_transformed.ApplyTransform(state.GetPose());

      EXPECT_THAT(lazy_transformed.GetPoints(),
                  ContainerEq(eager_transformed.GetPoints()))
          << "state: " << state;
    }
  }
}

TEST(LazyTransformedAugmentedHullTestCases,
     ComparisonToEagerHullWithoutDangerMargin) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  auto generator = [&]() { return uni_dist(rand_eng); };

  for (int hull_number = 0; hull_number < 5; ++hull_number) {
    ConvexHull orig = SampleConvexHull(4, generator);
    for (int motion_number = 0; motion_number < 5; ++motion_number) {
      LazyAugmentedConvexHull lazy_augmented(
          &diff_drive::testing::kTestDynamicLimits, &orig);
      LazyTransformedAugmentedConvexHull lazy_transformed(&lazy_augmented);

      constexpr double kDangerMargin = 0.0;
      diff_drive::State state;
      state.SetArcVelocity({generator(), generator()});
      state.SetPose({{generator(), generator()}, generator()});
      lazy_augmented.ApplyMotion(state);
      lazy_augmented.SetDangerMargin(kDangerMargin);
      lazy_transformed.ApplyTransform(state.GetPose());

      ConvexHull eager_augmented = AugmentConvexHull(
          diff_drive::testing::kTestDynamicLimits, state, orig);
      // Do not add danger margin.
      ConvexHull eager_transformed = eager_augmented;
      eager_transformed.ApplyTransform(state.GetPose());

      EXPECT_THAT(lazy_transformed.GetPoints(),
                  ContainerEq(eager_transformed.GetPoints()))
          << "state: " << state;
    }
  }
}

// Sets up a lazy transformed augmented hull containing Square() and a forward
// translated copy of it, shifted to the right.
class LazyTransformedAugmentedConvexHullTest : public ::testing::Test {
 protected:
  LazyTransformedAugmentedConvexHullTest()
      : orig_(Square()),
        lazy_augmented_(&diff_drive::testing::kTestDynamicLimits, &orig_),
        eager_transformed_(Square()),
        lazy_transformed_(&lazy_augmented_) {}

  void SetUp() override {
    // Apply motion.
    const auto motion = ForwardMotion();
    lazy_augmented_.ApplyMotion(motion);
    eager_transformed_ = AugmentConvexHull(
        diff_drive::testing::kTestDynamicLimits, motion, orig_);

    // Transform to the right.
    eigenmath::Pose2d pose_to_right;
    pose_to_right.translation() = eigenmath::Vector2d(0.0, 5.0);
    eager_transformed_.ApplyTransform(pose_to_right);
    lazy_transformed_.ApplyTransform(pose_to_right);
    EXPECT_FALSE(lazy_transformed_.HasTransformedCentroid());
    EXPECT_FALSE(lazy_transformed_.HasTransformedPoints());
    EXPECT_FALSE(lazy_augmented_.HasAugmentedBoundingCircle());
    EXPECT_FALSE(lazy_augmented_.HasAugmentedPoints());
  }

 private:
  ConvexHull orig_;
  LazyAugmentedConvexHull lazy_augmented_;

 protected:
  ConvexHull eager_transformed_;
  LazyTransformedAugmentedConvexHull lazy_transformed_;
};

TEST_F(LazyTransformedAugmentedConvexHullTest, ComparisonToEagerHull) {
  EXPECT_THAT(lazy_transformed_.GetPoints(),
              ContainerEq(eager_transformed_.GetPoints()));
}

TEST_F(LazyTransformedAugmentedConvexHullTest, Contains) {
  const eigenmath::Vector2d point_outside_circle{{-0.1, -5.1}};
  EXPECT_EQ(eager_transformed_.Contains(point_outside_circle),
            lazy_transformed_.Contains(point_outside_circle));
  EXPECT_TRUE(lazy_transformed_.HasTransformedCentroid());
  EXPECT_FALSE(lazy_transformed_.HasTransformedPoints());

  // Line reaching into the object.
  std::vector<eigenmath::Vector2d> points;
  for (double x = 0.0; x < 2.0; x += 0.3) {
    points.push_back({x, 5.5});
  }
  for (const eigenmath::Vector2d& point : points) {
    EXPECT_EQ(eager_transformed_.Contains(point),
              lazy_transformed_.Contains(point));
  }
  EXPECT_TRUE(lazy_transformed_.HasTransformedCentroid());
  EXPECT_TRUE(lazy_transformed_.HasTransformedPoints());
}

TEST_F(LazyTransformedAugmentedConvexHullTest, Distance) {
  const eigenmath::Vector2d point_outside{{-1.0, 4.0}};
  EXPECT_EQ(eager_transformed_.Distance(point_outside),
            lazy_transformed_.Distance(point_outside));
  EXPECT_TRUE(lazy_transformed_.HasTransformedPoints());

  const eigenmath::Vector2d point_inside{{0.5, 5.5}};
  EXPECT_EQ(eager_transformed_.Distance(point_inside),
            lazy_transformed_.Distance(point_inside));
}

TEST_F(LazyTransformedAugmentedConvexHullTest, DistanceIfPenetrating) {
  const eigenmath::Vector2d far_point{{-1.0, 4.0}};
  EXPECT_EQ(eager_transformed_.DistanceIfPenetrating(far_point),
            lazy_transformed_.DistanceIfPenetrating(far_point));
  EXPECT_TRUE(lazy_transformed_.HasTransformedCentroid());
  EXPECT_FALSE(lazy_transformed_.HasTransformedPoints());

  const eigenmath::Vector2d close_point{{-0.1, 5.5}};
  EXPECT_EQ(eager_transformed_.DistanceIfPenetrating(close_point),
            lazy_transformed_.DistanceIfPenetrating(close_point));
  EXPECT_TRUE(lazy_transformed_.HasTransformedCentroid());
  EXPECT_TRUE(lazy_transformed_.HasTransformedPoints());
}

void BM_ConvexHullContains(benchmark::State& state) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  std::vector<eigenmath::Vector2d> points_test1;
  for (int i = 0; i < state.range(0); ++i) {
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
  }
  eigenmath::Vector2d point_test2(uni_dist(rand_eng) * 2.0,
                                  uni_dist(rand_eng) * 2.0);
  ConvexHull hull_test1(points_test1);
  while (hull_test1.GetPoints().size() != state.range(0)) {
    points_test1 = hull_test1.GetPoints();
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test1 = ConvexHull(points_test1);
  }

  for (auto _ : state) {
    const eigenmath::Pose2d pose_test(
        eigenmath::Vector2d(uni_dist(rand_eng) * 2.0, uni_dist(rand_eng) * 2.0),
        uni_dist(rand_eng) * 2.0 * M_PI);
    const eigenmath::Vector2d point_test2_transformed = pose_test * point_test2;
    // Runs 10 times to drown out the cost of applying the transform.
    for (int i = 0; i < 1000; ++i) {
      CHECK_GT(static_cast<int>(hull_test1.Contains(point_test2_transformed)),
               -1);
    }
  }
}
BENCHMARK(BM_ConvexHullContains)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

void BM_ConvexHullDistanceIfPenetrating(benchmark::State& state) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  std::vector<eigenmath::Vector2d> points_test1;
  for (int i = 0; i < state.range(0); ++i) {
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
  }
  eigenmath::Vector2d point_test2(uni_dist(rand_eng) * 2.0,
                                  uni_dist(rand_eng) * 2.0);
  ConvexHull hull_test1(points_test1);
  while (hull_test1.GetPoints().size() != state.range(0)) {
    points_test1 = hull_test1.GetPoints();
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test1 = ConvexHull(points_test1);
  }

  for (auto _ : state) {
    const eigenmath::Pose2d pose_test(
        eigenmath::Vector2d(uni_dist(rand_eng) * 2.0, uni_dist(rand_eng) * 2.0),
        uni_dist(rand_eng) * 2.0 * M_PI);
    const eigenmath::Vector2d point_test2_transformed = pose_test * point_test2;
    // Runs 10 times to drown out the cost of applying the transform.
    for (int i = 0; i < 1000; ++i) {
      CHECK_GT(hull_test1.DistanceIfPenetrating(point_test2_transformed), -100);
    }
  }
}
BENCHMARK(BM_ConvexHullDistanceIfPenetrating)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

void BM_ConvexHullDistance(benchmark::State& state) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  std::vector<eigenmath::Vector2d> points_test1;
  for (int i = 0; i < state.range(0); ++i) {
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
  }
  eigenmath::Vector2d point_test2(uni_dist(rand_eng) * 2.0,
                                  uni_dist(rand_eng) * 2.0);
  ConvexHull hull_test1(points_test1);
  while (hull_test1.GetPoints().size() != state.range(0)) {
    points_test1 = hull_test1.GetPoints();
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test1 = ConvexHull(points_test1);
  }

  for (auto _ : state) {
    const eigenmath::Pose2d pose_test(
        eigenmath::Vector2d(uni_dist(rand_eng) * 2.0, uni_dist(rand_eng) * 2.0),
        uni_dist(rand_eng) * 2.0 * M_PI);
    const eigenmath::Vector2d point_test2_transformed = pose_test * point_test2;
    // Runs 10 times to drown out the cost of applying the transform.
    for (int i = 0; i < 1000; ++i) {
      CHECK_GT(hull_test1.Distance(point_test2_transformed), -100);
    }
  }
}
BENCHMARK(BM_ConvexHullDistance)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

void BM_ConvexHullAreOverlapping(benchmark::State& state) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  std::vector<eigenmath::Vector2d> points_test1, points_test2;
  for (int i = 0; i < state.range(0); ++i) {
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    points_test2.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
  }
  ConvexHull hull_test1(points_test1);
  while (hull_test1.GetPoints().size() != state.range(0)) {
    points_test1 = hull_test1.GetPoints();
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test1 = ConvexHull(points_test1);
  }
  ConvexHull hull_test2(points_test2);
  while (hull_test2.GetPoints().size() != state.range(0)) {
    points_test2 = hull_test2.GetPoints();
    points_test2.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test2 = ConvexHull(points_test2);
  }

  for (auto _ : state) {
    const eigenmath::Pose2d pose_test(
        eigenmath::Vector2d(uni_dist(rand_eng) * 2.0, uni_dist(rand_eng) * 2.0),
        uni_dist(rand_eng) * 2.0 * M_PI);
    hull_test1.ApplyTransform(pose_test);
    // Runs 10 times to drown out the cost of applying the transform.
    for (int i = 0; i < 10; ++i) {
      CHECK_GT(static_cast<int>(AreOverlapping(hull_test2, hull_test1)), -1);
    }
    hull_test1.ApplyTransform(pose_test.inverse());
  }
}
BENCHMARK(BM_ConvexHullAreOverlapping)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

void BM_ConvexHullDistanceBetween(benchmark::State& state) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  std::vector<eigenmath::Vector2d> points_test1, points_test2;
  for (int i = 0; i < state.range(0); ++i) {
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    points_test2.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
  }
  ConvexHull hull_test1(points_test1);
  while (hull_test1.GetPoints().size() != state.range(0)) {
    points_test1 = hull_test1.GetPoints();
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test1 = ConvexHull(points_test1);
  }
  ConvexHull hull_test2(points_test2);
  while (hull_test2.GetPoints().size() != state.range(0)) {
    points_test2 = hull_test2.GetPoints();
    points_test2.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test2 = ConvexHull(points_test2);
  }

  for (auto _ : state) {
    const eigenmath::Pose2d pose_test(
        eigenmath::Vector2d(uni_dist(rand_eng) * 2.0, uni_dist(rand_eng) * 2.0),
        uni_dist(rand_eng) * 2.0 * M_PI);
    hull_test1.ApplyTransform(pose_test);
    // Runs 10 times to drown out the cost of applying the transform.
    for (int i = 0; i < 10; ++i) {
      CHECK_LT(DistanceBetween(hull_test2, hull_test1),
               std::numeric_limits<double>::max());
    }
    hull_test1.ApplyTransform(pose_test.inverse());
  }
}
BENCHMARK(BM_ConvexHullDistanceBetween)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

void BM_ConvexHullDistanceBetweenSeparationOnly(benchmark::State& state) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  std::vector<eigenmath::Vector2d> points_test1, points_test2;
  for (int i = 0; i < state.range(0); ++i) {
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    points_test2.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
  }
  ConvexHull hull_test1(points_test1);
  while (hull_test1.GetPoints().size() != state.range(0)) {
    points_test1 = hull_test1.GetPoints();
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test1 = ConvexHull(points_test1);
  }
  ConvexHull hull_test2(points_test2);
  while (hull_test2.GetPoints().size() != state.range(0)) {
    points_test2 = hull_test2.GetPoints();
    points_test2.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test2 = ConvexHull(points_test2);
  }

  for (auto _ : state) {
    const eigenmath::Pose2d pose_test(
        eigenmath::Vector2d(uni_dist(rand_eng) * 2.0, uni_dist(rand_eng) * 2.0),
        uni_dist(rand_eng) * 2.0 * M_PI);
    hull_test1.ApplyTransform(pose_test);
    // Runs 10 times to drown out the cost of applying the transform.
    for (int i = 0; i < 10; ++i) {
      CHECK_LT(DistanceBetween(hull_test2, hull_test1, 0.0),
               std::numeric_limits<double>::max());
    }
    hull_test1.ApplyTransform(pose_test.inverse());
  }
}
BENCHMARK(BM_ConvexHullDistanceBetweenSeparationOnly)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32);

void BM_ConvexHullDistanceBetweenPenetrationOnly(benchmark::State& state) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  std::vector<eigenmath::Vector2d> points_test1, points_test2;
  for (int i = 0; i < state.range(0); ++i) {
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    points_test2.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
  }
  ConvexHull hull_test1(points_test1);
  while (hull_test1.GetPoints().size() != state.range(0)) {
    points_test1 = hull_test1.GetPoints();
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test1 = ConvexHull(points_test1);
  }
  ConvexHull hull_test2(points_test2);
  while (hull_test2.GetPoints().size() != state.range(0)) {
    points_test2 = hull_test2.GetPoints();
    points_test2.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
    hull_test2 = ConvexHull(points_test2);
  }

  for (auto _ : state) {
    const eigenmath::Pose2d pose_test(
        eigenmath::Vector2d(uni_dist(rand_eng) * 2.0, uni_dist(rand_eng) * 2.0),
        uni_dist(rand_eng) * 2.0 * M_PI);
    hull_test1.ApplyTransform(pose_test);
    // Runs 10 times to drown out the cost of applying the transform.
    for (int i = 0; i < 10; ++i) {
      CHECK_LT(DistanceBetween(hull_test2, hull_test1,
                               std::numeric_limits<double>::lowest(), 0.0),
               std::numeric_limits<double>::max());
    }
    hull_test1.ApplyTransform(pose_test.inverse());
  }
}
BENCHMARK(BM_ConvexHullDistanceBetweenPenetrationOnly)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32);

void BM_LazyConvexHullAugmentation(benchmark::State& state) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
  auto generator = [&]() { return uni_dist(rand_eng) * 2.0; };

  ConvexHull orig_hull = SampleConvexHull(state.range(0), generator);
  diff_drive::State dd_states[2];
  dd_states[0].SetArcVelocity({generator(), generator()});
  dd_states[1].SetArcVelocity({generator(), generator()});
  constexpr double kDangerMargins[] = {0.3, 0.5};

  int param_index = 0;
  auto augmented_hull = LazyAugmentedConvexHull(
      &diff_drive::testing::kTestDynamicLimits, &orig_hull);

  for (auto _ : state) {
    augmented_hull.ApplyMotion(dd_states[param_index]);
    augmented_hull.SetDangerMargin(kDangerMargins[param_index]);
    param_index = (param_index + 1) % 2;
    // Force evaluation of lazy augmentation.
    benchmark::DoNotOptimize(augmented_hull.GetPoints());
  }
}
BENCHMARK(BM_LazyConvexHullAugmentation)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

void BM_LazyConvexHullTransformation(benchmark::State& state) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
  auto generator = [&]() { return uni_dist(rand_eng) * 2.0; };

  ConvexHull orig_hull = SampleConvexHull(state.range(0), generator);
  eigenmath::Pose2d poses[2] = {{{generator(), generator()}, generator()},
                                {{generator(), generator()}, generator()}};

  int pose_index = 0;
  auto transformed_hull = LazyTransformedConvexHull(&orig_hull);

  for (auto _ : state) {
    transformed_hull.ApplyTransform(poses[pose_index]);
    pose_index = (pose_index + 1) % 2;
    // Force evaluation of lazy transformation.
    benchmark::DoNotOptimize(transformed_hull.GetPoints());
  }
}
BENCHMARK(BM_LazyConvexHullTransformation)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

void BM_GetMinAreaBoundingBox(benchmark::State& state) {
  std::default_random_engine rand_eng(1745897623);  // Fixed seed.
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

  std::vector<eigenmath::Vector2d> points_test1;
  for (int i = 0; i < state.range(0); ++i) {
    points_test1.emplace_back(uni_dist(rand_eng) * 2.0,
                              uni_dist(rand_eng) * 2.0);
  }

  for (auto _ : state) {
    ConvexHull hull2(points_test1);
    OrientedBox2d box = hull2.GetMinAreaBoundingBox();
    CHECK_GT(box.Area(), 0.0);
  }
}
BENCHMARK(BM_GetMinAreaBoundingBox)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

}  // namespace
}  // namespace mobility::collision

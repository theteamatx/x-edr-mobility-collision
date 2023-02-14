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

#include "collision/lattice_pose.h"

#include <array>

#include "eigenmath/distribution.h"
#include "eigenmath/matchers.h"
#include "eigenmath/pose2.h"
#include "eigenmath/sampling.h"
#include "eigenmath/types.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

using ::eigenmath::testing::IsApprox;
using ::testing::Eq;
using ::testing::UnorderedElementsAreArray;

const double kEpsilon = 0.0001;
const double kGridResolution = 0.1;
const int kLatticeAngleDivisions = 16;
const double kLatticeAngleResolution = 2.0 * M_PI / kLatticeAngleDivisions;

TEST(LatticeFrame, FrameSO2ToLatticeAngleIndex) {
  const GridFrame grid_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI),
                             kGridResolution);
  const LatticeFrame lat_frame(grid_frame, kLatticeAngleDivisions);

  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(M_PI)),
              Eq(0));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(
                  M_PI + kLatticeAngleResolution / 2.0 - kEpsilon)),
              Eq(0));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(
                  M_PI - kLatticeAngleResolution / 2.0 + kEpsilon)),
              Eq(0));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(
                  M_PI + kLatticeAngleResolution / 2.0 + kEpsilon)),
              Eq(1));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(
                  M_PI - kLatticeAngleResolution / 2.0 - kEpsilon)),
              Eq(15));
  EXPECT_THAT(
      lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(2.0 * M_PI)),
      Eq(8));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(0.0)),
              Eq(8));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(-M_PI)),
              Eq(0));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(
                  -M_PI + kLatticeAngleResolution / 2.0 - kEpsilon)),
              Eq(0));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(
                  -M_PI - kLatticeAngleResolution / 2.0 + kEpsilon)),
              Eq(0));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(
                  -M_PI + kLatticeAngleResolution / 2.0 + kEpsilon)),
              Eq(1));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndex(eigenmath::SO2d(
                  -M_PI - kLatticeAngleResolution / 2.0 - kEpsilon)),
              Eq(15));
}

TEST(LatticeFrame, FrameSO2ToLatticeAngleIndices) {
  const GridFrame grid_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI),
                             kGridResolution);
  const LatticeFrame lat_frame(grid_frame, kLatticeAngleDivisions);

  EXPECT_THAT(
      lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(M_PI + kEpsilon)),
      UnorderedElementsAreArray({0, 1}));
  EXPECT_THAT(
      lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(M_PI - kEpsilon)),
      UnorderedElementsAreArray({0, 15}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(
                  M_PI + kLatticeAngleResolution / 2.0 - kEpsilon)),
              UnorderedElementsAreArray({0, 1}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(
                  M_PI - kLatticeAngleResolution / 2.0 + kEpsilon)),
              UnorderedElementsAreArray({0, 15}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(
                  M_PI + kLatticeAngleResolution / 2.0 + kEpsilon)),
              UnorderedElementsAreArray({0, 1}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(
                  M_PI - kLatticeAngleResolution / 2.0 - kEpsilon)),
              UnorderedElementsAreArray({0, 15}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(
                  eigenmath::SO2d(2.0 * M_PI + kEpsilon)),
              UnorderedElementsAreArray({8, 9}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(
                  eigenmath::SO2d(2.0 * M_PI - kEpsilon)),
              UnorderedElementsAreArray({7, 8}));
  EXPECT_THAT(
      lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(kEpsilon)),
      UnorderedElementsAreArray({8, 9}));
  EXPECT_THAT(
      lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(-kEpsilon)),
      UnorderedElementsAreArray({7, 8}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(
                  eigenmath::SO2d(-M_PI + kEpsilon)),
              UnorderedElementsAreArray({0, 1}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(
                  eigenmath::SO2d(-M_PI - kEpsilon)),
              UnorderedElementsAreArray({0, 15}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(
                  -M_PI + kLatticeAngleResolution / 2.0 - kEpsilon)),
              UnorderedElementsAreArray({0, 1}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(
                  -M_PI - kLatticeAngleResolution / 2.0 + kEpsilon)),
              UnorderedElementsAreArray({0, 15}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(
                  -M_PI + kLatticeAngleResolution / 2.0 + kEpsilon)),
              UnorderedElementsAreArray({0, 1}));
  EXPECT_THAT(lat_frame.FrameSO2ToLatticeAngleIndices(eigenmath::SO2d(
                  -M_PI - kLatticeAngleResolution / 2.0 - kEpsilon)),
              UnorderedElementsAreArray({0, 15}));
}

TEST(LatticeFrame, FrameToLatticeToFrame) {
  const GridFrame grid_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI),
                             kGridResolution);
  const LatticeFrame lat_frame(grid_frame, kLatticeAngleDivisions);

  eigenmath::TestGenerator rng_gen{eigenmath::kGeneratorTestSeed};
  eigenmath::UniformDistributionPose2d pose_dist;

  for (int i = 0; i < 20; ++i) {
    const eigenmath::Pose2d p_orig = pose_dist(rng_gen);
    const eigenmath::Pose2d p_new =
        lat_frame.LatticeToFrame(lat_frame.FrameToLattice(p_orig));
    EXPECT_NEAR(p_new.translation().x(), p_orig.translation().x(),
                kGridResolution / 2.0);
    EXPECT_NEAR(p_new.translation().y(), p_orig.translation().y(),
                kGridResolution / 2.0);
    EXPECT_THAT(p_new.so2(),
                IsApprox(p_orig.so2(), kLatticeAngleResolution / 2.0));
  }
}

TEST(LatticeFrame, ClosestLatticePoses) {
  const GridFrame grid_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI),
                             kGridResolution);
  const LatticeFrame lat_frame(grid_frame, kLatticeAngleDivisions);

  eigenmath::TestGenerator rng_gen{eigenmath::kGeneratorTestSeed};
  eigenmath::UniformDistributionPose2d pose_dist;

  for (int i = 0; i < 20; ++i) {
    const eigenmath::Pose2d p_orig = pose_dist(rng_gen);
    const std::array<LatticePose, 8> lat_poses =
        lat_frame.ClosestLatticePoses(p_orig);
    for (const auto& lat_pose : lat_poses) {
      const eigenmath::Pose2d p_new = lat_frame.LatticeToFrame(lat_pose);
      EXPECT_NEAR(p_new.translation().x(), p_orig.translation().x(),
                  kGridResolution);
      EXPECT_NEAR(p_new.translation().y(), p_orig.translation().y(),
                  kGridResolution);
      EXPECT_THAT(p_new.so2(), IsApprox(p_orig.so2(), kLatticeAngleResolution));
    }
  }
}

}  // namespace
}  // namespace mobility::collision

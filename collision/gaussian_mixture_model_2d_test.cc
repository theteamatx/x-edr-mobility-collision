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

#include "collision/gaussian_mixture_model_2d.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "eigenmath/matchers.h"
#include "eigenmath/types.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

TEST(GaussianMixtureModelTest, TestDimensions) {
  constexpr double kNumComponents = 2;

  std::vector<eigenmath::Vector2d> samples({{0.0, 0.0}, {1.0, 1.0}});
  const auto model = FitGaussianMixture(samples, kNumComponents, 1);

  const auto& means = model.Means();
  EXPECT_EQ(means.size(), kNumComponents);

  const auto& covariances = model.Covariances();
  EXPECT_EQ(covariances.size(), kNumComponents);

  const auto& weights = model.Weights();
  EXPECT_EQ(weights.size(), kNumComponents);

  const auto& assignments = Cluster(model, samples);
  EXPECT_EQ(assignments.size(), kNumComponents);
}

TEST(GaussianMixtureModelTest, TestFitAndCluster) {
  // This test computes GMM fit on points sampled from two normal distributions
  // with means (0.0, 0.0) and (1.0, 1.0) respectively.

  constexpr double kNumComponents = 2;

  std::vector<eigenmath::Vector2d> samples = {
      {-0.42066539, 0.27245867},  {0.32020358, 0.70640669},
      {-1.22535812, 0.91400322},  {0.26636418, 0.63950791},
      {-0.57170825, -0.31391238}, {-0.4255136, 0.97587012},
      {0.4651654, -1.88530487},   {0.61080489, -1.00862093},
      {0.85237696, 0.01054289},   {-0.28556363, -1.18568488},
      {-0.16167106, 0.64708921},  {0.58001829, -0.49782324},
      {-0.0298887, -1.83847705},  {-1.56430565, 1.72306189},
      {-1.31749659, 0.23211178},  {-2.04743135, -0.88152512},
      {-1.08357499, -0.88632615}, {0.26472079, -2.82717255},
      {1.09790622, -0.05903765},  {-0.53740482, -0.08149446},
      {-0.73020647, -2.51619972}, {0.70544524, 0.69299419},
      {0.30204706, -0.33425917},  {-1.47922838, -0.48566066},
      {-0.85496614, 0.74981261},  {1.89050003, 1.10647472},
      {-0.87129168, 0.1953911},   {1.07138865, -0.95953823},
      {-0.05020272, 0.24693672},  {0.6274057, 1.19559289},
      {-1.9449457, -0.52901459},  {0.08078008, 0.43050216},
      {1.39480634, 0.24984036},   {-0.3035451, -0.94539186},
      {0.40955549, -0.28084566},  {0.55239178, 0.34710348},
      {-0.61389001, -1.56916603}, {0.28262493, -1.06387056},
      {2.30330922, 0.03940474},   {0.76154525, 0.04366192},
      {1.27190305, -1.23733831},  {1.93437665, -1.86087916},
      {0.11100488, 0.04434245},   {0.2793723, -0.31151413},
      {-1.28121141, 0.68181645},  {0.08480207, -1.77603346},
      {0.99127049, -1.16659381},  {0.03150228, 0.04575129},
      {0.21374219, -0.14452303},  {-0.93981773, -0.36073701},
      {0.01115141, -1.62733535},  {0.61767096, 1.60390311},
      {-1.18941638, 0.31547374},  {1.54943054, -0.8858462},
      {-1.00299496, 0.52212422},  {0.80744309, 0.70228723},
      {-0.96268394, -1.33309059}, {-0.84416434, -1.14865527},
      {-0.26030535, 0.92282488},  {0.20730095, 0.70881959},
      {2.27985429, 0.96084993},   {0.02693938, -0.06313706},
      {-0.21786218, -0.37215866}, {1.01055827, -0.90779647},
      {0.32031415, -1.90162071},  {-2.2416667, 0.28624352},
      {0.46543651, -1.29243201},  {0.8687528, 0.22361407},
      {-0.68060896, -0.32556877}, {0.03737989, 0.69137571},
      {1.65369511, 0.43261774},   {1.68452809, 0.16067183},
      {-1.075402, -0.88490652},   {-0.66149106, -0.90309182},
      {0.33750548, 0.50357859},   {0.30919747, 0.65040355},
      {-1.72250105, -0.58532124}, {-0.90165072, 0.08423747},
      {0.42527553, -0.26390384},  {-1.84238377, 1.06084025},
      {1.02737954, -0.83716781},  {0.60758958, 0.22792795},
      {0.65362507, 0.10988851},   {2.24726778, 1.25799088},
      {-0.45618912, 1.57844885},  {0.7521782, 0.5132088},
      {0.24316589, 1.52190264},   {0.74557951, -0.14695676},
      {1.0050385, 1.9623814},     {0.46827587, 0.00586674},
      {-0.8391659, -0.3643822},   {-0.59274133, 1.09449209},
      {-0.03145169, 0.36038483},  {-0.62626399, -0.99925616},
      {0.19160219, -0.35594769},  {0.34819933, 0.07360121},
      {-0.19091945, 1.14076783},  {1.14414559, -0.44558611},
      {-1.03445909, -0.83093086}, {-1.0471651, -0.29323532},
      {1.051435176, 0.977273678}, {0.935444358, 1.241166945},
      {1.092250707, 1.118323395}, {0.942486745, 0.898839535},
      {1.012574777, 0.996554614}, {1.136533378, 0.98854528},
      {1.02213101, 0.903460639},  {0.986806043, 0.930099787},
      {1.166824137, 1.003895218}, {1.00799984, 0.991097837},
      {0.863862335, 1.003672078}, {0.963799807, 0.946587001},
      {1.231225876, 1.13729442},  {0.924287254, 1.066368269},
      {0.930951682, 0.92109867},  {0.957932622, 0.911782892},
      {0.941531483, 0.977815843}, {0.895147409, 1.02061983},
      {1.055754037, 0.730383257}, {1.033838934, 0.957559019},
      {0.952049428, 0.910496867}, {0.942512163, 0.973040452},
      {1.131132789, 1.073463063}, {1.049030668, 1.037288458},
      {0.987202517, 0.987825182}, {1.10235945, 0.938999029},
      {0.812501296, 0.938659341}, {0.886231902, 1.047846607},
      {1.223980161, 0.98175046},  {0.938815303, 1.019920002},
      {1.034145353, 0.949227617}, {0.904251103, 0.945115731},
      {0.930720657, 0.731211188}, {0.842083973, 0.908283359},
      {1.090616645, 0.901153571}, {0.989048688, 1.025247852},
      {1.002117953, 0.949415839}, {1.168053756, 1.060843126},
      {1.059849828, 0.808425688}, {1.117188204, 0.792570665},
      {1.010506151, 1.007061846}, {0.897078258, 1.163980088},
      {1.032296421, 1.013705956}, {1.113896454, 0.944012749},
      {1.083633417, 0.986661662}, {0.858311168, 1.062858637},
      {0.855167341, 0.97605128},  {0.943421357, 1.088075468},
      {0.952120529, 0.978320334}, {1.067062644, 1.193395773},
      {0.882587922, 0.959408084}, {1.212738924, 0.894845844},
      {1.119643309, 0.98388094},  {1.03234755, 1.004354273},
      {0.940821909, 0.928584023}, {1.103732431, 1.154462476},
      {0.866104125, 1.016003698}, {1.109231386, 0.950301247},
      {1.015550832, 1.00337229},  {0.910652834, 0.949330461},
      {1.195937542, 0.897395319}, {1.032102341, 0.939036165},
      {0.83342259, 1.09937261},   {1.095243347, 1.094799205},
      {1.203755312, 0.707554011}, {1.052085036, 1.143540778},
      {0.976247423, 1.00029531},  {0.945539432, 0.981916158},
      {1.012885615, 1.010217617}, {1.068564167, 0.976643375},
      {1.033634721, 0.954902011}, {0.845661113, 0.968427655},
      {0.971745064, 0.891916251}, {1.06577676, 1.091409125},
      {0.92199538, 0.951549135},  {1.021781781, 1.085224097},
      {0.991182074, 1.120588063}, {1.065613876, 1.028446192},
      {0.968197545, 0.985074659}, {1.102082735, 0.87924356},
      {1.107093783, 1.042078656}, {1.010109637, 1.110347677},
      {0.881673442, 0.867132059}, {1.050652114, 1.180895485},
      {0.991932941, 1.064064861}, {1.081849858, 1.025233529},
      {0.968272242, 1.05133839},  {0.834525535, 1.038258125},
      {1.033459273, 0.951609231}, {1.088149252, 0.922831984},
      {0.912376541, 0.94136688},  {0.918544125, 1.249858736},
      {0.885918331, 0.941186925}, {1.057606003, 1.01081467},
      {1.152134095, 0.738816026}, {0.742934262, 1.029403666},
      {1.049431889, 1.096319551}, {1.010510269, 0.93011101},
      {0.931568817, 0.906486172}, {0.973480593, 0.904822669}};

  const auto model = FitGaussianMixture(samples, kNumComponents, 1000, 0.0);

  const auto& means = model.Means();
  ASSERT_EQ(means.size(), kNumComponents);

  constexpr double kEpsilon = 0.1;
  EXPECT_THAT(means, testing::UnorderedElementsAreArray(
                         {eigenmath::testing::IsApprox(
                              eigenmath::Vector2d({0.0, 0.0}), kEpsilon),
                          eigenmath::testing::IsApprox(
                              eigenmath::Vector2d({1.0, 1.0}), kEpsilon)}));

  std::vector<eigenmath::Vector2d> test_points;
  test_points.push_back({0.0, 0.0});
  test_points.push_back({1.0, 1.0});
  const auto& assignments = Cluster(model, test_points);
  ASSERT_EQ(assignments.size(), kNumComponents);
  ASSERT_THAT(assignments, testing::Each(testing::SizeIs(1)));

  EXPECT_THAT(assignments[0], testing::ElementsAreArray({0}));
  EXPECT_THAT(assignments[1], testing::ElementsAreArray({1}));

  const auto& weights = model.Weights();
  ASSERT_EQ(weights.size(), kNumComponents);

  EXPECT_THAT(weights, testing::Each(testing::DoubleNear(0.5, kEpsilon)));
}

std::vector<double> EvaluateEllipseDistances(
    const eigenmath::Vector2d& mean, const eigenmath::Matrix2d& covariance,
    const std::vector<eigenmath::Vector2d>& points) {
  // Compute max elliptical distance of points in the ellipse.
  const eigenmath::Matrix2d cov_inv = covariance.inverse();
  std::vector<double> result;
  for (const auto& point : points) {
    const eigenmath::Vector2d diff = point - mean;
    result.emplace_back(diff.transpose() * cov_inv * diff);
  }
  return result;
}

TEST(ScaleAndExpandEllipseTest, TestFit) {
  std::vector<eigenmath::Vector2d> cluster{{
      {0.0, 0.0},
      {1.0, 0.0},
      {2.0, 0.0},
      {3.0, 0.0},
      {-1.0, 0.0},
      {-2.0, 0.0},
      {-3.0, 0.0},
      {0.0, 0.5},
      {0.0, -0.5},
  }};

  const auto gmm = FitGaussianMixture(cluster, 1, 10, 0.0);

  auto assignments = Cluster(gmm, cluster);
  const auto& means = gmm.Means();
  const auto& covariances = gmm.Covariances();

  constexpr double kMargin = 0.2;

  eigenmath::Matrix2d ellipse_mat = ScaleAndExpandEllipse(
      means[0], covariances[0], assignments[0], cluster, kMargin);

  EXPECT_TRUE(ellipse_mat.allFinite());
  EXPECT_GT(ellipse_mat.determinant(), 0.0);

  std::vector<int> rev_assignments = assignments[0];
  std::reverse(rev_assignments.begin(), rev_assignments.end());
  eigenmath::Matrix2d rev_ellipse_mat = ScaleAndExpandEllipse(
      means[0], covariances[0], rev_assignments, cluster, kMargin);

  EXPECT_THAT(rev_ellipse_mat, eigenmath::testing::IsApprox(ellipse_mat));

  std::vector<eigenmath::Vector2d> test_points{{
      {3.0 + kMargin, 0.0},
      {-3.0 - kMargin, 0.0},
      {0.0, 0.5 + kMargin},
      {0.0, -0.5 - kMargin},
  }};

  EXPECT_THAT(EvaluateEllipseDistances(means[0], ellipse_mat, test_points),
              testing::Each(testing::DoubleNear(1.0, 1e-3)));
}

TEST(ScaleAndExpandEllipseTest, TestFitDegenerate) {
  std::vector<eigenmath::Vector2d> cluster{{
      {0.0, 0.0},
      {1.0, 0.0},
      {2.0, 0.0},
      {3.0, 0.0},
      {-1.0, 0.0},
      {-2.0, 0.0},
      {-3.0, 0.0},
  }};

  const auto& gmm = FitGaussianMixture(cluster, 1, 10, 0.0);

  auto assignments = Cluster(gmm, cluster);
  const auto& means = gmm.Means();
  const auto& covariances = gmm.Covariances();

  constexpr double kMargin = 0.2;

  eigenmath::Matrix2d ellipse_mat = ScaleAndExpandEllipse(
      means[0], covariances[0], assignments[0], cluster, kMargin);

  EXPECT_TRUE(ellipse_mat.allFinite());
  EXPECT_GT(ellipse_mat.determinant(), 0.0);

  std::vector<int> rev_assignments = assignments[0];
  std::reverse(rev_assignments.begin(), rev_assignments.end());
  eigenmath::Matrix2d rev_ellipse_mat = ScaleAndExpandEllipse(
      means[0], covariances[0], rev_assignments, cluster, kMargin);

  EXPECT_THAT(rev_ellipse_mat, eigenmath::testing::IsApprox(ellipse_mat));

  std::vector<eigenmath::Vector2d> test_points_x{{
      {3.0 + kMargin, 0.0},
      {-3.0 - kMargin, 0.0},
  }};
  EXPECT_THAT(EvaluateEllipseDistances(means[0], ellipse_mat, test_points_x),
              testing::Each(testing::DoubleNear(1.0, 1e-3)));

  // Be less strict about the degenerate axis.
  std::vector<eigenmath::Vector2d> test_points_y{{
      {0.0, kMargin},
      {0.0, -kMargin},
  }};
  EXPECT_THAT(EvaluateEllipseDistances(means[0], ellipse_mat, test_points_y),
              testing::Each(testing::DoubleNear(1.0, 2e-2)));
}

TEST(ScaleAndExpandEllipseTest, TestFitCorners) {
  std::vector<eigenmath::Vector2d> cluster{{
      {0.0, 0.0},
      {1.0, 0.0},
      {2.0, 0.0},
      {-1.0, 0.0},
      {-2.0, 0.0},
      {0.0, 1.0},
      {0.0, 2.0},
      {0.0, -1.0},
      {0.0, -2.0},
      {2.0, 2.0},
      {-2.0, 2.0},
      {2.0, -2.0},
      {-2.0, -2.0},
  }};

  const auto& gmm = FitGaussianMixture(cluster, 1, 10, 0.0);

  auto assignments = Cluster(gmm, cluster);
  const auto& means = gmm.Means();
  const auto& covariances = gmm.Covariances();

  constexpr double kMargin = 0.2;

  eigenmath::Matrix2d ellipse_mat = ScaleAndExpandEllipse(
      means[0], covariances[0], assignments[0], cluster, kMargin);

  EXPECT_TRUE(ellipse_mat.allFinite());
  EXPECT_GT(ellipse_mat.determinant(), 0.0);

  std::vector<int> rev_assignments = assignments[0];
  std::reverse(rev_assignments.begin(), rev_assignments.end());
  eigenmath::Matrix2d rev_ellipse_mat = ScaleAndExpandEllipse(
      means[0], covariances[0], rev_assignments, cluster, kMargin);

  EXPECT_THAT(rev_ellipse_mat, eigenmath::testing::IsApprox(ellipse_mat));

  std::vector<eigenmath::Vector2d> test_points{{
      {2.0 + kMargin * M_SQRT1_2, 2.0 + kMargin * M_SQRT1_2},
      {-2.0 - kMargin * M_SQRT1_2, 2.0 + kMargin * M_SQRT1_2},
      {2.0 + kMargin * M_SQRT1_2, -2.0 - kMargin * M_SQRT1_2},
      {-2.0 - kMargin * M_SQRT1_2, -2.0 - kMargin * M_SQRT1_2},
  }};
  EXPECT_THAT(EvaluateEllipseDistances(means[0], ellipse_mat, test_points),
              testing::Each(testing::DoubleNear(1.0, 1e-3)));

  std::vector<eigenmath::Vector2d> test_points_flanks{{
      {2.0 + kMargin, 0.0},
      {-2.0 - kMargin, 0.0},
      {0.0, 2.0 + kMargin},
      {0.0, -2.0 - kMargin},
  }};
  EXPECT_THAT(
      EvaluateEllipseDistances(means[0], ellipse_mat, test_points_flanks),
      testing::Each(testing::Lt(1.0)));
}

TEST(ScaleAndExpandEllipseTest, TestFitGeneralCase) {
  const eigenmath::Vector2d mean{-0.112679, -0.0108533};
  const eigenmath::Matrix2d covariance =
      (eigenmath::Matrix2d() << 0.0175166, 0.00541297, 0.00541297, 0.00567003)
          .finished();
  const std::vector<int> assignments = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
  };
  const std::vector<eigenmath::Vector2d> points = {
      {-0.4, -0.2}, {-0.3, -0.2}, {-0.2, -0.2}, {-0.4, -0.1}, {-0.3, -0.1},
      {-0.2, -0.1}, {-0.3, 0},    {0, 0.1},     {0.1, 0.1},   {0.1, 0.2},
      {-0.4, -0.2}, {0.19, 0.12}, {0.14, 0.2},
  };
  constexpr double kMargin = 0.1;

  eigenmath::Matrix2d ellipse_mat =
      ScaleAndExpandEllipse(mean, covariance, assignments, points, kMargin);

  EXPECT_GT(ellipse_mat.determinant(), 0.0);
  EXPECT_TRUE(ellipse_mat.allFinite());
  // Compare area to original algorithm which pessimistically scaled the
  // ellipse isometrically to fit the farthest single point only.
  EXPECT_LT(ellipse_mat.determinant(), 0.0179851);

  EXPECT_THAT(
      EvaluateEllipseDistances(mean, ellipse_mat,
                               {
                                   {0.1 + kMargin, 0.2 + kMargin},
                                   {-0.4 - kMargin, -0.2 - kMargin},
                               }),
      testing::Each(testing::AllOf(testing::Ge(1.0), testing::Le(1.1))));
}

}  // namespace
}  // namespace mobility::collision

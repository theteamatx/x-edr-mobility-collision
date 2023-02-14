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

#include "collision/grid_common.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "collision/occupancy_grid.h"
#include "eigenmath/distribution.h"
#include "eigenmath/matchers.h"
#include "eigenmath/sampling.h"
#include "eigenmath/types.h"
#include "genit/iterator_facade.h"
#include "genit/iterator_range.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mobility::collision {
namespace {

using ::eigenmath::testing::IsApprox;
using ::testing::Eq;
using ::testing::Not;
using ::testing::Pointwise;

std::string DumpCharGrid(const Grid<char> &grid,
                         const GridIndex *maybe_highlight) {
  std::ostringstream result;
  const GridRange &range = grid.Range();
  result << "  range " << range << "\n";
  if ((maybe_highlight != nullptr) && !range.Contains(*maybe_highlight)) {
    result << "  highlight " << maybe_highlight->transpose()
           << " out of range\n";
    maybe_highlight = nullptr;
  }
  if (maybe_highlight != nullptr) {
    result << "    ";
    for (int ix = range.lower.x(); ix < maybe_highlight->x(); ++ix) {
      result << " ";
    }
    result << "v\n";
  }

  for (GridIndex index{0, range.upper.y() - 1}; index.y() >= range.lower.y();
       --index.y()) {
    if ((maybe_highlight != nullptr) && index.y() == maybe_highlight->y()) {
      result << "   >";
    } else {
      result << "    ";
    }
    for (index.x() = range.lower.x(); index.x() < range.upper.x();
         ++index.x()) {
      result << grid.GetUnsafe(index);
    }
    if ((maybe_highlight != nullptr) && index.y() == maybe_highlight->y()) {
      result << "<";
    }
    result << "\n";
  }

  if (maybe_highlight != nullptr) {
    result << "    ";
    for (int ix = range.lower.x(); ix < maybe_highlight->x(); ++ix) {
      result << " ";
    }
    result << "^\n";
  }
  return result.str();
}

std::string DumpNonSpanningUnion(const std::vector<GridRange> &aas) {
  const GridRange range = GridRange::SpanningUnion(aas);
  std::ostringstream result;
  for (int i = range.lower.x(); i < range.upper.x(); ++i) {
    for (int j = range.lower.y(); j < range.upper.y(); ++j) {
      if (GridRange::Contains(aas, GridIndex{i, j})) {
        result << 'X';
      } else {
        result << '.';
      }
    }
    result << '\n';
  }
  return result.str();
}

Grid<char> CreateVennDiagram(const GridRange &aa, const GridRange &bb) {
  Grid<char> diagram;
  diagram.SetDefaultValue('?');
  diagram.Reshape(aa);
  if (aa.Empty() && bb.Empty()) {
    return diagram;
  }

  diagram.Reshape(GridRange::SpanningUnion(aa, bb));
  std::ostringstream os;
  for (GridIndex ii{0, diagram.Range().upper.y() - 1};
       ii.y() >= diagram.Range().lower.y(); --ii.y()) {
    for (ii.x() = diagram.Range().lower.x(); ii.x() < diagram.Range().upper.x();
         ++ii.x()) {
      if (aa.Contains(ii)) {
        if (bb.Contains(ii)) {
          diagram.Set(ii, 'x');
        } else {
          diagram.Set(ii, 'A');
        }
      } else {
        if (bb.Contains(ii)) {
          diagram.Set(ii, 'B');
        } else {
          diagram.Set(ii, '.');
        }
      }
    }
  }
  return diagram;
}

void EnumerateAllCombos(std::vector<GridRange> *aas,
                        std::vector<GridRange> *bbs) {
  // Double-thickness lines indicate the inclusive end of the bounding
  // box (i.e. lower.x() and lower.y()), and single-thickness lines
  // the exclusive ones (i.e. upper.x() and upper.z()). The set A is
  // in the center. The set B is constructed such that it covers all
  // cases of lower and upper bounds below, on, and above each of the
  // bounds of A.
  //
  // y^
  //  |  0  1  2  3  4  5  6  7 -->x
  // 17        #        |
  // 16        #        |
  // 15 -------#--------+------
  // 14        #AAAAAAAA|
  // 13        #AAAAAAAA|
  // 12 =======#========+======
  // 11        #        |
  // 10        #        |
  //
  // And we throw in empty sets for A and B to make sure we cover
  // those edge cases, too.
  *aas = {
      {{0, 0}, {-1, -1}},  // empty
      {{2, 12}, {5, 15}},
  };
  std::vector<int> lower_xs = {0, 2, 3, 5, 6};
  std::vector<int> upper_xs = {1, 2, 4, 5, 7};
  bbs->push_back(GridRange());  // empty
  for (int ii = 0; ii < lower_xs.size(); ++ii) {
    for (int jj = 0; jj < lower_xs.size(); ++jj) {
      const GridIndex lower(lower_xs[ii], lower_xs[jj] + 10);
      for (int kk = 0; kk < upper_xs.size(); ++kk) {
        for (int ll = 0; ll < upper_xs.size(); ++ll) {
          const GridIndex upper(upper_xs[kk], upper_xs[ll] + 10);
          const GridRange rr(lower, upper);
          if (!rr.Empty()) {
            bbs->emplace_back(rr);
          }
        }
      }
    }
  }
}

TEST(GridRange, GrowToInclude) {
  GridRange empty_range;
  empty_range.GrowToInclude(GridIndex{4, 2});
  EXPECT_TRUE(empty_range.Contains(GridIndex{4, 2}));
  EXPECT_EQ(empty_range.lower.x(), 4);
  EXPECT_EQ(empty_range.lower.y(), 2);
  EXPECT_EQ(empty_range.upper.x(), 5);
  EXPECT_EQ(empty_range.upper.y(), 3);

  GridRange nominal_range{GridIndex{2, 3}, GridIndex{6, 9}};
  nominal_range.GrowToInclude(GridIndex{-2, 2});
  EXPECT_TRUE(nominal_range.Contains(GridIndex{-2, 2}));
  EXPECT_EQ(nominal_range.lower.x(), -2);
  EXPECT_EQ(nominal_range.lower.y(), 2);
  EXPECT_EQ(nominal_range.upper.x(), 6);
  EXPECT_EQ(nominal_range.upper.y(), 9);
}

TEST(GridRange, GrowBy) {
  GridRange grown_range{GridIndex{2, 3}, GridIndex{6, 9}};
  grown_range.GrowBy(2);
  EXPECT_EQ(grown_range.lower.x(), 0);
  EXPECT_EQ(grown_range.lower.y(), 1);
  EXPECT_EQ(grown_range.upper.x(), 8);
  EXPECT_EQ(grown_range.upper.y(), 11);

  GridRange empty_range;
  empty_range.GrowBy(2);
  EXPECT_TRUE(empty_range.Empty());
}

TEST(GridRange, ShrinkBy) {
  GridRange shrunk_range{GridIndex{2, 3}, GridIndex{7, 9}};
  shrunk_range.ShrinkBy(2);
  EXPECT_EQ(shrunk_range.lower.x(), 4);
  EXPECT_EQ(shrunk_range.lower.y(), 5);
  EXPECT_EQ(shrunk_range.upper.x(), 5);
  EXPECT_EQ(shrunk_range.upper.y(), 7);

  GridRange too_small_range{GridIndex{2, 3}, GridIndex{4, 9}};
  too_small_range.ShrinkBy(2);
  EXPECT_TRUE(too_small_range.Empty());

  GridRange empty_range;
  empty_range.ShrinkBy(2);
  EXPECT_TRUE(empty_range.Empty());
}

TEST(GridRange, Complement) {
  GridRange nominal_range{GridIndex{2, 3}, GridIndex{7, 9}};
  GridRange::Quad nominal_comp = GridRange::Complement(nominal_range);
  for (const auto &comp : nominal_comp) {
    EXPECT_TRUE(GridRange::Intersect(nominal_range, comp).Empty());
  }

  GridRange empty_range;
  GridRange::Quad empty_comp = GridRange::Complement(empty_range);
  EXPECT_EQ(empty_comp.size(), 1);
  EXPECT_FALSE(empty_comp[0].Empty());
}

TEST(GridRange, Difference) {
  std::vector<GridRange> aas, bbs;
  EnumerateAllCombos(&aas, &bbs);

  // Verification happens by actually drawing out the sets A, B, and
  // A\B on a grid, and then checking that the difference operator
  // visits each cell of A\B exactly once and all the others exactly
  // never.
  for (const auto &aa : aas) {
    for (const auto &bb : bbs) {
      const Grid<char> expected = CreateVennDiagram(aa, bb);
      const GridRange::Quad a_without_b = GridRange::Difference(aa, bb);
      Grid<char> scratch = expected;
      for (const auto &subrange : a_without_b) {
        ASSERT_FALSE(subrange.Empty()) << "Empty subrange entry\n"
                                       << DumpCharGrid(expected, nullptr);
        for (const GridIndex &index : subrange) {
          char cc;
          ASSERT_TRUE(scratch.Get(index, &cc))
              << "Index " << index.transpose() << " out of expected range "
              << expected.Range() << "\n"
              << DumpCharGrid(expected, &index);
          ASSERT_NE('o', cc)
              << "Index " << index.transpose() << " visited at least twice\n"
              << DumpCharGrid(expected, &index);
          ASSERT_EQ('A', cc) << "Index " << index.transpose()
                             << " should not be in difference\n"
                             << DumpCharGrid(expected, &index);
          scratch.Set(index, 'o');
        }
      }
      for (const GridIndex &index : scratch.Range()) {
        char cc;
        ASSERT_TRUE(scratch.Get(index, &cc))
            << "Fixture error (failed to get) at index " << index.transpose()
            << "\nexpected\n"
            << DumpCharGrid(expected, &index) << "scratch\n"
            << DumpCharGrid(scratch, &index);
        ASSERT_NE('A', cc) << "Index " << index.transpose()
                           << " should have been in difference\nexpected\n"
                           << DumpCharGrid(expected, &index) << "scratch\n"
                           << DumpCharGrid(scratch, &index);
        ASSERT_NE('?', cc) << "Fixture error (failed to set) at index "
                           << index.transpose() << "\nexpected\n"
                           << DumpCharGrid(expected, &index) << "scratch\n"
                           << DumpCharGrid(scratch, &index);
      }
    }
  }
}

TEST(GridRange, NonSpanningUnion) {
  std::vector<GridRange> aas, bbs;
  EnumerateAllCombos(&aas, &bbs);

  // For verification, set counters to zero or one depending on
  // whether they are in either A or B. Then loop over the
  // non-spanning union, decrement each counter. At the end, they
  // should all be zero.
  for (const auto &aa : aas) {
    for (const auto &bb : bbs) {
      Grid<int> counters(GridFrame(), GridRange::SpanningUnion(aa, bb));
      counters.Fill(0);
      for (const GridIndex &index : aa) {
        EXPECT_TRUE(counters.Set(index, 1));
      }
      for (const GridIndex &index : bb) {
        EXPECT_TRUE(counters.Set(index, 1));
      }

      const GridRange::Quad non_spanning_union =
          GridRange::NonSpanningUnion(aa, bb);
      for (const auto &range : non_spanning_union) {
        for (const GridIndex &index : range) {
          int counter;
          EXPECT_TRUE(counters.Get(index, &counter));
          EXPECT_TRUE(counters.Set(index, counter - 1));
        }
      }

      for (const GridIndex &index : counters.Range()) {
        EXPECT_EQ(0, counters.GetUnsafe(index));
      }
    }
  }
}

TEST(GridRange, Intersect) {
  std::vector<GridRange> aas = {
      {{0, 0}, {-1, -1}},  // empty
      {{2, 12}, {5, 15}}, {{0, 14}, {5, 15}}, {{4, 5}, {5, 13}},
      {{4, 5}, {5, 13}},  {{4, 12}, {5, 20}}, {{0, 0}, {-1, -1}},  // empty
      {{1, 12}, {3, 18}}, {{3, 13}, {4, 14}}, {{3, 16}, {8, 18}},
  };

  const GridRange bb_1{{-10, -10}, {-5, -5}};
  const std::vector<GridRange> inter_1 = GridRange::Intersect(bb_1, aas);
  EXPECT_TRUE(inter_1.empty());

  const GridRange bb_2{{-1, -1}, {5, 8}};
  const std::vector<GridRange> inter_2 = GridRange::Intersect(bb_2, aas);
  ASSERT_EQ(inter_2.size(), 2);
  EXPECT_EQ(inter_2[0].lower.x(), 4);
  EXPECT_EQ(inter_2[0].lower.y(), 5);
  EXPECT_EQ(inter_2[0].upper.x(), 5);
  EXPECT_EQ(inter_2[0].upper.y(), 8);
  EXPECT_EQ(inter_2[1].lower.x(), 4);
  EXPECT_EQ(inter_2[1].lower.y(), 5);
  EXPECT_EQ(inter_2[1].upper.x(), 5);
  EXPECT_EQ(inter_2[1].upper.y(), 8);
}

TEST(GridRange, RemoveSelfIntersections) {
  std::vector<GridRange> aas = {
      {{0, 0}, {-1, -1}},  // empty
      {{2, 12}, {5, 15}}, {{0, 14}, {5, 15}}, {{4, 5}, {5, 13}},
      {{4, 5}, {5, 13}},  {{4, 12}, {5, 20}}, {{0, 0}, {-1, -1}},  // empty
      {{1, 12}, {3, 18}}, {{3, 13}, {4, 14}}, {{3, 16}, {8, 18}},
  };

  const GridRange original_range = GridRange::SpanningUnion(aas);
  const std::string original_diagram = DumpNonSpanningUnion(aas);

  GridRange::RemoveSelfIntersections(&aas);

  const GridRange resulting_range = GridRange::SpanningUnion(aas);
  const std::string resulting_diagram = DumpNonSpanningUnion(aas);

  EXPECT_EQ(original_range, resulting_range);
  EXPECT_EQ(original_diagram, resulting_diagram);

  for (int i = 0; i < aas.size(); ++i) {
    EXPECT_FALSE(aas[i].Empty());
    for (int j = 0; j < aas.size(); ++j) {
      if (i == j) {
        continue;
      }
      const GridRange intersection = GridRange::Intersect(aas[i], aas[j]);
      EXPECT_TRUE(intersection.Empty());
    }
  }
}

TEST(GridRange, IterationEmptyRanges) {
  GridRange empty{{0, 0}, {0, 0}};
  EXPECT_EQ(empty.begin(), empty.end());

  empty = GridRange{{0, 0}, {-2, 4}};
  EXPECT_EQ(empty.begin(), empty.end());

  empty = GridRange{{0, 0}, {2, -4}};
  EXPECT_EQ(empty.begin(), empty.end());
}

TEST(GridRange, Iteration) {
  GridRange range{{0, 0}, {2, 3}};
  const auto points = genit::CopyRange<std::vector<GridIndex>>(range);
  const GridIndex expected[] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}};
  using testing::Eq;
  using testing::Pointwise;
  EXPECT_THAT(points, Pointwise(Eq(), expected));

  std::vector<GridIndex> reverse_points;
  std::reverse_copy(range.begin(), range.end(),
                    std::back_inserter(reverse_points));
  std::reverse(reverse_points.begin(), reverse_points.end());
  EXPECT_THAT(reverse_points, Pointwise(Eq(), expected));
}

// Test iteration speed over various line lengths.
void BM_GridRangeIteration(benchmark::State &state) {
  const int length = state.range(0);
  const GridIndex from{-50, 3};
  const GridIndex to = from + GridIndex{length, 0.3 * length};
  for (auto _ : state) {
    const GridRange rectangle{from, to};
    for (const GridIndex &index : rectangle) {
      benchmark::DoNotOptimize(index);
    }
  }
}
BENCHMARK(BM_GridRangeIteration)->Arg(0)->Arg(1)->Arg(10)->Arg(100)->Arg(1000);

TEST(GridFrame, GridToFrameToGrid) {
  const double kGridResolution = 0.1;
  const GridFrame src_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI),
                            kGridResolution);

  EXPECT_THAT(src_frame.GridToFrame({4, 5}),
              IsApprox(eigenmath::Vector2d{0.6, 1.5}));
  EXPECT_THAT(src_frame.FrameToGrid({0.6, 1.5}), Eq(GridIndex{4, 5}));

  eigenmath::TestGenerator rng_gen{eigenmath::kGeneratorTestSeed};
  eigenmath::UniformDistributionVector2d vec_dist;

  for (int i = 0; i < 20; ++i) {
    const eigenmath::Vector2d v_orig = vec_dist(rng_gen);
    const eigenmath::Vector2d v_new =
        src_frame.GridToFrame(src_frame.FrameToGrid(v_orig));
    EXPECT_NEAR(v_new.x(), v_orig.x(), kGridResolution / 2.0);
    EXPECT_NEAR(v_new.y(), v_orig.y(), kGridResolution / 2.0);
  }
}

TEST(GridFrame, FrameToGridRange) {
  const double kGridResolution = 0.1;
  const GridFrame src_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI),
                            kGridResolution);

  auto expect_good_range = [](const GridRange &r) {
    EXPECT_TRUE(r.Contains(GridIndex{4, 5}));
    EXPECT_EQ(r.XSpan(), 2);
    EXPECT_EQ(r.YSpan(), 2);
  };

  expect_good_range(src_frame.FrameToGridRange({0.6, 1.5}));
  expect_good_range(src_frame.FrameToGridRange({0.59, 1.5}));
  expect_good_range(src_frame.FrameToGridRange({0.6, 1.49}));
  expect_good_range(src_frame.FrameToGridRange({0.61, 1.5}));
  expect_good_range(src_frame.FrameToGridRange({0.6, 1.51}));

  expect_good_range(src_frame.FrameToGridRange({0.6, 1.45000000001}));
  expect_good_range(src_frame.FrameToGridRange({0.6, 1.54999999999}));
  expect_good_range(src_frame.FrameToGridRange({0.55000000001, 1.5}));
  expect_good_range(src_frame.FrameToGridRange({0.64999999999, 1.5}));

  eigenmath::TestGenerator rng_gen{eigenmath::kGeneratorTestSeed};
  eigenmath::UniformDistributionVector2d vec_dist;

  for (int i = 0; i < 20; ++i) {
    const eigenmath::Vector2d v_orig = vec_dist(rng_gen);
    const GridRange v_range = src_frame.FrameToGridRange(v_orig);
    v_range.ForEachGridCoord([&](const GridIndex &index) {
      const eigenmath::Vector2d v_new = src_frame.GridToFrame(index);
      EXPECT_NEAR(v_new.x(), v_orig.x(), kGridResolution);
      EXPECT_NEAR(v_new.y(), v_orig.y(), kGridResolution);
    });
  }
}

TEST(GridFrame, GridToGrid) {
  const GridFrame src_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI), 0.1);

  const GridFrame down_sampled_frame(
      "foo", eigenmath::Pose2d({0.8, 1.6}, M_PI / 2.0), 0.2);
  EXPECT_THAT(GridFrame::GridToGrid(src_frame, {4, 5}, down_sampled_frame),
              Eq(GridIndex{-1, 1}));
  EXPECT_THAT(GridFrame::GridToGrid(src_frame, {4, 6}, down_sampled_frame),
              Eq(GridIndex{-1, 1}));
  EXPECT_THAT(GridFrame::GridToGrid(src_frame, {5, 7}, down_sampled_frame),
              Eq(GridIndex{-2, 2}));

  const GridFrame up_sampled_frame(
      "foo", eigenmath::Pose2d({1.05, 2.05}, M_PI / 2.0), 0.05);
  EXPECT_THAT(GridFrame::GridToGrid(src_frame, {4, 5}, up_sampled_frame),
              Eq(GridIndex{-11, 9}));
  EXPECT_THAT(GridFrame::GridToGrid(src_frame, {4, 6}, up_sampled_frame),
              Eq(GridIndex{-13, 9}));
  EXPECT_THAT(GridFrame::GridToGrid(src_frame, {5, 7}, up_sampled_frame),
              Eq(GridIndex{-15, 11}));
}

TEST(GridFrame, GridToGridFunctor) {
  const GridFrame src_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI), 0.1);

  const GridFrame down_sampled_frame(
      "foo", eigenmath::Pose2d({0.8, 1.6}, M_PI / 2.0), 0.2);
  const auto to_down_sampled =
      GridFrame::GridToGrid(src_frame, down_sampled_frame);
  EXPECT_THAT(to_down_sampled({4, 5}), Eq(GridIndex{-1, 1}));
  EXPECT_THAT(to_down_sampled({4, 6}), Eq(GridIndex{-1, 1}));
  EXPECT_THAT(to_down_sampled({5, 7}), Eq(GridIndex{-2, 2}));

  const GridFrame up_sampled_frame(
      "foo", eigenmath::Pose2d({1.05, 2.05}, M_PI / 2.0), 0.05);
  const auto to_up_sampled = GridFrame::GridToGrid(src_frame, up_sampled_frame);
  const GridRange test_range({-3, -3}, {4, 4});
  for (const GridIndex &index : test_range) {
    EXPECT_THAT(to_up_sampled(index),
                Eq(GridFrame::GridToGrid(src_frame, index, up_sampled_frame)))
        << "at index " << index;
  }
}

TEST(GridFrame, GridToGridInclusive) {
  const GridFrame src_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI), 0.1);

  const GridFrame down_sampled_frame(
      "foo", eigenmath::Pose2d({0.8, 1.6}, M_PI / 2.0), 0.2);
  EXPECT_THAT(
      GridFrame::GridToGridInclusive(src_frame, {4, 5}, down_sampled_frame),
      Eq(GridRange{{-2, 0}, {1, 3}}));
  EXPECT_THAT(
      GridFrame::GridToGridInclusive(src_frame, {4, 6}, down_sampled_frame),
      Eq(GridRange{{-2, 0}, {1, 3}}));
  EXPECT_THAT(
      GridFrame::GridToGridInclusive(src_frame, {5, 7}, down_sampled_frame),
      Eq(GridRange{{-3, 1}, {0, 4}}));

  const GridFrame up_sampled_frame(
      "foo", eigenmath::Pose2d({1.05, 2.05}, M_PI / 2.0), 0.05);
  EXPECT_THAT(
      GridFrame::GridToGridInclusive(src_frame, {4, 5}, up_sampled_frame),
      Eq(GridRange{{-13, 7}, {-8, 12}}));
  EXPECT_THAT(
      GridFrame::GridToGridInclusive(src_frame, {4, 6}, up_sampled_frame),
      Eq(GridRange{{-15, 7}, {-10, 12}}));
  EXPECT_THAT(
      GridFrame::GridToGridInclusive(src_frame, {5, 7}, up_sampled_frame),
      Eq(GridRange{{-17, 9}, {-12, 14}}));
}

TEST(GridFrame, GridToGridInclusiveCoverage) {
  const GridFrame src_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI / 2.5),
                            0.1);
  GridFrame up_sampled_frame("foo", eigenmath::Pose2d({1.05, 2.05}, M_PI / 4.0),
                             0.05);

  // Overlapping and coverage test for inclusive up-sampling.
  for (double resolution = 0.02; resolution < 0.1; resolution += 0.005434) {
    up_sampled_frame.resolution = resolution;
    const GridRange src_range = GridRange::OriginTo({11, 11});
    const GridIndex src_corners[] = {
        src_range.lower,
        {src_range.upper.x() - 1, src_range.lower.y()},
        {src_range.lower.x(), src_range.upper.y() - 1},
        src_range.upper - GridIndex::Constant(1)};
    GridRange src_corners_range;
    for (const GridIndex &src_corner : src_corners) {
      src_corners_range.SpanningUnion(GridFrame::GridToGridInclusive(
          src_frame, src_corner, up_sampled_frame));
    }
    Grid<int> counting_grid(up_sampled_frame, src_corners_range, 0);
    std::stringstream frame_info;
    frame_info << "src_frame = " << src_frame
               << " up_sampled_frame = " << up_sampled_frame
               << " counting_grid.Range() = " << counting_grid.Range();
    const std::string frame_info_str = frame_info.str();
    src_range.ForEachGridCoord([&](const GridIndex &src_index) {
      const GridRange dst_range = GridFrame::GridToGridInclusive(
          src_frame, src_index, up_sampled_frame);
      // Checks that every src_index in src_range yields a dst_range inside
      // the transformed bounds of src_range.
      EXPECT_FALSE(dst_range.Empty())
          << frame_info_str << " src_index = " << src_index.transpose()
          << " dst_range = " << dst_range;
      EXPECT_TRUE(counting_grid.Range().Contains(dst_range))
          << frame_info_str << " src_index = " << src_index.transpose()
          << " dst_range = " << dst_range;
      GridRange::Intersect(dst_range, counting_grid.Range())
          .ForEachGridCoord([&](const GridIndex &dst_index) {
            counting_grid.SetUnsafe(dst_index,
                                    counting_grid.GetUnsafe(dst_index) + 1);
            // Theoretically, the maximum overlap is 9 (e.g., src cell inscribed
            // in a 3x3 patch) plus 4 adjacent cells (diamond shape).
            EXPECT_LE(counting_grid.GetUnsafe(dst_index), 13)
                << frame_info_str << " src_index = " << src_index.transpose()
                << " dst_index = " << dst_index.transpose();
            const GridIndex remapped_src_index =
                GridFrame::GridToGrid(up_sampled_frame, dst_index, src_frame);
            EXPECT_LE(
                (remapped_src_index - src_index).lpNorm<Eigen::Infinity>(), 2)
                << frame_info_str << " src_index = " << src_index.transpose()
                << " dst_index = " << dst_index.transpose()
                << " remapped_src_index = " << remapped_src_index.transpose();
          });
      // Checks that there are no points outside of dst_range that map to
      // the src_index point.
      GridRange::Intersect(GridRange::GrowBy(dst_range, 5),
                           counting_grid.Range())
          .ForEachGridCoord([&](const GridIndex &dst_index) {
            if (dst_range.Contains(dst_index)) {
              return;
            }
            EXPECT_THAT(
                GridFrame::GridToGrid(up_sampled_frame, dst_index, src_frame),
                Not(Eq(src_index)));
          });
    });
    // Check that we have covered every cell of the counting_grid.
    src_corners_range.ForEachGridCoord([&](const GridIndex &dst_index) {
      // Only look at the interior region.
      if (!src_range.Contains(
              src_frame.FrameToGrid(up_sampled_frame.GridToFrame(dst_index)))) {
        return;
      }
      EXPECT_GT(counting_grid.GetUnsafe(dst_index), 0)
          << frame_info_str << " dst_index = " << dst_index.transpose();
    });
  }
}

TEST(SmallestTargetRangeCoveringSourceRange,
     IdentityTransformYieldsIdenticalRange) {
  const GridRange original = GridRange::OriginTo(7, 13);
  const GridFrame::GridToGridFunctor identity;
  const GridRange covering =
      SmallestTargetRangeCoveringSourceRange(original, identity);
  EXPECT_THAT(covering, original);
}

TEST(SmallestTargetRangeCoveringSourceRange,
     CoveringRangeIncludesCornerPointsButNoMore) {
  const GridRange original = GridRange::OriginTo(7, 13);
  Eigen::Affine2d transform = Eigen::Affine2d::Identity();
  transform.linear() << 1, 1, -1, 1;
  transform.translation() << 5.4, 7.2;
  const GridFrame::GridToGridFunctor grid_functor(transform);
  const GridRange covering =
      SmallestTargetRangeCoveringSourceRange(original, grid_functor);

  // Transformed indices are:
  // (5.4, 7.2)  (17.4, 19.2)  (23.4, 13.2)  (11.4, 1.2)
  EXPECT_THAT(covering.lower.x(), 5);
  EXPECT_THAT(covering.lower.y(), 1);
  EXPECT_THAT(covering.upper.x() - 1, 23);
  EXPECT_THAT(covering.upper.y() - 1, 19);
}

TEST(FullTargetRangeProjectingOntoSourceRange,
     IdentityTransformYieldsRangeContainingCellBoundary) {
  const GridRange original = GridRange::OriginTo(7, 13);
  const GridFrame::GridToGridFunctor identity;
  const GridRange covering =
      FullTargetRangeProjectingOntoSourceRange(original, identity);
  EXPECT_TRUE(covering.Contains(GridIndex(-0.5, -0.5)));
  EXPECT_TRUE(covering.Contains(GridIndex(-0.5, 13.5)));
  EXPECT_TRUE(covering.Contains(GridIndex(7.5, -0.5)));
  EXPECT_TRUE(covering.Contains(GridIndex(7.4, 13.5)));
}

TEST(FullTargetRangeProjectingOntoSourceRange,
     CoveringRangeContainsCellBoundary) {
  const GridRange original = GridRange::OriginTo(7, 13);
  Eigen::Affine2d transform = Eigen::Affine2d::Identity();
  transform.linear() << 1, 1, -1, 1;
  transform.translation() << 5.5, 7.2;
  const GridFrame::GridToGridFunctor grid_functor(transform);
  const GridRange covering =
      FullTargetRangeProjectingOntoSourceRange(original, grid_functor);
  const GridIndex corners[] = {
      grid_functor(GridIndex(-0.5, -0.5)), grid_functor(GridIndex(-0.5, 13.5)),
      grid_functor(GridIndex(7.5, -0.5)), grid_functor(GridIndex(7.4, 13.5))};
  EXPECT_TRUE(covering.Contains(corners[0]));
  EXPECT_TRUE(covering.Contains(corners[1]));
  EXPECT_TRUE(covering.Contains(corners[2]));
  EXPECT_TRUE(covering.Contains(corners[3]));
}

TEST(GridLine, SingleCell) {
  const GridIndex a{3, 4};
  const GridIndex points[] = {a};

  const GridLine line{a, a};
  const auto line_points = genit::CopyRange<std::vector<GridIndex>>(line);
  EXPECT_THAT(line_points, Pointwise(Eq(), points));
}

TEST(GridLine, HorizontalLine) {
  const GridIndex a{3, -2};
  const GridIndex b{3, 5};
  const GridIndex points[] = {{3, -2}, {3, -1}, {3, 0}, {3, 1},
                              {3, 2},  {3, 3},  {3, 4}, {3, 5}};

  const GridLine forward{a, b};
  const auto forward_points = genit::CopyRange<std::vector<GridIndex>>(forward);
  EXPECT_THAT(forward_points, Pointwise(Eq(), points));

  const GridLine backward{b, a};
  auto backward_points = genit::CopyRange<std::vector<GridIndex>>(backward);
  std::reverse(backward_points.begin(), backward_points.end());
  EXPECT_THAT(backward_points, Pointwise(Eq(), points));
}

TEST(GridLine, VerticalLine) {
  const GridIndex a{-2, -1};
  const GridIndex b{5, -1};
  const GridIndex points[] = {{-2, -1}, {-1, -1}, {0, -1}, {1, -1},
                              {2, -1},  {3, -1},  {4, -1}, {5, -1}};

  const GridLine forward{a, b};
  const auto forward_points = genit::CopyRange<std::vector<GridIndex>>(forward);
  EXPECT_THAT(forward_points, Pointwise(Eq(), points));

  const GridLine backward{b, a};
  auto backward_points = genit::CopyRange<std::vector<GridIndex>>(backward);
  std::reverse(backward_points.begin(), backward_points.end());
  EXPECT_THAT(backward_points, Pointwise(Eq(), points));
}

TEST(GridLine, DiagonalLine) {
  const GridIndex a{-2, 1};
  const GridIndex b{3, 6};
  const GridIndex points[] = {{-2, 1}, {-1, 2}, {0, 3}, {1, 4}, {2, 5}, {3, 6}};

  const GridLine forward{a, b};
  const auto forward_points = genit::CopyRange<std::vector<GridIndex>>(forward);
  EXPECT_THAT(forward_points, Pointwise(Eq(), points));

  const GridLine backward{b, a};
  auto backward_points = genit::CopyRange<std::vector<GridIndex>>(backward);
  std::reverse(backward_points.begin(), backward_points.end());
  EXPECT_THAT(backward_points, Pointwise(Eq(), points));
}

TEST(GridLine, DiagonalLineLengthen) {
  const GridIndex a{-1, 2};
  const GridIndex b{2, 5};
  const GridIndex points[] = {{-2, 1}, {-1, 2}, {0, 3}, {1, 4}, {2, 5}, {3, 6}};

  GridLine forward{a, b};
  forward.LengthenOnBothEnds(1);
  const auto forward_points = genit::CopyRange<std::vector<GridIndex>>(forward);
  EXPECT_THAT(forward_points, Pointwise(Eq(), points));

  GridLine backward{b, a};
  backward.LengthenOnBothEnds(1);
  auto backward_points = genit::CopyRange<std::vector<GridIndex>>(backward);
  std::reverse(backward_points.begin(), backward_points.end());
  EXPECT_THAT(backward_points, Pointwise(Eq(), points));
}

TEST(GridLine, SlantedLine) {
  const GridIndex a{3, 0};
  const GridIndex b{-2, 2};
  const GridIndex points[] = {{3, 0}, {2, 0}, {1, 1}, {0, 1}, {-1, 2}, {-2, 2}};

  const GridLine forward{a, b};
  const auto forward_points = genit::CopyRange<std::vector<GridIndex>>(forward);
  EXPECT_THAT(forward_points, Pointwise(Eq(), points));

  const GridLine backward{b, a};
  auto backward_points = genit::CopyRange<std::vector<GridIndex>>(backward);
  std::reverse(backward_points.begin(), backward_points.end());
  EXPECT_THAT(backward_points, Pointwise(Eq(), points));
}

TEST(GridLine, JumpingOnLine) {
  const GridIndex a{0, 0};
  const GridIndex b{10, 10};
  const GridLine line{a, b};
  auto it = line.begin();
  it += 4;
  EXPECT_THAT(it - line.begin(), 4);
  EXPECT_THAT(*it, Eq(GridIndex{4, 4}));
  it += (line.end() - it);
  EXPECT_THAT(it, Eq(line.end()));
}

// Test iteration speed over various line lengths.
void BM_GridLineRangeIteration(benchmark::State &state) {
  const int length = state.range(0);
  const GridIndex from{-50, 3};
  const GridIndex to = from + GridIndex{length, 0.3 * length};
  for (auto _ : state) {
    const GridLine line{from, to};
    for (const GridIndex index : line) {
      benchmark::DoNotOptimize(index);
    }
  }
}

BENCHMARK(BM_GridLineRangeIteration)
    ->Arg(0)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000);

void BM_StrictModulus(benchmark::State &state) {
  const auto index_range = genit::IteratorRange(genit::IndexIterator(-1000),
                                                genit::IndexIterator(1000));
  constexpr int kLength = 100;
  for (auto _ : state) {
    for (const int index : index_range) {
      benchmark::DoNotOptimize(StrictModulus(index, kLength));
    }
  }
}
BENCHMARK(BM_StrictModulus);

void BM_StrictModulusAlternative(benchmark::State &state) {
  const auto index_range = genit::IteratorRange(genit::IndexIterator(-1000),
                                                genit::IndexIterator(1000));
  constexpr int kLength = 100;
  for (auto _ : state) {
    for (const int index : index_range) {
      benchmark::DoNotOptimize((index % kLength + kLength) % kLength);
    }
  }
}
BENCHMARK(BM_StrictModulusAlternative);

void BM_GridToGridPerIndex(benchmark::State &state) {
  const GridFrame src_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI), 0.1);
  const GridFrame down_sampled_frame(
      "foo", eigenmath::Pose2d({0.8, 1.6}, M_PI / 2.0), 0.2);
  const GridRange sample_range = GridRange::OriginTo({103, 107});
  for (auto _ : state) {
    for (const GridIndex &index : sample_range) {
      benchmark::DoNotOptimize(
          GridFrame::GridToGrid(src_frame, index, down_sampled_frame));
    }
  }
}
BENCHMARK(BM_GridToGridPerIndex);

void BM_GridToGridViaFunctor(benchmark::State &state) {
  const GridFrame src_frame("foo", eigenmath::Pose2d({1.0, 2.0}, M_PI), 0.1);
  const GridFrame down_sampled_frame(
      "foo", eigenmath::Pose2d({0.8, 1.6}, M_PI / 2.0), 0.2);
  const GridRange sample_range = GridRange::OriginTo({103, 107});
  const auto to_down_sampled =
      GridFrame::GridToGrid(src_frame, down_sampled_frame);
  for (auto _ : state) {
    for (const GridIndex &index : sample_range) {
      benchmark::DoNotOptimize(to_down_sampled(index));
    }
  }
}
BENCHMARK(BM_GridToGridViaFunctor);

}  // namespace
}  // namespace mobility::collision

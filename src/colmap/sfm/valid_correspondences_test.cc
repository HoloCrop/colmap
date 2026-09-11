#include "colmap/sfm/valid_correspondences.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <array>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ValidCorrespondences, VisitsRetainedMatchesExactlyOnce) {
  auto database = Database::Open(CreateTestDir() / "database.db");
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 2;
  options.num_frames_per_rig = 3;
  options.num_points3D = 20;
  options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(options, &reconstruction, database.get());
  const auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());
  const auto& graph = *cache->CorrespondenceGraph();
  PoseGraph poses;
  poses.Load(graph);
  ASSERT_GT(poses.NumEdges(), 1);
  poses.SetInvalidEdge(poses.Edges().begin()->first);

  auto image_ids = reconstruction.RegImageIds();
  std::sort(image_ids.begin(), image_ids.end());
  FlatHashMap<image_t, size_t> indices;
  for (size_t index = 0; index < image_ids.size(); ++index) {
    indices.emplace(image_ids[index], index);
  }
  using Match = std::array<size_t, 4>;
  std::vector<Match> expected;
  FeatureMatches matches;
  for (const auto& pair : poses.ValidEdges()) {
    const auto [first, second] = PairIdToImagePair(pair.first);
    graph.ExtractMatchesBetweenImages(first, second, matches);
    for (const auto& match : matches) {
      expected.push_back({indices.at(first),
                          match.point2D_idx1,
                          indices.at(second),
                          match.point2D_idx2});
    }
  }
  std::vector<Match> actual;
  ForEachValidCorrespondence(
      graph,
      poses,
      reconstruction,
      image_ids,
      indices,
      [&](size_t first, point2D_t point1, size_t second, point2D_t point2) {
        actual.push_back({first, point1, second, point2});
      });
  ASSERT_FALSE(expected.empty());
  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(actual, expected);
}

}  // namespace
}  // namespace colmap

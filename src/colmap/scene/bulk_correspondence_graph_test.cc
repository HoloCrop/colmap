#include "colmap/scene/bulk_correspondence_graph.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

void ExpectEquivalent(const CorrespondenceGraph& first,
                      const CorrespondenceGraph& second,
                      const NodeHashMap<image_t, Image>& images) {
  EXPECT_EQ(first.NumImages(), second.NumImages());
  EXPECT_EQ(first.NumMatchesBetweenAllImages(),
            second.NumMatchesBetweenAllImages());
  for (const auto& [id, image] : images) {
    EXPECT_EQ(first.NumObservationsForImage(id),
              second.NumObservationsForImage(id));
    EXPECT_EQ(first.NumCorrespondencesForImage(id),
              second.NumCorrespondencesForImage(id));
    for (point2D_t point = 0; point < image.NumPoints2D(); ++point) {
      const auto left = first.FindCorrespondences(id, point);
      const auto right = second.FindCorrespondences(id, point);
      std::vector<std::pair<image_t, point2D_t>> left_matches, right_matches;
      for (auto corr = left.beg; corr != left.end; ++corr)
        left_matches.emplace_back(corr->image_id, corr->point2D_idx);
      for (auto corr = right.beg; corr != right.end; ++corr)
        right_matches.emplace_back(corr->image_id, corr->point2D_idx);
      std::sort(left_matches.begin(), left_matches.end());
      std::sort(right_matches.begin(), right_matches.end());
      EXPECT_EQ(left_matches, right_matches);
    }
  }
  for (const image_pair_t pair : first.ImagePairs()) {
    const auto [id1, id2] = PairIdToImagePair(pair);
    // Reverse extraction exercises geometry inversion as well as match indices.
    const auto left = first.ExtractTwoViewGeometry(id2, id1, true);
    const auto right = second.ExtractTwoViewGeometry(id2, id1, true);
    ASSERT_EQ(left.inlier_matches.size(), right.inlier_matches.size());
    for (size_t index = 0; index < left.inlier_matches.size(); ++index) {
      EXPECT_EQ(left.inlier_matches[index].point2D_idx1,
                right.inlier_matches[index].point2D_idx1);
      EXPECT_EQ(left.inlier_matches[index].point2D_idx2,
                right.inlier_matches[index].point2D_idx2);
    }
    EXPECT_EQ(left.config, right.config);
    ASSERT_EQ(left.F.has_value(), right.F.has_value());
    if (left.F.has_value()) EXPECT_TRUE(left.F->isApprox(*right.F));
    ASSERT_EQ(left.cam2_from_cam1.has_value(),
              right.cam2_from_cam1.has_value());
    if (left.cam2_from_cam1.has_value()) {
      EXPECT_TRUE(left.cam2_from_cam1->translation().isApprox(
          right.cam2_from_cam1->translation()));
    }
  }
}

TEST(BulkCorrespondenceGraph, PreservesValidationGeometryAndSubsets) {
  NodeHashMap<image_t, Image> images;
  for (image_t id : {2, 100, 501, 10000}) {
    images[id].SetImageId(id);
    images[id].SetPoints2D(
        std::vector<Eigen::Vector2d>(4, Eigen::Vector2d::Zero()));
  }
  TwoViewGeometry geometry;
  geometry.config = TwoViewGeometry::CALIBRATED;
  geometry.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 2, 3));
  geometry.F = Eigen::Matrix3d::Identity();
  geometry.inlier_matches = {{0, 1}, {0, 1}, {2, 3}, {8, 0}, {1, 9}, {0, 2}};
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> pairs;
  pairs.emplace_back(ImagePairToPairId(2, 100), geometry);
  geometry.inlier_matches = {{0, 1}, {2, 3}};
  pairs.emplace_back(ImagePairToPairId(2, 501), geometry);
  geometry.inlier_matches.clear();
  pairs.emplace_back(ImagePairToPairId(100, 501), geometry);
  pairs.emplace_back(ImagePairToPairId(2, 2), geometry);

  CorrespondenceGraph reference;
  for (const auto& [id, image] : images)
    reference.AddImage(id, image.NumPoints2D());
  for (const auto& [pair, value] : pairs) {
    const auto [first, second] = PairIdToImagePair(pair);
    reference.AddTwoViewGeometry(first, second, value);
  }
  reference.Finalize();
  const auto bulk = BulkCorrespondenceGraph::FromPairs(images, pairs);
  ExpectEquivalent(reference, *bulk, images);

  images.erase(501);
  CorrespondenceGraph reference_subset;
  for (const auto& [id, image] : images)
    reference_subset.AddImage(id, image.NumPoints2D());
  reference_subset.AddTwoViewGeometry(
      2, 100, reference.ExtractTwoViewGeometry(2, 100, true));
  reference_subset.Finalize();
  const auto subset = BulkCorrespondenceGraph::Subset(*bulk, images);
  ExpectEquivalent(reference_subset, *subset, images);
}

}  // namespace
}  // namespace colmap

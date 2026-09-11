#include "colmap/scene/bulk_correspondence_graph.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <numeric>

namespace colmap {

std::shared_ptr<CorrespondenceGraph> BulkCorrespondenceGraph::FromPairs(
    const NodeHashMap<image_t, Image>& images,
    std::vector<std::pair<image_pair_t, TwoViewGeometry>>& pairs) {
  auto graph = std::make_shared<CorrespondenceGraph>();
  for (const auto& [id, image] : images) {
    graph->images_[id].flat_corr_begs.resize(image.NumPoints2D() + 1, 0);
  }

  FlatHashSet<uint64_t> seen;
  for (auto& [pair_id, geometry] : pairs) {
    const auto [first, second] = PairIdToImagePair(pair_id);
    if (first == second) {
      LOG(WARNING) << "Cannot use self-matches for image_id=" << first;
      FeatureMatches().swap(geometry.inlier_matches);
      continue;
    }
    auto& first_image = graph->images_.at(first);
    auto& second_image = graph->images_.at(second);
    seen.clear();
    seen.reserve(geometry.inlier_matches.size());
    auto& matches = geometry.inlier_matches;
    matches.erase(
        std::remove_if(
            matches.begin(),
            matches.end(),
            [&](const FeatureMatch& match) {
              const bool valid_first =
                  match.point2D_idx1 < first_image.flat_corr_begs.size() - 1;
              const bool valid_second =
                  match.point2D_idx2 < second_image.flat_corr_begs.size() - 1;
              if (!valid_first || !valid_second) {
                LOG(WARNING)
                    << "Ignoring invalid correspondence for image pair "
                    << pair_id;
                return true;
              }
              const uint64_t key =
                  (uint64_t{match.point2D_idx1} << 32) | match.point2D_idx2;
              if (!seen.insert(key).second) {
                LOG(WARNING)
                    << "Ignoring duplicate correspondence for image pair "
                    << pair_id;
                return true;
              }
              ++first_image.flat_corr_begs[match.point2D_idx1 + 1];
              ++second_image.flat_corr_begs[match.point2D_idx2 + 1];
              return false;
            }),
        matches.end());
    const auto [entry, inserted] = graph->image_pairs_.try_emplace(pair_id);
    THROW_CHECK(inserted)
        << "Two view geometry for image pair was already added";
    entry->second.num_matches = matches.size();
  }

  for (auto& [id, image] : graph->images_) {
    auto& offsets = image.flat_corr_begs;
    image.num_observations =
        std::count_if(offsets.begin(), offsets.end(), [](point2D_t degree) {
          return degree != 0;
        });
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    image.num_correspondences = offsets.back();
    image.flat_corrs.resize(offsets.back());
  }
  for (auto& [pair_id, geometry] : pairs) {
    const auto [first, second] = PairIdToImagePair(pair_id);
    if (first == second) continue;
    auto& first_image = graph->images_.at(first);
    auto& second_image = graph->images_.at(second);
    for (const auto& match : geometry.inlier_matches) {
      first_image.flat_corrs[first_image.flat_corr_begs[match.point2D_idx1]++] =
          {second, match.point2D_idx2};
      second_image
          .flat_corrs[second_image.flat_corr_begs[match.point2D_idx2]++] = {
          first, match.point2D_idx1};
    }
    FeatureMatches().swap(geometry.inlier_matches);
    graph->image_pairs_.at(pair_id).two_view_geometry = std::move(geometry);
  }
  for (auto& [id, image] : graph->images_) {
    auto& offsets = image.flat_corr_begs;
    // Filling advanced each start offset to its end. Restore the starts.
    std::move_backward(offsets.begin(), offsets.end() - 1, offsets.end());
    offsets.front() = 0;
  }
  graph->finalized_ = true;
  return graph;
}

std::shared_ptr<CorrespondenceGraph> BulkCorrespondenceGraph::Subset(
    const CorrespondenceGraph& source,
    const NodeHashMap<image_t, Image>& images) {
  THROW_CHECK(source.finalized_);
  auto graph = std::make_shared<CorrespondenceGraph>();
  for (const auto& [id, image] : images) {
    auto& target = graph->images_[id];
    const auto& original = source.images_.at(id);
    target.flat_corr_begs.reserve(original.flat_corr_begs.size());
    const auto retained = [&](const CorrespondenceGraph::Correspondence& corr) {
      return images.count(corr.image_id) != 0;
    };
    target.flat_corrs.reserve(std::count_if(
        original.flat_corrs.begin(), original.flat_corrs.end(), retained));
    for (point2D_t point = 0; point < image.NumPoints2D(); ++point) {
      target.flat_corr_begs.push_back(target.flat_corrs.size());
      const auto range = source.FindCorrespondences(id, point);
      for (auto corr = range.beg; corr != range.end; ++corr) {
        if (retained(*corr)) target.flat_corrs.push_back(*corr);
      }
      target.num_observations +=
          target.flat_corrs.size() != target.flat_corr_begs.back();
    }
    target.num_correspondences = target.flat_corrs.size();
    target.flat_corr_begs.push_back(target.flat_corrs.size());
  }
  for (const auto& [pair_id, pair] : source.image_pairs_) {
    const auto [first, second] = PairIdToImagePair(pair_id);
    if (images.count(first) != 0 && images.count(second) != 0) {
      graph->image_pairs_.emplace(pair_id, pair);
    }
  }
  graph->finalized_ = true;
  return graph;
}

}  // namespace colmap

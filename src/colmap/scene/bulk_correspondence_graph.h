#pragma once

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"

#include <memory>

namespace colmap {

// Builds the finalized cache representation directly. Matching continues to use
// CorrespondenceGraph's mutable API. This class alone bridges its private
// layout.
class BulkCorrespondenceGraph {
 public:
  static std::shared_ptr<CorrespondenceGraph> FromPairs(
      const NodeHashMap<image_t, Image>& images,
      std::vector<std::pair<image_pair_t, TwoViewGeometry>>& pairs);
  static std::shared_ptr<CorrespondenceGraph> Subset(
      const CorrespondenceGraph& source,
      const NodeHashMap<image_t, Image>& images);
};

}  // namespace colmap

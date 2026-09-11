#include "pycolmap/pipeline/fixed_rig_global_mapping.h"
#include "pycolmap/pipeline/prepared_global_mapping.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindExtractFeatures(py::module& m);
void BindImages(py::module& m);
void BindMatchFeatures(py::module& m);
#if defined(COLMAP_MVS_ENABLED)
void BindMeshing(py::module& m);
void BindMVS(py::module& m);
#endif
void BindSfM(py::module& m);
void BindSourceCoordinateRestoration(py::module& m);

void BindPipeline(py::module& m) {
  BindImages(m);
  BindSourceCoordinateRestoration(m);
  BindExtractFeatures(m);
  BindMatchFeatures(m);
  BindSfM(m);
  BindPreparedGlobalMapping(m);
  BindFixedRigGlobalMapping(m);
#if defined(COLMAP_MVS_ENABLED)
  BindMVS(m);
  BindMeshing(m);
#endif
}

#pragma once

#include <cuda_runtime.h>

namespace caspar {

cudaError_t ConstImageFromWorldStackedToCaspar(const float* stacked_data,
                                               float* cas_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects);

cudaError_t ConstImageFromWorldCasparToStacked(const float* cas_data,
                                               float* stacked_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects);

cudaError_t ConstLogScalePriorSqrtInformationStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstLogScalePriorSqrtInformationCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstLogScalePriorTargetStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstLogScalePriorTargetCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeCalibStackedToCaspar(const float* stacked_data,
                                             float* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholeCalibCasparToStacked(const float* cas_data,
                                             float* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholeFocalStackedToCaspar(const float* stacked_data,
                                             float* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholeFocalCasparToStacked(const float* cas_data,
                                             float* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholePoseStackedToCaspar(const float* stacked_data,
                                            float* cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t ConstPinholePoseCasparToStacked(const float* cas_data,
                                            float* stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeSensorCalibrationStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeSensorCalibrationCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeSensorFromRigStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeSensorFromRigCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPixelStackedToCaspar(const float* stacked_data,
                                      float* cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPixelCasparToStacked(const float* cas_data,
                                      float* stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPointStackedToCaspar(const float* stacked_data,
                                      float* cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPointCasparToStacked(const float* cas_data,
                                      float* stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPositionLossScaleStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPositionLossScaleCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPositionSqrtInformationStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPositionSqrtInformationCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstReferencePositionStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstReferencePositionCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstReprojectionLossScaleStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstReprojectionLossScaleCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstRigPointStackedToCaspar(const float* stacked_data,
                                         float* cas_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstRigPointCasparToStacked(const float* cas_data,
                                         float* stacked_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocalAndExtraStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocalAndExtraCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPoseStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPoseCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialSensorFromRigStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialSensorFromRigCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSurfacePositionStackedToCaspar(const float* stacked_data,
                                                float* cas_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t ConstSurfacePositionCasparToStacked(const float* cas_data,
                                                float* stacked_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t ConstSurfaceRobustStackedToCaspar(const float* stacked_data,
                                              float* cas_data,
                                              const unsigned int cas_stride,
                                              const unsigned int cas_offset,
                                              const unsigned int num_objects);

cudaError_t ConstSurfaceRobustCasparToStacked(const float* cas_data,
                                              float* stacked_data,
                                              const unsigned int cas_stride,
                                              const unsigned int cas_offset,
                                              const unsigned int num_objects);

cudaError_t ConstSurfaceSqrtWeightStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSurfaceSqrtWeightCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholeCalibStackedToCaspar(const float* stacked_data,
                                        float* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeCalibCasparToStacked(const float* cas_data,
                                        float* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeFocalStackedToCaspar(const float* stacked_data,
                                        float* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeFocalCasparToStacked(const float* cas_data,
                                        float* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholePoseStackedToCaspar(const float* stacked_data,
                                       float* cas_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t PinholePoseCasparToStacked(const float* cas_data,
                                       float* stacked_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t PinholePrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholePrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PointStackedToCaspar(const float* stacked_data,
                                 float* cas_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t PointCasparToStacked(const float* cas_data,
                                 float* stacked_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t SensorFromRigLogScaleStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SensorFromRigLogScaleCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialCalibStackedToCaspar(const float* stacked_data,
                                             float* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t SimpleRadialCalibCasparToStacked(const float* cas_data,
                                             float* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndExtraStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndExtraCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialPoseStackedToCaspar(const float* stacked_data,
                                            float* cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t SimpleRadialPoseCasparToStacked(const float* cas_data,
                                            float* stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

}  // namespace caspar
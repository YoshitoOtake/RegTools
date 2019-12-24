/*
  Author(s):  Yoshito Otake, Ali Uneri
  Created on: 2013-03-01
*/

#ifndef REG_TOOLS_MATLAB_H
#define REG_TOOLS_MATLAB_H

#ifdef _WINDOWS
#  define EXPORT extern __declspec(dllexport)
#else
#  define EXPORT
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#include "ProjectionParameterStructures.h"

struct RegToolsInstance { void * InstancePtr; };

//: function declarations
EXPORT int CreateRegToolsInstance(struct RegToolsInstance* instancePtr);
EXPORT int DestroyRegToolsInstance(struct RegToolsInstance instance, char **messages);
EXPORT int InitializeRegToolsThread(struct RegToolsInstance instance, int* deviceIDs, int numDevice, double *deviceLoadList, char **messages);
EXPORT int InitializeProjectionParametersArray(struct RegToolsInstance instance, int numProjections);
EXPORT int SetCurrentGeometrySetting(struct RegToolsInstance instance, int getmetry_id);
EXPORT int GetCurrentGeometrySetting(struct RegToolsInstance instance);
EXPORT int DeleteProjectionParametersArray(struct RegToolsInstance instance, int geometry_id);
EXPORT int SetProjectionParameter_objectOriented(struct RegToolsInstance instance, int projection_number /* note: this is 0-base index */, 
                                                 struct ProjectionParameters_objectOriented projectionParams);
EXPORT int SetProjectionParameter_3x4PM(struct RegToolsInstance instance, int projection_number /* note: this is 0-base index */, 
                                                 double *pm3x4_row_major, double pixel_width, double pixel_height, double u_dim, double v_dim, double down_sample_ratio_u, double down_sample_ratio_v);
EXPORT int SetProjectionParameter_3x4PM_multi(struct RegToolsInstance instance, int numProj, 
                                                 double *pm3x4_row_major, double pixel_width, double pixel_height, double u_dim, double v_dim);
EXPORT int CreateVolumePlan_cudaArray(struct RegToolsInstance instance, struct VolumePlan_cudaArray *plan, bool isCudaArray);
EXPORT int GetVolumePlan_cudaArrayVolumeInfo(struct RegToolsInstance instance, int plan_id, int *volume_dim, double *voxel_size, int *numVolumes);
EXPORT int GetVolumePlan_cudaArrayVolume(struct RegToolsInstance instance, int plan_id, float *h_volume, bool isCudaArray, int volume_index_1_base);
EXPORT int GetGPUProjection(struct RegToolsInstance instance, struct ProjectionResult *result, int volume_index_1_base);
EXPORT int MultVolumePlan(struct RegToolsInstance instance, int plan_id, float val);
EXPORT int DeleteVolumePlan_cudaArray(struct RegToolsInstance instance, int plan_id);
EXPORT int SetVolumePlan_cudaArrayVolume(struct RegToolsInstance instance, int plan_id, float *h_volume, bool isCudaArray, int volume_index_1_base);
EXPORT int CreateSimilarityMeasureComputationPlan(struct RegToolsInstance instance, struct SimilarityMeasureComputationPlan *plan, double *normalization_factor);
EXPORT int GetSimilarityMeasureComputationPlanImageInfo(struct RegToolsInstance instance, int plan_id, int GPU_ID, int *image_dim, double *normalization_factor);
EXPORT int GetSimilarityMeasureComputationPlanImages(struct RegToolsInstance instance, int plan_id, int GPU_ID, float *images, int image_type, int frame_no);
EXPORT int DeleteSimilarityMeasureComputationPlan(struct RegToolsInstance instance, int plan_id);

EXPORT void ComputeSimilarityMeasure(struct RegToolsInstance instance, int plan_id, int similarity_type, int numImageSet, double *sm, float *elapsed_time);
EXPORT void ComputeSimilarityMeasure2(struct RegToolsInstance instance, int plan_id1, int plan_id2, int similarity_type, int numImageSet, double *sm, float *elapsed_time);
EXPORT int ComputeBoxProjectionBoundingSquare(struct RegToolsInstance instance, int *projected_square_left_bottom, int *projected_size, int *in_out, double *box_center, double *box_size, int margin);
EXPORT int CropAllProjections(struct RegToolsInstance instance, int *left_bottom /* numbefOfProjections*2 element array, 0-base */, int *crop_size /* 2 element array */);

//: Set transformation from world to volume coordinate (transform matrix should be 4x4 column major matrix) 
EXPORT int SetWorldToVolumeTransform_4x4(struct RegToolsInstance instance, const double *transform_col);
EXPORT int SetWorldToVolumeTransform_6(struct RegToolsInstance instance, double tx, double ty, double tz, double rx, double ry, double rz);
EXPORT int AddLogFile(struct RegToolsInstance instance, char* filename);
EXPORT int RemoveLogFile(struct RegToolsInstance instance);

EXPORT int ForwardProjection(struct RegToolsInstance instance, struct ProjectionResult *result, const float *volume);
EXPORT int ForwardProjection_with3x4ProjectionMatrices(struct RegToolsInstance instance, struct ProjectionResult *result, const int plan_id, const double *pm_3x4);
EXPORT int ForwardProjection_withPlan(struct RegToolsInstance instance, struct ProjectionResult *result, const int plan_id, int numGlobals, const double *transformations_global
                                      , int numView, int numLocalTrans, const double *transformations_local, const int memory_store_mode);
//EXPORT int Interpolation(struct RegToolsInstance instance, struct ProjectionResult *result, const float *volume, const float *transform, int type, int order, float bicubic_a, float back_ground, float *volume_center, float penalty_beta, int volumePlanID);
EXPORT int Interpolation_withPlan(struct RegToolsInstance instance, struct ProjectionResult *result, int plan_id, const float *transforms, const int num_transform_element, int num_transform, int type, int order, float bicubic_a, float back_ground, float *volume_center);
EXPORT int ApplyDeformationField(struct RegToolsInstance instance, struct ProjectionResult *result, int target_volume_id, int *warps_tex, int num_dims, int type, int order, 
                                 float bicubic_a, float back_ground, float *volume_center, int scattered_pnts_plan, float *transforms_3x4xN, int num_transform_element, int num_transforms);
EXPORT int SetWarpTextures(struct RegToolsInstance instance, int *warps_tex);
EXPORT int ComputeLinearCombination(struct RegToolsInstance instance, int warp_device, int def_mode_device, int mode_weight);
EXPORT int CopyDeviceMemoryToCudaArray(struct RegToolsInstance instance, int cudaArrayID, int deviceMemoryID, int isCopyToAllDevices, int volume_index_tex_1_base, int volume_index_dev_1_base);
EXPORT int CopyDeviceMemoryToCudaArray_Multi(struct RegToolsInstance instance, int *cudaArrayIDs, int numCudaArrayID, int deviceMemoryID);

//: Set dimensions of the volume and the size of one voxel (in whatever unit, e.g "mm")
EXPORT int SetVolumeInfo(struct RegToolsInstance instance, int volume_dim_x, int volume_dim_y, int volume_dim_z, float voxelSize_x, float voxelSize_y, float voxelSize_z);
EXPORT int GetVolumeInfo(struct RegToolsInstance instance, int *volume_dim_x, int *volume_dim_y, int *volume_dim_z, float *voxelSize_x, float *voxelSize_y, float *voxelSize_z);

EXPORT int SetStepSize(struct RegToolsInstance instance, float step_size);
EXPORT int GetStepSize(struct RegToolsInstance instance, float *step_size);
EXPORT int SetRayCastingLOD(struct RegToolsInstance instance, int lod);
EXPORT int GetRayCastingLOD(struct RegToolsInstance instance, int *lod);
EXPORT int SetRayCastingThreshold(struct RegToolsInstance instance, float threshold);
EXPORT int GetRayCastingThreshold(struct RegToolsInstance instance, float *threshold);
EXPORT int SetRayCastingDistanceFalloffCoefficient(struct RegToolsInstance instance, float coeff);
EXPORT int GetRayCastingDistanceFalloffCoefficient(struct RegToolsInstance instance, float *coeff);
EXPORT int SetCountNonIntersectedPixel(struct RegToolsInstance instance, int count_non_intersected_pixel);
EXPORT int GetCountNonIntersectedPixel(struct RegToolsInstance instance, int *count_non_intersected_pixel);
EXPORT int SetDifferentVolumePerProjectionSet(struct RegToolsInstance instance, int different_volume_per_projection_set);
EXPORT int GetDifferentVolumePerProjectionSet(struct RegToolsInstance instance, int *different_volume_per_projection_set);

EXPORT int GetNumberOfProjections(struct RegToolsInstance instance);
EXPORT int GetNumberOfEnabledProjections(struct RegToolsInstance instance);
EXPORT int SetProjectorMode(struct RegToolsInstance instance, int projectorMode);
EXPORT int GetProjectorMode(struct RegToolsInstance instance);
EXPORT int SetProjectionDim(struct RegToolsInstance instance, int width, int height);
EXPORT int GetProjectionDim(struct RegToolsInstance instance, int *width, int *height);
EXPORT int GetProjectionMatrices(struct RegToolsInstance instance, double *pm);
EXPORT int GetPixelSize(struct RegToolsInstance instance, double *pixel_width, double *pixel_height);
EXPORT int SetNumberOfProjectionSets(struct RegToolsInstance instance, int num_projection_sets);
EXPORT int GetNumberOfProjectionSets(struct RegToolsInstance instance);
EXPORT int SetTransferBlockSize(struct RegToolsInstance instance, int numBlockSize);
EXPORT int SetSubSamplingVector(struct RegToolsInstance instance, int *sub_sampling_vector, int numElements);  // 0: disabled projection, 1: enabled projection
EXPORT int GetSubSamplingVector(struct RegToolsInstance instance, int *sub_sampling_vector, int numElements);  // 0: disabled projection, 1: enabled projection
EXPORT int EraseDisabledProjections(struct RegToolsInstance instance);
EXPORT int ReplicateProjections(struct RegToolsInstance instance, int num_rep);
EXPORT int convertRotTransTo4x4(struct RegToolsInstance instance, double *in_1x6vec, double *out_4x4_col);
EXPORT int convertTransRotTo4x4(struct RegToolsInstance instance, double *in_1x6vec, double *out_4x4_col);
EXPORT int convert4x4ToRotTrans(struct RegToolsInstance instance, double *in_4x4, double *out_1x6vec);
EXPORT int convert4x4ToTransRot(struct RegToolsInstance instance, double *in_4x4, double *out_1x6vec);

EXPORT double GPUmemCheck(struct RegToolsInstance instance, const char* message, int threadID);
EXPORT int GetGPUList(char** name_list, int max_list_length, int max_name_length);

EXPORT int CMAES_popuation(struct RegToolsInstance instance, int arz_ID, int arx_ID, int arxvalid_ID, int xmean_ID, int diagD_ID, int lbounds_ID, int ubounds_ID);

EXPORT void SetInitialProjectionOnDevice(struct RegToolsInstance instance, float* h_proj, double len);

EXPORT void ClearInitialProjectionOnDevice(struct RegToolsInstance instance);

#if defined(__cplusplus)
}  // extern "C"
#endif

#endif

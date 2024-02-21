/*
  Author(s):  Yoshito Otake, Ali Uneri
  Created on: 2011-02-21
*/

#include "RegToolsMatlab.h"
#include "RegTools.h"

int CreateRegToolsInstance(struct RegToolsInstance* instancePtr)
{
#if defined RegTools_VERBOSE_MESSAGE
  printf("RegTools - Generating projector instance...\n");
#endif
  RegTools* internalInstance = new RegTools();
  if (internalInstance == 0)  return 0;
  else                        instancePtr->InstancePtr = internalInstance;
  return 1;
}

int DestroyRegToolsInstance(struct RegToolsInstance instance, char **messages)
{
#if defined RegTools_VERBOSE_MESSAGE
  printf("RegTools - Destroying projector instance...\n");
#endif
  if ( instance.InstancePtr == 0 ){
    printf("RegTools - error at DestroyRegToolsInstance: instance argument not initialized\n");
    return 0;
  }
  delete reinterpret_cast<RegTools *>(instance.InstancePtr);
  instance.InstancePtr = 0;
  int messages_ptr = 0;
  add_str(messages[0], messages_ptr, "RegTools closed.\n");
  return 1;
}

int InitializeRegToolsThread(struct RegToolsInstance instance, int* deviceIDs, int numDevice, double *deviceLoadList, char **messages)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->InitializeRegToolsThread(deviceIDs, numDevice, deviceLoadList, false, messages);
}

//: create an array of projection parameters 
int InitializeProjectionParametersArray(struct RegToolsInstance instance, int numProjections)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->InitializeProjectionParametersArray(numProjections);
}

int SetCurrentGeometrySetting(struct RegToolsInstance instance, int geometry_id)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetCurrentGeometrySetting(geometry_id);
}

int GetCurrentGeometrySetting(struct RegToolsInstance instance)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetCurrentGeometrySetting();
}

int DeleteProjectionParametersArray(struct RegToolsInstance instance, int geometry_id)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->DeleteProjectionParametersArray(geometry_id);
}

int AddLogFile(struct RegToolsInstance instance, char* filename)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->AddLogFile(filename);
}

int RemoveLogFile(struct RegToolsInstance instance)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->RemoveLogFile();
}

//: user should call this function for each projection image.
int SetProjectionParameter_objectOriented(struct RegToolsInstance instance, int projection_number /* note: this is 0-base index */, 
                                          struct ProjectionParameters_objectOriented projectionParams)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetProjectionParameter_objectOriented(projection_number, projectionParams);
}

int SetProjectionParameter_3x4PM(struct RegToolsInstance instance, int projection_number /* note: this is 0-base index */, 
                                          double *pm3x4_row_major, double pixel_width, double pixel_height, double u_dim, double v_dim, double down_sample_ratio_u, double down_sample_ratio_v)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetProjectionParameter_3x4PM(projection_number, pm3x4_row_major, pixel_width, pixel_height, u_dim, v_dim, down_sample_ratio_u, down_sample_ratio_v);
}

int SetProjectionParameter_3x4PM_multi(struct RegToolsInstance instance, int numProj, 
                                          double *pm3x4_row_major, double pixel_width, double pixel_height, double u_dim, double v_dim)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetProjectionParameter_3x4PM_multi(numProj, pm3x4_row_major, pixel_width, pixel_height, u_dim, v_dim);
}

int CreateVolumePlan_cudaArray(struct RegToolsInstance instance, struct VolumePlan_cudaArray *plan, bool isCudaArray)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->CreateVolumePlan_cudaArray(plan, isCudaArray);
}

int GetGPUProjection(struct RegToolsInstance instance, struct ProjectionResult *result, int volume_index_1_base)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetGPUProjection(result->Data, volume_index_1_base-1);
}

int GetVolumePlan_cudaArrayVolumeInfo(struct RegToolsInstance instance, int plan_id, int *volume_dim, double *voxel_size, int *numVolumes)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetVolumePlan_cudaArrayVolumeInfo(plan_id, volume_dim, voxel_size, numVolumes);
}

int GetVolumePlan_cudaArrayVolume(struct RegToolsInstance instance, int plan_id, float *h_volume, bool isCudaArray, int volume_index_1_base)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetVolumePlan_cudaArrayVolume(plan_id, h_volume, isCudaArray, volume_index_1_base-1); // 1-base index -> 0-base index
}

int SetVolumePlan_cudaArrayVolume(struct RegToolsInstance instance, int plan_id, float *h_volume, bool isCudaArray, int volume_index_1_base)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetVolumePlan_cudaArrayVolume(plan_id, h_volume, isCudaArray, volume_index_1_base-1); // 1-base index -> 0-base index
}

int MultVolumePlan(struct RegToolsInstance instance, int plan_id, float val)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->MultVolumePlan(plan_id, val);
}

int DeleteVolumePlan_cudaArray(struct RegToolsInstance instance, int plan_id)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->DeleteVolumePlan_cudaArray(plan_id);
}

int CreateSimilarityMeasureComputationPlan(struct RegToolsInstance instance, struct SimilarityMeasureComputationPlan *plan, double *normalization_factor)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->CreateSimilarityMeasureComputationPlan(plan, normalization_factor);
}

int GetSimilarityMeasureComputationPlanImageInfo(struct RegToolsInstance instance, int plan_id, int GPU_ID, int *image_dim, double *normalization_factor)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetSimilarityMeasureComputationPlanImageInfo(plan_id, GPU_ID, image_dim, normalization_factor);
}

int GetSimilarityMeasureComputationPlanImages(struct RegToolsInstance instance, int plan_id, int GPU_ID, float *images, int image_type, int frame_no)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetSimilarityMeasureComputationPlanImages(plan_id, GPU_ID, images, image_type, frame_no);
}

int DeleteSimilarityMeasureComputationPlan(struct RegToolsInstance instance, int plan_id)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->DeleteSimilarityMeasureComputationPlan(plan_id);
}

int CreateLCNComputationPlan(struct RegToolsInstance instance, struct LCN_computation_plan *plan)
{
	return reinterpret_cast<RegTools*>(instance.InstancePtr)->CreateLCNComputationPlan(plan);
}

int DeleteLCNComputationPlan(struct RegToolsInstance instance, int plan_id)
{
	return reinterpret_cast<RegTools*>(instance.InstancePtr)->DeleteLCNComputationPlan(plan_id);
}
void ComputeSimilarityMeasure(struct RegToolsInstance instance, int plan_id, int similarity_type, int numImageSet, double *sm, float *elapsed_time)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->ComputeSimilarityMeasure(plan_id, similarity_type, numImageSet, sm, elapsed_time);
}

void ComputeSimilarityMeasure2(struct RegToolsInstance instance, int plan_id1, int plan_id2, int similarity_type, int numImageSet, double *sm, float *elapsed_time)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->ComputeSimilarityMeasure(plan_id1, plan_id2, similarity_type, numImageSet, sm, elapsed_time);
}

int ComputeBoxProjectionBoundingSquare(struct RegToolsInstance instance, int *projected_square_left_bottom, int *projected_size, int *in_out, double *box_center, double *box_size, int margin)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->ComputeBoxProjectionBoundingSquare(projected_square_left_bottom, projected_size, in_out, box_center, box_size, margin);
}

int CropAllProjections(struct RegToolsInstance instance, int *left_bottom /* numbefOfProjections*2 element array */, int *crop_size /* 2 element array */)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->CropAllProjections(left_bottom, crop_size);
}

int SetWorldToVolumeTransform_4x4(struct RegToolsInstance instance, const double *transform_col)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetWorldToVolumeTransform(transform_col);
}

int SetWorldToVolumeTransform_6(struct RegToolsInstance instance, double tx, double ty, double tz, double rx, double ry, double rz)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetWorldToVolumeTransform(tx, ty, tz, rx, ry, rz);
}

int ForwardProjection(struct RegToolsInstance instance, struct ProjectionResult *result, const float *volume)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->ForwardProjection(*result, volume);
}

int ForwardProjection_with3x4ProjectionMatrices(struct RegToolsInstance instance, struct ProjectionResult *result, const int plan_id, const double *pm_3x4)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->ForwardProjection_with3x4ProjectionMatrices(*result, plan_id, pm_3x4);
}

int ForwardProjection_withPlan(struct RegToolsInstance instance, struct ProjectionResult *result, const int plan_id, int numGlobals, const double *transformations_global, int numView
                               , int numLocalTrans, const double *transformations_local, const int memory_store_mode, const int LCN_plan_ID)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->ForwardProjection_withPlan(*result, plan_id, numGlobals, transformations_global, numView, numLocalTrans, transformations_local, memory_store_mode, LCN_plan_ID);
}

int Interpolation_withPlan(struct RegToolsInstance instance, struct ProjectionResult *result, int plan_id, const float *transforms, const int num_transform_element, int num_transforms, int type, int order, float bicubic_a, float back_ground, float *volume_center)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->Interpolation_withPlan(*result, plan_id, transforms, num_transform_element, num_transforms, type, order, bicubic_a, back_ground, volume_center);
}

int ApplyDeformationField(struct RegToolsInstance instance, struct ProjectionResult *result, int target_volume_id, int *warps_tex, int num_dims, int type, int order, 
                          float bicubic_a, float back_ground, float *volume_center, int scattered_pnts_plan, float *transforms_3x4xN, int num_transform_element, int num_transforms)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->ApplyDeformationField(*result, target_volume_id, warps_tex, num_dims, type, order, bicubic_a, back_ground, 
                                                                                                volume_center, scattered_pnts_plan, transforms_3x4xN, num_transform_element, num_transforms);
}

int SetWarpTextures(struct RegToolsInstance instance, int *warps_tex)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetWarpTextures(warps_tex); 
}

int ComputeLinearCombination(struct RegToolsInstance instance, int warp_device, int def_mode_device, int mode_weight)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->ComputeLinearCombination(warp_device, def_mode_device, mode_weight);
}

int CopyDeviceMemoryToCudaArray(struct RegToolsInstance instance, int cudaArrayID, int deviceMemoryID, int isCopyToAllDevices, int volume_index_tex_1_base, int volume_index_dev_1_base)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->CopyDeviceMemoryToCudaArray(cudaArrayID, deviceMemoryID, isCopyToAllDevices, volume_index_tex_1_base-1, volume_index_dev_1_base-1);  // 1-base index -> 0-base index
}

int CopyDeviceMemoryToCudaArray_Multi(struct RegToolsInstance instance, int *cudaArrayIDs, int numCudaArrayID, int deviceMemoryID)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->CopyDeviceMemoryToCudaArray_Multi(cudaArrayIDs, numCudaArrayID, deviceMemoryID);
}

int GetNumberOfProjections(struct RegToolsInstance instance)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetNumberOfProjections();
}

int GetNumberOfEnabledProjections(struct RegToolsInstance instance)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetNumberOfEnabledProjections();
}

int SetVolumeInfo(struct RegToolsInstance instance, int volume_dim_x, int volume_dim_y, int volume_dim_z, float voxelSize_x, float voxelSize_y, float voxelSize_z)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetVolumeInfo(volume_dim_x, volume_dim_y, volume_dim_z, voxelSize_x, voxelSize_y, voxelSize_z);
}

int GetVolumeInfo(struct RegToolsInstance instance, int *volume_dim_x, int *volume_dim_y, int *volume_dim_z, float *voxelSize_x, float *voxelSize_y, float *voxelSize_z)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetVolumeInfo(volume_dim_x, volume_dim_y, volume_dim_z, voxelSize_x, voxelSize_y, voxelSize_z);
}

int SetStepSize(struct RegToolsInstance instance, float step_size)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetStepSize(step_size);
}

int GetStepSize(struct RegToolsInstance instance, float *step_size)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetStepSize(step_size);
}

int SetLCNSigma(struct RegToolsInstance instance, float LCN_sigma)
{
	return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetLCNSigma(LCN_sigma);
}

int SetRayCastingLOD(struct RegToolsInstance instance, int lod)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetRayCastingLOD(lod);
}

int GetRayCastingLOD(struct RegToolsInstance instance, int *lod)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetRayCastingLOD(lod);
}

int SetRayCastingThreshold(struct RegToolsInstance instance, float threshold)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetRayCastingThreshold(threshold);
}

int GetRayCastingThreshold(struct RegToolsInstance instance, float *threshold)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetRayCastingThreshold(threshold);
}

int SetRayCastingDistanceFalloffCoefficient(struct RegToolsInstance instance, float coeff)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetRayCastingDistanceFalloffCoefficient(coeff);
}

int GetRayCastingDistanceFalloffCoefficient(struct RegToolsInstance instance, float *coeff)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetRayCastingDistanceFalloffCoefficient(coeff);
}

int SetCountNonIntersectedPixel(struct RegToolsInstance instance, int count_non_intersected_pixel)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetCountNonIntersectedPixel(count_non_intersected_pixel);
}

int GetCountNonIntersectedPixel(struct RegToolsInstance instance, int *count_non_intersected_pixel)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetCountNonIntersectedPixel(count_non_intersected_pixel);
}

int SetDifferentVolumePerProjectionSet(struct RegToolsInstance instance, int different_volume_per_projection_set)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetDifferentVolumePerProjectionSet(different_volume_per_projection_set);
}

int GetDifferentVolumePerProjectionSet(struct RegToolsInstance instance, int *different_volume_per_projection_set)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetDifferentVolumePerProjectionSet(different_volume_per_projection_set);
}

int SetDepthMapBack(struct RegToolsInstance instance, int depth_map_back)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetDepthMapBack(depth_map_back);
}

int GetDepthMapBack(struct RegToolsInstance instance, int* depth_map_back)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetDepthMapBack(depth_map_back);
}

int SetRayCastingIntensityGradDim(struct RegToolsInstance instance, int dim)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetRayCastingIntensityGradDim(dim);
}

int GetRayCastingIntensityGradDim(struct RegToolsInstance instance, int* dim)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetRayCastingIntensityGradDim(dim);
}

int SetProjectionDim(struct RegToolsInstance instance, int width, int height)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetProjectionDim(width, height);
}

int GetProjectionDim(struct RegToolsInstance instance, int *width, int *height)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetProjectionDim(width, height);
}

int GetProjectionMatrices(struct RegToolsInstance instance, double *pm)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetProjectionMatrices(pm);
}

int GetPixelSize(struct RegToolsInstance instance, double *pixel_width, double *pixel_height)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetPixelSize(pixel_width, pixel_height);
}

int SetNumberOfProjectionSets(struct RegToolsInstance instance, int num_projection_sets)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetNumberOfProjectionSets(num_projection_sets);
}

int GetNumberOfProjectionSets(struct RegToolsInstance instance)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetNumberOfProjectionSets();
}

int SetTransferBlockSize(struct RegToolsInstance instance, int numBlockSize)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetTransferBlockSize(numBlockSize);
}

int SetSubSamplingVector(struct RegToolsInstance instance, int *sub_sampling_vector, int numElements)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetSubSamplingVector(sub_sampling_vector, numElements);
}

int GetSubSamplingVector(struct RegToolsInstance instance, int *sub_sampling_vector, int numElements)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetSubSamplingVector(sub_sampling_vector, numElements);
}

int SetProjectorMode(struct RegToolsInstance instance, int projectorMode)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetProjectorMode(projectorMode);
}

int GetProjectorMode(struct RegToolsInstance instance)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GetProjectorMode();
}

int EraseDisabledProjections(struct RegToolsInstance instance)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->EraseDisabledProjections();
}

int ReplicateProjections(struct RegToolsInstance instance, int num_rep)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->ReplicateProjections(num_rep);
}

int convertRotTransTo4x4(struct RegToolsInstance instance, double *in_1x6vec, double *out_4x4_col)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->convertRotTransTo4x4(in_1x6vec, out_4x4_col);
}

int convertTransRotTo4x4(struct RegToolsInstance instance, double *in_1x6vec, double *out_4x4_col)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->convertTransRotTo4x4(in_1x6vec, out_4x4_col);
}

int convert4x4ToRotTrans(struct RegToolsInstance instance, double *in_4x4, double *out_1x6vec)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->convert4x4ToRotTrans(in_4x4, out_1x6vec);
}

int convert4x4ToTransRot(struct RegToolsInstance instance, double *in_4x4, double *out_1x6vec)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->convert4x4ToTransRot(in_4x4, out_1x6vec);
}

double GPUmemCheck(struct RegToolsInstance instance, const char* message, int threadID)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->GPUmemCheck(message, threadID);
}

int CMAES_popuation(struct RegToolsInstance instance, int arz_ID, int arx_ID, int arxvalid_ID, int xmean_ID, int diagD_ID, int lbounds_ID, int ubounds_ID)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->CMAESPopulation(arz_ID, arx_ID, arxvalid_ID, xmean_ID, diagD_ID, lbounds_ID, ubounds_ID);
}

int GetGPUList(char** name_list, int max_list_length, int max_name_length)
{
  int device_count;
  cudaDeviceProp deviceProp;
 	cudaGetDeviceCount( &device_count );

  for(int i=0;i<max_list_length && i<device_count;i++){
    cudaGetDeviceProperties( &deviceProp, i );
    sprintf_s(name_list[i], max_name_length, "GPU no.%d: %s, total global memory: %d MB, compute capability: %d.%d, %d CUDA cores"
      , i, deviceProp.name, deviceProp.totalGlobalMem/(1024*1024), deviceProp.major, deviceProp.minor
      , ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
/*
    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
        "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
        "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
        "Unknown",
        NULL
    };
    sprintf_s(name_list[i], max_name_length, "GPU no.%d: %s, total global memory: %d MB, compute capability: %d.%d, number of multiprocessor: %d, computeMode: %s"
      , i, deviceProp.name, deviceProp.totalGlobalMem/(1024*1024), deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount, sComputeMode[deviceProp.computeMode]);
*/
  }
  return device_count;
}

void SetInitialProjectionOnDevice(struct RegToolsInstance instance, float* h_proj, double len)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->SetInitialProjectionOnDevice(h_proj, static_cast<size_t>(len));
}

void ClearInitialProjectionOnDevice(struct RegToolsInstance instance)
{
  return reinterpret_cast<RegTools*>(instance.InstancePtr)->ClearInitialProjectionOnDevice();
}

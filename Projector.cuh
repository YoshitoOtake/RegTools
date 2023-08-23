/*
  Author(s):  Yoshito Otake
  Created on: 2013-03-01
*/

#ifndef PROJECTOR_CUH
#define PROJECTOR_CUH

#include "RegTools.h"
#include "ProjectionParameterStructures.h"
#include "my_cutil_math.h"
#include <curand.h>
#include <cublas.h>

struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};

// host functions
extern "C" void copyProjectorMode(int projectorMode); // ProjectorMode_LinearInterpolation = 0, ProjectorMode_RayCasting = 5
extern "C" void copyVolumeDimensions(unsigned int volume_width, unsigned int volume_height, unsigned int volume_depth);
extern "C" void copyVoxelSize(float3 voxel_size);
extern "C" void copyVolumeCorner(float3 voxel_corner_mm);
extern "C" void copyProjectionDim(int width, int height);
extern "C" void copyNumberOfProjectionSets(int num_projection_sets);
extern "C" void copyTileSize(int tileX, int tileY);
extern "C" void copyProjectionBlockSize(int block_size);
extern "C" void copyStepSize(float step_size);
extern "C" void copyMemoryStoreMode(int memory_store_mode);
extern "C" void copyRayCastingLOD(int lod);
extern "C" void copyRayCastingThreshold(float thresh);
extern "C" void copyRayCastingDistanceFalloffCoefficient(float coefficient);
extern "C" void copyCountNonIntersectedPixel(bool count_non_intersected_pixel);
extern "C" void copyDifferentVolumePerProjectionSet(bool different_volume_per_projection_set);
extern "C" void copyDepthMapBack(bool depth_map_back);
extern "C" void cudaMallocForTexture(struct cudaArray **d_projArray, cudaExtent extent);

extern "C" void bindCUDATexture(struct cudaArray *d_inputDataArray);
extern "C" void unBindCUDATexture();
extern "C" void bindWarpTexture(struct cudaArray *d_warpXArray, struct cudaArray *d_warpYArray, struct cudaArray *d_warpZArray);
extern "C" void unBindWarpTexture();
extern "C" void bindPrecomputedMatrixTexture(float *d_PrecomputedMatrix_array, int numProjections);
extern "C" void unBindPrecomputedMatrixTexture();
extern "C" void initCudaTexture(bool normalized, bool isLinear);
extern "C" void print_all_constant_vars(void);
extern "C" void ComputeMaxMin(float *d_data, int size, float *h_maxValue, float *h_minValue);
extern "C" void NormalizeData(float *d_data, int size, float h_normMax, float h_normMin);
extern "C" void FillData(float *d_data, int size, float val);
extern "C" void FillData_i(int *d_data, int size, int val);
extern "C" void MultScalar(float *d_data, float scalar_val, int size);// data <- data * scalar_val
extern "C" void MultData(float *d_data1, float *d_data2, float *d_out, int size); // out <- data1 * data2
extern "C" void DivData(float *d_numerator, float *d_denominator, float *d_out, int size); // out <- d_numerator / d_denominator
extern "C" void SqrtData(float *d_data, float *d_out, int size); // out <- sqrt(d_data)
extern "C" void randn_CURAND(curandGenerator_t generator, float *d_data, int size);
extern "C" void generateCMAESPopulation(float *d_arz, float *d_arx, float *d_arxvalid, float *d_xmean, float *d_diagD, float *d_lbounds, float *d_ubounds, float *d_OneVector, cublasHandle_t cublasHandle, int nRows, int nCols);

extern "C" void launch_LinearInterpolationProjector(float *d_projection_local, int *d_ZeroPixelCount, int number_of_projections_in_one_set, dim3 grid, dim3 block);
extern "C" void launch_LinearInterpolationDeformableProjector(float *d_projection_local, int *d_ZeroPixelCount, int number_of_projections_in_one_set, dim3 grid, dim3 block);
extern "C" void launch_RayCastingProjector(float *d_projection_local, dim3 grid, dim3 block);
extern "C" void launch_DepthMapProjector(float* d_projection_local, dim3 grid, dim3 block);
extern "C" void launch_SiddonProjector(float *d_projection_local, size_t pitch, dim3 grid, dim3 block, int *d_random_sequence);
extern "C" void launch_Interpolator(float* d_data_out, const float transform[6], 
                                    const int type, const int order, const float bicubic_a, const float back_ground, float *volume_center);
extern "C" void launch_Interpolator_BBoxCheck(float* d_data_out, int in_volumeDim[3], double in_voxelSize[3], const float* transform, const int num_transform_element, 
                                              const int type, const int order, const float bicubic_a, float back_ground_value, float *volume_center,
                                              float *scattered_pnts, int num_scattered_pnts, bool isWarp, int num_transforms);
extern "C" void launch_LocalContrastNormalization(float *d_projection_local, int number_of_projections_in_one_set, dim3 grid, dim3 block);

// common device functions
__device__ bool getPixelIndex(unsigned int &x, unsigned int &y, unsigned int &z);

__device__ Ray computeNormalizedRay( const float x_pix, const float y_pix, const int numProjection);

__global__ void LinearInterpolationProjection(float* d_projection, int* d_ZeroPixelCount, int number_of_projections_in_one_set);
__global__ void LinearInterpolationDeformableProjection(float* d_projection, int* d_ZeroPixelCount, int number_of_projections_in_one_set);
__global__ void SiddonProjection(float* d_projection, size_t pitch, int *d_random_gridID = NULL);
__global__ void RayCastingProjection(float* d_projection);
__global__ void DepthMapProjection(float* d_projection);
__global__ void LocalContrastNormalization(float* d_projection, int number_of_projections_in_one_set);

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
// This code is based on volumeRender demo in CUDA SDK
inline __device__ bool intersectBoxRay(const float3 boxmin, const float3 boxmax, const Ray ray, float &tnear, float &tfar)
{
  // compute intersection of ray with all six planes
  float3 tbot = (boxmin - ray.o) / ray.d;
  float3 ttop = (boxmax - ray.o) / ray.d;
  // re-order intersections to find smallest and largest on each axis
  float3 tmin = fminf(ttop, tbot);
  float3 tmax = fmaxf(ttop, tbot);

  // find the largest tmin and the smallest tmax
  tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
  tfar = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
	return tfar > tnear;
}

inline __device__ bool intersectBoxRay_withPlane(const float3 boxmin, const float3 boxmax, const Ray ray, float &tnear, float &tfar, int &near_plane)
{
  // compute intersection of ray with all six planes
  float3 tbot = (boxmin - ray.o) / ray.d;
  float3 ttop = (boxmax - ray.o) / ray.d;
  // re-order intersections to find smallest and largest on each axis
  float3 tmin = fminf(ttop, tbot);
  float3 tmax = fmaxf(ttop, tbot);

  // find the largest tmin and the smallest tmax
  tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
  tfar = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
  if(tnear == tmin.x)      near_plane = 0;
  else if(tnear == tmin.y) near_plane = 1;
  else                     near_plane = 2;
	return tfar > tnear;
}

#endif

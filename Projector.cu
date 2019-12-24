/*
  Author(s):  Yoshito Otake
  Created on: 2013-03-01
*/

#include "RegTools.h"
#include "Projector.cuh"
#include "my_cutil_math.h"
#define _USE_MATH_DEFINES
#include <math.h>  // for M_PI_2, cos, sin

// for thrust functions
#include <thrust/fill.h> // thrust functions
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

extern FILE *m_LogFile;

__constant__ cudaExtent c_VolumeDim;
__constant__ float3 c_VolumeDim_f;
__constant__ float3 c_VoxelSize_mm;
__constant__ float3 c_VolumeCorner_mm;
__constant__ float c_ImageWidth, c_ImageHeight;
__constant__ int c_ImageWidth_i, c_ImageHeight_i;
__constant__ int c_NumberOfProjectionSets;
__constant__ int c_ProjectionBlockSize_i;
__constant__ int c_RayCastingLOD;
__constant__ float c_StepSize_mm;
__constant__ int c_TileX, c_TileY;
__constant__ int c_MemoryStoreMode; // MemoryStoreMode_Replace = 0, MemoryStoreMode_Additive = 1, MemoryStoreMode_Multiplicative = 2
__constant__ int c_ProjectorMode;  // ProjectorMode_LinearInterpolation = 0, ProjectorMode_Siddon = 1, ProjectorMode_SF_TR = 2, ProjectorMode_SF_TT, ProjectorMode_SF_DD = 4
__constant__ float c_RayCastingThreshold;
__constant__ float c_RayCastingDistanceFalloffCoefficient;
__constant__ bool c_CountNonIntersectedPixel;
__constant__ bool c_DifferentVolumePerProjectionSet;

// device texture memory
// need to define this in the source file, since syntax is not compatible with SWIG
texture<float, 3, cudaReadModeElementType> d_texture_in;      // input data
texture<float, 3, cudaReadModeElementType> d_texture_warpX;   // deformation field (X-direction)
texture<float, 3, cudaReadModeElementType> d_texture_warpY;   // deformation field (Y-direction)
texture<float, 3, cudaReadModeElementType> d_texture_warpZ;   // deformation field (Z-direction)

// "pre computed" matrix
// 3 vectors representing: 1) source -> left-bottom corner, 2) left-bottom corner -> right-bottom corner, 3) left-bottom corner -> left-top corner
texture<float, 1, cudaReadModeElementType> d_PrecomputedMatrix; // texture pointer for precomputed matrices

#define LOAD_PRE(i) tex1Dfetch(d_PrecomputedMatrix, i)

bool isTextureBound = false, isWarpTextureBound = false;

extern "C" void copyProjectorMode(int projectorMode) // ProjectorMode_LinearInterpolation = 0, ProjectorMode_Siddon, ProjectorMode_SF_TR = 2, ProjectorMode_SF_TT, ProjectorMode_SF_DD = 4
{
  cutilSafeCall( cudaMemcpyToSymbol(c_ProjectorMode, &projectorMode, sizeof(int), 0, cudaMemcpyHostToDevice ) );
}

extern "C" void copyVolumeDimensions(unsigned int volume_width, unsigned int volume_height, unsigned int volume_depth)
{
  cudaExtent h_volumeDimensions = make_cudaExtent(volume_width, volume_height, volume_depth);
  cutilSafeCall( cudaMemcpyToSymbol(c_VolumeDim, &h_volumeDimensions, sizeof(cudaExtent)) );
  float3 vol_dim = make_float3(static_cast<float>(volume_width), static_cast<float>(volume_height), static_cast<float>(volume_depth));
  cutilSafeCall( cudaMemcpyToSymbol(c_VolumeDim_f, &vol_dim, sizeof(float3)) );
//  print_and_log("copyVolumeDimensions(), c_VolumeDim_f: %f, %f, %f\n", vol_dim.x, vol_dim.y, vol_dim.z);
}

extern "C" void copyNumberOfProjectionSets(int num_projection_sets)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_NumberOfProjectionSets, &num_projection_sets, sizeof(int)) );
//  print_and_log("copyNumberOfProjectionSets(), c_NumberOfProjectionSets: %d\n", num_projection_sets);
}

extern "C" void copyVoxelSize(float3 voxel_size)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_VoxelSize_mm, &voxel_size, sizeof(float3)) );
}

extern "C" void copyVolumeCorner(float3 voxel_corner_mm)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_VolumeCorner_mm, &voxel_corner_mm, sizeof(float3)) );
}

extern "C" void copyProjectionDim(int width, int height)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_ImageWidth_i, &width, sizeof(int)) );
  cutilSafeCall( cudaMemcpyToSymbol(c_ImageHeight_i,&height, sizeof(int)) );
  float width_f = static_cast<float>(width), height_f = static_cast<float>(height);
  cutilSafeCall( cudaMemcpyToSymbol(c_ImageWidth, &width_f, sizeof(float)) );
  cutilSafeCall( cudaMemcpyToSymbol(c_ImageHeight,&height_f, sizeof(float)) );
}

extern "C" void copyTileSize(int tileX, int tileY)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_TileX, &tileX, sizeof(int), 0, cudaMemcpyHostToDevice ) );
  cutilSafeCall( cudaMemcpyToSymbol(c_TileY, &tileY, sizeof(int), 0, cudaMemcpyHostToDevice ) );
}

extern "C" void copyProjectionBlockSize(int block_size)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_ProjectionBlockSize_i, &block_size, sizeof(int), 0, cudaMemcpyHostToDevice ) );
}

extern "C" void copyStepSize(float step_size)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_StepSize_mm, &step_size, sizeof(float), 0, cudaMemcpyHostToDevice ) );
}

extern "C" void copyMemoryStoreMode(int memory_store_mode)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_MemoryStoreMode, &memory_store_mode, sizeof(int), 0, cudaMemcpyHostToDevice ) );
}

extern "C" void copyRayCastingLOD(int lod)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_RayCastingLOD, &lod, sizeof(int)) );
//  print_and_log("copyRayCastingLOD(%d)\n", lod);
}

extern "C" void copyRayCastingThreshold(float thresh)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_RayCastingThreshold, &thresh, sizeof(float)) );
}

extern "C" void copyRayCastingDistanceFalloffCoefficient(float coefficient)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_RayCastingDistanceFalloffCoefficient, &coefficient, sizeof(float)) );
}

extern "C" void copyCountNonIntersectedPixel(bool count_non_intersected_pixel)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_CountNonIntersectedPixel, &count_non_intersected_pixel, sizeof(bool)) );
}

extern "C" void copyDifferentVolumePerProjectionSet(bool different_volume_per_projection_set)
{
  cutilSafeCall( cudaMemcpyToSymbol(c_DifferentVolumePerProjectionSet, &different_volume_per_projection_set, sizeof(bool)) );
//  print_and_log("copyDifferentVolumePerProjectionSet(), %d\n", different_volume_per_projection_set);
}

extern "C" void print_all_constant_vars(void)
{
  // retrieve variables from constant memory on the device to the host memory
  int width, height, numberOfProjectionSets, tileX, tileY;
  float step_size;
  cutilSafeCall( cudaMemcpyFromSymbol(&width, c_ImageWidth_i, sizeof(int), 0, cudaMemcpyDeviceToHost ) );
  cutilSafeCall( cudaMemcpyFromSymbol(&height, c_ImageHeight_i, sizeof(int), 0, cudaMemcpyDeviceToHost ) );
  cutilSafeCall( cudaMemcpyFromSymbol(&numberOfProjectionSets, c_NumberOfProjectionSets, sizeof(int), 0, cudaMemcpyDeviceToHost ) );
  cutilSafeCall( cudaMemcpyFromSymbol(&tileX, c_TileX, sizeof(int), 0, cudaMemcpyDeviceToHost ) );
  cutilSafeCall( cudaMemcpyFromSymbol(&tileY, c_TileY, sizeof(int), 0, cudaMemcpyDeviceToHost ) );
  cutilSafeCall( cudaMemcpyFromSymbol(&step_size, c_StepSize_mm, sizeof(float), 0, cudaMemcpyDeviceToHost ) );
  cudaExtent volumeDim;
  cutilSafeCall( cudaMemcpyFromSymbol(&volumeDim, c_VolumeDim, sizeof(cudaExtent), 0, cudaMemcpyDeviceToHost ) );
  float3 voxel_size;
  cutilSafeCall( cudaMemcpyFromSymbol(&voxel_size, c_VoxelSize_mm, sizeof(float3), 0, cudaMemcpyDeviceToHost ) );

  // print
  print_and_log("Projection image size: (%d, %d, %d), tile: (%d, %d)\n", width, height, numberOfProjectionSets, tileX, tileY)
  print_and_log("Volume dimension: (%d, %d, %d), voxel size: (%f, %f, %f)\n", (int)volumeDim.width, (int)volumeDim.height, (int)volumeDim.depth, voxel_size.x, voxel_size.y, voxel_size.z)
  print_and_log("Step size: %f\n", step_size)
}

extern "C" void ComputeMaxMin(float *d_data, int size, float *h_maxValue, float *h_minValue)
{
  if(h_maxValue && h_minValue){
    thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> result;
    result = thrust::minmax_element(thrust::device_pointer_cast(d_data), thrust::device_pointer_cast(d_data) + size);
    (*h_maxValue) = *(result.second);
    (*h_minValue) = *(result.first);
//    print_and_log("min: %f, max: %f\n", (*h_minValue), (*h_maxValue));
  }
}

struct normalize_crop
{  
  const float maxVal, minVal;
  normalize_crop(float _maxVal, float _minVal) : maxVal(_maxVal), minVal(_minVal) {}

  __host__ __device__     
  float operator()(const float& x) const{
    float norm = (x-minVal)/(maxVal-minVal);
    return norm < 0.0f ? 0.0f : (norm > 1.0f ? 1.0f : norm);
  }  
};  

extern "C" void NormalizeData(float *d_data, int size, float h_normMax, float h_normMin)
{
  thrust::transform( thrust::device_ptr<float>(d_data), thrust::device_ptr<float>(d_data) + size 
                   , thrust::device_ptr<float>(d_data), normalize_crop(h_normMax, h_normMin) );
}

extern "C" void FillData(float *d_data, int size, float val)
{
  thrust::fill( thrust::device_ptr<float>(d_data), thrust::device_ptr<float>(d_data)+size, val );
}

extern "C" void FillData_i(int *d_data, int size, int val)
{
  thrust::fill( thrust::device_ptr<int>(d_data), thrust::device_ptr<int>(d_data)+size, val );
}

extern "C" void MultScalar(float *d_data, float scalar_val, int size)
{
  // data <- data * scalar_val
  thrust::transform( thrust::device_ptr<float>(d_data), thrust::device_ptr<float>(d_data)+size, thrust::constant_iterator<float>(scalar_val), 
                     thrust::device_ptr<float>(d_data), thrust::multiplies<float>() );
}

extern "C" void MultData(float *d_data1, float *d_data2, float *d_out, int size)
{
  // out <- data1 * data2
  thrust::transform( thrust::device_ptr<float>(d_data1), thrust::device_ptr<float>(d_data1) + size, 
                     thrust::device_ptr<float>(d_data2), thrust::device_ptr<float>(d_out), thrust::multiplies<float>());
}

extern "C" void randn_CURAND(curandGenerator_t generator, float *d_data, int size)
{
  CURAND_CALL( curandGenerateNormal(generator, d_data, size, 0.0, 1.0f) );
}

struct CMAESPopulation_clamp {
  template <typename Tuple>
  __host__ __device__
  float operator()(Tuple v) {
    // 0:arx, 1:lbounds, 2:ubounds -> arxvalid
    return fmaxf(thrust::get<1>(v), fminf(thrust::get<0>(v), thrust::get<2>(v)));
  } 
};

extern "C" void generateCMAESPopulation(float *d_arz, float *d_arx, float *d_arxvalid, float *d_xmean, float *d_diagD, float *d_lbounds, float *d_ubounds, float *d_OneVector, cublasHandle_t cublasHandle, int nRows, int nCols)
{
  // multiply diagD to d_arz and store to d_arx
//  print_and_log("generateCMAESPopulation(), nRows: %d, nCols:%d\n", nRows, nCols);
//  cudaMemcpy(d_arx, d_arz, nRows*nCols*sizeof(float), cudaMemcpyDeviceToDevice);
  cublasSdgmm(cublasHandle, CUBLAS_SIDE_LEFT, nRows, nCols, d_arz, nRows, d_diagD, 1, d_arx, nRows);

  // add xmean to each column in d_arx (column-wise addition)
  float alpha = 1.0f, beta = 1.0f;
  cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, nRows, nCols, 1, &alpha, d_xmean, nRows, d_OneVector, 1, &beta, d_arx, nRows);
  // clamp to bounds
  typedef thrust::device_vector<float>::iterator      Iterator;
  typedef thrust::tuple<Iterator, Iterator, Iterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple>         ZipIterator;
  typedef thrust::device_ptr<float>                   V;
  thrust::transform( ZipIterator(thrust::make_tuple( V(d_arx), V(d_lbounds), V(d_ubounds) )), 
                     ZipIterator(thrust::make_tuple( V(d_arx)+nRows*nCols, V(d_lbounds)+nRows*nCols, V(d_ubounds)+nRows*nCols )), 
                     V(d_arxvalid), CMAESPopulation_clamp() );
}

extern "C" void cudaMallocForTexture(struct cudaArray **d_projArray, cudaExtent extent)
{
//  print_and_log("cudaMallocForTexture(), extent: (%d, %d, %d)\n", extent.width, extent.height, extent.depth);
  cutilSafeCall( cudaMalloc3DArray(d_projArray, &d_texture_in.channelDesc, extent) );
}

extern "C" void bindCUDATexture(struct cudaArray *d_inputDataArray)
{
  // bind array to 3D texture
  if(d_inputDataArray){
    cutilSafeCall(cudaBindTextureToArray(d_texture_in, d_inputDataArray, d_texture_in.channelDesc));
    isTextureBound = true;
  }
}

extern "C" void unBindCUDATexture()
{
  if(isTextureBound){
    // unbind array from 3D texture
    cutilSafeCall(cudaUnbindTexture(d_texture_in));
    isTextureBound = false;
  }
}

extern "C" void bindWarpTexture(struct cudaArray *d_warpXArray, struct cudaArray *d_warpYArray, struct cudaArray *d_warpZArray)
{
  // bind array to 3D texture
  if(d_warpXArray && d_warpYArray && d_warpZArray){
    cutilSafeCall(cudaBindTextureToArray(d_texture_warpX, d_warpXArray, d_texture_warpX.channelDesc));
    cutilSafeCall(cudaBindTextureToArray(d_texture_warpY, d_warpYArray, d_texture_warpY.channelDesc));
    cutilSafeCall(cudaBindTextureToArray(d_texture_warpZ, d_warpZArray, d_texture_warpZ.channelDesc));
    isWarpTextureBound = true;
  }
}

extern "C" void unBindWarpTexture()
{
  if(isWarpTextureBound){
    // unbind array from 3D texture
    cutilSafeCall(cudaUnbindTexture(d_texture_warpX));
    cutilSafeCall(cudaUnbindTexture(d_texture_warpY));
    cutilSafeCall(cudaUnbindTexture(d_texture_warpZ));
    isWarpTextureBound = false;
  }
}

extern "C" void bindPrecomputedMatrixTexture(float *d_PrecomputedMatrix_array, int numProjections)
{
  // bind array to 3D texture
  if(d_PrecomputedMatrix_array)
    cutilSafeCall(cudaBindTexture(0, d_PrecomputedMatrix, d_PrecomputedMatrix_array, d_PrecomputedMatrix.channelDesc, numProjections*15*sizeof(float)));
}

extern "C" void unBindPrecomputedMatrixTexture()
{
  // unbind label texture array from 3D texture
  cutilSafeCall(cudaUnbindTexture(d_PrecomputedMatrix));
}

extern "C" void initCudaTexture(bool normalized, bool isLinear)
{
  // set texture parameters
  d_texture_in.normalized = normalized;
  if(isLinear)  d_texture_in.filterMode = cudaFilterModeLinear;    // linear interpolation
  else          d_texture_in.filterMode = cudaFilterModePoint;     // point (nearest-neighbor) interpolation
  d_texture_in.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
  d_texture_in.addressMode[1] = cudaAddressModeClamp;
  d_texture_in.addressMode[2] = cudaAddressModeClamp;

  d_PrecomputedMatrix.normalized = false;
  d_PrecomputedMatrix.filterMode = cudaFilterModePoint;       // always use point (nearest-neighbor) interpolation
  d_PrecomputedMatrix.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
  d_PrecomputedMatrix.addressMode[1] = cudaAddressModeClamp;
}

extern "C" void launch_LinearInterpolationProjector(float *d_projection_local, int *d_ZeroPixelCount, int number_of_projections_in_one_set, dim3 grid, dim3 block)
{
#if defined RegTools_VERBOSE_MESSAGE
  print_all_constant_vars();
  print_and_log("start linear interpolation projection\n");
#endif

  LinearInterpolationProjection<<< grid, block, 0 >>> (d_projection_local, d_ZeroPixelCount, number_of_projections_in_one_set);
}

extern "C" void launch_LinearInterpolationDeformableProjector(float *d_projection_local, int *d_ZeroPixelCount, int number_of_projections_in_one_set, dim3 grid, dim3 block)
{
#if defined RegTools_VERBOSE_MESSAGE
  print_all_constant_vars();
  print_and_log("start linear interpolation deformable projection\n");
#endif

  LinearInterpolationDeformableProjection<<< grid, block, 0 >>> (d_projection_local, d_ZeroPixelCount, number_of_projections_in_one_set);
}

extern "C" void launch_SiddonProjector(float *d_projection_local, size_t pitch, dim3 grid, dim3 block, int *d_random_sequence)
{
#if defined RegTools_VERBOSE_MESSAGE
  print_all_constant_vars();
  print_and_log("start Siddon projection\n");
#endif

  SiddonProjection<<< grid, block, 0 >>> (d_projection_local, pitch, d_random_sequence);
}

extern "C" void launch_RayCastingProjector(float *d_projection_local, dim3 grid, dim3 block)
{
#if defined RegTools_VERBOSE_MESSAGE
  print_all_constant_vars();
  print_and_log("start ray casting projection\n");
#endif

  RayCastingProjection<<< grid, block, 0 >>> (d_projection_local);
}

__device__ Ray computeNormalizedRay( const float x_pix, const float y_pix, const int p )
{
  // compute a unit vector connecting the source and the pixel (x_pix, y_pix) on the imaging plane using pre-computed corner points.
  Ray ray;
  float x = x_pix/c_ImageWidth, y = y_pix/c_ImageHeight;
  ray.d = normalize( make_float3( LOAD_PRE(p*12+0)+LOAD_PRE(p*12+3)*x+LOAD_PRE(p*12+6)*y, 
                                  LOAD_PRE(p*12+1)+LOAD_PRE(p*12+4)*x+LOAD_PRE(p*12+7)*y, 
                                  LOAD_PRE(p*12+2)+LOAD_PRE(p*12+5)*x+LOAD_PRE(p*12+8)*y ) );
  ray.o = make_float3( LOAD_PRE(p*12+9), LOAD_PRE(p*12+10), LOAD_PRE(p*12+11) );
  return ray;
}

#if !defined(RegTools_ENABLE_CUDA20_CAPABILITY_FEATURES)
// if 3D grid is not available, we tile the projection images (ray-tracing threads) in 2D
__device__ bool getPixelIndex(unsigned int &x, unsigned int &y, unsigned int &z)
{
  // compute index of the pixel on the projection image
  x = blockIdx.x * blockDim.x + threadIdx.x;  // locality-oriented
  y = blockIdx.y * blockDim.y + threadIdx.y;
  int tile_x = x/c_ImageWidth, tile_y = y/c_ImageHeight;
  // check if (tile_x,tile_y) index of the image in the large tiled image is exceeded (c_TileX, c_TileY) or not
  if(tile_x>=c_TileX || tile_y>=c_TileY)  return false;
  x = x % c_ImageWidth_i;
  y = y % c_ImageHeight_i;
  if(x>=c_ImageWidth_i || y>=c_ImageHeight_i) return false;
  z = tile_y * c_TileX + tile_x;    // index of the image
  return true;
}
#endif

__global__ void LinearInterpolationProjection(float* d_projection, int* d_ZeroPixelCount, int number_of_projections_in_one_set)
{
  // ray-driven simple linear interpolation-based forward projection
  unsigned int x, y, z;
#if defined(RegTools_ENABLE_CUDA20_CAPABILITY_FEATURES)
  x = blockIdx.x * blockDim.x + threadIdx.x;  // locality-oriented
  y = blockIdx.y * blockDim.y + threadIdx.y;
  z = blockIdx.z * blockDim.z + threadIdx.z;
  if(x>=c_ImageWidth_i || y>=c_ImageHeight_i || z>=c_ProjectionBlockSize_i) return; // this doesn't create divergent branches if number of pixels is multiple of warpSize
#else
  if(!getPixelIndex(x, y, z)) return;                                               // this doesn't create divergent branches if number of pixels is multiple of warpSize
#endif
  Ray ray = computeNormalizedRay(((float)x)+0.5f, ((float)y)+0.5f, z);
  float tnear, tfar, RPL = 0.0f;

  if(!intersectBoxRay(-c_VolumeCorner_mm, c_VolumeCorner_mm, ray, tnear, tfar)){
    if(c_CountNonIntersectedPixel)  atomicAdd(d_ZeroPixelCount+z, 1);
    return;
  }

  float z_offset = 0;
  if(c_DifferentVolumePerProjectionSet){
    // TODO: clean-up this interface
    // use different volume for each projection set (size of the volume needs to be nx x ny x nz*numProjectionSets
    z_offset = floor((float)z/(float)number_of_projections_in_one_set) * c_VolumeDim_f.z;    
  }

  // compute Radiological Path Length (RPL) by trilinear interpolation (texture fetching)
  float3 cur = (ray.o + tnear * ray.d + c_VolumeCorner_mm) / c_VoxelSize_mm;  // object coordinate (mm) -> texture (voxel) coordinate
  float3 delta_dir = c_StepSize_mm * ray.d / c_VoxelSize_mm;                 // object coordinate (mm) -> texture (voxel) coordinate
  for(float travelled_length = 0; travelled_length < (tfar-tnear); travelled_length += c_StepSize_mm, cur += delta_dir){
    // pick the density value at the current point and accumulate it (Note: currently consider only single input volume case)
    RPL += tex3D(d_texture_in, cur.x, cur.y, cur.z + z_offset) * c_StepSize_mm;   // access to register memory and texture memory (filterMode of texture should be 'linear')
  }

  if(c_MemoryStoreMode == MemoryStoreMode_Replace || c_MemoryStoreMode == MemoryStoreMode_Additive)
    d_projection[z*c_ImageWidth_i*c_ImageHeight_i+y*c_ImageWidth_i+x] += RPL;    // access to global memory
  else if(c_MemoryStoreMode == MemoryStoreMode_Multiplicative)
    d_projection[z*c_ImageWidth_i*c_ImageHeight_i+y*c_ImageWidth_i+x] *= RPL;    // access to global memory
}

__global__ void LinearInterpolationDeformableProjection(float* d_projection, int* d_ZeroPixelCount, int number_of_projections_in_one_set)
{
  // ray-driven simple linear interpolation-based forward projection
  unsigned int x, y, z;
#if defined(RegTools_ENABLE_CUDA20_CAPABILITY_FEATURES)
  x = blockIdx.x * blockDim.x + threadIdx.x;  // locality-oriented
  y = blockIdx.y * blockDim.y + threadIdx.y;
  z = blockIdx.z * blockDim.z + threadIdx.z;
  if(x>=c_ImageWidth_i || y>=c_ImageHeight_i || z>=c_ProjectionBlockSize_i) return; // this doesn't create divergent branches if number of pixels is multiple of warpSize
#else
  if(!getPixelIndex(x, y, z)) return;                                               // this doesn't create divergent branches if number of pixels is multiple of warpSize
#endif
  Ray ray = computeNormalizedRay(((float)x)+0.5f, ((float)y)+0.5f, z);
  float tnear, tfar, RPL = 0.0f;
//  intersectBoxRay(-c_VolumeCorner_mm, c_VolumeCorner_mm, ray, tnear, tfar);
  if(!intersectBoxRay(-c_VolumeCorner_mm, c_VolumeCorner_mm, ray, tnear, tfar)){
    if(c_CountNonIntersectedPixel)  atomicAdd(d_ZeroPixelCount+z, 1);
    return;
  }

  // consider every (number_of_projections) projections as a set
  float volume_index = floor((float)z/(float)number_of_projections_in_one_set); // this is the index of deformation volume

  // compute Radiological Path Length (RPL) by trilinear interpolation (texture fetching)
  float3 cur = (ray.o + tnear * ray.d + c_VolumeCorner_mm) / c_VoxelSize_mm;  // object coordinate (mm) -> texture (voxel) coordinate
  float3 delta_dir = c_StepSize_mm * ray.d / c_VoxelSize_mm;                 // object coordinate (mm) -> texture (voxel) coordinate
  for(float travelled_length = 0; travelled_length < (tfar-tnear); travelled_length += c_StepSize_mm, cur += delta_dir){
    // pick the density value at the current point and accumulate it (Note: currently consider only single input volume case)
    // compute transformation
    RPL += tex3D(d_texture_in, cur.x + tex3D(d_texture_warpX, cur.x, cur.y, cur.z + volume_index*c_VolumeDim_f.z) + 0.5
                             , cur.y + tex3D(d_texture_warpY, cur.x, cur.y, cur.z + volume_index*c_VolumeDim_f.z) + 0.5
                             , cur.z + tex3D(d_texture_warpZ, cur.x, cur.y, cur.z + volume_index*c_VolumeDim_f.z) + 0.5) * c_StepSize_mm;   // access to register memory and texture memory (filterMode of texture should be 'linear')
  }
  if(c_MemoryStoreMode == MemoryStoreMode_Replace || c_MemoryStoreMode == MemoryStoreMode_Additive)
    d_projection[z*c_ImageWidth_i*c_ImageHeight_i+y*c_ImageWidth_i+x] += RPL;    // access to global memory
  else if(c_MemoryStoreMode == MemoryStoreMode_Multiplicative)
    d_projection[z*c_ImageWidth_i*c_ImageHeight_i+y*c_ImageWidth_i+x] *= RPL;    // access to global memory
}

#define MAX_RAYCASTING_LOD  20  // if this number is set to a huge number, it crashes GPU seriously. Thus we need a maximum limit.
__global__ void RayCastingProjection(float* d_projection)
{
  // ray casting forward projection
  unsigned int x, y, z;
#if defined(RegTools_ENABLE_CUDA20_CAPABILITY_FEATURES)
  x = blockIdx.x * blockDim.x + threadIdx.x;  // locality-oriented
  y = blockIdx.y * blockDim.y + threadIdx.y;
  z = blockIdx.z * blockDim.z + threadIdx.z;
  if(x>=c_ImageWidth_i || y>=c_ImageHeight_i || z>=c_ProjectionBlockSize_i) return; // this doesn't create divergent branches if number of pixels is multiple of warpSize
#else
  if(!getPixelIndex(x, y, z)) return;                                               // this doesn't create divergent branches if number of pixels is multiple of warpSize
#endif

  Ray ray = computeNormalizedRay(((float)x)+0.5f, ((float)y)+0.5f, z);
  ray.d = -ray.d;
  float tnear_mm, tfar_mm;
  if(!intersectBoxRay(-c_VolumeCorner_mm, c_VolumeCorner_mm, ray, tnear_mm, tfar_mm)) return;
  if(tnear_mm<0){
    if(tfar_mm<0){  // the ray goes backward
      d_projection[z*c_ImageWidth_i*c_ImageHeight_i+y*c_ImageWidth_i+x] = 0; return;
    } else {        // the ray starts from inside the volume 
      tnear_mm = 0;
    }
  }

  // find the first voxel that hits to the ray
  float3 cur = (ray.o + tnear_mm * ray.d + c_VolumeCorner_mm) / c_VoxelSize_mm; // object coordinate (mm) -> texture (voxel) coordinate
  float3 delta_dir = c_StepSize_mm * ray.d / c_VoxelSize_mm;                    // object coordinate (mm) -> texture (voxel) coordinate

  float cur_val = 0.0f, prev_val = 0.0f, alpha = 1.0f, finite_diff = 0.5f;
  for(float travelled_length_mm = 0; travelled_length_mm < (tfar_mm-tnear_mm); travelled_length_mm += c_StepSize_mm, cur += delta_dir){
    if((cur_val = tex3D(d_texture_in, cur.x, cur.y, cur.z)) > c_RayCastingThreshold){
      float3 half_step = delta_dir;
      for(int count=0;count<c_RayCastingLOD && count<MAX_RAYCASTING_LOD;count++){
        half_step /= 2;
        cur -= half_step;
        float temp = cur_val;
        if((cur_val = tex3D(d_texture_in, cur.x, cur.y, cur.z)) < c_RayCastingThreshold){
          prev_val = cur_val;
          cur_val = temp;
          cur += half_step;
        }
      }
      float offset_ratio = (cur_val-c_RayCastingThreshold)/(cur_val-prev_val);// find the threshold surface by linear interpolation between previous and current intensty
      cur = cur - half_step * offset_ratio;                                   // surface point
      float R = tnear_mm + travelled_length_mm - c_StepSize_mm*offset_ratio;  // distance between camera center and the surface
      float3 intensity_grad = normalize( make_float3(
        tex3D(d_texture_in, cur.x+finite_diff, cur.y, cur.z)-tex3D(d_texture_in, cur.x-finite_diff, cur.y, cur.z), 
        tex3D(d_texture_in, cur.x, cur.y+finite_diff, cur.z)-tex3D(d_texture_in, cur.x, cur.y-finite_diff, cur.z),
        tex3D(d_texture_in, cur.x, cur.y, cur.z+finite_diff)-tex3D(d_texture_in, cur.x, cur.y, cur.z-finite_diff) ) );
      if(travelled_length_mm==0){
        d_projection[z*c_ImageWidth_i*c_ImageHeight_i+y*c_ImageWidth_i+x] = 1.0;
      } else if(travelled_length_mm>0){
        float dot_prod = dot(ray.d, intensity_grad);
        if(dot_prod<0) dot_prod = dot(ray.d, -intensity_grad);
        if(isnan(dot_prod)) dot_prod = 0.0f;
        if(c_RayCastingDistanceFalloffCoefficient>0)
          d_projection[z*c_ImageWidth_i*c_ImageHeight_i+y*c_ImageWidth_i+x] = fminf( c_RayCastingDistanceFalloffCoefficient * dot_prod / (M_PI * R * R), 1.0f);
        else
          d_projection[z*c_ImageWidth_i*c_ImageHeight_i+y*c_ImageWidth_i+x] = dot_prod;
      }
      break;
    }
    prev_val = cur_val;
  }
}

#define MAX_NUM_PROJECTION_SETS 1
__global__ void SiddonProjection(float* d_projection, size_t pitch, int *d_random_sequence)
//__global__ void SiddonProjection(cudaPitchedPtr d_projection, int *d_random_sequence) // attempt to use "pitched" array -> slower on GTX470
{
  // ray-driven Siddon projector
  // see, for example,
  // Siddon RL. , "Fast calculation of the exact radiological path for a three-dimensional CT array," Med.Phys. Mar-Apr 12(2), 252-255 (1985).
  if(d_projection == NULL)  return;
  unsigned int x, y, z;

#if defined(RegTools_ENABLE_CUDA20_CAPABILITY_FEATURES)
  x = blockIdx.x * blockDim.x + threadIdx.x;  // locality-oriented
  y = blockIdx.y * blockDim.y + threadIdx.y;
  z = blockIdx.z * blockDim.z + threadIdx.z;
  if(x>=c_ImageWidth_i || y>=c_ImageHeight_i || z>=c_ProjectionBlockSize_i) return; // this doesn't create divergent branches if number of pixels is multiple of warpSize
#else
  if(!getPixelIndex(x, y, z)) return;                                               // this doesn't create divergent branches if number of pixels is multiple of warpSize
#endif
  
  unsigned int pixel_index = z*c_ImageWidth_i*c_ImageHeight_i + y*c_ImageWidth_i + x;
  float3 volume_corner = (c_VolumeDim_f * c_VoxelSize_mm) / 2;
  float tnear, tfar;
  int near_plane;
  Ray ray = computeNormalizedRay(((float)x)+0.5f, ((float)y)+0.5f, z);

  if(!intersectBoxRay_withPlane(-volume_corner, volume_corner, ray, tnear, tfar, near_plane)) return; 
  float total_len_mm = tfar-tnear;       // distance between front and back in "mm" unit

  // delta: one step length that the ray walk throught a voxel. normalized by back-front distance.
  // This means that when the ray walk until it intersects with next Y-Z plane, the "alpha (normalized walk length)" is increased delta.x.
  // Similarly for X-Y plane (delta.z), for X-Z plane (delta.y).
  float3 delta = make_float3(1.0f / total_len_mm) / fabs(ray.d);

  float3 front = (ray.o + tnear * ray.d + volume_corner) / c_VoxelSize_mm;  // position of the front point in volume coordinate (normalized by the voxel size)
  float3 voxel_index_int = floor( front ); // "integer" index of the first voxel to be traced
  front = front - voxel_index_int;         // residual (move origin to the corner of voxel)
  voxel_index_int = clamp( voxel_index_int, make_float3(0.0f, 0.0f, 0.0f), make_float3(c_VolumeDim.width-1, c_VolumeDim.height-1, c_VolumeDim.depth-1) );

  // find the initial value of "alphaNext" (next voxel edge)
  float3 alphaNext;
  if(near_plane == 0){        // ray is crossed with Y-Z plane
    // the following two lines are needed to increase numerical precision (dividing and multiplying by voxel size result unstable +/-1.0f)
    voxel_index_int.x = (ray.d.x>0) ? 0 : (c_VolumeDim.width-1);
    alphaNext.x = 1.0;       
    alphaNext.y = (ray.d.y>0) ? (1.0f-front.y) : front.y;
    alphaNext.z = (ray.d.z>0) ? (1.0f-front.z) : front.z;
  } else if(near_plane == 1){ // ray is crossed with Z-X plane
    voxel_index_int.y = (ray.d.y>0) ? 0 : (c_VolumeDim.height-1);
    alphaNext.x = (ray.d.x>0) ? (1.0f-front.x) : front.x;
    alphaNext.y = 1.0f;
    alphaNext.z = (ray.d.z>0) ? (1.0f-front.z) : front.z;
  } else {                    // ray is crossed with X-Y plane
    voxel_index_int.z = (ray.d.z>0) ? 0 : (c_VolumeDim.depth-1);
    alphaNext.x = (ray.d.x>0) ? (1.0f-front.x) : front.x;
    alphaNext.y = (ray.d.y>0) ? (1.0f-front.y) : front.y;
    alphaNext.z = 1.0f;
  }

  alphaNext = alphaNext * delta * c_VoxelSize_mm; // convert to 'mm' unit
  delta = delta * c_VoxelSize_mm;                 // convert to 'mm' unit

  float detector_value = 0.0f;

  float line_segment_length_normalized;   // length of the line segment that the ray has travelled through ONE VOXEL (in normalized coordinate)
  float alphaCurrent = 0.0f;
  float3 next_voxel_index = voxel_index_int;
  float3 voxel_index_inc = make_float3( copysign( 1.0f, ray.d.x ), copysign( 1.0f, ray.d.y ), copysign( 1.0f, ray.d.z ) );
//  if(c_NumberOfLabels<=0){
    // non voxel-selective (conventional) projection

    float RPL = 0.0f;
    for(bool loop_end = false; !loop_end ; ){
      if(((alphaNext.x < alphaNext.y) && (alphaNext.x < alphaNext.z))){
        // alphaNext.x is minimum (ray intersects Y-Z plane), move one voxel in X direction
        line_segment_length_normalized = (alphaNext.x - alphaCurrent);
        alphaCurrent = alphaNext.x;
        alphaNext.x += delta.x;             // next intersection with Y-Z plane
        next_voxel_index.x += voxel_index_inc.x;
        if(next_voxel_index.x>=c_VolumeDim.width || next_voxel_index.x<0)  loop_end = true;
      } else if( (alphaNext.y < alphaNext.z) ) {
        // alphaNext.y is minimum (ray intersects X-Z plane), move one voxel in Y direction
        line_segment_length_normalized = (alphaNext.y - alphaCurrent);
        alphaCurrent = alphaNext.y;
        alphaNext.y += delta.y;             // next intersection with X-Z plane
        next_voxel_index.y += voxel_index_inc.y;
        if(next_voxel_index.y>=c_VolumeDim.height || next_voxel_index.y<0) loop_end = true;
      } else {
        // alphaNext.z is minimum (ray intersects X-Y plane), move one voxel in Z direction
        line_segment_length_normalized = (alphaNext.z - alphaCurrent);
        alphaCurrent = alphaNext.z;
        alphaNext.z += delta.z;             // next intersection with X-Y plane
        next_voxel_index.z += voxel_index_inc.z;
        if(next_voxel_index.z>=c_VolumeDim.depth || next_voxel_index.z<0)  loop_end = true;
      }

      float intersection_length = line_segment_length_normalized * total_len_mm;
//      if(c_ProjectorComputationMode == ProjectorComputationMode_Square) intersection_length *= intersection_length;
//      if(c_ProcessingMode == ProcessingMode_ForwardProjection){
        // fetch the intensity from 3D texture memory and accumulate (access to register memory and texture memory)
        RPL += tex3D(d_texture_in, voxel_index_int.x+0.5f, voxel_index_int.y+0.5f, voxel_index_int.z+0.5f) * intersection_length;
//      }
      voxel_index_int = next_voxel_index;
    }

//    if(c_ProcessingMode == ProcessingMode_ForwardProjection){
      if(c_MemoryStoreMode == MemoryStoreMode_Replace || c_MemoryStoreMode == MemoryStoreMode_Additive)
        d_projection[pixel_index] += RPL;   // access to global memory and texture memory
      else if(c_MemoryStoreMode == MemoryStoreMode_Multiplicative)
        d_projection[pixel_index] *= RPL;   // access to global memory and texture memory
//    }
  /*} else {

  // voxel-selective projection (in case of foward projection, we accumulate in local array to save global memory access)
    float RPLs[MAX_NUMBER_OF_LABELS];   // temporary space to accumulate one ray used for forward projection (only fixed size array is allowed in device function)
    int num_labels = min(c_NumberOfLabels, MAX_NUMBER_OF_LABELS);
    for(int i=0;i<num_labels;i++) RPLs[i] = 0.0f;   // initialize

    for(bool loop_end = false; !loop_end ; ){
      if(((alphaNext.x < alphaNext.y) && (alphaNext.x < alphaNext.z))){
        // alphaNext.x is minimum (ray intersects Y-Z plane), move one voxel in X direction
        line_segment_length_normalized = (alphaNext.x - alphaCurrent);
        alphaCurrent = alphaNext.x;
        alphaNext.x += delta.x;             // next intersection with Y-Z plane
        next_voxel_index.x += voxel_index_inc.x;
        if(next_voxel_index.x>=c_VolumeDim.width || next_voxel_index.x<0)  loop_end = true;
      } else if( (alphaNext.y < alphaNext.z) ) {
        // alphaNext.y is minimum (ray intersects X-Z plane), move one voxel in Y direction
        line_segment_length_normalized = (alphaNext.y - alphaCurrent);
        alphaCurrent = alphaNext.y;
        alphaNext.y += delta.y;             // next intersection with X-Z plane
        next_voxel_index.y += voxel_index_inc.y;
        if(next_voxel_index.y>=c_VolumeDim.height || next_voxel_index.y<0) loop_end = true;
      } else {
        // alphaNext.z is minimum (ray intersects X-Y plane), move one voxel in Z direction
        line_segment_length_normalized = (alphaNext.z - alphaCurrent);
        alphaCurrent = alphaNext.z;
        alphaNext.z += delta.z;             // next intersection with X-Y plane
        next_voxel_index.z += voxel_index_inc.z;
        if(next_voxel_index.z>=c_VolumeDim.depth || next_voxel_index.z<0)  loop_end = true;
      }

      float intersection_length = line_segment_length_normalized * total_len_mm;
//      if(c_ProjectorComputationMode == ProjectorComputationMode_Square) intersection_length *= intersection_length;

      int label = (int)(tex3D(d_label_texture, voxel_index_int.x+0.5f, voxel_index_int.y+0.5f, voxel_index_int.z+0.5f));
      if(label>0 && label<=num_labels){
        if(c_ProcessingMode == ProcessingMode_ForwardProjection){
          // fetch the intensity from 3D texture memory and accumulate (access to register memory and texture memory)
            RPLs[label-1] += tex3D(d_texture_in, voxel_index_int.x+0.5f, voxel_index_int.y+0.5f, voxel_index_int.z+0.5f) * intersection_length;
        }
      }
      voxel_index_int = next_voxel_index;
    }
    if(c_ProcessingMode == ProcessingMode_ForwardProjection){
      for(int i=0;i<num_labels;i++){
        if(c_MemoryStoreMode == MemoryStoreMode_Replace || c_MemoryStoreMode == MemoryStoreMode_Additive)
          d_projection[pixel_index + i * c_ImageWidth_i * c_ImageHeight_i * (int)c_ProjectionBlockSize_f] += RPLs[i];    // access to global memory
        else if(c_MemoryStoreMode == MemoryStoreMode_Multiplicative)
          d_projection[pixel_index + i * c_ImageWidth_i * c_ImageHeight_i * (int)c_ProjectionBlockSize_f] *= RPLs[i];   // access to global memory
      }
    }
  }
    */
}

#include "Interpolator.cuh"

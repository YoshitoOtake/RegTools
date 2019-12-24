/*
  Author(s):  Ali Uneri, Sureerat Reaungamornrat, Yoshito Otake
  Created on: 2010-11-11
*/

#include <fstream>
#include <math.h>
#include <cuda.h>

__constant__ static float d_translation[3];        // translation vector
__constant__ static float d_r_mat[9];              // rotation matrix
__constant__ static float d_in_voxel_size[3];      // voxel size of the input volume
__constant__ static float d_in_volume_center[3];   // volume center of the input volume (unit in 'mm')
__constant__ static float d_out_volume_center[3];  // volume center of the output volume (unit in 'mm')
__constant__ static int d_in_volume_dim[3];        // dimensions of the input volume (unit in 'voxel')
__constant__ static unsigned int d_bbox_max[3], d_bbox_min[3], d_bbox_size[3]; // bounding box of the input volume (unit in 'voxel')

__device__ float sgn(const float &x) {
  if (x > 0.0f) return 1.0f;
  else if (x < 0.0f) return -1.0f;
  else return 0.0f;
}

__device__ float bicubic_K(const float &p, const float a) {
  const float pabs = fabsf(p);
  if (pabs <= 1.0f) {
    return ((a + 2.0f) * pabs * pabs * pabs) - ((a + 3.0f) * pabs * pabs) + 1.0f;
  } else if (pabs <= 2.0f) {
    return (a * pabs * pabs * pabs) - (5.0f * a * pabs * pabs) + (8.0f * a * pabs) - (4.0f * a);
  } else {
    return 0.0f;
  }
}

__device__ float bicubic_Kd(const float &p, const float a) {
  const short psgn = sgn(p);
  const float pabs = fabsf(p);
  if (pabs <= 1.0f) {
    return (3.0f * (a + 2.0f) * pabs * pabs * psgn) - (2.0f * (a + 3.0f) * p);
  } else if (pabs <= 2.0f) {
    return (3.0f * a * pabs * pabs * psgn) - (10.0f * a * p) + (8.0f * a * psgn);
  } else {
    return 0.0f;
  }
}

__device__ float bspline_K(const float &p) {
  const float pabs = fabsf(p);
  if (pabs <= 1.0f) {
    return (2.0f / 3.0f) - ((1 / 2.0f) * pabs * pabs * (2 - pabs));
  } else if (pabs <= 2.0f) {
    return (1.0f / 6.0f) * (2 - pabs) * (2 - pabs) * (2 - pabs);
  } else {
    return 0.0f;
  }
}

__device__ float bspline_Kd(const float &p) {
  const short psgn = sgn(p);
  const float pabs = fabsf(p);
  if (pabs <= 1.0f) {
    return ((pabs * pabs * psgn) / 2.0f) - (p * (2 - pabs));
  } else if (pabs <= 2.0f) {
    return -(1.0f / 2.0f) * (2 - pabs)*(2 - pabs) * psgn;
  } else {
    return 0.0f;
  }
}

__device__ float filter(const float (&v)[3], const int type, const int order, const float bicubic_a) {
  int vn [3];          // rounded voxel indices
  float offset;        // offset from local grid center
  float w [3][4];      // weights
  float b [3][4];      // interpolated values
  float result = 0.0;  // resulting interpolated voxel value

  // compute weights
  for (short i = 0; i < 3; ++i) {
    vn[i] = (int)floorf(v[i]);
    offset = 0.5f - (v[i] - vn[i]);
    for (short j = 0; j < 4; ++j) {
      if (i == (order - 1)) {
        if (type == InterpolatorType_Bicubic) {
          w[i][j] = bicubic_Kd(offset - 1.5f + j, bicubic_a);
        } else if (type == InterpolatorType_Bspline) {
          w[i][j] = bspline_Kd(offset - 1.5f + j);
        } else {
          w[i][j] = 0.0f;
        }
      } else {
        if (type == InterpolatorType_Bicubic) {
          w[i][j] = bicubic_K(offset - 1.5f + j, bicubic_a);
        } else if (type == InterpolatorType_Bspline) {
          w[i][j] = bspline_K(offset - 1.5f + j);
        } else {
          w[i][j] = 0.0f;
        }
      }
    }
  }

  // take weighted sum of interpolated values
#pragma unroll
  for (short i = 0; i < 4; ++i) {
    b[2][i] = 0.0f;
#pragma unroll
    for (short j = 0; j < 4; ++j) {
      b[1][j] = 0.0f;
#pragma unroll
      for (short k = 0; k < 4; ++k) {
        b[0][k] = tex3D(d_texture_in, vn[0]-1+k, vn[1]-1+j, vn[2]-1+i);
        b[1][j] += w[0][k] * b[0][k];
      }
      b[2][i] += w[1][j] * b[1][j];
    }
    result += w[2][i] * b[2][i];
  }
  return result;
}

__device__ bool getVoxelIndex(unsigned int &x, unsigned int &y, unsigned int &z)
{
  x = blockIdx.x * blockDim.x + threadIdx.x;  // locality-oriented
  y = blockIdx.y * blockDim.y + threadIdx.y;
  z = blockIdx.z * blockDim.z + threadIdx.z;
  if(x>=c_VolumeDim.width || y>=c_VolumeDim.height || z>=c_VolumeDim.depth) 
    return false; // check if the target voxel is not out of the volume
  else 
    return true;
}

__device__ float filter_image(float *data_out, unsigned int v[3], const float (&vt)[3], const int type, const int order, const float bicubic_a)
{
  // filter image
  if(vt[0] >= 0 && vt[0] < d_in_volume_dim[0] && vt[1] >= 0 && vt[1] < d_in_volume_dim[1] && vt[2] >= 0 && vt[2] < d_in_volume_dim[2]){    
    if (type == InterpolatorType_Bilinear || type == InterpolatorType_NearestNeighbor)
      data_out[v[2]*c_VolumeDim.width*c_VolumeDim.height + v[1]*c_VolumeDim.width + v[0]] = tex3D(d_texture_in, vt[0], vt[1], vt[2]);
    else                                                                              
      data_out[v[2]*c_VolumeDim.width*c_VolumeDim.height + v[1]*c_VolumeDim.width + v[0]] = filter(vt, type, order, bicubic_a);
  }
}

__global__ void interpolator_kernel(float *data_out, const int type, const int order, const float bicubic_a)
{
  // type
  // 0: bilinear interpolation
  // 1: bicubic interpolation
  // 2: B-spline interpolation
  // 5: nearest-neighbor interpolation
  //

  unsigned int v [3]; // voxel coordinates
  float vt [3];       // transformed voxel coordinates

  // compute voxel coordinates
#if defined(RegTools_ENABLE_CUDA20_CAPABILITY_FEATURES)
  if(!getVoxelIndex(v[0], v[1], v[2])) return;
#else
  // not supported (GPU needs to have >2.0 compute capability for 3D grid)
  return;
#endif

  // compute transformation
  float p[3] = { ((float)v[0])*c_VoxelSize_mm.x-d_out_volume_center[0], ((float)v[1])*c_VoxelSize_mm.y-d_out_volume_center[1], ((float)v[2])*c_VoxelSize_mm.z-d_out_volume_center[2] };
  vt[0] = d_r_mat[0]*p[0] + d_r_mat[3]*p[1] + d_r_mat[6]*p[2] + d_translation[0] + d_in_volume_center[0];
  vt[1] = d_r_mat[1]*p[0] + d_r_mat[4]*p[1] + d_r_mat[7]*p[2] + d_translation[1] + d_in_volume_center[1];
  vt[2] = d_r_mat[2]*p[0] + d_r_mat[5]*p[1] + d_r_mat[8]*p[2] + d_translation[2] + d_in_volume_center[2];
  vt[0] /= d_in_voxel_size[0]; vt[1] /= d_in_voxel_size[1]; vt[2] /= d_in_voxel_size[2];  // convert unit from 'mm' to 'voxel'

  filter_image(data_out, v, vt, type, order, bicubic_a);
}

__global__ void warp_kernel(float *data_out, const int type, const int order, const float bicubic_a, const int nz)
{
  // type
  // 0: bilinear interpolation
  // 1: bicubic interpolation
  // 2: B-spline interpolation
  // 5: nearest-neighbor interpolation
  //

  unsigned int v [3]; // voxel coordinates
  float vt [3];       // transformed voxel coordinates

  // not supported (GPU needs to have >2.0 compute capability for 3D grid)
#if defined(RegTools_ENABLE_CUDA20_CAPABILITY_FEATURES)
  if(!getVoxelIndex(v[0], v[1], v[2])) return;
#else
  // not supported  
  return;
#endif

  for(int i=0;i<ceil((float)nz/(float)c_VolumeDim.depth);i++, v[2] += c_VolumeDim.depth){
    // compute transformation
    // if (nz < c_VolumeDim.depth), the transformation is applied to the same volume (nz slices) multiple times
    // if (nz > c_VolumeDim.depth), a set of nz slices is considered as one volume
    vt[0] = (float)v[0] + tex3D(d_texture_warpX, v[0], v[1], v[2]) + 0.5;
    vt[1] = (float)v[1] + tex3D(d_texture_warpY, v[0], v[1], v[2]) + 0.5;
    vt[2] = fmodf((float)v[2],(float)nz) + tex3D(d_texture_warpZ, v[0], v[1], v[2]) + 0.5;
    if(vt[2]<i*c_VolumeDim.depth || vt[2]>(i+1)*c_VolumeDim.depth)  continue;   // out of the volume (to avoid pulling a voxel in the neighboring volume)

    filter_image(data_out, v, vt, type, order, bicubic_a);
  }
}

__global__ void warp_trans_kernel(float *data_out, const int type, const int order, const float bicubic_a, const int nz, float *transform)
{
  unsigned int v [3]; // voxel coordinates
  float vt[3], tw[3]; // transformed voxel coordinates
  if(!getVoxelIndex(v[0], v[1], v[2])) return;
  
  // concatenate warp and rigid transformation
  float v2 = fmodf((float)v[2],nz); // multiple volumes are stucked along Z-direction
  int id = v[2]/nz;

  float p[3] = { ((float)v[0])*c_VoxelSize_mm.x-d_out_volume_center[0], ((float)v[1])*c_VoxelSize_mm.y-d_out_volume_center[1], ((float)v2)*c_VoxelSize_mm.z-d_out_volume_center[2] };
  tw[0] = transform[0+id*12]*p[0] + transform[3+id*12]*p[1] + transform[6+id*12]*p[2] + transform[9 +id*12] + d_in_volume_center[0];
  tw[1] = transform[1+id*12]*p[0] + transform[4+id*12]*p[1] + transform[7+id*12]*p[2] + transform[10+id*12] + d_in_volume_center[1];
  tw[2] = transform[2+id*12]*p[0] + transform[5+id*12]*p[1] + transform[8+id*12]*p[2] + transform[11+id*12] + d_in_volume_center[2];
  tw[0] /= d_in_voxel_size[0]; tw[1] /= d_in_voxel_size[1]; tw[2] /= d_in_voxel_size[2];  // convert unit from 'mm' to 'voxel'
  tw[0] -= v[0]; tw[1] -= v[1]; tw[2] -= v2;

  for(int i=0;i<ceil((float)nz/(float)c_VolumeDim.depth);i++, v[2] += c_VolumeDim.depth){
    // compute transformation
    // if (nz < c_VolumeDim.depth), the transformation is applied to the same volume (nz slices) multiple times
    // if (nz > c_VolumeDim.depth), a set of nz slices is considered as one volume
    vt[0] = (float)v[0] + tw[0] + tex3D(d_texture_warpX, v[0]+tw[0], v[1]+tw[1], v[2]+tw[2]) + 0.5;
    vt[1] = (float)v[1] + tw[1] + tex3D(d_texture_warpY, v[0]+tw[0], v[1]+tw[1], v[2]+tw[2]) + 0.5;
    vt[2] = v2          + tw[2] + tex3D(d_texture_warpZ, v[0]+tw[0], v[1]+tw[1], v[2]+tw[2]) + 0.5;
    if(vt[2]<i*c_VolumeDim.depth || vt[2]>(i+1)*c_VolumeDim.depth)  continue;   // out of the volume (to avoid pulling a voxel in the neighboring volume)

    //  data_out[v[2]*c_VolumeDim.width*c_VolumeDim.height + v[1]*c_VolumeDim.width + v[0]] = id;
    filter_image(data_out, v, vt, type, order, bicubic_a);
  }
}

__global__ void scattered_point_interpolation_kernel(float *d_out_pnts, const float *d_in_pnts, int num_pnts)
{
  // currently, only support bilinear interpolation
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x>=num_pnts) return;
  d_out_pnts[x*3+0] = tex3D(d_texture_warpX, d_in_pnts[x*3+0], d_in_pnts[x*3+1], d_in_pnts[x*3+2]);
  d_out_pnts[x*3+1] = tex3D(d_texture_warpY, d_in_pnts[x*3+0], d_in_pnts[x*3+1], d_in_pnts[x*3+2]);
  d_out_pnts[x*3+2] = tex3D(d_texture_warpZ, d_in_pnts[x*3+0], d_in_pnts[x*3+1], d_in_pnts[x*3+2]);
}

extern "C" void launch_Interpolator_BBoxCheck(float* d_data_out, int in_volumeDim[3], double in_voxelSize[3], const float* transform, int num_transform_element
                                                , const int type, const int order, const float bicubic_a, float back_ground_value, float *volume_center
                                                , float *scattered_pnts, int num_scattered_pnts, bool isWarp, int num_transforms)
{
//  print_and_log("num_scattered_pnts %d\n", num_scattered_pnts);
  if(num_scattered_pnts>0){
    d_texture_warpX.filterMode = cudaFilterModeLinear;
    d_texture_warpY.filterMode = cudaFilterModeLinear;
    d_texture_warpZ.filterMode = cudaFilterModeLinear;
    dim3 block(512,1,1);
    dim3 grid(iDivUp(num_scattered_pnts, block.x), 1, 1);
//    print_and_log("scattered_point_interpolation, number of points: %d, grid: (%d,%d,%d), block: (%d,%d,%d)\n", num_scattered_pnts, grid.x, grid.y, grid.z, block.x, block.y, block.z);
    scattered_point_interpolation_kernel<<<grid, block>>>(d_data_out, scattered_pnts, num_scattered_pnts);
    return;
  }
  cudaExtent vol_dim;
  float3 voxel_size;
  cutilSafeCall( cudaMemcpyFromSymbol(&vol_dim, c_VolumeDim, sizeof(cudaExtent), 0, cudaMemcpyDeviceToHost ) );
  unsigned int out_volumeDim[3] = {vol_dim.width, vol_dim.height, vol_dim.depth };
  cutilSafeCall( cudaMemcpyFromSymbol(&voxel_size, c_VoxelSize_mm, sizeof(float3), 0, cudaMemcpyDeviceToHost ) );
//  print_and_log("in volumeDim: %d, %d, %d\n", in_volumeDim[0], in_volumeDim[1], in_volumeDim[2]);
//  print_and_log("out volumeDim: %d, %d, %d\n", out_volumeDim[0], out_volumeDim[1], out_volumeDim[2]);
  thrust::fill( thrust::device_ptr<float>(d_data_out), thrust::device_ptr<float>(d_data_out)+out_volumeDim[0]*out_volumeDim[1]*out_volumeDim[2], back_ground_value );

  if(type == 0) d_texture_in.filterMode = cudaFilterModeLinear;    // linear interpolation
  else          d_texture_in.filterMode = cudaFilterModePoint;     // point (nearest-neighbor) interpolation

  // find bounding box of the input volume
  int bbox_max[3] = {-INT_MAX, -INT_MAX, -INT_MAX}, bbox_min[3] = {INT_MAX, INT_MAX, INT_MAX};
  float in_volDim_mm[3] = { in_voxelSize[0]*in_volumeDim[0], in_voxelSize[1]*in_volumeDim[1], in_voxelSize[2]*in_volumeDim[2] };
  float in_voxelSize_f[3] = { in_voxelSize[0], in_voxelSize[1], in_voxelSize[2] };
  float in_half_volumeSize[3] = { in_volDim_mm[0]/2, in_volDim_mm[1]/2, in_volDim_mm[2]/2 };
  float in_volume_center[3] = { in_half_volumeSize[0]-in_voxelSize[0]/2.0f, in_half_volumeSize[1]-in_voxelSize[1]/2.0f, in_half_volumeSize[2]-in_voxelSize[2]/2.0f };

  float out_volume_center[3];
  if(volume_center){
//    print_and_log("use specified volume center\n");
    memcpy(out_volume_center, volume_center, sizeof(float)*3);
  } else {
//    print_and_log("use default volume center\n");
    float c2[3] = { (out_volumeDim[0]-1)*voxel_size.x/2, (out_volumeDim[1]-1)*voxel_size.y/2, (out_volumeDim[2]-1)*voxel_size.z/2 };
    if(isWarp) c2[2] = (out_volumeDim[2]/num_transforms-1)*voxel_size.z/2;
    memcpy(out_volume_center, c2, sizeof(float)*3);
  }
  cudaMemcpyToSymbol(d_out_volume_center, out_volume_center, sizeof(d_out_volume_center), 0, cudaMemcpyHostToDevice);

  float out_voxel_size[3] = {voxel_size.x, voxel_size.y, voxel_size.z};

  float rot[9], translation[3];
  if(num_transform_element == 6){
    // rotation -> translation
    // vt = c - t + Rx * Ry * Rz * (v-c)
    // angles (sr[0], sr[1], sr[2]) is nagated compared to a typical definition (e.g. http://en.wikipedia.org/wiki/Euler_angles)
    float sr [3] = { sinf(transform[3]), sinf(transform[4]), sinf(transform[5]) };  // note: rotation angles are in radians
    float cr [3] = { cosf(transform[3]), cosf(transform[4]), cosf(transform[5]) };  // note: rotation angles are in radians
//    print_and_log("rotation angle: (%f, %f, %f), sin_rotation: (%f, %f, %f), cos_rotation: (%f, %f, %f)\n"
//      , transform[3], transform[4], transform[5], sr[0], sr[1], sr[2], cr[0], cr[1], cr[2]);
    rot[0] = cr[1]*cr[2];                     rot[3] = cr[1]*sr[2];                   rot[6] = -sr[1];      translation[0] = transform[0]; // transform[0-2] are in 'mm' unit
    rot[1] = -cr[0]*sr[2]+cr[2]*sr[0]*sr[1];  rot[4] = cr[0]*cr[2]+sr[0]*sr[1]*sr[2]; rot[7] = cr[1]*sr[0]; translation[1] = transform[1];
    rot[2] =  sr[0]*sr[2]+cr[0]*cr[2]*sr[1];  rot[5] = cr[0]*sr[1]*sr[2]-cr[2]*sr[0]; rot[8] = cr[0]*cr[1]; translation[2] = transform[2];
  } else if(num_transform_element == 12 && transform) {
    for(int i=0;i<9;i++)  rot[i] = transform[i];
    for(int i=0;i<3;i++)  translation[i] = transform[9+i]*in_voxelSize[i];
  }
  cudaMemcpyToSymbol(d_r_mat, rot, sizeof(float)*9, 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_translation, translation, sizeof(d_translation), 0, cudaMemcpyHostToDevice);
/*
  print_and_log("num_transform_element: %d\n", num_transform_element);
  for(int i=0;i<3;i++) print_and_log("%f %f %f %f\n", rot[i], rot[i+3], rot[i+6], translation[i]);
  print_and_log("Translation in_volume_center: %f %f %f\n", in_volume_center[0], in_volume_center[1], in_volume_center[2]);
  float original_edge[3], moved_edge[3];
  float edge_w[8][3] = { {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1} };
  for(int i=0;i<8;i++){
    for(int j=0;j<3;j++) original_edge[j] = (edge_w[i][j]*in_volDim_mm[j]+(edge_w[i][j]-0.5)*2*2*in_voxelSize[j]);  // 2 voxels margin for each direction
    transform_point(original_edge, rot, (float*)transform, out_volume_center, in_half_volumeSize, moved_edge);
//    print_and_log("original_edge:(%f, %f, %f) mm, moved_edge: (%f, %f, %f) mm\n", original_edge[0], original_edge[1], original_edge[2], moved_edge[0], moved_edge[1], moved_edge[2]);
    for(int j=0;j<3;j++){
      moved_edge[j] /= out_voxel_size[j];
      if(moved_edge[j]<bbox_min[j]) bbox_min[j] = floor(moved_edge[j]);
      if(moved_edge[j]>bbox_max[j]) bbox_max[j] = ceil(moved_edge[j]);
    }
  }
  for(int i=0;i<3;i++){ bbox_min[i] = MIN( MAX(bbox_min[i],0), volumeDimArray[i] ); bbox_max[i] = MIN( MAX(bbox_max[i],0), volumeDimArray[i] ); }   // force positive
*/
  for(int i=0;i<3;i++){ bbox_min[i] = 0; bbox_max[i] = out_volumeDim[i]; }
  unsigned int u_bbox_min[3] = {bbox_min[0], bbox_min[1], bbox_min[2]}, u_bbox_max[3] = {bbox_max[0], bbox_max[1], bbox_max[2]};
  unsigned int bbox_size[3] = {bbox_max[0]-bbox_min[0], bbox_max[1]-bbox_min[1], bbox_max[2]-bbox_min[2]};
//  print_and_log("volumeDim: (%d, %d, %d)\n", volumeDimArray[0], volumeDimArray[1], volumeDimArray[2]);
//  print_and_log("bbox: (%d,%d,%d)-(%d,%d,%d), size:(%d, %d, %d)\n", bbox_min[0], bbox_min[1], bbox_min[2], bbox_max[0], bbox_max[1], bbox_max[2], bbox_size[0], bbox_size[1], bbox_size[2]);

  cudaMemcpyToSymbol(d_bbox_size, bbox_size, sizeof(d_bbox_size), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_bbox_min, u_bbox_min, sizeof(d_bbox_min), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_in_voxel_size, in_voxelSize_f, sizeof(d_in_voxel_size), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_in_volume_center, in_volume_center, sizeof(d_in_volume_center), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_in_volume_dim, in_volumeDim, sizeof(d_in_volume_dim), 0, cudaMemcpyHostToDevice);

  // run kernel

  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 grid(iDivUp(out_volumeDim[0], block.x), iDivUp(out_volumeDim[1], block.y), iDivUp(out_volumeDim[2], block.z));

  #if defined RegTools_VERBOSE_MESSAGE
  print_and_log("start interpolator kernel, grid: (%d, %d), block: (%d, %d), type: %d, input voxel size: (%f, %f, %f)\n"
    , grid.x, grid.y, block.x, block.y, type, in_voxelSize_f[0], in_voxelSize_f[1], in_voxelSize_f[2]);
  print_and_log("transform: (%f, %f, %f, %f, %f, %f)\n", transform[0], transform[1], transform[2], transform[3], transform[4], transform[5]);
  print_and_log("Interpolator, rotation matrix:\n");
/*
  float *cr = cos_rotation, *sr = sin_rotation;
  print_and_log("%f %f %f\n", cr[1]*cr[2], cr[1]*sr[2], - sr[1]);
  print_and_log("%f %f %f\n", (-cr[0]*sr[2]+cr[2]*sr[0]*sr[1]), (cr[0]*cr[2]+sr[0]*sr[1]*sr[2]), cr[1]*sr[0]);
  print_and_log("%f %f %f\n", (sr[0]*sr[2]+cr[0]*cr[2]*sr[1]), (cr[0]*sr[1]*sr[2]-cr[2]*sr[0]), cr[0]*cr[1]);
*/
  print_and_log("Translation in_volume_center: %f %f %f, transform: %f %f %f\n", in_half_volumeSize[0], in_half_volumeSize[1], in_half_volumeSize[2], transform[0], transform[1], transform[2]);
  #endif

  if(isWarp){
    // apply warp
    d_texture_warpX.filterMode = cudaFilterModePoint; //cudaFilterModeLinear;
    d_texture_warpY.filterMode = cudaFilterModePoint; //cudaFilterModeLinear;
    d_texture_warpZ.filterMode = cudaFilterModePoint; //cudaFilterModeLinear;
//    print_and_log("grid: (%d,%d,%d), block: (%d,%d,%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    if(!transform){
//      print_and_log("Warp without transform, in_volumeDim = (%d, %d, %d), out_volumeDim = (%d, %d, %d)\n"
//        , in_volumeDim[0], in_volumeDim[1], in_volumeDim[2], out_volumeDim[0], out_volumeDim[1], out_volumeDim[2]);
      warp_kernel<<<grid, block>>>(d_data_out, type, order, bicubic_a, in_volumeDim[2]);
    } else {
//      print_and_log("Warp with transform, in_volumeDim = (%d, %d, %d), out_volumeDim = (%d, %d, %d)\n"
//        , in_volumeDim[0], in_volumeDim[1], in_volumeDim[2], out_volumeDim[0], out_volumeDim[1], out_volumeDim[2]);
//      print_and_log("volume center: (%f %f %f)\n", out_volume_center[0], out_volume_center[1], out_volume_center[2]);
//      print_and_log("out volume dim: (%d %d %d, %d)\n", out_volumeDim[0], out_volumeDim[1], out_volumeDim[2], num_transforms);
//      print_and_log("in voxel size: (%f, %f, %f)\n", in_voxelSize_f[0], in_voxelSize_f[1], in_voxelSize_f[2]);
//      print_and_log("out voxel size: (%f, %f, %f)\n", voxel_size.x, voxel_size.y, voxel_size.z);
//      print_and_log("launch_Interpolator_BBoxCheck(), isWarp = %d, num_transform_element = %d, num_transforms = %d\n", isWarp, num_transform_element, num_transforms);
//      for(int i=0;i<num_transforms;i++){
//        for(int j=0;j<3;j++){ print_and_log("%f, %f, %f, %f\n", transform[i*12+j], transform[i*12+3+j], transform[i*12+6+j], transform[i*12+9+j]) }
//      }
      float *d_transform;
      cutilSafeCall( cudaMalloc(&d_transform, num_transforms*12*sizeof(float)) );
      cutilSafeCall( cudaMemcpy(d_transform, transform, num_transforms*12*sizeof(float), cudaMemcpyHostToDevice) );
      warp_trans_kernel<<<grid, block>>>(d_data_out, type, order, bicubic_a, in_volumeDim[2], d_transform );
      cutilSafeCall( cudaFree(d_transform) );
    }
  } else {
    // interpolate
    interpolator_kernel<<<grid, block>>>(d_data_out, type, order, bicubic_a);
  }

  cudaThreadSynchronize();
  cudaError_t kernel_error = cudaGetLastError();
  if (kernel_error != cudaSuccess) {
    print_and_log("interpolator.cu::run_kernel> ERROR! %s\n", cudaGetErrorString(kernel_error));
    return;
  }
  /*
  #include<thrust/reduce.h>
  thrust::device_ptr<float> D = thrust::device_pointer_cast(d_data_out);
  int num_voxels = in_volumeDim[0] * in_volumeDim[1] * in_volumeDim[2];
  float sum = thrust::reduce(D, D + num_voxels);
  print_and_log("launch_Interpolator_BBoxCheck(), sum(d_Result):%f\n", sum);
  */

}

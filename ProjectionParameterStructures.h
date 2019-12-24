/*
  Author(s):  Yoshito Otake
  Created on: 2013-03-01
*/

#ifndef PROJECTION_PARAMETER_STRUCTURES_H
#define PROJECTION_PARAMETER_STRUCTURES_H

#include "config.h"

#ifdef __CUDACC__
    typedef float2 fComplex;
#else
    typedef struct{
        float x;
        float y;
    } fComplex;
#endif

struct ProjectionParameters
{
  double *ProjectionMatrix_3x4; // row-major
  double *FOV_mm;               // field of view in millimeter (2 element vector)
};

struct ProjectionParameters_objectOriented
{
  // This representation of projection geometry is intended for use in an "imaging" application (forward/back projection).
  // The position of the source and detector with respect to the object coordinate are specified as well as
  // the detector's field of view
  double* DetectorFrame;      // 4x4 matrix representing a transformation from the object coordinate to the detector coordinate
  double* SourcePosition;     // source position (x, y, z) in the object coordinate (in mm)
  double* FOV;                // horizontal (X axis) and vertical (Y axis) field of view of the detector (in mm)
  double* Pixel_aspect_ratio; // aspect ratio of pixel size ( pixel_width / pixel_height )
  double* Skew_angle_deg;     // angle between the two image axes in degree (default: 90 degree)
  double Pixel_width;         // pixel width (1-element array). this is used for voxel-driven algorithm only
//  int IsPerspective;        // True: perspective projection, False: orthographic (parallel) projection
};

struct ProjectionResult
{
  float *Data;
  float *projectionTime;      // ellapsed time for the forward projection (in milliseconds)
  float *minValue, *maxValue; // minimum/maximum value of Data (set NULL if these are not needed)
  int dDataID;                // id to specify pre-allocated device memory for the projection result (optional)
  int numGPU;                 // number of GPUs that is used for projection (-1: use all GPUs)
  float *initData;            // the data used to initialize the projection result in the MemoryStoreMode_Replace case, can be null
};

struct VolumePlan_cudaArray
{
  struct cudaArray **d_volume;     // array of device pointer to the volume on each device (should be casted to 'struct cudaArray*' type)
  float **d_raw_volume;            // array of device pointer to the 'raw' volume on each device (not cudaArray)
  float *h_volume, *h_volume_set;  // host pointer to the volume
  unsigned int numVolumes;         // number of volumes (multiple volumes with the same dimension)
  unsigned int volumeIndex;        // used in get/set function to specify index of the volume
  int VolumeDim[3];
  double VoxelSize[3];
};

// cufftHandle is a handle type used to store and access CUFFT plans.
#if (CUDA_VERSION_MAJOR>=5)
  typedef int cufftHandle;
#else
  typedef unsigned int cufftHandle;
#endif

struct SimilarityMeasureComputationPlan
{
  int m_CudaDeviceID_Sequential;
  float *d_fixed_images;   // device pointer to the fixed images
  float *h_fixed_images;   // host pointer to the volume
  float *d_fixed_measurement;
  float *h_fixed_measurement;
  float *d_fixed_normalized;
  float *h_fixed_normalized;
  float *d_floating_normalized;
  float *h_floating_normalized;
  float *d_fixed_Xgrad, *d_fixed_Ygrad;
  float *h_fixed_Xgrad, *h_fixed_Ygrad;
  float *d_floating_Xgrad, *d_floating_Ygrad;
  float *h_floating_Xgrad, *h_floating_Ygrad;
  float *d_intermediate_images;  // device pointer to the intermediate image
  float *h_intermediate_images;  // host pointer to the intermediate image
  int ImageDim[3];
  int MaxNumImageSets;
  int h_get_frame_no;
  float Sigma;    // if this is less than zero, we don't use gradient-based similarity measeure
  int SimilarityMeasureType;
  double *SimilarityMeasure;
  double NormalizationFactor;
  float NormalizeMax_floating, NormalizeMin_floating; // max and min pixel value for normalization in computation of MI and NMI (both zero: compute on-the-fly)
  float NormalizeMax_fixed, NormalizeMin_fixed; // max and min pixel value for normalization in computation of MI and NMI (both zero: compute on-the-fly)
  double I0;

  float *d_temp_images;   // temporary memory for similarity measure computation (for now, onlye GI_STD_NORM uses this)
  float *d_temp_kernel;
  float *d_temp_padded;
  float *d_temp_SM, *h_temp_SM;
  float *d_WeightVector;
  float *h_joint_pdf;               // probability distribution function (normalized histogram) for MI, NMI computation
  float *d_pdf_buf, *d_joint_pdf;   // 1D&2D probability distribution function (normalized histogram) for MI, NMI computation
  int   *d_hist_buf, *d_joint_hist; // 1D&2D histogram
  float h_fixed_entropy;  // pre-computed entropy of the fixed images
  fComplex *d_temp_spectrum;
  fComplex *d_X_kernel_spectrum, *d_Y_kernel_spectrum;
  cufftHandle *fftPlanFwd, *fftPlanInv, *fftPlanManyFwd, *fftPlanManyInv;

  float *h_mask_weight; // note: currently mask weight is applied only on Gradient Information
  float *d_mask_weight;
  float h_GI_threshold; // ignore (zero gradient) if pixel value is less than this value (default: -FLT_MAX)
  float h_SSIM_DynamicRange;
};

#endif // PROJECTION_PARAMETER_STRUCTURES_H
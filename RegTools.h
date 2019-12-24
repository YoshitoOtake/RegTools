/*
  Author(s):  Yoshito Otake, Ali Uneri
  Created on: 2011-02-21
*/

#ifndef REG_TOOLS_H
#define REG_TOOLS_H

// standard library headers
#include <vector>
#include <map>
#include <float.h>

// headers shared with Matlab
#include <config.h>
#include <ProjectionParameterStructures.h>

// CUDA kernel/function headers
#include <Projector.cuh>
#include <SimilarityMeasureComputation/SimilarityMeasures.cuh>

// CUDA library headers
#include "multithreading.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <driver_functions.h>
#include <vector_functions.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <curand.h>

// other headers
#if defined (__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#define cutilSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cufftSafeCall(err) __cufftSafeCall(err, __FILE__, __LINE__)

#ifndef MAX
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#define print_and_log(...) { \
                              printf (__VA_ARGS__);  \
                              if(m_LogFile){ fprintf(m_LogFile, __VA_ARGS__); fflush(m_LogFile); }\
                           }
#define CURAND_CALL(x) { if((x)!=CURAND_STATUS_SUCCESS) { print_and_log("Error at %s:%d\n",__FILE__,__LINE__);  }}

#define MAX_MESSAGE_LENGTH 2048

#if _WIN32
#define add_str(ptr,counter,...) { counter += sprintf_s(ptr+counter, MAX_MESSAGE_LENGTH-counter,__VA_ARGS__); }
#else
#define add_str(ptr,counter,...) { counter += snprintf(ptr+counter, MAX_MESSAGE_LENGTH-counter,__VA_ARGS__); }
#endif

enum {  ProcessingMode_ForwardProjection = 0, 
        ProcessingMode_Interpolator, 
        ProcessingMode_VolumePlan,
        ProcessingMode_VolumePlan_cudaArray,
        ProcessingMode_MultVolume,
        ProcessingMode_CMAESPopulation,
        ProcessingMode_SimilarityMeasureComputation,
        ProcessingMode_CopyDeviceMemoryToCudaArray,
        ProcessingMode_CopyDeviceMemoryToCudaArray_Multi,
        ProcessingMode_SetPBO,
        ProcessingMode_MemInfoQuery,
        ProcessingMode_Initialize,
        ProcessingMode_ComputeLinearCombination,
        ProcessingMode_GetGPUProjection,
        ProcessingMode_CopyHostInitProjectionToDevice,
        ProcessingMode_ClearDeviceInitProjection,
        ProcessingMode_DoNothing
      };  // processing mode

enum {  InterpolatorType_Bilinear = 0,
        InterpolatorType_Bicubic = 1,
        InterpolatorType_Bspline = 2,
        InterpolatorType_NearestNeighbor = 5,
      };  // interpolator type

enum { ProjectorMode_LinearInterpolation = 0, ProjectorMode_Siddon = 1, ProjectorMode_RayCasting = 5, ProjectorMode_LinearInterpolationDeformable = 6 }; // Projector mode
enum {  SIMILARITY_MEASURE_MI = 1, SIMILARITY_MEASURE_NMI = 2, SIMILARITY_MEASURE_GI = 3, SIMILARITY_MEASURE_GI_SINGLE = 4, 
        SIMILARITY_MEASURE_GC = 5, SIMILARITY_MEASURE_NCC = 6, SIMILARITY_MEASURE_MSE = 7, SIMILARITY_MEASURE_LogLikelihood = 8, 
        SIMILARITY_MEASURE_GI_SINGLE_ALWAYS_FLOATING_NORM = 9, SIMILARITY_MEASURE_SSIM = 10, SIMILARITY_MEASURE_GI_STD_NORM = 11 };
enum { MemoryStoreMode_Replace = 0, MemoryStoreMode_Additive = 1, MemoryStoreMode_Multiplicative = 2 };

struct RegToolsThreadParam {
  int m_CudaDeviceID;
  int m_CudaDeviceID_Sequential;
  bool m_WithGL;
  double m_CudaDeviceLoad;
  char **m_Messages;
  int m_MessagesPtr;
  cublasHandle_t m_CublasHandle;
  curandGenerator_t m_CurandGenerator;

  unsigned int m_NumTotalProjections;       // number of total projections (this is the same as number of projection matrices)
  unsigned int m_NumAllocatedProjections;   // number of allocated projections (this excludes projections that were disabled by SetSubsamplingVector() from above)
  unsigned int m_NumEnabledProjections;     // number of 'enabled' projections (this is used for multi-GPU projection. subset of 'allocated' projections are 'enabled' on each GPU.)
    // note: m_NumTotalProjections > m_NumAllocatedProjections > m_NumEnabledProjections
  int *m_EnabledProjection;
  int m_ProcessingMode, m_ProjectorMode;
  ProjectionParameters *m_ProjectionParameters;
  float *m_Volume, *m_Volume2;
  float *m_Projections;
  unsigned int m_ProjectionWidth, m_ProjectionHeight, m_NumProjectionSets, m_ProjectionSetIndex;
  unsigned short m_ProjectionOption;
  cudaExtent m_VolumeDim;
  float3 m_VoxelSize_mm;
  unsigned int m_NumVolumes;
  unsigned int m_TransferBlockSize;
  float m_StepSize;

  CUTThread m_ProjectorDataReadyEvent, m_ProjectorCompletedEvent;     // TODO: Linux compatibility
  bool m_ThreadCompleted;
  CUTThread *m_ThreadHandle;
  float* m_ElapsedTime;
  double* m_World_Volume_col;
  size_t m_FreeMem, m_TotalMem;
  int m_MemoryStoreMode;  // 0: allocate new memory and store the result to the new memory, 1: additive (no allocation), 2: multiplicative (no allocation)
  bool m_CountNonIntersectedPixel;
  int *m_ZeroPixelCount;
  bool m_DifferentVolumePerProjectionSet;

  // for maximum and minimum value to return
  float *m_MinValue, *m_MaxValue;

  // for normalization of resulted images (do nothing if min>=max)
  float m_NormalizeMin, m_NormalizeMax;

  // for Interpolator
  float *m_Interpolator_transform;
  int m_Interpolator_num_transform_element, m_Interpolator_num_transforms;
  int m_Interpolator_type;
  int m_Interpolator_order;
  float m_Interpolator_bicubic_a;
  float m_Interpolator_back_ground;
  float *m_Interpolator_volume_center;
  float *m_Interpolator_scattered_pnts;
  int m_Interpolator_num_scattered_pnts;
  bool m_Interpolator_IsWarp;
  VolumePlan_cudaArray *m_VolumePlan_cudaArray, *m_VolumePlan_cudaArray_warpX, *m_VolumePlan_cudaArray_warpY, *m_VolumePlan_cudaArray_warpZ;
  VolumePlan_cudaArray *m_VolumePlan_cudaArray_out;

  // for Similarity Measure Comoputation
  SimilarityMeasureComputationPlan *m_SimilarityMeasureComputationPlan, *m_SimilarityMeasureComputationPlan2;
  int m_SimilarityMeasureComputationImageOffset;
  int m_SimilarityMeasure_NumberOfImageSet;

  // for OpenGL interface
  unsigned int m_PBO;
  cudaGraphicsResource **m_ProjectionImagePBOResource;
  int m_PBO_index;

  // for each thread run
  struct cudaArray *d_ProjectionTextureArray, *d_VolumeTextureArray;
  float *d_Projections, *d_Volume, *d_Volume2;  // global memory space to keep projection images and volume (projection results are stored here)
  size_t previousResultWidth, previousResultHeight, pitch;
  size_t previousResultWidth2, previousResultHeight2, pitch2;   // previous memory size for d_Volume2
  cudaEvent_t h_timer_start, h_timer_stop;

  // for volume manipulation
  float m_ScalarVal;

  // for ray-casting
  int m_RayCastingLOD;
  float m_RayCastingThreshold, m_RayCastingDistanceFalloffCoefficient;

  // for computation of linear combination
  int m_NumModeDims, m_NumModes;
  float *d_WarpArray, *d_ModeArray, *d_ModeWeights;
  int m_ModeArray_NumberOfModes;

  VolumePlan_cudaArray **m_VolumePlans;
  int m_NumCudaArrays;

  // for generation of population in CMAES
  float *d_arx, *d_arxvalid, *d_CMAES_xmean, *d_CMAES_diagD, *d_CMAES_lbounds, *d_CMAES_ubounds;

  // for multiple bodies registration - keeping the initial projection stored when the transformation remains fixed for certain components
  float *m_ProjectionsInit;
  size_t m_ProjectionsInit_len;
  int m_NumViews;
  float* d_ProjectionsInit;
  size_t d_ProjectionsInit_capacity_bytes;
};

struct ProjectionParametersSetting {
  int m_NumProjections, m_NumProjectionSets, m_NumEnabledProjections;
  ProjectionParameters *m_ProjectionParameters;
  int *m_SubSamplingArray;  // vector of int type variables that represents which image is enabled in projection process (default: all enabled)
  int m_ProjectionWidth, m_ProjectionHeight;
  int *m_ZeroPixelCount;    // vector to store number of zero pixels (optional)
};

class RegTools {

public:
  RegTools(void);
  ~RegTools();
  bool InitializeRegToolsThread_withGL(void);
  bool InitializeRegToolsThread(int* deviceIDs, int numDevice, double *deviceLoadList = NULL, bool withGL = false, char **messages = NULL);

  int InitializeProjectionParametersArray(int numProjections);
  bool SetCurrentGeometrySetting(int geometry_id);
  int GetCurrentGeometrySetting(void);
  bool DeleteProjectionParametersArray(int geometry_id);
  bool AddLogFile(char* filename);
  bool RemoveLogFile(void);
  int CopyProjectionParametersStruct(struct ProjectionParameters *dst, struct ProjectionParameters *src);
  int InitializeProjectionParametersStruct(struct ProjectionParameters *projectionParams);
  int DeleteProjectionParametersStruct(struct ProjectionParameters *projectionParams);
  int CopyProjectionParametersStruct_objectOriented(struct ProjectionParameters_objectOriented *dst, struct ProjectionParameters_objectOriented *src);
  int InitializeProjectionParametersStruct_objectOriented(struct ProjectionParameters_objectOriented *projectionParams);
  int DeleteProjectionParametersStruct_objectOriented(struct ProjectionParameters_objectOriented *projectionParams);

  int SetProjectionParameter_objectOriented(int projection_number /* note: this is 0-base index */, struct ProjectionParameters_objectOriented projectionParams);
  int SetProjectionParameter_3x4PM(int projection_number /* note: this is 0-base index */, double *pm3x4_row_major, double pixel_width, double pixel_height, double u_dim, double v_dim, double down_sample_ratio_u = 1, double down_sample_ratio_v = 1);
  int SetProjectionParameter_3x4PM_multi(int numProj, double *pm3x4_row_major, double pixel_width, double pixel_height, double u_dim, double v_dim);
  int Downsample3x4ProjectionMatrix(double *pm3x4_row_major, double down_sample_ratio_u, double down_sample_ratio_v, double u_dim, double v_dim);
  int Scale3x4ProjectionMatrix(double *pm3x4_row_major, double pixel_width, double *pixel_height = NULL);
  int SetWorldToVolumeTransform(const double *transform_col);
  int SetWorldToVolumeTransform(double tx, double ty, double tz, double rx, double ry, double rz); // (tx, ty, tz): in mm, (rx, ry, rz): in radians
  int CreateProjectionImagePBO(int num);
  int SetProjectionImagePBO(unsigned int pbo);
  int SetProjectionImagePBOs(unsigned int *pbos, int start, int num);
  void DeleteProjectionImagePBO(void);

  //: Set dimensions of the volume and the size of one voxel (in whatever unit, e.g "mm")
  bool SetVolumeInfo(int volume_dim_x, int volume_dim_y, int volume_dim_z, float voxelSize_x, float voxelSize_y, float voxelSize_z);
  bool GetVolumeInfo(int *volume_dim_x, int *volume_dim_y, int *volume_dim_z, float *voxelSize_x, float *voxelSize_y, float *voxelSize_z);
  int SetProjectorMode(int projectorMode){ m_ProjectorMode = projectorMode; return 1; }
  int GetProjectorMode(void){ return m_ProjectorMode; }
  int SetRayCastingLOD(int lod){ m_RayCastingLOD = lod; return 1; }
  int GetRayCastingLOD(int *lod){ *lod = m_RayCastingLOD; return 1; }
  int SetRayCastingThreshold(float thresh){ m_RayCastingThreshold = thresh; return 1; }
  int GetRayCastingThreshold(float *thresh){ *thresh = m_RayCastingThreshold; return 1; }
  int SetCountNonIntersectedPixel(int count_non_intersected_pixel){ m_CountNonIntersectedPixel = count_non_intersected_pixel; return 1; }
  int GetCountNonIntersectedPixel(int *count_non_intersected_pixel){ *count_non_intersected_pixel = m_CountNonIntersectedPixel; return 1; }
  int SetRayCastingDistanceFalloffCoefficient(float coefficient){ m_RayCastingDistanceFalloffCoefficient = coefficient; return 1;}
  int GetRayCastingDistanceFalloffCoefficient(float *coefficient){ *coefficient = m_RayCastingDistanceFalloffCoefficient; return 1;}
  int SetDifferentVolumePerProjectionSet(int different_volume_per_projection_set){ m_DifferentVolumePerProjectionSet = different_volume_per_projection_set; return 1; }
  int GetDifferentVolumePerProjectionSet(int *different_volume_per_projection_set){ *different_volume_per_projection_set = m_DifferentVolumePerProjectionSet; return 1; }

  bool SetStepSize(float step_size);
  bool GetStepSize(float *step_size);
  void SetNormalizeMaxMin(float norm_max, float norm_min){ m_NormalizeMax = norm_max; m_NormalizeMin = norm_min; }
  int GetProjectionWidth(void);
  int GetProjectionHeight(void);
  int GetNumberOfProjections(void);
  int GetNumberOfEnabledProjections(void);
  int SetProjectionDim(int width, int height);
  int GetProjectionDim(int *width, int *height);
  int GetProjectionMatrices(double *pm);
  int GetPixelSize(double *pixel_width, double *pixel_height);
  int SetNumberOfProjectionSets(int num_projection_set);
  int GetNumberOfProjectionSets(void);
  int SetTransferBlockSize(int numBlockSize);
  static int ShiftProjectionMatrices(double *pm_all, int num_proj, int *left_bottom /* numbefOfProjections*2 element array */);
  int SetSubSamplingVector(int *sub_sampling_vector, int numElements);
  int GetSubSamplingVector(int *sub_sampling_vector, int numElements);
  int EraseDisabledProjections(void);
  int ReplicateProjections(int num_rep);

  int ForwardProjection(ProjectionResult &result, const float *volume);
  int ForwardProjection_with3x4ProjectionMatrices(ProjectionResult &result, const int plan_id, const double *pm_3x4);
  int ForwardProjection_withPlan(ProjectionResult &result, const int plan_id);
  int ForwardProjection_withPlan(ProjectionResult &result, const int plan_id, int numGlobals, const double *transformations_global, int numView
    , int numLocalTrans, const double *transformations_local, const int memory_store_mode = MemoryStoreMode_Replace);
  int Interpolation_withPlan(struct ProjectionResult &result, int plan_id, const float *transforms, const int num_transform_element, int num_transform, const int type, const int order, const float bicubic_a, 
                              const float back_ground, float *volume_center, const int isRGBA = 0, float *color_map = NULL, int num_color_map = 0, int label_overlay_mode = 0);
  int ApplyDeformationField(struct ProjectionResult &result, int target_volume_id, int *warps_tex, int num_dims, int type, int order, float bicubic_a, float back_ground, 
                              float *volume_center, int scattered_pnts_plan, float *transforms_3x4xN, int num_transform_element, int num_transforms);
  int RegTools::SetWarpTextures(int *warps_tex);
  int CreatePCAInstance(struct ProjectionResult &result, int target_volume_id, int *def_modes, double *mode_weights, int num_modes, int type, int order, float bicubic_a, float back_ground, float *volume_center);
  int ComputeLinearCombination(int warp_device, int def_mode_device, int mode_weight);
  int GetGPUProjection(float *h_projection, int projection_index_0_base);
  VolumePlan_cudaArray* FindVolumePlan(int planID);
  int CopyDeviceMemoryToCudaArray(int cudaArrayID, int deviceMemoryID, int isCopyToAllDevices = 1, int volume_index_tex_0_base = 0, int volume_index_dev_0_base = 0);
  int CopyDeviceMemoryToCudaArray_Multi(int *cudaArrayIDs, int numCudaArrayID, int deviceMemoryID);

  void SetInitialProjectionOnDevice(const float* h_proj, const size_t len);
  void ClearInitialProjectionOnDevice();

  static CUT_THREADPROC RegToolsThread(RegToolsThreadParam *in_param);
  static void RegToolsThread_startup(RegToolsThreadParam *in_param);
  static void RegToolsThread_Initialize(RegToolsThreadParam *in_param);
  static void RegToolsThread_main(RegToolsThreadParam *in_param);
  static void RegToolsThread_cleanup(RegToolsThreadParam *in_param);
  static void RegToolsThread_InitializeMemory(RegToolsThreadParam *in_param);
  static void RegToolsThread_SimilarityMeasureComputation(RegToolsThreadParam *in_param, float *d_Projections, float *d_Volume);
  static void RegToolsThread_UpdateSimilarityMeasureComputationPlan(SimilarityMeasureComputationPlan *plan, RegToolsThreadParam *in_param);
  static void RegToolsThread_MultVolume(RegToolsThreadParam *in_param);
  static void RegToolsThread_CMAESPopulation(RegToolsThreadParam *in_param);
  static void RegToolsThread_VolumePlan(RegToolsThreadParam *in_param);
  static void RegToolsThread_VolumePlan_cudaArray(RegToolsThreadParam *in_param);
  static void RegToolsThread_CopyVarToCudaMemory(RegToolsThreadParam *in_param);
  static void RegToolsThread_CopyDeviceMemoryToCudaArray(RegToolsThreadParam *in_param);
  static void RegToolsThread_CopyDeviceMemoryToCudaArray_Multi(RegToolsThreadParam *in_param);
  static void RegToolsThread_SetPBO(unsigned int pbo, cudaGraphicsResource **resource);
  static void RegToolsThread_RunInterpolator(RegToolsThreadParam *in_param, float *d_ProjectionResult);
  static void RegToolsThread_ComputeLinearCombination(RegToolsThreadParam *in_param);
  static void RegToolsThread_GetGPUProjection(RegToolsThreadParam *in_param);
  static int RegToolsThread_MemGetInfo(size_t &free, size_t &total);
  static void ConstructProjectionParameterArray(int index, struct ProjectionParameters *projectionParam
                                                  , double *w_v_col /* T_World_Volume */, double *volume_size, int u_pix, int v_pix
                                                  , float *h_PreComputedMatrix_array);
  static void ComputeBoxProjection(double *pm, double *box_center, double *box_size, double *bsquare_min, double *bsquare_max);
  static void RegToolsThread_CopyHostInitProjectionToDevice(RegToolsThreadParam *in_param);
  static void RegToolsThread_ClearDeviceInitProjection(RegToolsThreadParam *in_param);

  double GPUmemCheck(const char* message, int threadID = 0);
  int CreateVolumePlan_cudaArray(struct VolumePlan_cudaArray *plan, bool isCudaArray = true);
  int GetVolumePlan_cudaArrayVolumeInfo(int plan_id, int *volume_dim, double *voxel_size, int *numVolumes);
  int GetVolumePlan_cudaArrayVolume(int plan_id, float *h_volume, bool isCudaArray = true, int volume_index_0_base = 0);
  int SetVolumePlan_cudaArrayVolume(int plan_id, float *h_volume, bool isCudaArray = true, int volume_index_0_base = 0);
  int MultVolumePlan(int plan_id, float value);
  int DeleteVolumePlan_cudaArray(int plan_id);
  void InitializeSimilarityMeasureComputationPlan(struct SimilarityMeasureComputationPlan *plan);
  int CreateSimilarityMeasureComputationPlan(struct SimilarityMeasureComputationPlan *plan, double *normalization_factor = NULL);
  int GetSimilarityMeasureComputationPlanImageInfo(int plan_id, int GPU_ID, int *image_dim, double *normalization_factor = NULL);
  int GetSimilarityMeasureComputationPlanImages(int plan_id, int GPU_ID, float *images, int image_type, int frame_no = 0);
  int DeleteSimilarityMeasureComputationPlan(int plan_id);
  void ComputeNumberOfProjectionForEachGPU(RegToolsThreadParam *threadParams, int numThreads, int totalProj, int *num_projs, int *start_projs = NULL);
  void ComputeSimilarityMeasure(int plan_id, int similarity_type, int numImageSet, double *sm, float *elapsed_time = NULL);
  void ComputeSimilarityMeasure(int plan_id1, int plan_id2, int similarity_type, int numImageSet, double *sm, float *elapsed_time = NULL);
  int ComputeBoxProjectionBoundingSquare(int *projected_square_left_bottom, int *projected_size, int *in_out, double *box_center, double *box_size, int margin);
  int CropAllProjections(int *left_bottom /* numbefOfProjections*2 element array */, int *crop_size /* 2 element array */);

  // Matrix computation
  static double Determinant3x3d(const double *in);
  static double Determinant4x4d(const double *in);
  static bool Inverse4x4d(const double *in, double *out);
  static bool Inverse3x3d(const double *in, double *out);
  static float3 MultMatrixf_3x3_col(const double *R_3x3_col, const double x, const double y, const double z);
  static void MultMatrix_3x3_col(const double *R_3x3_col, const double in[3], double *out);
  static double ComputeFocalLength(const double *pm_3x4_row, double *pixel_size);
  static void ComputeSourcePosition(const double *pm_3x4_row, double source_position[3]);
  bool ComposeProjectionMatrix_3x4(const double *F_Object_Source_4x4, const double *intrinsics_3x3, const double *scale, double *out_row_major_3x4);
  bool DecomposeProjectionMatrix_3x4(const double *in_column_major_3x4, double *F_Object_Source_4x4, double *intrinsics_3x3, double *scale);
  bool QRDecomposition_square(const double *in_column_major, const int n, double *Q, double *R);
  static bool CrossProduct3(const double *in1, const double *in2, double *out);
  double DotProduct3(const double *in1, const double *in2);
  static void Normalize3(double *in_out);
  static void MultMatrixd_col(const double *mat, const double *mul, double *out);
  static void TransposeMatrixd(double *mat);
  static void LoadIdentity_4x4d(double *mat);
  static void RotateX_col(double *mat, double angle_rad);
  static void RotateY_col(double *mat, double angle_rad);
  static void RotateZ_col(double *mat, double angle_rad);
  static void Translate_col(double *mat, double x, double y, double z);
  static void ApplyTransformation_col(double *mat_4x4, double *vec);
  int convertRotTransTo4x4(double *in_1x6vec, double *out_4x4_col);
  int convertTransRotTo4x4(double *in_1x6vec, double *out_4x4_col);
  int convert4x4ToRotTrans(double *in_4x4, double *out_1x6vec);
  int convert4x4ToTransRot(double *in_4x4, double *out_1x6vec);
  void SetPBORenderingStartIndex(int index){ m_PBO_rendering_start_index = index; }
  int CMAESPopulation(int arz_ID, int arx_ID, int arxvalid_ID, int xmean_ID, int diagD_ID, int lbounds_ID, int ubounds_ID);

protected:
  RegToolsThreadParam *m_RegToolsThreadParams;
  CUTThread *m_RegToolsThreads;
  int m_NumRegToolsThreads;
  CUTThread *m_RegToolsThread_AllCompletedEvent;  // TODO: Linux compatibility
  cudaGraphicsResource **m_ProjectionImagePBOResource;
  int m_NumPBO;
  bool m_WithGL;  // use OpenGL interoperability or not

  float* m_Volume;
  float* m_Projections;
  float* m_ProjectionResult;
  unsigned short m_ProjectionOption;
  float m_NormalizeRatio;
  cudaExtent m_VolumeDim;
  float3 m_VoxelSize_mm;
  unsigned int m_NumVolumes;
  int m_TransferBlockSize;
  int m_MainCudaDevice;
  float m_StepSize;   // step size for simple linear interpolation-based forward projector
  double m_VolumeTransform_col[16]; // transformation between World coordinate and the volume coordinate (column-major order)
  float m_NormalizeMin, m_NormalizeMax; // for normalization of resulted images (do nothing if min>=max)
  int m_ProjectorMode;
  bool m_CountNonIntersectedPixel;
  bool m_DifferentVolumePerProjectionSet;

  // for PBO rendering
  int m_PBO_rendering_start_index;

  // for geometry settings
  int m_CurrentProjectionParametersSettingID;
  std::map<int, ProjectionParametersSetting*> m_ProjectionParametersSetting;
  
  // for Interpolator plan
  std::map<int, VolumePlan_cudaArray*> m_VolumePlan_cudaArrays;

  // for Similarity Measure Computation plan (for multi-GPU environment, the same ID is used for all plan. each plan has cudaDeviceID to distinguish each other)
  std::multimap<int, SimilarityMeasureComputationPlan*> m_SimilarityMeasureComputationPlans;

  // for ray-casting
  int m_RayCastingLOD;
  float m_RayCastingThreshold, m_RayCastingDistanceFalloffCoefficient;

  int ComputeSubSamplingArray(int *sub_sampling_array);
  void PrepareForRegToolsThread(RegToolsThreadParam *param);

  static int cutGetMaxGflopsDeviceId(void);
  static int InitializeCudaDevice(int deviceID, double deviceLoad, bool withGL, char **messages, int &messages_ptr);
  static void prime_factorization(int x, std::vector<int> &primes);
  static void findTileSize(int numProjections, int imageWidth, int imageHeight, int &tileX, int &tileY, int &tiledImageX, int &tiledImageY);
  static void copyDataToCudaArray(struct cudaArray *d_array, float *h_data, int dst_index, int src_index, cudaExtent dimension, int numVolumes, cudaMemcpyKind kind);
  static void copyDataFromCudaArray(float *h_data, struct cudaArray *d_array, int dst_index, int src_index, cudaExtent dimension, int numVolumes);

  int ConvertProjectionMatrix_ObjectOrientedTo3x4(const ProjectionParameters_objectOriented in, const double pixel_width, double *pm3x4_row_major);

  void RunRegToolsThreads(int threadID = -1);
};

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}

// I want to make this function (__cudaSafeCall) inline
// But I don't know how to handle log output appropriately
void __cudaSafeCall( cudaError err, const char *file, const int line );
void __cufftSafeCall( cufftResult err, const char *file, const int line );

#define CUT_SAFE_CALL( call)                                               \
    if( true != call) {                                                   \
        fprintf(stderr, "Cut error in file '%s' in line %i.\n",              \
                __FILE__, __LINE__);                                         \
    }
#define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    } \
    }
#define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);                                            \
//! Check for CUDA error
#ifdef _DEBUG
#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = cudaThreadSynchronize();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }
#else
#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
    }                                                                        \
    }
#endif

////////////////////////////////////////////////////////////////////////////
//! Check for OpenGL error
//! @return CUTTrue if no GL error has been encountered, otherwise 0
//! @param file  __FILE__ macro
//! @param line  __LINE__ macro
//! @note The GL error is listed on stderr
//! @note This function should be used via the CHECK_ERROR_GL() macro
////////////////////////////////////////////////////////////////////////////
inline bool cutCheckErrorGL( const char* file, const int line) 
{
	bool ret_val = true;

	// check for error
	GLenum gl_error = glGetError();
	if (gl_error != GL_NO_ERROR) 
	{
#ifdef _WIN32
		char tmpStr[512];
		// NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
		// when the user double clicks on the error line in the Output pane. Like any compile error.
		sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", file, line, gluErrorString(gl_error));
		OutputDebugString(tmpStr);
#endif
		fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
		fprintf(stderr, "%s\n", gluErrorString(gl_error));
		ret_val = false;
	}
	return ret_val;
}
#define CUT_CHECK_ERROR_GL()                                               \
	if( false == cutCheckErrorGL( __FILE__, __LINE__)) {                  \
	    exit(EXIT_FAILURE);                                                  \
	}

// we just ignore timer functions that defined in cutil (use other timer functions)
inline bool cutCreateTimer( unsigned int* name){ return true; }
inline bool cutDeleteTimer( unsigned int name){ return true; }
inline bool cutStartTimer( const unsigned int name){ return true; }
inline bool cutStopTimer( const unsigned int name){ return true; }
inline bool cutResetTimer( const unsigned int name){ return true; }
inline float cutGetTimerValue( const unsigned int name){ return 0; }


inline int ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{ { 0x10,  8  }, // Tesla Generation (SM 1.0) G80 class
	  { 0x11,  8  }, // Tesla Generation (SM 1.1) G8x class
	  { 0x12,  8  }, // Tesla Generation (SM 1.2) G9x class
	  { 0x13,  8  }, // Tesla Generation (SM 1.3) GT200 class
	  { 0x20, 32  }, // Fermi Generation (SM 2.0) GF100 class
	  { 0x21, 48  }, // Fermi Generation (SM 2.1) GF10x class
	  { 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
	  { 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
	  { 0x37, 192 },
	  { 0x50, 128 },
	  { 0x52, 128 },
    {   -1, -1 }
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
	return -1;
}


#endif

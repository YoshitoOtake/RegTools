/*
  Author(s):  Yoshito Otake, Ali Uneri
  Created on: 2011-02-21
*/

#include "RegTools.h"
#define _USE_MATH_DEFINES
#include <math.h>  // for M_PI_2, cos, sin
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <cudaGL.h>

FILE *m_LogFile = NULL;

// error handling
void __cudaSafeCall( cudaError err, const char *file, const int line )
{
  if( cudaSuccess != err) {
    print_and_log("%s(%i) : cudaSafeCall() Runtime API error : %s.\n", file, line, cudaGetErrorString(err) )
  }
}

void __cufftSafeCall( cufftResult err, const char *file, const int line )
{
  if( CUFFT_SUCCESS != err) {
    print_and_log("%s(%i) : cufftSafeCall() CUFFT error %d.\n", file, line, err)
  }
}

RegTools::RegTools(void)
{
//  m_ProjectionParameters = NULL;
//  m_NumProjectionSets = 1;
  m_Volume = NULL;
  m_Projections = NULL;
  m_NumVolumes = 1;
//  m_SubSamplingArray = NULL;
  m_TransferBlockSize = DEFAULT_TRANSFER_BLOCK_SIZE;
  m_StepSize = 1.0f;
  m_CurrentProjectionParametersSettingID = -1;
  m_ProjectorMode = ProjectorMode_LinearInterpolation;

  LoadIdentity_4x4d(m_VolumeTransform_col);
  m_RegToolsThreadParams = NULL;
  m_RegToolsThreads = NULL;
  m_RegToolsThread_AllCompletedEvent = NULL;
  m_NumRegToolsThreads = 0;
  m_NumPBO = 0;
  m_ProjectionImagePBOResource = NULL;
  m_ProjectionOption = 0;
  m_NormalizeMin = 0.0; // default: no nomalization
  m_NormalizeMax = 0.0;
  m_RayCastingLOD = 5;
  m_RayCastingThreshold = -500;
  m_RayCastingDistanceFalloffCoefficient = -1;
  m_CountNonIntersectedPixel = false;
  m_DifferentVolumePerProjectionSet = false;

  m_PBO_rendering_start_index = 0;
}

bool RegTools::InitializeRegToolsThread_withGL(void)
{
  int deviceIDs[1] = {-1};
  return InitializeRegToolsThread(deviceIDs, 1, NULL, true);
}

bool RegTools::InitializeRegToolsThread(int* deviceIDs, int numDevice, double *deviceLoadList, bool withGL, char **messages)
{
  char tmp_msg[MAX_MESSAGE_LENGTH];
  int tmp_msg_total_count = 0;

  add_str(tmp_msg, tmp_msg_total_count, "RegTools - GPU-accelerated Registration Toolkit. Compiled at %s, %s\n", __DATE__, __TIME__);
  add_str(tmp_msg, tmp_msg_total_count, "RegTools - initializing %d devices ( ", numDevice);
  for (int i = 0; i < numDevice; i++)
  {
    add_str(tmp_msg, tmp_msg_total_count, "%d ", deviceIDs[i]);
  }
  add_str(tmp_msg, tmp_msg_total_count, ")\n" );

  print_and_log( "%s", tmp_msg );
  int messages_ptr = 0;
  if (messages)
  {
    add_str(messages[0], messages_ptr, tmp_msg);
  }

  m_NumRegToolsThreads = numDevice;
  m_WithGL = withGL;
  if (m_RegToolsThreadParams)
  {
    delete[] m_RegToolsThreadParams;
  }
  if (m_RegToolsThreads)
  {
    delete[] m_RegToolsThreads;
  }
  
  if (m_RegToolsThread_AllCompletedEvent)
  {
    delete[] m_RegToolsThread_AllCompletedEvent;
  }

  if(!m_WithGL)
  {
#if _WIN32
    // run on separate thread. one thread per device (GPU)
    m_RegToolsThreadParams = new RegToolsThreadParam[m_NumRegToolsThreads];
    m_RegToolsThreads = new CUTThread[m_NumRegToolsThreads];
    m_RegToolsThread_AllCompletedEvent = new CUTThread[m_NumRegToolsThreads];

    for(int i = 0; i < m_NumRegToolsThreads; i++)
    {
      m_RegToolsThreadParams[i].m_CudaDeviceID = deviceIDs[i];
      m_RegToolsThreadParams[i].m_CudaDeviceID_Sequential = i;
      m_RegToolsThreadParams[i].m_CudaDeviceLoad = (deviceLoadList == NULL) ? 1.0 / (double)m_NumRegToolsThreads : deviceLoadList[i] / 100.0;
      m_RegToolsThreadParams[i].m_WithGL = m_WithGL;
      m_RegToolsThreadParams[i].m_ProjectionImagePBOResource = NULL;
      
      m_RegToolsThreadParams[i].m_ProjectionsInit = 0;
      m_RegToolsThreadParams[i].m_NumViews = 0;
      m_RegToolsThreadParams[i].d_ProjectionsInit = 0;
      m_RegToolsThreadParams[i].d_ProjectionsInit_capacity_bytes = 0;
      
      const int max_event_name_string = 256;
      char data_ready_event_name[max_event_name_string], completed_event_name[max_event_name_string];
      sprintf_s(data_ready_event_name, max_event_name_string, "ProjectorDataReadyEvent%d", deviceIDs[i]);
      sprintf_s(completed_event_name, max_event_name_string, "ProjectorCompletedEvent%d", deviceIDs[i]);
      m_RegToolsThreadParams[i].m_ThreadCompleted = false;
      m_RegToolsThreadParams[i].m_ProjectorDataReadyEvent = ::CreateEvent( 
          NULL,               // default security attributes
          TRUE,               // manual-reset event
          FALSE,              // initial state
          data_ready_event_name  // object name
          );
      m_RegToolsThreadParams[i].m_ProjectorCompletedEvent = ::CreateEvent(NULL, TRUE, FALSE, completed_event_name);
      m_RegToolsThread_AllCompletedEvent[i] = m_RegToolsThreadParams[i].m_ProjectorCompletedEvent;

      m_RegToolsThreadParams[i].m_Messages = messages;
      m_RegToolsThreadParams[i].m_MessagesPtr = messages_ptr;
      m_RegToolsThreads[i] = cutStartThread((CUT_THREADROUTINE)RegToolsThread, &(m_RegToolsThreadParams[i]));
      ::WaitForSingleObject(m_RegToolsThreadParams[i].m_ProjectorCompletedEvent, INFINITE);
      ::ResetEvent(m_RegToolsThreadParams[i].m_ProjectorCompletedEvent);
      messages_ptr = m_RegToolsThreadParams[i].m_MessagesPtr;

      m_RegToolsThreadParams[i].m_ProcessingMode = ProcessingMode_Initialize;
    }
    RunRegToolsThreads();

#endif
  }
  else
  {
    // run on the same thread on the first device (no need for multi-thread related initialization)
    m_RegToolsThreadParams = new RegToolsThreadParam[1];
    m_RegToolsThreads = m_RegToolsThread_AllCompletedEvent = NULL;
    m_RegToolsThreadParams[0].m_CudaDeviceID = deviceIDs[0];
    m_RegToolsThreadParams[0].m_CudaDeviceID_Sequential = 0;
    m_RegToolsThreadParams[0].m_CudaDeviceLoad = 1.0;
    m_RegToolsThreadParams[0].m_WithGL = m_WithGL;
    m_RegToolsThreadParams[0].m_ProjectionImagePBOResource = NULL;
    m_RegToolsThreadParams[0].m_ProjectionsInit = 0;
    m_RegToolsThreadParams[0].m_NumViews = 0;
    m_RegToolsThreadParams[0].d_ProjectionsInit = 0;
    m_RegToolsThreadParams[0].d_ProjectionsInit_capacity_bytes = 0;
    m_RegToolsThreadParams[0].m_Messages = NULL;
    RegToolsThread_startup(&(m_RegToolsThreadParams[0]));
  }

#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("GPU initialization on the projector thread is completed\n")
#endif

  return true;
}

RegTools::~RegTools()
{
  // delete all interpolator plans
  while (m_VolumePlan_cudaArrays.size() > 0)
  {
    DeleteVolumePlan_cudaArray(m_VolumePlan_cudaArrays.begin()->first);
  }

  // delete all similarity measure computation plans
  while(m_SimilarityMeasureComputationPlans.size() > 0)
  {
    DeleteSimilarityMeasureComputationPlan(m_SimilarityMeasureComputationPlans.begin()->first);
  }

  // delete all projection parameters array
  while(m_ProjectionParametersSetting.size() > 0)
  {
    DeleteProjectionParametersArray(m_ProjectionParametersSetting.begin()->first);
  }

  // clean up PBO
  DeleteProjectionImagePBO();

  // terminate projector thread
  if(!m_WithGL)
  {
#if _WIN32
    // clean up process run on separate thread
    for(int i = 0; i < m_NumRegToolsThreads; i++)
    {
      m_RegToolsThreadParams[i].m_ThreadCompleted = true;
      ::SetEvent(m_RegToolsThreadParams[i].m_ProjectorDataReadyEvent);
    }
#endif
  //  print_and_log("waiting for thread completion...\n");
  //  ::WaitForMultipleObjects(m_NumRegToolsThreads, m_RegToolsThread_AllCompletedEvent, true, INFINITE);

    for(int i = 0; i < m_NumRegToolsThreads; i++)
    {
      cutEndThread(m_RegToolsThreads[i]);
    }
  //  print_and_log("all thread were completed and ended\n");
  }
  else
  {
    // cleanup process for single device run
    RegToolsThread_cleanup(&(m_RegToolsThreadParams[0]));
  }

  if (m_RegToolsThreadParams)
  {
    delete[] m_RegToolsThreadParams;
  }
  
  if(m_RegToolsThreads)       delete[] m_RegToolsThreads;
  
  if(m_RegToolsThread_AllCompletedEvent) delete[] m_RegToolsThread_AllCompletedEvent;

  RemoveLogFile();
  print_and_log("RegTools closed.\n")
}

bool RegTools::AddLogFile(char* filename)
{
  if(m_LogFile){
    print_and_log("Log file already exists.\n")
    return false;
  }
#if _WIN32
  if( fopen_s(&m_LogFile, filename, "w+") != 0){
#else
  if( (m_LogFile = fopen(filename, "w+")) == NULL){
#endif
    print_and_log("cannot open log file: %s\n", filename)
    return false;
  };
  return true;
}

bool RegTools::RemoveLogFile(void)
{
  if(m_LogFile){
    fclose(m_LogFile);
    m_LogFile = NULL;
  }
  return true;
}

void RegTools::prime_factorization(int x, std::vector<int> &primes)
{
	int c = x;			// remaining product to factor

	while ((c % 2) == 0) {
    primes.push_back(2);
		c = c / 2;
	}

	int i = 3;			// counter
	while (i <= (sqrt((double)c)+1)) {
		if ((c % i) == 0) {
			primes.push_back(i);
			c = c / i;
		}
		else  i = i + 2;
	}

	if (c > 1) primes.push_back(c);
}

void RegTools::findTileSize(int numProjections, int imageWidth, int imageHeight, int &tileX, int &tileY, int &tiledImageX, int &tiledImageY)
{
  // find an appropriate tile size from number of projections and (width, height) of an image
  // we want to get a large image that is close to "square"
  std::vector<int> primes;
  prime_factorization(numProjections, primes);
  int sqrt_totalPixels = static_cast<int>( sqrt((double)(numProjections*imageWidth*imageHeight)) );
  tileX = tileY = 1;
  for(int i=0;i<primes.size();i++){
    if(tileY*imageHeight*primes[i]<sqrt_totalPixels) tileY *= primes[i];
    else                                             tileX *= primes[i];
  }
  tiledImageX = imageWidth * tileX;
  tiledImageY = imageHeight * tileY;
}

void RegTools::copyDataFromCudaArray(float *h_data, struct cudaArray *d_array, int dst_index, int src_index, cudaExtent dimension, int numVolumes)
{
  // copy host data to 3D array
  int image_size = static_cast<int>(dimension.width*dimension.height);
  dimension.depth *= numVolumes;
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcArray = d_array;
  copyParams.srcPos.z = src_index;
  copyParams.dstPtr   = make_cudaPitchedPtr((void*)h_data, dimension.width*sizeof(float), dimension.width, dimension.height);
  copyParams.dstPos.x = 0;
  copyParams.dstPos.y = 0;
  copyParams.dstPos.z = dst_index;
  copyParams.extent   = dimension;
  copyParams.kind     = cudaMemcpyDeviceToHost;
  cutilSafeCall( cudaMemcpy3D(&copyParams) );
  cutilSafeCall( cudaThreadSynchronize() );
}

int RegTools::InitializeCudaDevice(int deviceID, double deviceLoad, bool withGL, char **messages, int &messages_ptr)
{
  // CUDA initialization
  if(deviceID < 0)  deviceID = cutGetMaxGflopsDeviceId();
  if(!withGL) cutilSafeCall( cudaSetDevice( deviceID ) );

  // check CUDA devices
  int deviceCount;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
    print_and_log("error at cudaGetDeviceCount()\n");
    return -1;
  }
  cudaDeviceProp deviceProp;
#if defined RegTools_VERBOSE_MESSAGE
  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaGetDeviceProperties(&deviceProp, dev);
    if (dev == 0) {
		// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
      if (deviceProp.major == 9999 && deviceProp.minor == 9999){
        print_and_log("There is no device supporting CUDA\n")
      } else if (deviceCount == 1) {
        print_and_log("There is 1 device supporting CUDA\n")
      } else {
        print_and_log("There are %d devices supporting CUDA\n", deviceCount)
      }
    }
    print_and_log("Device %d: %s\n", dev, deviceProp.name)
  }
#endif
  cudaGetDeviceProperties( &deviceProp, deviceID );
/*
  CUcontext contextID;
  if(!withGL){
    cuCtxCreate( &contextID, CU_CTX_SCHED_AUTO, deviceID ); // need to create context for first call of cuMemGetInfo()
  } else {
    cuGLCtxCreate( &contextID, CU_CTX_SCHED_AUTO, deviceID ); // need to create context for first call of cuMemGetInfo()
  }
*/
  size_t free_memory, total_memory;
  RegToolsThread_MemGetInfo(free_memory, total_memory);
  char tmp_msg[MAX_MESSAGE_LENGTH];
  int tmp_msg_total_count = 0;
  add_str(tmp_msg, tmp_msg_total_count, "RegTools - Using device %d: %s, Global memory: %.2f MB (%.2f available), compute capability: %d.%d, UVA: %d, %d CUDA cores, %.1f percent load\n"
    , deviceID, deviceProp.name, (double)deviceProp.totalGlobalMem/(1024.0f*1024.0f), (double)free_memory/(1024.0f*1024.0f), deviceProp.major, deviceProp.minor, deviceProp.unifiedAddressing
    , ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount, deviceLoad*100 );

  print_and_log("%s", tmp_msg);
  if(messages){ add_str(messages[0], messages_ptr, tmp_msg); }
  return deviceID;
}

int RegTools::GetProjectionWidth(void)
{
  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  return it != m_ProjectionParametersSetting.end() ? it->second->m_ProjectionWidth : 0; 
}

int RegTools::GetProjectionHeight(void)
{
  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  return it != m_ProjectionParametersSetting.end() ? it->second->m_ProjectionHeight : 0; 
}

int RegTools::GetNumberOfProjections(void)
{
  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  return it != m_ProjectionParametersSetting.end() ? it->second->m_NumProjections : 0; 
}

int RegTools::GetNumberOfEnabledProjections(void)
{ 
  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  return it != m_ProjectionParametersSetting.end() ? it->second->m_NumEnabledProjections : 0;
}

int RegTools::GetNumberOfProjectionSets(void)
{
  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  return it != m_ProjectionParametersSetting.end() ? it->second->m_NumProjectionSets : 1; 
}

int RegTools::InitializeProjectionParametersArray(int numProjections)
{
  ProjectionParametersSetting *new_setting = new ProjectionParametersSetting;
  new_setting->m_ProjectionWidth = new_setting->m_ProjectionHeight = 0;   // user need to specify with SetProjectionDim() 
  new_setting->m_NumProjections = new_setting->m_NumEnabledProjections = numProjections;
  new_setting->m_NumProjectionSets = 1;
  new_setting->m_ProjectionParameters = new ProjectionParameters[numProjections];
  new_setting->m_SubSamplingArray = new int[numProjections];
  new_setting->m_ZeroPixelCount = new int[numProjections];
//  print_and_log("new m_ZeroPixelCount: %d\n", numProjections);
  for(int i=0;i<numProjections;i++){
    InitializeProjectionParametersStruct(&new_setting->m_ProjectionParameters[i]);
    new_setting->m_SubSamplingArray[i] = i;
  }
  // find smallest 'un-used' id
  int id = 0;
  while( m_ProjectionParametersSetting.find(id) != m_ProjectionParametersSetting.end() ) id++;
  m_ProjectionParametersSetting.insert( std::pair<int, ProjectionParametersSetting*>( id, new_setting ) );
  m_CurrentProjectionParametersSettingID = id;

#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("RegTools - %d projection parameters array was created\n", GetNumberOfProjections())
#endif
  return m_CurrentProjectionParametersSettingID;
}

bool RegTools::SetCurrentGeometrySetting(int geometry_id)
{
  m_CurrentProjectionParametersSettingID = geometry_id;
  return true;
}

int RegTools::GetCurrentGeometrySetting(void)
{
  return m_CurrentProjectionParametersSettingID;
}

bool RegTools::DeleteProjectionParametersArray(int id)
{
  /*
  if(m_ProjectionParameters == NULL){
    return false;
  }
  */
  std::map<int, ProjectionParametersSetting*>::iterator it = m_ProjectionParametersSetting.find(id);
  for(int i=0;i<it->second->m_NumProjections;i++) DeleteProjectionParametersStruct(&it->second->m_ProjectionParameters[i]);
  delete[] it->second->m_ProjectionParameters;
  delete[] it->second->m_SubSamplingArray;
  delete[] it->second->m_ZeroPixelCount;
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("RegTools - %d projection parameters array was deleted\n", GetNumberOfProjections());
#endif
  m_ProjectionParametersSetting.erase(it);
  return true;
}

int RegTools::InitializeProjectionParametersStruct(struct ProjectionParameters *projectionParams)
{
  projectionParams->ProjectionMatrix_3x4 = new double[12];
  projectionParams->FOV_mm = new double[2];
  return true;
}

int RegTools::CopyProjectionParametersStruct(struct ProjectionParameters *dst, struct ProjectionParameters *src)
{
  memcpy( dst->ProjectionMatrix_3x4, src->ProjectionMatrix_3x4, sizeof(double)*12 );
  memcpy( dst->FOV_mm, src->FOV_mm, sizeof(double)*2 );
  return true;
}

int RegTools::DeleteProjectionParametersStruct(struct ProjectionParameters *projectionParams)
{
  if(projectionParams->ProjectionMatrix_3x4)    delete projectionParams->ProjectionMatrix_3x4;
  if(projectionParams->FOV_mm)                  delete projectionParams->FOV_mm;
  return true;
}

int RegTools::SetProjectionParameter_objectOriented(int projection_number /* note: this is 0-base index */, struct ProjectionParameters_objectOriented projectionParams)
{
  if(projection_number >= GetNumberOfProjections()){
    print_and_log("RegTools::SetProjectionParameter_objectOriented error. projection number exceeds number of allocated array\n")
    return false;
  }
  ProjectionParameters *param = &(m_ProjectionParametersSetting[m_CurrentProjectionParametersSettingID]->m_ProjectionParameters[projection_number]);
  memcpy(param->FOV_mm, projectionParams.FOV, 2*sizeof(double));
  int result = ConvertProjectionMatrix_ObjectOrientedTo3x4(projectionParams, projectionParams.Pixel_width, param->ProjectionMatrix_3x4);
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("pm3x4(row major) for projection #%d:\n", projection_number);
  double *pm = param->ProjectionMatrix_3x4;
  for(int i=0;i<3;i++)  print_and_log("%f %f %f %f\n", pm[i*4], pm[i*4+1], pm[i*4+2], pm[i*4+3]);
  print_and_log("detector frame 4x4(row major) for projection #%d:\n", projection_number);
  print_and_log("FOV in mm for projection #%d: (%f, %f)\n", projection_number, param->FOV_mm[0], param->FOV_mm[1]);
#endif
  return result;
}

int RegTools::SetProjectionParameter_3x4PM(int projection_number /* note: this is 0-base index */, double *pm3x4_row_major, double pixel_width, double pixel_height
                                           , double u_dim, double v_dim, double down_sample_ratio_u, double down_sample_ratio_v)
{
  if(projection_number >= GetNumberOfProjections()){
    print_and_log("RegTools::SetProjectionParameter_objectOriented error. projection number exceeds number of allocated array\n")
    return false;
  }
//  double pixel_height;
  Downsample3x4ProjectionMatrix(pm3x4_row_major, down_sample_ratio_u, down_sample_ratio_v, u_dim, v_dim);
//  Scale3x4ProjectionMatrix(pm3x4_row_major, pixel_width, &pixel_height);
  Scale3x4ProjectionMatrix(pm3x4_row_major, pixel_width);
//  struct ProjectionParameters_objectOriented converted;
//  InitializeProjectionParametersStruct_objectOriented(&converted);
//  ConvertProjectionMatrix_3x4ToObjectOriented(pm3x4_row_major, pixel_width, u_dim, v_dim, converted, down_sample_ratio);
  ProjectionParameters *param = &(m_ProjectionParametersSetting[m_CurrentProjectionParametersSettingID]->m_ProjectionParameters[projection_number]);
  memcpy(param->ProjectionMatrix_3x4, pm3x4_row_major, 12*sizeof(double)); // Note: if down_sample_ratio != 1.0, ConvertProjectionMatrix_3x4ToObjectOriented() changes pm
  param->FOV_mm[0] = pixel_width * u_dim;   param->FOV_mm[1] = pixel_height * v_dim;
//  memcpy(param->FOV_mm, converted.FOV, 2*sizeof(double));
//  DeleteProjectionParametersStruct_objectOriented(&converted);
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("pm3x4(row major) for projection #%d:\n", projection_number);
  double *pm = param->ProjectionMatrix_3x4;
  for(int i=0;i<3;i++)  print_and_log("%f %f %f %f\n", pm[i*4], pm[i*4+1], pm[i*4+2], pm[i*4+3]);
  print_and_log("detector frame 4x4(row major) for projection #%d:\n", projection_number);
  print_and_log("pixel height: %f\n", pixel_height);
#endif
  return true;
}

int RegTools::SetProjectionParameter_3x4PM_multi(int numProj, double *pm3x4_row_major, double pixel_width, double pixel_height, double u_dim, double v_dim)
{
//  double pixel_height;
//  Downsample3x4ProjectionMatrix(pm3x4_row_major, down_sample_ratio, u_dim, v_dim);
  for(int i=0;i<numProj && i<GetNumberOfProjections();i++){
    ProjectionParameters *param = &(m_ProjectionParametersSetting[m_CurrentProjectionParametersSettingID]->m_ProjectionParameters[i]);
    Scale3x4ProjectionMatrix(pm3x4_row_major+12*i, pixel_width);
    memcpy(param->ProjectionMatrix_3x4, pm3x4_row_major+12*i, 12*sizeof(double));
    param->FOV_mm[0] = pixel_width * u_dim;   param->FOV_mm[1] = pixel_height * v_dim;
  }
  return true;
}

int RegTools::Downsample3x4ProjectionMatrix(double *pm3x4_row_major, double down_sample_ratio_u, double down_sample_ratio_v, double u_dim, double v_dim)
{
  // down-sample intrinsic parameters (in-plane resolution)
  // image center stays the same with respect to the center of the image (u_dim/2, v_dim/2)
  // TODO: this should be modified to a simpler way
  if(down_sample_ratio_u != 1.0 || down_sample_ratio_v != 1.0){
    double pm3x4[12];
    for(int i=0;i<4;i++)  for(int j=0;j<3;j++)  pm3x4[i*3+j] = pm3x4_row_major[j*4+i];  // convert pm3x4 to column-major
//    print_and_log("pm\n");
//    for(int i=0;i<3;i++)  print_and_log("%f %f %f %f\n", pm3x4[i], pm3x4[i+3], pm3x4[i+6], pm3x4[i+9]);

    double F_Object_Camera[16], intrinsics[9], scale;
    DecomposeProjectionMatrix_3x4(pm3x4, F_Object_Camera, intrinsics, &scale);
//    print_and_log("intrinsics\n");
//    for(int i=0;i<3;i++)  print_and_log("%f %f %f\n", intrinsics[i], intrinsics[i+3], intrinsics[i+6]);

    intrinsics[0] /= down_sample_ratio_u; intrinsics[4] /= down_sample_ratio_v; intrinsics[6] /= down_sample_ratio_u; intrinsics[7] /= down_sample_ratio_v;
    ComposeProjectionMatrix_3x4(F_Object_Camera, intrinsics, &scale, pm3x4_row_major);  // returned pm is row-major  
//    print_and_log("F_Object_Camera\n");
//    for(int i=0;i<4;i++)  print_and_log("%f %f %f %f\n", F_Object_Camera[i], F_Object_Camera[i+4], F_Object_Camera[i+8], F_Object_Camera[i+12]);
//    print_and_log("intrinsics\n");
//    for(int i=0;i<3;i++)  print_and_log("%f %f %f\n", intrinsics[i], intrinsics[i+3], intrinsics[i+6]);
  }
  return true;
}

int RegTools::Scale3x4ProjectionMatrix(double *pm3x4_row, double pixel_width, double *pixel_height)
{
  // scale projection matrix so that z=1 at the plane where width of one pixel becomes pixel_width
  // this is just for the purpose of distance weight in linear interpolation back projector...
  // maybe this just introduces confusion. distance weight should be taken into acount by another way
/*  
  double pm3x3_col[9] = {pm3x4_row[0], pm3x4_row[4], pm3x4_row[8], pm3x4_row[1], pm3x4_row[5], pm3x4_row[9], pm3x4_row[2], pm3x4_row[6], pm3x4_row[10]}, inv_col[9];
  RegTools::Inverse3x3d(pm3x3_col, inv_col);
  double scale = sqrt(inv_col[0]*inv_col[0] + inv_col[1]*inv_col[1] + inv_col[2]*inv_col[2]) / pixel_width;
  if(pixel_height)  // return pixel height if needed
    (*pixel_height) = sqrt(inv_col[3]*inv_col[3] + inv_col[4]*inv_col[4] + inv_col[5]*inv_col[5]) / scale;
  for(int i=0;i<12;i++) pm3x4_row[i] *= scale;
*/  
  return true;
}

int RegTools::SetWorldToVolumeTransform(const double *transform_col)
{
  memcpy(m_VolumeTransform_col, transform_col, sizeof(double)*16);
  return true;
}

int RegTools::SetWorldToVolumeTransform(double tx, double ty, double tz, double rx, double ry, double rz)
{
  LoadIdentity_4x4d(m_VolumeTransform_col);
  // rotation -> translation
  RotateX_col(m_VolumeTransform_col, rx);  // rx is in radians
  RotateY_col(m_VolumeTransform_col, ry);  // ry is in radians
  RotateZ_col(m_VolumeTransform_col, rz);  // rz is in radians
  Translate_col(m_VolumeTransform_col, tx, ty, tz);
  return true;
}

int RegTools::CopyProjectionParametersStruct_objectOriented(struct ProjectionParameters_objectOriented *dst, struct ProjectionParameters_objectOriented *src)
{
  memcpy( dst->DetectorFrame, src->DetectorFrame, sizeof(double)*16 );
  memcpy( dst->FOV, src->FOV, sizeof(double)*2 );
  memcpy( dst->Pixel_aspect_ratio, src->Pixel_aspect_ratio, sizeof(double)*1 );
  *dst->Pixel_aspect_ratio = *src->Pixel_aspect_ratio;
  memcpy( dst->Skew_angle_deg, src->Skew_angle_deg, sizeof(double)*1 );
  memcpy( dst->SourcePosition, src->SourcePosition, sizeof(double)*3 );
  return true;
}

int RegTools::InitializeProjectionParametersStruct_objectOriented(struct ProjectionParameters_objectOriented *projectionParams)
{
  projectionParams->DetectorFrame = (double*)malloc( sizeof(double)*16 );
  projectionParams->FOV =           (double*)malloc( sizeof(double)*2 );
  projectionParams->Pixel_aspect_ratio = (double*)malloc( sizeof(double)*1 );
  *projectionParams->Pixel_aspect_ratio = 1.0f;
  projectionParams->Skew_angle_deg = (double*)malloc( sizeof(double)*1 );
  *projectionParams->Skew_angle_deg = 90.0f;
  projectionParams->SourcePosition = (double*)malloc( sizeof(double)*3 );
  projectionParams->SourcePosition[0] = projectionParams->SourcePosition[1] = projectionParams->SourcePosition[2] = 0.0;
//  projectionParams->IsPerspective = true;
  return true;
}

int RegTools::DeleteProjectionParametersStruct_objectOriented(struct ProjectionParameters_objectOriented *projectionParams)
{
  if(projectionParams->DetectorFrame) free(projectionParams->DetectorFrame);
  if(projectionParams->FOV)           free(projectionParams->FOV);
  if(projectionParams->Pixel_aspect_ratio) free(projectionParams->Pixel_aspect_ratio);
  if(projectionParams->Skew_angle_deg)     free(projectionParams->Skew_angle_deg);
  if(projectionParams->SourcePosition)     free(projectionParams->SourcePosition);
  return true;
}

int RegTools::ConvertProjectionMatrix_ObjectOrientedTo3x4(const ProjectionParameters_objectOriented in, const double pixel_width, double *pm3x4_row_major)
{
  // pm3x4_row_major: row-major
  // Although we internally use column-major matrix, the matrix stored in ProjectionParameters_OpenGL is row-major for compatibility of other C-based software
  // See the following book for detail
  // David A. Forsyth, Jean Ponce; Computer Vision - A modern approach -,
  // Prentice Hall Series in Artificial Intelligence
  //
  if(in.Skew_angle_deg == NULL || in.Pixel_aspect_ratio == NULL)  return 0;

  double m[16], D_on_S[3], intrinsic[9];
  // compute inverse of 'World to Source (which is [DetectorFrame(1:3,1:3), SourcePosition; [0 0 0 1]])'.
  // both input and output are column-major
  m[0] = in.DetectorFrame[0];  m[4] = in.DetectorFrame[1];  m[8] = in.DetectorFrame[2];  
  m[1] = in.DetectorFrame[4];  m[5] = in.DetectorFrame[5];  m[9] = in.DetectorFrame[6];  
  m[2] = in.DetectorFrame[8];  m[6] = in.DetectorFrame[9];  m[10]= in.DetectorFrame[10]; 
  m[12] = -(in.DetectorFrame[0]*in.SourcePosition[0]+in.DetectorFrame[1]*in.SourcePosition[1]+in.DetectorFrame[2]*in.SourcePosition[2]);
  m[13] = -(in.DetectorFrame[4]*in.SourcePosition[0]+in.DetectorFrame[5]*in.SourcePosition[1]+in.DetectorFrame[6]*in.SourcePosition[2]);
  m[14] = -(in.DetectorFrame[8]*in.SourcePosition[0]+in.DetectorFrame[9]*in.SourcePosition[1]+in.DetectorFrame[10]*in.SourcePosition[2]);
  m[3] = 0.0;   m[7] = 0.0;   m[11]= 0.0;   m[15] = 1.0;
  // compute detector position in source frame (sFw * d)
  for(int i=0;i<3;i++)  D_on_S[i] = m[i]*in.DetectorFrame[12] + m[i+4]*in.DetectorFrame[13] + m[i+8]*in.DetectorFrame[14] + m[i+12];

  // form intrinsic parameter matrix
  double pixel_height = pixel_width/(*in.Pixel_aspect_ratio);
  double u0 = (in.FOV[0]/2 - D_on_S[0])/pixel_width, v0 = (in.FOV[1]/2 - D_on_S[1])/pixel_height;
  double skew_rad = (*in.Skew_angle_deg) * (M_PI/180);
  double cot_skew = 1 / tan(skew_rad);  // cotangent of skew angle
  double alpha = D_on_S[2]/pixel_width, beta = D_on_S[2]/pixel_height;
  intrinsic[0] = alpha;   intrinsic[3] = -alpha*cot_skew;     intrinsic[6] = u0;
  intrinsic[1] = 0;       intrinsic[4] = beta/sin(skew_rad);  intrinsic[7] = v0;
  intrinsic[2] = 0;       intrinsic[5] = 0;                   intrinsic[8] = 1;

  // compute intrinsic * extrinsic and store to pm3x4 in row-major order
  // note: both intrinsic and m[] (extrinsic) is in column-major at this point
  for(int row=0;row<3;row++)  for(int column=0;column<3;column++)
    pm3x4_row_major[row*4+column] = intrinsic[row]*m[column*4+0]+intrinsic[row+3]*m[column*4+1]+intrinsic[row+6]*m[column*4+2];
  for(int row=0;row<3;row++)
    pm3x4_row_major[row*4+3] = intrinsic[row]*m[12]+intrinsic[3+row]*m[13]+intrinsic[6+row]*m[14];

  Scale3x4ProjectionMatrix(pm3x4_row_major, pixel_width);
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("FOV: %f, %f\n", in.FOV[0], in.FOV[1]);
#endif
  return 1;
}

bool RegTools::ComposeProjectionMatrix_3x4(const double *F_Object_Source_4x4, const double *in, const double *scale, double *out_row_major_3x4)
{
  // compute scale * [intrinsics [0;0;0]] * inv(F_Object_Source)
  double ext[16];
  Inverse4x4d(F_Object_Source_4x4, ext);
  for(int i=0;i<4;i++)  for(int j=0;j<3;j++)  out_row_major_3x4[j*4+i] = (in[j]*ext[i*4]+in[j+3]*ext[i*4+1]+in[j+6]*ext[i*4+2]) * (*scale);
  return true;
}

bool RegTools::DecomposeProjectionMatrix_3x4(const double *in_column_major_3x4, double *F_Object_Source_4x4, double *intrinsics_3x3, double *scale)
{
  // Decompose camera projection matrix
  // Note that we want (upper triangular matrix) * (orthonormal matrix)
  // which is opposite order to QR decomposition.
  // So we use the following relation: inv(QR) = inv(R) * inv(Q).
  // See, for example,
  // http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node5.html
  // for detail of the decomposition of camera projection matrix

  // since QR decomposition doesn't work for double-precision stably, we do complicated type-cast many times
  // but if we use external linear algebra library that can do double-precision QR decomposition stably, this would become much simpler

  double prm_3x3d[9], inv_prm3x3d[9];
  for(int i=0;i<3;i++)  for(int j=0;j<3;j++)  prm_3x3d[i*3+j] = in_column_major_3x4[i*3+j]; // copy 3x3 part of the input matrix (column-major -> colum-major)

  if(!Inverse3x3d(prm_3x3d, inv_prm3x3d)) return false;
  double Q_column_major[9], R_column_major[9];
  if(!QRDecomposition_square(inv_prm3x3d, 3, Q_column_major, R_column_major)) return false;

  if(!Inverse3x3d(R_column_major, intrinsics_3x3)) return false;
  double extrinsic[9];
  for(int i=0;i<3;i++)  for(int j=0;j<3;j++)  extrinsic[i*3+j] = Q_column_major[j*3+i]; // transpose
  
  // We change the sign and scale of the upper triangular matrix 
  // in order to make it appropriate form of intrinsic parameters
  // (diagonal elements are both negative and (3,3) element is positive one)
#if defined RegTools_VERBOSE_MESSAGE
    print_and_log("DecomposeProjectionMatrix_3x4, intrinsics_3x3 after QR\n");  // column-major
    for(int i=0;i<3;i++)  print_and_log("%f %f %f\n", intrinsics_3x3[i], intrinsics_3x3[i+3], intrinsics_3x3[i+6]);
#endif
    for(int i=0;i<3;i++){
    if((i<2 && intrinsics_3x3[i*3+i]>0) || (i==2 && intrinsics_3x3[i*3+i]<0)){
      for(int j=0;j<3;j++){
        intrinsics_3x3[i*3+j] = -intrinsics_3x3[i*3+j];
        extrinsic[j*3+i] = -extrinsic[j*3+i];
      }
    }
  }
  *scale = intrinsics_3x3[8];
  for(int i=0;i<9;i++)  intrinsics_3x3[i] /= (*scale);

  // if the extrinsic parameter is left handed, we change the sign to make
  // it right handed
  double cross_prod[3];
  CrossProduct3(extrinsic, extrinsic+3, cross_prod);
  if(DotProduct3(extrinsic+6, cross_prod) < 0){
    for(int i=0;i<9;i++)  extrinsic[i] *= -1;
    (*scale) *= -1;
  }

  // translation part of the extrinsic parameter can be computed by
  // multiplying inverse of the intrinsic parameter and 4th column of
  // the camera projection matrix
  // P = s C [ R | T ] -> T = inv(C) * (P(:,4) / s)
  double inv_int[9], trans[3], intrinsics_3x3_column_major[9];
  for(int i=0;i<3;i++)  for(int j=0;j<3;j++)  intrinsics_3x3_column_major[i*3+j] = intrinsics_3x3[j*3+i];
  if(!Inverse3x3d(intrinsics_3x3_column_major, inv_int)) return false;
  trans[0] = inv_int[0]*in_column_major_3x4[9] + inv_int[1]*in_column_major_3x4[10] + inv_int[2]*in_column_major_3x4[11];
  trans[1] = inv_int[3]*in_column_major_3x4[9] + inv_int[4]*in_column_major_3x4[10] + inv_int[5]*in_column_major_3x4[11];
  trans[2] = inv_int[6]*in_column_major_3x4[9] + inv_int[7]*in_column_major_3x4[10] + inv_int[8]*in_column_major_3x4[11];
  for(int i=0;i<3;i++)  trans[i] /= (*scale);

  // store return variables
  double camera_object[16]; // column-major
  for(int i=0;i<3;i++)  for(int j=0;j<3;j++)  camera_object[j*4+i] = extrinsic[i*3+j];  // row-major -> column-major
  for(int i=0;i<3;i++)  camera_object[i*4+3] = trans[i];
  camera_object[12] = camera_object[13] = camera_object[14] = 0.0;
  camera_object[15] = 1.0;
  if(!Inverse4x4d(camera_object, F_Object_Source_4x4)) return false;
  TransposeMatrixd(F_Object_Source_4x4);
  return true;
}

bool RegTools::SetVolumeInfo(int volume_dim_x, int volume_dim_y, int volume_dim_z, float voxelSize_x, float voxelSize_y, float voxelSize_z)
{
  m_VolumeDim = make_cudaExtent(volume_dim_x, volume_dim_y, volume_dim_z);
  m_VoxelSize_mm = make_float3(voxelSize_x, voxelSize_y, voxelSize_z);

#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("RegTools - volume dimensions = (%d, %d, %d), voxel size = (%f, %f, %f)\n", volume_dim_x, volume_dim_y, volume_dim_z, voxelSize_x, voxelSize_y, voxelSize_z);
#endif
  return true;
}

bool RegTools::GetVolumeInfo(int *volume_dim_x, int *volume_dim_y, int *volume_dim_z, float *voxelSize_x, float *voxelSize_y, float *voxelSize_z)
{
  *volume_dim_x = static_cast<int>(m_VolumeDim.width);  *volume_dim_y = static_cast<int>(m_VolumeDim.height); *volume_dim_z = static_cast<int>(m_VolumeDim.depth);
  *voxelSize_x = m_VoxelSize_mm.x;    *voxelSize_y = m_VoxelSize_mm.y;    *voxelSize_z = m_VoxelSize_mm.z;
  return true;
}

bool SetStepSize(float step_size);
bool GetStepSize(float *step_size);

bool RegTools::SetStepSize(float step_size)
{
  m_StepSize = step_size;
  return true;
}

bool RegTools::GetStepSize(float *step_size)
{
  *step_size = m_StepSize;
  return true;
}

int RegTools::SetProjectionDim(int width, int height)
{
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("RegTools - SetProjectionImageSize: (%d, %d)\n", width, height);
#endif
  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  if(it == m_ProjectionParametersSetting.end()){
    print_and_log("error: RegTools::SetProjectionDim, invalid projection parameters array\n")
  } else {
    it->second->m_ProjectionWidth = width;
    it->second->m_ProjectionHeight = height;
  }
  return true;  // TODO: error handling
}

int RegTools::GetPixelSize(double *pixel_width, double *pixel_height)
{
  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  if(it != m_ProjectionParametersSetting.end() && it->second->m_NumProjections>0){
    *pixel_width = it->second->m_ProjectionParameters[0].FOV_mm[0]/it->second->m_ProjectionWidth;
    *pixel_height = it->second->m_ProjectionParameters[0].FOV_mm[1]/it->second->m_ProjectionHeight;
  } else {
    *pixel_width = *pixel_height = 0;
  }
  return true; 
}

int RegTools::GetProjectionDim(int *width, int *height)
{
  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  if(it == m_ProjectionParametersSetting.end()){
    print_and_log("error: RegTools::SetProjectionDim, invalid projection parameters array\n");
  } else {
    (*width) = GetProjectionWidth();
    (*height)= GetProjectionHeight();
  }
  return true;  // TODO: error handling
}

int RegTools::GetProjectionMatrices(double *pm)
{
  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  if(it == m_ProjectionParametersSetting.end()){
    print_and_log("error: RegTools::SetProjectionDim, invalid projection parameters array\n")
  } else {
    for(int i=0;i<it->second->m_NumProjections;i++)
      memcpy(pm+12*i, it->second->m_ProjectionParameters[i].ProjectionMatrix_3x4, sizeof(double)*12);
  }
  return true;  // TODO: error handling
}

int RegTools::SetNumberOfProjectionSets(int num_projection_set)
{
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("RegTools - SetNumberOfProjectionSets: (%d)\n", num_projection_set);
#endif
  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  if(it != m_ProjectionParametersSetting.end())
    m_ProjectionParametersSetting[m_CurrentProjectionParametersSettingID]->m_NumProjectionSets = num_projection_set;
  return true;
}

int RegTools::SetTransferBlockSize(int numBlockSize)
{
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("RegTools - SetTransferBlockSize: %d\n", numBlockSize);
#endif
  m_TransferBlockSize = numBlockSize;
  return true;
}

int RegTools::ComputeBoxProjectionBoundingSquare(int *projected_square_left_bottom, int *projected_size, int *in_out, double *box_center, double *box_size, int margin)
{
  // compute size and location of squares for each projection that contains projection of the given box
  // the size is the same for all projection
  int num_proj = GetNumberOfProjections();
  double *projection_centers = new double[num_proj*2];
  double max_square_size[2] = {-DBL_MAX, -DBL_MAX};
  int projection_width, projection_height;
  GetProjectionDim(&projection_width, &projection_height);
  for(int i=0;i<num_proj;i++){
    // compute projection of 8 vertices of the given box
    double *pm = m_ProjectionParametersSetting[m_CurrentProjectionParametersSettingID]->m_ProjectionParameters[i].ProjectionMatrix_3x4, bsquare_min[2], bsquare_max[2];
    ComputeBoxProjection(pm, box_center, box_size, bsquare_min, bsquare_max);
    // compute center of the bounding square
    projection_centers[i*2+0] = (bsquare_min[0]+bsquare_max[0])/2.0f;
    projection_centers[i*2+1] = (bsquare_min[1]+bsquare_max[1])/2.0f;
    if(bsquare_min[0]<0) bsquare_min[0] = 0; else if(bsquare_min[0]>=projection_width) bsquare_min[0] = projection_width-1;
    if(bsquare_max[0]<0) bsquare_max[0] = 0; else if(bsquare_max[0]>=projection_width) bsquare_max[0] = projection_width-1;
    if(bsquare_min[1]<0) bsquare_min[1] = 0; else if(bsquare_min[1]>=projection_height) bsquare_min[1] = projection_height-1;
    if(bsquare_max[1]<0) bsquare_max[1] = 0; else if(bsquare_max[1]>=projection_height) bsquare_max[1] = projection_height-1;
//	print_and_log("proj #%d, bsquare_min:(%f,%f), bsquare_max:(%f,%f), width&height:(%d,%d)\n", i, bsquare_min[0], bsquare_min[1], bsquare_max[0], bsquare_max[1], projection_width, projection_height);
/*
    // addjust the bounding box so that the box_center becomes exact center of the image
    double projected_vert[3];
    for(int k=0;k<3;k++)  projected_vert[k] = pm[k*4+0]*box_center[0] + pm[k*4+1]*box_center[1] + pm[k*4+2]*box_center[2] + pm[k*4+3];
    projection_centers[i*2+0] = projected_vert[0]/projected_vert[2]; projection_centers[i*2+1] = projected_vert[1]/projected_vert[2];
    for(int k=0;k<1;k++){
      if(bsquare_max[k]-projection_centers[i*2+k]>projection_centers[i*2+k]-bsquare_min[k])
        bsquare_min[k] = projection_centers[i*2+k]-(bsquare_max[k]-projection_centers[i*2+k]);
      else
        bsquare_max[k] = projection_centers[i*2+k]+(projection_centers[i*2+k]-bsquare_min[k]);
    }
*/
    // find the maximum size of the bounding square
    double square_size[2] = {bsquare_max[0]-bsquare_min[0], bsquare_max[1]-bsquare_min[1]};
    if(max_square_size[0] < square_size[0]) max_square_size[0] = square_size[0];
    if(max_square_size[1] < square_size[1]) max_square_size[1] = square_size[1];
  }
  // compute size of the square that contains all projected box
  projected_size[0] = static_cast<int>(ceil(max_square_size[0]+1)); projected_size[1] = static_cast<int>(ceil(max_square_size[1]+1));
  if(projected_size[0]%2>0) projected_size[0]++;  // force even size
  if(projected_size[1]%2>0) projected_size[1]++;
  // if the square size is larger than the projection image, change the size to fit the projection
  if(projected_size[0]>projection_width)  projected_size[0] = projection_width;
  if(projected_size[1]>projection_height)  projected_size[1] = projection_height;

  for(int i=0;i<num_proj;i++){
    // compute left bottom pixel for each image
//    projected_square_left_bottom[i*2+0] = static_cast<int>( ceil(projection_centers[i*2+0]-max_square_size[0]/2) );
//    projected_square_left_bottom[i*2+1] = static_cast<int>( ceil(projection_centers[i*2+1]-max_square_size[1]/2) );
    projected_square_left_bottom[i*2+0] = static_cast<int>( ceil(projection_centers[i*2+0]-projected_size[0]/2) );
    projected_square_left_bottom[i*2+1] = static_cast<int>( ceil(projection_centers[i*2+1]-projected_size[1]/2) );
    // check if the entire box is inside of the cropped projection or outside
//    in_out[i] = (projected_square_left_bottom[i*2+0]>=projection_width || projected_square_left_bottom[i*2+1]>=projection_height
//          || (projected_square_left_bottom[i*2+0]+projected_size[0])<0 || (projected_square_left_bottom[i*2+1]+projected_size[1])<0) ? 0 : 1;
    // check if the projection of the box center is inside of the cropped projection or outside
    in_out[i] = (projection_centers[i*2+0]>=(projection_width-margin) || projection_centers[i*2+1]>=(projection_height-margin)
          || projection_centers[i*2+0]<margin || projection_centers[i*2+1]<margin) ? 0 : 1;
  }

  // if left bottom pixel is negative, move the image center accordingly
  for(int i=0;i<num_proj;i++){
    if(projected_square_left_bottom[i*2+0]<0) projected_square_left_bottom[i*2+0] = 0;
    if(projected_square_left_bottom[i*2+1]<0) projected_square_left_bottom[i*2+1] = 0;
    if(projected_square_left_bottom[i*2+0]+projected_size[0]>projection_width)  projected_square_left_bottom[i*2+0] = projection_width-projected_size[0];
    if(projected_square_left_bottom[i*2+1]+projected_size[1]>projection_height) projected_square_left_bottom[i*2+1] = projection_height-projected_size[1];
  }
  
  delete[] projection_centers;
  return true;
}

int RegTools::CropAllProjections(int *left_bottom /* numbefOfProjections*2 element array */, int *crop_size /* 2 element array */)
{
  if(m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID) == m_ProjectionParametersSetting.end()) return false;
  struct ProjectionParametersSetting *setting = m_ProjectionParametersSetting[m_CurrentProjectionParametersSettingID];
  struct ProjectionParameters_objectOriented objectOriented;
  InitializeProjectionParametersStruct_objectOriented(&objectOriented);
  int num_proj = GetNumberOfProjections();
  for(int n=0;n<num_proj;n++){
    ProjectionParameters *param = &(setting->m_ProjectionParameters[n]);
    // viewport transform
    ShiftProjectionMatrices(param->ProjectionMatrix_3x4, 1, left_bottom+n*2);

    double pixel_size[2] = {param->FOV_mm[0]/setting->m_ProjectionWidth, param->FOV_mm[1]/setting->m_ProjectionHeight}; // pixel size doesn't change
    param->FOV_mm[0] = pixel_size[0]*crop_size[0];  param->FOV_mm[1] = pixel_size[1]*crop_size[1];
/*
    print_and_log("cropped pm3x4 for projection %d\n", n);    // row-major
    for(int i=0;i<3;i++)  print_and_log("%f %f %f %f\n", param->ProjectionMatrix_3x4[i*4], param->ProjectionMatrix_3x4[i*4+1], param->ProjectionMatrix_3x4[i*4+2], param->ProjectionMatrix_3x4[i*4+3]);
    print_and_log("FOV_mm: (%f, %f)\n", param->FOV_mm[0], param->FOV_mm[1]);
    print_and_log("Left-bottom: (%d, %d)\n", left_bottom[n*2+0], left_bottom[n*2+1]);
*/
  }
  setting->m_ProjectionWidth = crop_size[0];  setting->m_ProjectionHeight = crop_size[1];
  DeleteProjectionParametersStruct_objectOriented(&objectOriented);
  return true;
}

int RegTools::ShiftProjectionMatrices(double *pm_all, int num_proj, int *left_bottom /* numbefOfProjections*2 element array, 0-base */)
{
  for(int n=0;n<num_proj;n++){
    // viewport transform
    double *pm = pm_all+n*12;
    pm[0] -= left_bottom[n*2+0]*pm[8]; pm[1] -= left_bottom[n*2+0]*pm[9]; pm[2] -= left_bottom[n*2+0]*pm[10]; pm[3] -= left_bottom[n*2+0]*pm[11]; 
    pm[4] -= left_bottom[n*2+1]*pm[8]; pm[5] -= left_bottom[n*2+1]*pm[9]; pm[6] -= left_bottom[n*2+1]*pm[10]; pm[7] -= left_bottom[n*2+1]*pm[11]; 
  }
  return true;
}

int RegTools::SetSubSamplingVector(int *sub_sampling_vector, int numElements)
{
  ProjectionParametersSetting *setting = m_ProjectionParametersSetting[m_CurrentProjectionParametersSettingID];
  if(setting->m_SubSamplingArray) delete[] setting->m_SubSamplingArray;
  if(numElements != setting->m_NumProjections){
    print_and_log("number of elements should be the same as number of projection images")
    return false;
  }
  // count enabled projections
  setting->m_NumEnabledProjections = 0;
  for(int i=0;i<setting->m_NumProjections;i++) if(sub_sampling_vector[i])  setting->m_NumEnabledProjections++;
  setting->m_SubSamplingArray = new int[setting->m_NumEnabledProjections];
  for(int i=0, count=0;i<setting->m_NumProjections;i++)
    if(sub_sampling_vector[i])  setting->m_SubSamplingArray[count++] = i;
  return true;
}

int RegTools::GetSubSamplingVector(int *sub_sampling_vector, int numElements)
{
  std::map<int,ProjectionParametersSetting*>::const_iterator geom = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  for(int i=0;i<numElements;i++) sub_sampling_vector[i] = geom->second->m_SubSamplingArray[i];
//  print_and_log("m_SubSamplingArray = ");
//  for(int i=0;i<geom->second->m_NumEnabledProjections;i++){ print_and_log("%d ", geom->second->m_SubSamplingArray[i]); }
//  print_and_log("\n");
  return true;
}

int RegTools::EraseDisabledProjections(void)
{
  ProjectionParametersSetting *setting = m_ProjectionParametersSetting[m_CurrentProjectionParametersSettingID];
  int enabled_indx = 0;
  ProjectionParameters* newProjectionParameters = new ProjectionParameters[setting->m_NumEnabledProjections];
  for(int i=0;i<setting->m_NumProjections;i++){ 
    if(enabled_indx < setting->m_NumEnabledProjections && i==setting->m_SubSamplingArray[enabled_indx]){
//      print_and_log("Not delete projection #%d\n", i);
      newProjectionParameters[enabled_indx] = setting->m_ProjectionParameters[i];
      enabled_indx++;
    } else {
//      print_and_log("Delete projection #%d\n", i);
      DeleteProjectionParametersStruct(&setting->m_ProjectionParameters[i]);
    }
  }
  delete[] setting->m_ProjectionParameters;
  delete[] setting->m_SubSamplingArray;
  setting->m_ProjectionParameters = newProjectionParameters;
  setting->m_SubSamplingArray = new int[setting->m_NumEnabledProjections];
  for(int i=0;i<setting->m_NumEnabledProjections;i++) setting->m_SubSamplingArray[i] = i;
  setting->m_NumProjections = setting->m_NumEnabledProjections;
//  print_and_log("setting->m_NumProjections: %d\n", setting->m_NumProjections);
  return true;
}

int RegTools::ReplicateProjections(int num_rep)
{
  ProjectionParametersSetting *setting = m_ProjectionParametersSetting[m_CurrentProjectionParametersSettingID];
  int enabled_indx = 0;
  ProjectionParameters* newProjectionParameters = new ProjectionParameters[setting->m_NumProjections*num_rep];
  int *newSubSamplingArray = new int[setting->m_NumProjections*num_rep];
  delete[] setting->m_ZeroPixelCount; setting->m_ZeroPixelCount = new int[setting->m_NumProjections*num_rep];
//  print_and_log("new ZeroPixelCount: %d\n", setting->m_NumProjections*num_rep);
  for(int j=0;j<num_rep;j++){
    for(int i=0;i<setting->m_NumProjections;i++){
      InitializeProjectionParametersStruct(&(newProjectionParameters[j*setting->m_NumProjections+i]));
      CopyProjectionParametersStruct(&(newProjectionParameters[j*setting->m_NumProjections+i]), &(setting->m_ProjectionParameters[i]));

      newSubSamplingArray[j*setting->m_NumProjections+i] = j*setting->m_NumProjections+i; //setting->m_SubSamplingArray[i];
    }
  }
  for(int i=0;i<setting->m_NumProjections;i++) DeleteProjectionParametersStruct(&setting->m_ProjectionParameters[i]);
  delete[] setting->m_ProjectionParameters;
  delete[] setting->m_SubSamplingArray;

  setting->m_ProjectionParameters = newProjectionParameters;
  setting->m_SubSamplingArray = newSubSamplingArray;
  setting->m_NumProjections *= num_rep;
  setting->m_NumEnabledProjections *= num_rep;
  return true;
}

int RegTools::ForwardProjection(ProjectionResult &result, const float *volume)
{
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("RegTools - ForwardProjection, dim: proj(%d, %d, %d, %d), vol(%d, %d, %d)\n"
    , GetProjectionWidth(), GetProjectionHeight(), GetNumberOfProjections(), GetNumberOfProjectionSets(), m_VolumeDim.width, m_VolumeDim.height, m_VolumeDim.depth);
  print_and_log("RegTools - ForwardProjection, transfer block size: %d\n", m_TransferBlockSize);
#endif
  const float *input_volume = volume;

  int numberOfRunningThreads = m_NumRegToolsThreads;
  if(result.numGPU>0)  numberOfRunningThreads = result.numGPU;
//  print_and_log("number of running threads: %d\n", numberOfRunningThreads);

  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  if(it == m_ProjectionParametersSetting.end()) return false; // error
  float *minValues = NULL, *maxValues = NULL;
  if(result.minValue){ 
    minValues = new float[numberOfRunningThreads]; maxValues = new float[numberOfRunningThreads]; 
    *(result.minValue) = FLT_MAX; *(result.maxValue) = -FLT_MAX;
  }
  for(int i=0,start_proj=0,nProj=0;i<numberOfRunningThreads;i++, start_proj+=nProj){
    // calculate number of projection that is computed on this device
    if(i == numberOfRunningThreads-1) nProj = it->second->m_NumEnabledProjections-start_proj;
    else nProj = ceil((double)it->second->m_NumEnabledProjections * m_RegToolsThreadParams[i].m_CudaDeviceLoad);
//    print_and_log("%d forward projections on device %d\n", nProj, i);

    m_RegToolsThreadParams[i].m_ProcessingMode = ProcessingMode_ForwardProjection;
    m_RegToolsThreadParams[i].m_Volume = const_cast<float*>(input_volume);
    m_RegToolsThreadParams[i].m_VolumePlan_cudaArray = NULL;
    m_RegToolsThreadParams[i].m_VolumePlan_cudaArray_out = FindVolumePlan(result.dDataID);
    m_RegToolsThreadParams[i].m_Projections = result.Data;
    // TODO: m_RegToolsThreadParams[i].d_ProjectionsInit = result.dev_init_data;
//    print_and_log("result.Data: %d, input_volume: %d\n", result.Data, input_volume);
    if(m_RegToolsThreadParams[i].m_Projections) m_RegToolsThreadParams[i].m_Projections += (it->second->m_ProjectionWidth*it->second->m_ProjectionHeight*start_proj);
    m_RegToolsThreadParams[i].m_ElapsedTime = result.projectionTime;
    if(result.minValue){ m_RegToolsThreadParams[i].m_MinValue = &(minValues[i]); m_RegToolsThreadParams[i].m_MaxValue = &(maxValues[i]); }
    else               { m_RegToolsThreadParams[i].m_MinValue = m_RegToolsThreadParams[i].m_MaxValue = NULL; }

    PrepareForRegToolsThread(&m_RegToolsThreadParams[i]);
    m_RegToolsThreadParams[i].m_ProjectionsInit = result.initData;
    m_RegToolsThreadParams[i].m_EnabledProjection += start_proj; // offset
    m_RegToolsThreadParams[i].m_NumAllocatedProjections = it->second->m_NumEnabledProjections;  // note: bad naming...
    m_RegToolsThreadParams[i].m_NumEnabledProjections = nProj;
  }

//  print_and_log("RunRegToolsThreads\n");
  if(result.numGPU>0){
//    print_and_log("single GPU mode\n");
    for(int i=0;i<result.numGPU;i++)  RunRegToolsThreads(i);  // this is inefficient if numGPU is larger than 2
  } else {
    RunRegToolsThreads();
  }
//  print_and_log("Forward projection completed\n");
  if(result.minValue){
    for(int i=0;i<numberOfRunningThreads;i++){
      if(*(result.minValue)>minValues[i])  *(result.minValue) = minValues[i];
      if(*(result.maxValue)<maxValues[i])  *(result.maxValue) = maxValues[i];
    }
    delete[] minValues; delete[] maxValues;
  }
  return true;
}

int RegTools::ForwardProjection_with3x4ProjectionMatrices(ProjectionResult &result, const int plan_id, const double *pm_3x4)
{
  std::map<int,ProjectionParametersSetting*>::const_iterator geom = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  if(geom == m_ProjectionParametersSetting.end()){
    print_and_log("error: RegTools::ForwardProjection_withPlan, invalid geometry setting\n")
    return 0;
  } else {
    for(int i=0;i<geom->second->m_NumEnabledProjections;i++){
      int indx = geom->second->m_SubSamplingArray[i];
      memcpy(geom->second->m_ProjectionParameters[indx].ProjectionMatrix_3x4, pm_3x4+12*i, sizeof(double)*12);
    }

    std::map<int, VolumePlan_cudaArray*>::iterator it = m_VolumePlan_cudaArrays.find(plan_id);
    if(it == m_VolumePlan_cudaArrays.end()){
      print_and_log("RegTools::ForwardProjection_with3x4ProjectionMatrices(), Invalid plan ID\n")
      return false;
    } else {
      SetVolumeInfo(it->second->VolumeDim[0], it->second->VolumeDim[1], it->second->VolumeDim[2], it->second->VoxelSize[0], it->second->VoxelSize[1], it->second->VoxelSize[2]);
      for(int j=0,start_proj=0,nProj=0;j<m_NumRegToolsThreads;j++, start_proj+=nProj){
        // calculate number of projection that is computed on this device
        nProj = ceil((double)geom->second->m_NumEnabledProjections * m_RegToolsThreadParams[j].m_CudaDeviceLoad);
        if((start_proj+nProj)>geom->second->m_NumEnabledProjections)  nProj = geom->second->m_NumEnabledProjections-start_proj;
//        print_and_log("%d forward projections on device %d\n", nProj, j);

        if(nProj==0)  m_RegToolsThreadParams[j].m_ProcessingMode = ProcessingMode_DoNothing;
        else {
          m_RegToolsThreadParams[j].m_ProcessingMode = ProcessingMode_ForwardProjection;
          m_RegToolsThreadParams[j].m_Volume = NULL;
          m_RegToolsThreadParams[j].m_VolumePlan_cudaArray = it->second;
          m_RegToolsThreadParams[j].m_VolumePlan_cudaArray_out = FindVolumePlan(result.dDataID);
          m_RegToolsThreadParams[j].m_Projections = result.Data;
          // TODO: handle device pointer
          if(m_RegToolsThreadParams[j].m_Projections) m_RegToolsThreadParams[j].m_Projections += (geom->second->m_ProjectionWidth*geom->second->m_ProjectionHeight*start_proj);
          m_RegToolsThreadParams[j].m_ElapsedTime = result.projectionTime;
          PrepareForRegToolsThread(&(m_RegToolsThreadParams[j]));
		  m_RegToolsThreadParams[j].m_ProjectionsInit = result.initData;
          m_RegToolsThreadParams[j].m_EnabledProjection += start_proj; // offset
          m_RegToolsThreadParams[j].m_NumAllocatedProjections = geom->second->m_NumEnabledProjections;  // note: bad naming...
          m_RegToolsThreadParams[j].m_NumEnabledProjections = nProj;
          m_RegToolsThreadParams[j].m_World_Volume_col = NULL;  // disable world to volume transform for now for speeding up
        }
      }
      RunRegToolsThreads();
      for(int j=0;j<m_NumRegToolsThreads;j++) m_RegToolsThreadParams[j].m_World_Volume_col = m_VolumeTransform_col;
      return true;
    }
  }
}

int RegTools::ForwardProjection_withPlan(ProjectionResult &result, const int plan_id)
{
  // forward projection with plan using default settings (no local & global transform) with "1" view
  double I[16] = {1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1};  // identity transformation
  return ForwardProjection_withPlan(result, plan_id, 1, I, 1 /* numView is 1 !!! */, 1, I);
}

int RegTools::ForwardProjection_withPlan(ProjectionResult &result, const int plan_id, int numGlobals, const double *transformations_global, int numView
                                           , int numLocalTrans, const double *transformations_local, const int memory_store_mode)
{
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("RegTools - ForwardProjection, dim: proj(%d, %d, %d, %d), vol(%d, %d, %d), numGlobals: %d, numView: %d, numLocalTrans: %d\n"
    , GetProjectionWidth(), GetProjectionHeight(), GetNumberOfProjections(), GetNumberOfProjectionSets(), m_VolumeDim.width, m_VolumeDim.height, m_VolumeDim.depth
    , numGlobals, numView, numLocalTrans);
  print_and_log("RegTools - ForwardProjection, transfer block size: %d\n", m_TransferBlockSize);
#endif
  if(numGlobals<1)  return false;
  double *cur_pm;
  std::map<int,ProjectionParametersSetting*>::const_iterator geom = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  if(geom == m_ProjectionParametersSetting.end()){
    print_and_log("error: RegTools::ForwardProjection_withPlan, invalid geometry setting\n")
    return 0;
  } else {
    // "SubSamplingArray" should be set appropriately prior to this call!
    // only numView*numGlobals projection should be enabled
/*
    print_and_log("RegTools - ForwardProjection_withPlan, numGlobals: %d, numEnabledProjections: %d\n", numGlobals, geom->second->m_NumEnabledProjections);
    print_and_log("Global trans:\n");
    for(int i=0;i<4;i++)  print_and_log("%f %f %f %f\n", transformations_global[i], transformations_global[i+4], transformations_global[i+8], transformations_global[i+12]);
    print_and_log("\n");
    print_and_log("m_SubSamplingArray = ");
    for(int i=0;i<geom->second->m_NumEnabledProjections;i++){ print_and_log("%d ", geom->second->m_SubSamplingArray[i]); }
    print_and_log("\n");
*/    

    // store first m_NumEnabledProjections projection matrices (for voxel driven projector) & detector frames (for ray driven projector)
    cur_pm = new double[12*geom->second->m_NumEnabledProjections];
    for(int i=0;i<geom->second->m_NumEnabledProjections;i++){
      int indx = geom->second->m_SubSamplingArray[i];
      memcpy(cur_pm+12*i, geom->second->m_ProjectionParameters[indx].ProjectionMatrix_3x4, sizeof(double)*12);
    }

    for(int p=0;p<numGlobals;p++){
      // consecutive numView views have the same "global" transformations
      const double *tg = transformations_global+16*p;
      for(int v=0;v<numView;v++){
        int v_org_idx = p*numView+v;
        int v_idx = geom->second->m_SubSamplingArray[v_org_idx];
//        print_and_log("Modify projection parameters for #%d (#%d)\n", v_idx, v_org_idx);
        // concatenate transformations_global (column-major) to 3x4 projection matrix (row-major)
        double *new_pm_row = geom->second->m_ProjectionParameters[v_idx].ProjectionMatrix_3x4, *pm_row = cur_pm+12*(v_org_idx);
        for(int i=0;i<3;i++) for(int j=0;j<4;j++) new_pm_row[i*4+j] = pm_row[i*4]*tg[j*4] + pm_row[i*4+1]*tg[j*4+1] + pm_row[i*4+2]*tg[j*4+2] + pm_row[i*4+3]*tg[j*4+3];
      }
    }
    SetNumberOfProjectionSets(numGlobals*numLocalTrans);

    double temp_transform[16];
    // forward projection
    for(int i=0;i<numLocalTrans;i++){
      std::map<int, VolumePlan_cudaArray*>::iterator it = m_VolumePlan_cudaArrays.find(plan_id);
      if(it == m_VolumePlan_cudaArrays.end()){
        print_and_log("RegTools::ForwardProjection_withPlan(), Invalid plan ID\n")
        return false;
      } else {
        cudaExtent cur_volumeDim = m_VolumeDim;
        float3 cur_voxelSize = m_VoxelSize_mm;
        double cur_transform[16];
        memcpy(cur_transform, m_VolumeTransform_col, sizeof(double)*16);
        SetVolumeInfo(it->second->VolumeDim[0], it->second->VolumeDim[1], it->second->VolumeDim[2], it->second->VoxelSize[0], it->second->VoxelSize[1], it->second->VoxelSize[2]);
        MultMatrixd_col(cur_transform, transformations_local+16*i, temp_transform);
        SetWorldToVolumeTransform(temp_transform);

        int numberOfRunningThreads = m_NumRegToolsThreads;
        if(result.numGPU>0)  numberOfRunningThreads = result.numGPU;

        float *minValues = NULL, *maxValues = NULL;
        if(result.minValue){ 
          minValues = new float[numberOfRunningThreads]; maxValues = new float[numberOfRunningThreads]; 
          *(result.minValue) = FLT_MAX; *(result.maxValue) = -FLT_MAX;
        }
        for(int j=0,start_proj=0,nProj=0;j<numberOfRunningThreads;j++, start_proj+=nProj){
          // calculate number of projection that is computed on this device
          nProj = ceil((double)numGlobals * m_RegToolsThreadParams[j].m_CudaDeviceLoad) * numView; // Note: this is slightly different from ForwardProjecton()
          if((start_proj+nProj)>geom->second->m_NumEnabledProjections)  nProj = geom->second->m_NumEnabledProjections-start_proj;
//          print_and_log("%d forward projections on device %d\n", nProj, j);

          if(nProj==0)  m_RegToolsThreadParams[j].m_ProcessingMode = ProcessingMode_DoNothing;
          else {
            m_RegToolsThreadParams[j].m_ProcessingMode = ProcessingMode_ForwardProjection;
            m_RegToolsThreadParams[j].m_Volume = NULL;
            m_RegToolsThreadParams[j].m_VolumePlan_cudaArray = it->second;
            m_RegToolsThreadParams[j].m_VolumePlan_cudaArray_out = FindVolumePlan(result.dDataID);
            m_RegToolsThreadParams[j].m_Projections = result.Data;
			      m_RegToolsThreadParams[j].m_NumViews = numView;
            // TODO: handle device init
            if(m_RegToolsThreadParams[j].m_Projections) m_RegToolsThreadParams[j].m_Projections += (geom->second->m_ProjectionWidth*geom->second->m_ProjectionHeight*start_proj);
            m_RegToolsThreadParams[j].m_ElapsedTime = result.projectionTime;
            if(result.minValue){ m_RegToolsThreadParams[j].m_MinValue = &(minValues[j]); m_RegToolsThreadParams[j].m_MaxValue = &(maxValues[j]); }
            else               { m_RegToolsThreadParams[j].m_MinValue = m_RegToolsThreadParams[j].m_MaxValue = NULL; }

            PrepareForRegToolsThread(&(m_RegToolsThreadParams[j]));
			m_RegToolsThreadParams[j].m_ProjectionsInit = result.initData;
            m_RegToolsThreadParams[j].m_EnabledProjection += start_proj; // offset
            m_RegToolsThreadParams[j].m_NumAllocatedProjections = geom->second->m_NumEnabledProjections;  // note: bad naming...
            m_RegToolsThreadParams[j].m_NumEnabledProjections = nProj;
            m_RegToolsThreadParams[j].m_MemoryStoreMode = (i==0) ? memory_store_mode : MemoryStoreMode_Additive;   // 'additive' mode, the interpolation result is added to the existing value
            //print_and_log("RegTools::ForwardProjection_withPlan(), i=%d, j=%d, m_RegToolsThreadParams[j].m_MemoryStoreMode=%d\n", i, j, m_RegToolsThreadParams[j].m_MemoryStoreMode);
		  }
        }
//        for(int j=0;j<m_RegToolsThreadParams[0].m_NumEnabledProjections;j++)
//          print_and_log("m_EnabledProjection[%d] = %d\n", j, m_RegToolsThreadParams[0].m_EnabledProjection[j]);

        if(result.numGPU>0){
          for(int i=0;i<result.numGPU;i++)  RunRegToolsThreads(i);  // this is unefficient if numGPU is larger than 2
        } else {
          RunRegToolsThreads();
        }

        if(result.minValue){
          for(int i=0;i<numberOfRunningThreads;i++){
            if(*(result.minValue)>minValues[i])  *(result.minValue) = minValues[i];
            if(*(result.maxValue)<maxValues[i])  *(result.maxValue) = maxValues[i];
          }
          delete[] minValues; delete[] maxValues;
        }

        SetVolumeInfo(cur_volumeDim.width, cur_volumeDim.height, cur_volumeDim.depth, cur_voxelSize.x, cur_voxelSize.y, cur_voxelSize.z);
        SetWorldToVolumeTransform(cur_transform);
      }
    }

    // recover current projection matrices & detector frames
    for(int i=0;i<geom->second->m_NumEnabledProjections;i++){
      int indx = geom->second->m_SubSamplingArray[i];
      memcpy(geom->second->m_ProjectionParameters[indx].ProjectionMatrix_3x4, cur_pm+12*i, sizeof(double)*12);
    }
    SetNumberOfProjectionSets(1);
    delete[] cur_pm;
  }
//  print_and_log("RegTools::ForwardProjection_withPlan, dim_Projections: (%d,%d,%d)\n", 
//    m_RegToolsThreadParam.m_ProjectionWidth, m_RegToolsThreadParam.m_ProjectionHeight, m_RegToolsThreadParam.m_NumEnabledProjections);
  return true;
}

VolumePlan_cudaArray* RegTools::FindVolumePlan(int planID)
{
  // find VolumePlan if planID is positive
  // currently, only supported single-GPU
  VolumePlan_cudaArray *ret = NULL;
  if(planID >= 0){
    std::map<int, VolumePlan_cudaArray*>::iterator it = m_VolumePlan_cudaArrays.find(planID);
    if(it == m_VolumePlan_cudaArrays.end()){
      print_and_log("RegTools::FindVolumePlan(), Invalid plan ID\n")
      return NULL;
    } else {
//      print_and_log("found valid plan, %d\n", it->second->d_raw_volume[0]);
      ret = it->second;
    }
  }
  return ret;
}

int RegTools::Interpolation_withPlan(ProjectionResult &result, int plan_id, const float *transforms, const int num_transform_element, int num_transform, const int type, const int order, const float bicubic_a, 
                                       const float back_ground, float *volume_center, const int isRGBA, float *color_map, int num_color_map, int label_overlay_mode)
{
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("RegTools - Interpolation with plan, dim: proj(%d, %d, %d, %d), vol(%d, %d, %d)\n"
    , GetProjectionWidth(), GetProjectionHeight(), GetNumberOfProjections(), GetNumberOfProjectionSets(), m_VolumeDim.width, m_VolumeDim.height, m_VolumeDim.depth);
  print_and_log("RegTools - Interpolation with plan, transfer block size: %d\n", m_TransferBlockSize);
#endif
  // note: rotation angles are in radians

  std::map<int, VolumePlan_cudaArray*>::iterator it = m_VolumePlan_cudaArrays.find(plan_id);
  if(it == m_VolumePlan_cudaArrays.end()){
    print_and_log("RegTools::Interpolation_withPlan(), Invalid plan ID\n")
    return false;
  } else {  
    // support only single GPU for now
    if(m_NumRegToolsThreads<1)  return false;

    for(int i=0; i<num_transform; i++){
      m_RegToolsThreadParams[0].m_ProcessingMode = ProcessingMode_Interpolator;
      m_RegToolsThreadParams[0].m_Volume = NULL;
      m_RegToolsThreadParams[0].m_VolumePlan_cudaArray = it->second;
      m_RegToolsThreadParams[0].m_VolumePlan_cudaArray_out = NULL; // TODO: output to CUDA array by FindVolumePlan(volumePlanID);
      if(result.Data)       m_RegToolsThreadParams[0].m_Projections = result.Data + (m_VolumeDim.width*m_VolumeDim.height*m_VolumeDim.depth*i);
      else                  m_RegToolsThreadParams[0].m_Projections = NULL;
      m_RegToolsThreadParams[0].m_ElapsedTime = result.projectionTime;
      m_RegToolsThreadParams[0].m_MinValue = result.minValue;
      m_RegToolsThreadParams[0].m_MaxValue = result.maxValue;
      m_RegToolsThreadParams[0].m_Interpolator_transform = const_cast<float*>(transforms + i*num_transform_element);
      m_RegToolsThreadParams[0].m_Interpolator_num_transform_element = num_transform_element;
      m_RegToolsThreadParams[0].m_Interpolator_num_transforms = 1;
      m_RegToolsThreadParams[0].m_Interpolator_type = type;
      m_RegToolsThreadParams[0].m_Interpolator_order = order;
      m_RegToolsThreadParams[0].m_Interpolator_bicubic_a = bicubic_a;
      m_RegToolsThreadParams[0].m_Interpolator_back_ground = back_ground;
      m_RegToolsThreadParams[0].m_Interpolator_volume_center = volume_center;
      m_RegToolsThreadParams[0].m_Interpolator_IsWarp = false;
      PrepareForRegToolsThread(&(m_RegToolsThreadParams[0]));
      m_RegToolsThreadParams[0].m_PBO_index = m_PBO_rendering_start_index + i;
      //print_and_log("RegTools::Interpolation_withPlan(), Projections: %d\n", m_RegToolsThreadParams[0].m_Projections);
      RunRegToolsThreads(0);
	  /*
	  int num_voxels = m_RegToolsThreadParams[0].m_VolumeDim.width*m_RegToolsThreadParams[0].m_VolumeDim.height*m_RegToolsThreadParams[0].m_VolumeDim.depth*m_RegToolsThreadParams[0].m_NumProjectionSets;
	  float *temp = new float[num_voxels];
	  cutilSafeCall(cudaMemcpy(temp, m_RegToolsThreadParams[0].d_Volume, num_voxels * sizeof(float), cudaMemcpyDeviceToHost));
	  float sum = 0;
	  for (int i = 0; i < num_voxels; i++) sum += temp[i];
	  print_and_log("RegTools::RegToolsThread_main(), m_VolumeDim: (%d, %d, %d), sum: %f\n", m_RegToolsThreadParams[0].m_VolumeDim.width, m_RegToolsThreadParams[0].m_VolumeDim.height, m_RegToolsThreadParams[0].m_VolumeDim.depth, sum);
	  free(temp);
	  */
	}
    return true;
  }
}

int RegTools::SetWarpTextures(int *warps_tex)
{
  if(!warps_tex){
    for(int i=0;i<m_NumRegToolsThreads;i++){
      m_RegToolsThreadParams[i].m_VolumePlan_cudaArray_warpX = NULL;
      m_RegToolsThreadParams[i].m_VolumePlan_cudaArray_warpY = NULL;
      m_RegToolsThreadParams[i].m_VolumePlan_cudaArray_warpZ = NULL;
    }
//    print_and_log("SetWarpTextures(), set NULL\n")
    return false;
  }
  // currently, only 3-dimensional warp is supported
  std::map<int, VolumePlan_cudaArray*>::iterator wX = m_VolumePlan_cudaArrays.find(warps_tex[0]);
  std::map<int, VolumePlan_cudaArray*>::iterator wY = m_VolumePlan_cudaArrays.find(warps_tex[1]);
  std::map<int, VolumePlan_cudaArray*>::iterator wZ = m_VolumePlan_cudaArrays.find(warps_tex[2]);
  if(wX == m_VolumePlan_cudaArrays.end() || wY == m_VolumePlan_cudaArrays.end() || wZ == m_VolumePlan_cudaArrays.end()){
    print_and_log("Invalid warp plan ID: (%d, %d, %d)\n", warps_tex[0], warps_tex[1], warps_tex[2])
    return false;
  }
  for(int i=0;i<m_NumRegToolsThreads;i++){
    m_RegToolsThreadParams[i].m_VolumePlan_cudaArray_warpX = wX->second;
    m_RegToolsThreadParams[i].m_VolumePlan_cudaArray_warpY = wY->second;
    m_RegToolsThreadParams[i].m_VolumePlan_cudaArray_warpZ = wZ->second;
  }
//  print_and_log("SetWarpTextures(), successful\n")
  return true;
}

int RegTools::ApplyDeformationField(ProjectionResult &result, int target_volume_id_tex, int *warps_tex, int num_dims, int type, int order, float bicubic_a, 
                          float back_ground, float *volume_center, int scattered_pnts_plan, float *transforms_3x4xN, int num_transform_element, int num_transforms)
{
  // currently, only 3-dimensional warp is supported
  std::map<int, VolumePlan_cudaArray*>::iterator target = m_VolumePlan_cudaArrays.find(target_volume_id_tex);
  std::map<int, VolumePlan_cudaArray*>::iterator wX = m_VolumePlan_cudaArrays.find(warps_tex[0]);
  std::map<int, VolumePlan_cudaArray*>::iterator wY = m_VolumePlan_cudaArrays.find(warps_tex[1]);
  std::map<int, VolumePlan_cudaArray*>::iterator wZ = m_VolumePlan_cudaArrays.find(warps_tex[2]);
  std::map<int, VolumePlan_cudaArray*>::iterator scat = m_VolumePlan_cudaArrays.find(scattered_pnts_plan);

  if(wX == m_VolumePlan_cudaArrays.end() || wY == m_VolumePlan_cudaArrays.end() || wZ == m_VolumePlan_cudaArrays.end()){
    print_and_log("Invalid warp plan ID: (%d, %d, %d)\n", warps_tex[0], warps_tex[1], warps_tex[2])
    return false;
  } else if(target == m_VolumePlan_cudaArrays.end() && scat == m_VolumePlan_cudaArrays.end()){
    print_and_log("Invalid data source ID\n")
    return false;
  } else {  
    // support only single GPU for now
    if(m_NumRegToolsThreads<1)  return false;

    m_RegToolsThreadParams[0].m_ProcessingMode = ProcessingMode_Interpolator;
    m_RegToolsThreadParams[0].m_Volume = NULL;
    m_RegToolsThreadParams[0].m_VolumePlan_cudaArray = (target != m_VolumePlan_cudaArrays.end()) ? target->second : NULL;
    m_RegToolsThreadParams[0].m_VolumePlan_cudaArray_out = FindVolumePlan(result.dDataID);
    if(result.Data)       m_RegToolsThreadParams[0].m_Projections = result.Data;
    else                  m_RegToolsThreadParams[0].m_Projections = NULL;
    m_RegToolsThreadParams[0].m_ElapsedTime = result.projectionTime;
    m_RegToolsThreadParams[0].m_MinValue = result.minValue;
    m_RegToolsThreadParams[0].m_MaxValue = result.maxValue;
    m_RegToolsThreadParams[0].m_Interpolator_transform = transforms_3x4xN;
    m_RegToolsThreadParams[0].m_Interpolator_num_transform_element = num_transform_element;
    m_RegToolsThreadParams[0].m_Interpolator_num_transforms = num_transforms;
    m_RegToolsThreadParams[0].m_Interpolator_type = type;
    m_RegToolsThreadParams[0].m_Interpolator_order = order;
    m_RegToolsThreadParams[0].m_Interpolator_bicubic_a = bicubic_a;
    m_RegToolsThreadParams[0].m_Interpolator_back_ground = back_ground;
    m_RegToolsThreadParams[0].m_Interpolator_volume_center = volume_center;
    m_RegToolsThreadParams[0].m_Interpolator_IsWarp = true;
    PrepareForRegToolsThread(&(m_RegToolsThreadParams[0]));

    m_RegToolsThreadParams[0].m_VolumePlan_cudaArray_warpX = wX->second;
    m_RegToolsThreadParams[0].m_VolumePlan_cudaArray_warpY = wY->second;
    m_RegToolsThreadParams[0].m_VolumePlan_cudaArray_warpZ = wZ->second;
    if(scat != m_VolumePlan_cudaArrays.end()){
      m_RegToolsThreadParams[0].m_Interpolator_scattered_pnts = scat->second->d_raw_volume[0];
      m_RegToolsThreadParams[0].m_Interpolator_num_scattered_pnts = scat->second->VolumeDim[1];  // the array size should be (3 x N x 1) (N: number of points)
//      print_and_log("scattered point interpolation: plan = %d, dimensions: (%d, %d, %d)\n", scattered_pnts_plan, scat->second->VolumeDim[0], scat->second->VolumeDim[1], scat->second->VolumeDim[2]);
      m_RegToolsThreadParams[0].m_VolumeDim = make_cudaExtent( scat->second->VolumeDim[0], scat->second->VolumeDim[1], scat->second->VolumeDim[2] );
      if(m_RegToolsThreadParams[0].d_Volume){  cutilSafeCall( cudaFree(m_RegToolsThreadParams[0].d_Volume) ); m_RegToolsThreadParams[0].d_Volume = NULL; }
    } else {
      m_RegToolsThreadParams[0].m_Interpolator_scattered_pnts = NULL;
      m_RegToolsThreadParams[0].m_Interpolator_num_scattered_pnts = 0;
    }

    m_RegToolsThreadParams[0].m_PBO_index = -1;
    RunRegToolsThreads(0);

    // clean up
    m_RegToolsThreadParams[0].m_Interpolator_scattered_pnts = NULL;
    m_RegToolsThreadParams[0].m_Interpolator_num_scattered_pnts = 0;
    return true;
  }
}

int RegTools::ComputeLinearCombination(int warp_device, int def_mode_device, int mode_weight)
{
  // compute linear combination of def_modes_device with mode_weights and store the results to texture array warps_tex (warps_device is used temorarily)
  // input: 
  //      warp_device: plan id pointing to (nx x ny x nz*num_dims x num_subjects) element array
  //      def_mode_device: plan id pointing to (nx x ny x nz*num_modes x num_dims) element array
  //      mode_weights: (num_modes x num_subjects) element array
  //
  int device_id = 0;  // currently, we use only 1 device

  m_RegToolsThreadParams[device_id].d_WarpArray = m_VolumePlan_cudaArrays.find(warp_device)->second->d_raw_volume[device_id];
  m_RegToolsThreadParams[device_id].d_ModeArray = m_VolumePlan_cudaArrays.find(def_mode_device)->second->d_raw_volume[device_id];
  m_RegToolsThreadParams[device_id].m_ModeArray_NumberOfModes = m_VolumePlan_cudaArrays.find(def_mode_device)->second->VolumeDim[1];  // number of modes in input array (not necessarily equal to the number of columns of mode weight array)
  m_RegToolsThreadParams[device_id].d_ModeWeights = m_VolumePlan_cudaArrays.find(mode_weight)->second->d_raw_volume[device_id];;

  // compute linear combination of def_modes on the device
  m_RegToolsThreadParams[device_id].m_ProcessingMode = ProcessingMode_ComputeLinearCombination;
  m_RegToolsThreadParams[device_id].m_VolumeDim = m_VolumeDim;
  m_RegToolsThreadParams[device_id].m_NumModeDims = m_VolumePlan_cudaArrays.find(def_mode_device)->second->VolumeDim[2];
  m_RegToolsThreadParams[device_id].m_NumModes = m_VolumePlan_cudaArrays.find(mode_weight)->second->VolumeDim[0];
  m_RegToolsThreadParams[device_id].m_NumVolumes = m_VolumePlan_cudaArrays.find(mode_weight)->second->VolumeDim[1];
  RunRegToolsThreads(device_id);

  return true;
}

int RegTools::CopyDeviceMemoryToCudaArray_Multi(int *cudaArrayIDs, int numCudaArrayID, int deviceMemoryID)
{
  if(numCudaArrayID==0) return 0;
  int device_id = 0;
  m_RegToolsThreadParams[device_id].m_ProcessingMode = ProcessingMode_CopyDeviceMemoryToCudaArray_Multi;
  m_RegToolsThreadParams[device_id].m_VolumePlans = new VolumePlan_cudaArray*[numCudaArrayID];
  m_RegToolsThreadParams[device_id].m_NumCudaArrays = numCudaArrayID;
  for(int i=0;i<numCudaArrayID;i++){
    m_RegToolsThreadParams[device_id].m_VolumePlans[i] = FindVolumePlan(cudaArrayIDs[i]);
//    print_and_log("ReconTools::CopyDeviceMemoryToCudaArray_Multi(), cudaArray %d, ID: %d, copy entire volume, texture volume size: (%d,%d,%d,%d)\n"
//      , i, cudaArrayIDs[i], FindVolumePlan(cudaArrayIDs[i])->VolumeDim[0], FindVolumePlan(cudaArrayIDs[i])->VolumeDim[1], FindVolumePlan(cudaArrayIDs[i])->VolumeDim[2], FindVolumePlan(cudaArrayIDs[i])->numVolumes);
  }
  VolumePlan_cudaArray *device_plan = FindVolumePlan(deviceMemoryID);
//  print_and_log("ReconTools::CopyDeviceMemoryToCudaArray_Multi(), copy entire volume, device volume size: (%d,%d,%d,%d)\n"
//    , device_plan->VolumeDim[0], device_plan->VolumeDim[1], device_plan->VolumeDim[2], device_plan->numVolumes);
  m_RegToolsThreadParams[device_id].m_VolumePlan_cudaArray = device_plan;
  RunRegToolsThreads(device_id);
  delete[] m_RegToolsThreadParams[device_id].m_VolumePlans;
  m_RegToolsThreadParams[device_id].m_NumCudaArrays = 0;
  return 1;
}

int RegTools::CopyDeviceMemoryToCudaArray(int cudaArrayID, int deviceMemoryID, int isCopyToAllDevices, int volume_index_tex_0_base, int volume_index_dev_0_base)
{
  std::map<int,ProjectionParametersSetting*>::const_iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  VolumePlan_cudaArray *device_plan = FindVolumePlan(deviceMemoryID), *texture_plan = FindVolumePlan(cudaArrayID);
  device_plan->volumeIndex = volume_index_dev_0_base;
  texture_plan->volumeIndex = volume_index_tex_0_base;
  for(int i=0,start_proj=0,nProj=0;i<m_NumRegToolsThreads;i++, start_proj+=nProj){
    m_RegToolsThreadParams[i].m_ProcessingMode = ProcessingMode_CopyDeviceMemoryToCudaArray;
    m_RegToolsThreadParams[i].m_VolumePlan_cudaArray = device_plan;
    m_RegToolsThreadParams[i].m_VolumePlan_cudaArray_out = texture_plan;
    m_RegToolsThreadParams[i].m_NumVolumes = 1;
//    m_RegToolsThreadParams[i].m_VolumePlan_cudaArray_out2 = NULL;
    if(isCopyToAllDevices){
      m_RegToolsThreadParams[i].m_VolumeDim = make_cudaExtent(device_plan->VolumeDim[0], device_plan->VolumeDim[1], device_plan->VolumeDim[2]);
    } else {
      // calculate number of projection that is computed on this device
      if(i == m_NumRegToolsThreads-1) nProj = device_plan->VolumeDim[2]-start_proj;
      else nProj = floor((double)device_plan->VolumeDim[2] * m_RegToolsThreadParams[i].m_CudaDeviceLoad);
      if(i>0) device_plan->d_raw_volume[i] += device_plan->VolumeDim[0]*device_plan->VolumeDim[1]*start_proj;  // add offset
      m_RegToolsThreadParams[i].m_VolumeDim = make_cudaExtent(device_plan->VolumeDim[0], device_plan->VolumeDim[1], nProj);
//      print_and_log("ReconTools::CopyDeviceMemoryToCudaArray(), copy part of volume, volume size: (%d,%d,%d)\n", device_plan->VolumeDim[0], device_plan->VolumeDim[1], nProj);
    }
  }
  RunRegToolsThreads();
  if(!isCopyToAllDevices){
    for(int i=0,start_proj=0,nProj=0;i<m_NumRegToolsThreads;i++, start_proj+=nProj){
      if(i == m_NumRegToolsThreads-1) nProj = it->second->m_NumEnabledProjections-start_proj;
      else nProj = floor((double)it->second->m_NumEnabledProjections * m_RegToolsThreadParams[i].m_CudaDeviceLoad);
      device_plan->d_raw_volume[i] -= device_plan->VolumeDim[0]*device_plan->VolumeDim[1]*start_proj; // reset offset
    }
  }
  return true;
}

int RegTools::CreateVolumePlan_cudaArray(struct VolumePlan_cudaArray *plan, bool isCudaArray)
{
  // generate new VolumePlan_cudaArray and store to the map
  VolumePlan_cudaArray *new_plan = new VolumePlan_cudaArray;
  new_plan->d_volume = new struct cudaArray*[m_NumRegToolsThreads];
  new_plan->d_raw_volume = new float*[m_NumRegToolsThreads];
  for(int i=0;i<m_NumRegToolsThreads;i++){
    new_plan->d_volume[i] = NULL;
    new_plan->d_raw_volume[i] = NULL;
  }
  new_plan->h_volume = NULL;  // internal instance doesn't use the host memory
  new_plan->h_volume_set = NULL;
  memcpy(new_plan->VolumeDim, plan->VolumeDim, sizeof(int)*3);
  memcpy(new_plan->VoxelSize, plan->VoxelSize, sizeof(double)*3);
  new_plan->numVolumes = plan->numVolumes;
  new_plan->volumeIndex = 0;
  // find smallest 'un-used' id
  int id = 0;
  while( m_VolumePlan_cudaArrays.find(id) != m_VolumePlan_cudaArrays.end() ) id++;

  // copy the volume from host (CPU) to GPU device memory (cudaArray)
  if(m_NumRegToolsThreads<1)  return -1;
  for(int i=0;i<m_NumRegToolsThreads;i++){
    if(isCudaArray) m_RegToolsThreadParams[i].m_ProcessingMode = ProcessingMode_VolumePlan_cudaArray;
    else            m_RegToolsThreadParams[i].m_ProcessingMode = ProcessingMode_VolumePlan;
    m_RegToolsThreadParams[i].m_Volume = plan->h_volume;
    m_RegToolsThreadParams[i].m_VolumeDim = make_cudaExtent(new_plan->VolumeDim[0], new_plan->VolumeDim[1], new_plan->VolumeDim[2]);
    m_RegToolsThreadParams[i].m_VoxelSize_mm = make_float3(static_cast<float>(new_plan->VoxelSize[0]), static_cast<float>(new_plan->VoxelSize[1]), static_cast<float>(new_plan->VoxelSize[2]));
    m_RegToolsThreadParams[i].m_NumVolumes = plan->numVolumes;
    m_RegToolsThreadParams[i].m_VolumePlan_cudaArray = new_plan;
  }
  RunRegToolsThreads();
  m_VolumePlan_cudaArrays.insert( std::pair<int, VolumePlan_cudaArray*>(id, new_plan) );

  return id;  // return the ID for the created VolumePlan_cudaArray
}

int RegTools::GetVolumePlan_cudaArrayVolumeInfo(int plan_id, int *volume_dim, double *voxel_size, int *numVolumes)
{
  std::map<int, VolumePlan_cudaArray*>::iterator it = m_VolumePlan_cudaArrays.find(plan_id);
  if(it != m_VolumePlan_cudaArrays.end()){
    for(int i=0;i<3;i++){ volume_dim[i] = it->second->VolumeDim[i]; voxel_size[i] = it->second->VoxelSize[i]; (*numVolumes) = it->second->numVolumes; }
  } else {
    for(int i=0;i<3;i++){ volume_dim[i] = 0; voxel_size[i] = 0.0; }
  }
  return true;
}

int RegTools::GetGPUProjection(float *h_projection, int projection_index_0_base)
{
  m_RegToolsThreadParams[0].m_ProcessingMode = ProcessingMode_GetGPUProjection;
  m_RegToolsThreadParams[0].m_Projections = h_projection;
  m_RegToolsThreadParams[0].m_ProjectionSetIndex = projection_index_0_base;
  RunRegToolsThreads(0);
  return true;
}

int RegTools::GetVolumePlan_cudaArrayVolume(int plan_id, float *h_volume, bool isCudaArray, int volume_index_0_base)
{
  // get volume on device memory from the first device
  std::map<int, VolumePlan_cudaArray*>::iterator it = m_VolumePlan_cudaArrays.find(plan_id);
  if(m_NumRegToolsThreads<1)  return false; // thread is not initialized
  VolumePlan_cudaArray *new_plan = new VolumePlan_cudaArray;
  new_plan->d_volume = new struct cudaArray*[1];
  new_plan->d_raw_volume = new float*[1];
  new_plan->d_volume[0] = NULL;
  new_plan->d_raw_volume[0] = new_plan->h_volume_set = NULL;
  new_plan->h_volume = h_volume;
  new_plan->volumeIndex = volume_index_0_base;
//  print_and_log("RegTools::GetVolumePlan_cudaArrayVolume(), volume_index_0_base = %d\n", volume_index_0_base);
  new_plan->numVolumes = 1; // get only one of the volumes
  m_RegToolsThreadParams[0].m_ProcessingMode = ProcessingMode_VolumePlan;
  if(it != m_VolumePlan_cudaArrays.end()){
    // get volume from the specified volume plan (otherwise get the default volume)
    if(isCudaArray){
        new_plan->d_volume[0] = it->second->d_volume[0];
        m_RegToolsThreadParams[0].m_ProcessingMode = ProcessingMode_VolumePlan_cudaArray;
    } else
      new_plan->d_raw_volume[0] = it->second->d_raw_volume[0];
    m_RegToolsThreadParams[0].m_VolumeDim = make_cudaExtent(it->second->VolumeDim[0], it->second->VolumeDim[1], it->second->VolumeDim[2]);
  } else {
    m_RegToolsThreadParams[0].m_VolumeDim = m_VolumeDim;
  }
  m_RegToolsThreadParams[0].m_VolumePlan_cudaArray = new_plan;
  RunRegToolsThreads(0);
  delete[] new_plan->d_raw_volume;
  delete[] new_plan->d_volume;
  delete new_plan;
  return true;
}

int RegTools::SetVolumePlan_cudaArrayVolume(int plan_id, float *h_volume, bool isCudaArray, int volume_index_0_base)
{
  // set volume on device memory of all the devices
  std::map<int, VolumePlan_cudaArray*>::iterator it = m_VolumePlan_cudaArrays.find(plan_id);
  if(m_NumRegToolsThreads<1)  return false; // thread is not initialized
  if(it != m_VolumePlan_cudaArrays.end()){
    VolumePlan_cudaArray *new_plan = new VolumePlan_cudaArray;
    new_plan->d_volume = new struct cudaArray*[m_NumRegToolsThreads];
    new_plan->d_raw_volume = new float*[m_NumRegToolsThreads];
    for(int i=0;i<m_NumRegToolsThreads;i++){
      if(isCudaArray){
        new_plan->d_volume[i] = it->second->d_volume[i];  new_plan->d_raw_volume[i] = NULL;
      } else {
        new_plan->d_volume[i] = NULL;                     new_plan->d_raw_volume[i] = it->second->d_raw_volume[i];
      }
    }
    new_plan->h_volume = NULL;
    new_plan->h_volume_set = h_volume;
    new_plan->volumeIndex = volume_index_0_base;
    new_plan->numVolumes = 1; // get only one of the volumes

    for(int i=0;i<m_NumRegToolsThreads;i++){
      if(isCudaArray) m_RegToolsThreadParams[i].m_ProcessingMode = ProcessingMode_VolumePlan_cudaArray;
      else            m_RegToolsThreadParams[i].m_ProcessingMode = ProcessingMode_VolumePlan;
      m_RegToolsThreadParams[i].m_VolumeDim = make_cudaExtent(it->second->VolumeDim[0], it->second->VolumeDim[1], it->second->VolumeDim[2]);
      m_RegToolsThreadParams[i].m_VolumePlan_cudaArray = new_plan;
    }
    RunRegToolsThreads();
    delete[] new_plan->d_raw_volume;
    delete[] new_plan->d_volume;
    delete new_plan;
  } else {
    h_volume = NULL;
  }
  return true;
}

int RegTools::MultVolumePlan(int plan_id, float value)
{
  std::map<int, VolumePlan_cudaArray*>::iterator it = m_VolumePlan_cudaArrays.find(plan_id);
  if(m_NumRegToolsThreads<1)  return false; // thread is not initialized
  if(it != m_VolumePlan_cudaArrays.end()){
    int array_size = it->second->VolumeDim[0]*it->second->VolumeDim[1]*it->second->VolumeDim[2];
//    print_and_log("multiply %f to the volume (size: %d)\n", value, array_size);
    for(int i=0;i<m_NumRegToolsThreads;i++){
      m_RegToolsThreadParams[i].m_ProcessingMode = ProcessingMode_MultVolume;
      m_RegToolsThreadParams[i].m_ScalarVal = value;
      m_RegToolsThreadParams[i].m_VolumeDim = make_cudaExtent(it->second->VolumeDim[0], it->second->VolumeDim[1], it->second->VolumeDim[2]);
      m_RegToolsThreadParams[i].m_VolumePlan_cudaArray = it->second;
    }
    RunRegToolsThreads();
    return false;
  } else {
    print_and_log("Volume plan was not found\n");
    return false;
  }
}

int RegTools::DeleteVolumePlan_cudaArray(int plan_id)
{
  std::map<int, VolumePlan_cudaArray*>::iterator it = m_VolumePlan_cudaArrays.find(plan_id);
  if(it != m_VolumePlan_cudaArrays.end()){
    // delete cudaArray and erase the element from map
    if(m_NumRegToolsThreads<1)  return -1;
    for(int i=0;i<m_NumRegToolsThreads;i++){
      m_RegToolsThreadParams[i].m_ProcessingMode = ProcessingMode_VolumePlan_cudaArray;
      m_RegToolsThreadParams[i].m_Volume = NULL;
      m_RegToolsThreadParams[i].m_VolumePlan_cudaArray = it->second;
    }
    RunRegToolsThreads();     // delete device memory
    delete[] it->second->d_raw_volume;
    delete[] it->second->d_volume;
    delete it->second;
    m_VolumePlan_cudaArrays.erase(it);
  }
  return true;
}

void RegTools::InitializeSimilarityMeasureComputationPlan(struct SimilarityMeasureComputationPlan *plan)
{
  // initialize all images to NULL
  plan->h_fixed_images = plan->d_fixed_images = plan->h_intermediate_images = plan->d_intermediate_images = plan->d_temp_images = NULL;
  plan->h_fixed_normalized = plan->d_fixed_normalized = plan->h_floating_normalized = plan->d_floating_normalized = NULL;
  plan->h_joint_pdf = plan->d_pdf_buf = plan->d_joint_pdf = NULL;
  plan->d_hist_buf = plan->d_joint_hist = NULL;
  plan->h_fixed_measurement = plan->d_fixed_measurement = NULL;
  plan->h_fixed_Xgrad = plan->d_fixed_Xgrad = plan->h_fixed_Ygrad = plan->d_fixed_Ygrad = NULL;
  plan->h_floating_Xgrad = plan->d_floating_Xgrad = plan->h_floating_Ygrad = plan->d_floating_Ygrad = NULL;
  plan->d_temp_padded = NULL;
  plan->d_temp_spectrum = plan->d_X_kernel_spectrum = plan->d_Y_kernel_spectrum = NULL;
  plan->d_temp_SM = plan->d_WeightVector = NULL;
  plan->h_temp_SM = NULL;
  plan->h_mask_weight = plan->d_mask_weight = NULL;

  plan->Sigma = 1.0;
  plan->NormalizeMax_fixed = plan->NormalizeMin_fixed = plan->NormalizeMax_floating = plan->NormalizeMin_floating = -1.0;
  plan->I0 = -1;

  plan->SimilarityMeasure = NULL;
  plan->h_GI_threshold = -FLT_MAX;
  plan->h_SSIM_DynamicRange = 1.0f;
}

int RegTools::CreateSimilarityMeasureComputationPlan(struct SimilarityMeasureComputationPlan *plan, double *normalization_factor)
{
  // generate new SimilarityMeasureComputationPlan and store to the map
  SimilarityMeasureComputationPlan **new_plans = new SimilarityMeasureComputationPlan*[m_NumRegToolsThreads];
  for(int i=0;i<m_NumRegToolsThreads;i++) new_plans[i] = new SimilarityMeasureComputationPlan;
  // find smallest 'un-used' id
  int id = 0;
  while( m_SimilarityMeasureComputationPlans.find(id) != m_SimilarityMeasureComputationPlans.end() ) id++;

  int *num_image_sets = new int[m_NumRegToolsThreads];
  ComputeNumberOfProjectionForEachGPU(m_RegToolsThreadParams, m_NumRegToolsThreads, plan->MaxNumImageSets, num_image_sets);
  // initialize all images to NULL
  for(int i=0;i<m_NumRegToolsThreads;i++){
    InitializeSimilarityMeasureComputationPlan(new_plans[i]);
    new_plans[i]->h_fixed_images = plan->h_fixed_images;  // only h_fixedImages is non-NULL -> create plan
    new_plans[i]->h_mask_weight = plan->h_mask_weight;
    memcpy(new_plans[i]->ImageDim, plan->ImageDim, sizeof(int)*3);

    new_plans[i]->SimilarityMeasureType = plan->SimilarityMeasureType;
    new_plans[i]->MaxNumImageSets = num_image_sets[i];
    new_plans[i]->Sigma = plan->Sigma;
    new_plans[i]->NormalizeMax_floating = plan->NormalizeMax_floating;  new_plans[i]->NormalizeMin_floating = plan->NormalizeMin_floating;
    new_plans[i]->NormalizeMax_fixed = plan->NormalizeMax_fixed;        new_plans[i]->NormalizeMin_fixed = plan->NormalizeMin_fixed;
    new_plans[i]->I0 = plan->I0;
    new_plans[i]->h_GI_threshold = plan->h_GI_threshold;
    new_plans[i]->h_SSIM_DynamicRange = plan->h_SSIM_DynamicRange;
    new_plans[i]->fftPlanFwd = new_plans[i]->fftPlanInv = new_plans[i]->fftPlanManyFwd = new_plans[i]->fftPlanManyInv = NULL;
    new_plans[i]->m_CudaDeviceID_Sequential = i;
    m_SimilarityMeasureComputationPlans.insert( std::pair<int, SimilarityMeasureComputationPlan*>(id, new_plans[i]) );
  }
  delete[] num_image_sets;

  // generate gradient image of the given image (in host memory) and store it in device memory
  for(int i=0;i<m_NumRegToolsThreads;i++){
    m_RegToolsThreadParams[i].m_ProcessingMode = ProcessingMode_SimilarityMeasureComputation;
    m_RegToolsThreadParams[i].m_ElapsedTime = NULL;
    m_RegToolsThreadParams[i].m_SimilarityMeasureComputationPlan = new_plans[i];
  }
  RunRegToolsThreads();
  for(int i=0;i<m_NumRegToolsThreads;i++){
    new_plans[i]->h_fixed_images = NULL;  // all host memory pointer should be NULL
    new_plans[i]->h_mask_weight = NULL;
  }
  if(normalization_factor)  (*normalization_factor) = new_plans[0]->NormalizationFactor;  // all new_plans have the same normalization factor (since the fixed images are common)
  delete new_plans;
  return id;  // return the ID for the created SimilarityMeasureComputationPlan
}

int RegTools::GetSimilarityMeasureComputationPlanImageInfo(int plan_id, int GPU_ID, int *image_dim, double *normalization_factor)
{
  std::pair<std::multimap<int,SimilarityMeasureComputationPlan*>::iterator,std::multimap<int,SimilarityMeasureComputationPlan*>::iterator> ret = 
    m_SimilarityMeasureComputationPlans.equal_range(plan_id);   // find ALL plans that has the same id
  if(ret.first == m_SimilarityMeasureComputationPlans.end())  return -1;
  memset(image_dim, 0, sizeof(int)*3);
  int device_id_sequential = -1;
  for(int i=0;i<m_NumRegToolsThreads;i++) if(m_RegToolsThreadParams[i].m_CudaDeviceID == GPU_ID) device_id_sequential = m_RegToolsThreadParams[i].m_CudaDeviceID_Sequential;
  if(GPU_ID<0)  device_id_sequential = 0;
  for(std::multimap<int, SimilarityMeasureComputationPlan*>::iterator it=ret.first;it != ret.second;it++){
    if(it->second->m_CudaDeviceID_Sequential == device_id_sequential){
      memcpy(image_dim, it->second->ImageDim, sizeof(int)*3);
      if(normalization_factor)  (*normalization_factor) = it->second->NormalizationFactor;
    }
  }
  return true;
}

int RegTools::GetSimilarityMeasureComputationPlanImages(int plan_id, int GPU_ID, float *images, int image_type, int frame_no)
{
  std::pair<std::multimap<int,SimilarityMeasureComputationPlan*>::iterator,std::multimap<int,SimilarityMeasureComputationPlan*>::iterator> ret = 
    m_SimilarityMeasureComputationPlans.equal_range(plan_id);   // find ALL plans that has the same id
  if(ret.first == m_SimilarityMeasureComputationPlans.end())  return false;
  if(m_NumRegToolsThreads<1)  return false; // thread is not initialized
  int device_id_sequential = -1;
  for(int i=0;i<m_NumRegToolsThreads;i++) if(m_RegToolsThreadParams[i].m_CudaDeviceID == GPU_ID) device_id_sequential = m_RegToolsThreadParams[i].m_CudaDeviceID_Sequential;
  if(GPU_ID<0)  device_id_sequential = 0;
  bool isImageSet = false;
  for(std::multimap<int, SimilarityMeasureComputationPlan*>::iterator it=ret.first;it != ret.second;it++){
    if(it->second->m_CudaDeviceID_Sequential == device_id_sequential){
      SimilarityMeasureComputationPlan *new_plan = new SimilarityMeasureComputationPlan;
      // initialize all images to NULL
      InitializeSimilarityMeasureComputationPlan(new_plan);

      new_plan->h_get_frame_no = frame_no;
      if(image_type == 0){         // get fixed images
        new_plan->d_fixed_images = it->second->d_fixed_images;  new_plan->h_fixed_images = images;
      } else if(image_type == 1){  // get intermediate images
        new_plan->d_intermediate_images = it->second->d_intermediate_images; new_plan->h_intermediate_images = images;
      } else if(image_type == 2){  // get X gradient of fixed images
        new_plan->d_fixed_Xgrad = it->second->d_fixed_Xgrad; new_plan->h_fixed_Xgrad = images;
      } else if(image_type == 3){  // get Y gradient of fixed images
        new_plan->d_fixed_Ygrad = it->second->d_fixed_Ygrad; new_plan->h_fixed_Ygrad = images;
      } else if(image_type == 4){  // get X gradient of floating images
        new_plan->d_floating_Xgrad = it->second->d_floating_Xgrad; new_plan->h_floating_Xgrad = images;
      } else if(image_type == 5){  // get Y gradient of floating images
        new_plan->d_floating_Ygrad = it->second->d_floating_Ygrad; new_plan->h_floating_Ygrad = images;
      } else if(image_type == 6){  // get Normalized fixed images
        new_plan->d_fixed_normalized = it->second->d_fixed_normalized; new_plan->h_fixed_normalized = images;
      } else if(image_type == 7){  // get Normalized floating images
        new_plan->d_floating_normalized = it->second->d_floating_normalized; new_plan->h_floating_normalized = images;
      } else if(image_type == 9){  // get joint pdf for MI, NMI computation
        new_plan->d_joint_pdf = it->second->d_joint_pdf; new_plan->h_joint_pdf = images;
      } else if(image_type == 10){  // get mask weight
        new_plan->d_mask_weight = it->second->d_mask_weight; new_plan->h_mask_weight = images;
      }
      for(int i=0;i<3;i++){ new_plan->ImageDim[i] = it->second->ImageDim[i]; }
  //    print_and_log("RegTools::GetSimilarityMeasureComputationPlanImages(), image_type: %d, dim(%d, %d, %d)\n", image_type, new_plan->ImageDim[0], new_plan->ImageDim[1], new_plan->ImageDim[2]);

      // run only one thread
      m_RegToolsThreadParams[it->second->m_CudaDeviceID_Sequential].m_ProcessingMode = ProcessingMode_SimilarityMeasureComputation;
      m_RegToolsThreadParams[it->second->m_CudaDeviceID_Sequential].m_SimilarityMeasureComputationPlan = new_plan;
      RunRegToolsThreads(it->second->m_CudaDeviceID_Sequential);

      delete new_plan;
      isImageSet = true;
    }
  }
  if(!isImageSet)  images = NULL;

  return true;
}

int RegTools::DeleteSimilarityMeasureComputationPlan(int plan_id)
{
  std::pair<std::multimap<int,SimilarityMeasureComputationPlan*>::iterator,std::multimap<int,SimilarityMeasureComputationPlan*>::iterator> ret = 
    m_SimilarityMeasureComputationPlans.equal_range(plan_id);   // delete ALL plans that has the same id
  if(ret.first == m_SimilarityMeasureComputationPlans.end())  return false;
  for(std::multimap<int, SimilarityMeasureComputationPlan*>::iterator it=ret.first;it != ret.second;it++){
    // delete float array and erase the element from map
    m_RegToolsThreadParams[it->second->m_CudaDeviceID_Sequential].m_ProcessingMode = ProcessingMode_SimilarityMeasureComputation;
    m_RegToolsThreadParams[it->second->m_CudaDeviceID_Sequential].m_SimilarityMeasureComputationPlan = it->second;
    m_RegToolsThreadParams[it->second->m_CudaDeviceID_Sequential].m_SimilarityMeasureComputationPlan->SimilarityMeasure = NULL;
  }
  RunRegToolsThreads();     // delete device memory
  m_SimilarityMeasureComputationPlans.erase(ret.first, ret.second);
  return true;
}

void RegTools::ComputeNumberOfProjectionForEachGPU(RegToolsThreadParam *threadParams, int numThreads, int totalProj, int *num_projs, int *start_projs)
{
  for(int i=0,start_proj=0,nProj=0;i<numThreads;i++, start_proj+=nProj){
    // calculate number of projection sets that is computed on this device
    nProj = ceil((double)totalProj * m_RegToolsThreadParams[i].m_CudaDeviceLoad);
    if((start_proj+nProj)>totalProj)  nProj = totalProj-start_proj;
    num_projs[i] = nProj;
    if(start_projs) start_projs[i] = start_proj;
  }
}

void RegTools::ComputeSimilarityMeasure(int plan_id, int similarity_type, int numImageSet, double *sm, float *elapsed_time)
{
  std::pair<std::multimap<int,SimilarityMeasureComputationPlan*>::iterator,std::multimap<int,SimilarityMeasureComputationPlan*>::iterator> ret = 
    m_SimilarityMeasureComputationPlans.equal_range(plan_id);   // find ALL plans that has the same id
  std::multimap<int, SimilarityMeasureComputationPlan*>::iterator it = m_SimilarityMeasureComputationPlans.find(plan_id);
  if(m_NumRegToolsThreads<1)  return;                               // thread is not initialized
  if(ret.first == m_SimilarityMeasureComputationPlans.end())  return; // no similarity measure plans were found

  // compute number of image sets that each GPU is responsible for (this is needed for multi-GPU environment)
  int *num_image_sets = new int[m_NumRegToolsThreads], *start_projs = new int[m_NumRegToolsThreads];
  ComputeNumberOfProjectionForEachGPU(m_RegToolsThreadParams, m_NumRegToolsThreads, numImageSet, num_image_sets, start_projs);

  for(std::multimap<int, SimilarityMeasureComputationPlan*>::iterator it=ret.first;it != ret.second;it++){
    int dev = it->second->m_CudaDeviceID_Sequential;
    if(num_image_sets[dev]==0){
      m_RegToolsThreadParams[dev].m_ProcessingMode = ProcessingMode_DoNothing;
    } else {
      m_RegToolsThreadParams[dev].m_ProcessingMode = ProcessingMode_SimilarityMeasureComputation;
      m_RegToolsThreadParams[dev].m_SimilarityMeasureComputationPlan = it->second;
      m_RegToolsThreadParams[dev].m_SimilarityMeasureComputationPlan2 = NULL;
      m_RegToolsThreadParams[dev].m_SimilarityMeasureComputationPlan->SimilarityMeasureType = similarity_type;
      //print_and_log("RegTools::ComputeSimilarityMeasure, dim_Plan: (%d,%d,%d), dim_Projections: (%d,%d,%d), number of image set: %d\n", 
      //  it->second->ImageDim[0], it->second->ImageDim[1], it->second->ImageDim[2], 
      //  m_RegToolsThreadParams[0].m_ProjectionWidth, m_RegToolsThreadParams[0].m_ProjectionHeight, m_RegToolsThreadParams[0].m_NumEnabledProjections, numImageSet);
      m_RegToolsThreadParams[dev].m_SimilarityMeasureComputationImageOffset = 0;
      m_RegToolsThreadParams[dev].m_SimilarityMeasure_NumberOfImageSet = num_image_sets[dev];
      m_RegToolsThreadParams[dev].m_SimilarityMeasureComputationPlan->SimilarityMeasure = sm + start_projs[dev]; //new double[num_image_sets[dev]];
      m_RegToolsThreadParams[dev].m_ElapsedTime = elapsed_time;
    }
    //print_and_log("ComputeSimilarityMeasure(): device %d, %d projection sets\n", dev, num_image_sets[dev]);
  }
  RunRegToolsThreads();

  delete[] num_image_sets;
  delete[] start_projs;
  return;
}

void RegTools::ComputeSimilarityMeasure(int plan_id1, int plan_id2, int similarity_type, int numImageSet, double *sm, float *elapsed_time)
{
  std::multimap<int, SimilarityMeasureComputationPlan*>::iterator it1 = m_SimilarityMeasureComputationPlans.find(plan_id1);
  std::multimap<int, SimilarityMeasureComputationPlan*>::iterator it2 = m_SimilarityMeasureComputationPlans.find(plan_id2);
  if(m_NumRegToolsThreads<1)  return; // thread is not initialized
  if(it1 != m_SimilarityMeasureComputationPlans.end() && it2 != m_SimilarityMeasureComputationPlans.end()){
    m_RegToolsThreadParams[0].m_ProcessingMode = ProcessingMode_SimilarityMeasureComputation;
    m_RegToolsThreadParams[0].m_SimilarityMeasureComputationPlan = it1->second;
    m_RegToolsThreadParams[0].m_SimilarityMeasureComputationPlan2 = it2->second;
    m_RegToolsThreadParams[0].m_SimilarityMeasureComputationPlan->SimilarityMeasureType = similarity_type;
    m_RegToolsThreadParams[0].m_SimilarityMeasureComputationImageOffset = 0;
    m_RegToolsThreadParams[0].m_SimilarityMeasure_NumberOfImageSet = numImageSet;
    m_RegToolsThreadParams[0].m_SimilarityMeasureComputationPlan->SimilarityMeasure = new double[numImageSet];
    m_RegToolsThreadParams[0].m_ElapsedTime = elapsed_time;
    RunRegToolsThreads(0);
    memcpy(sm, m_RegToolsThreadParams[0].m_SimilarityMeasureComputationPlan->SimilarityMeasure, sizeof(double)*numImageSet);
    delete[] m_RegToolsThreadParams[0].m_SimilarityMeasureComputationPlan->SimilarityMeasure;
    m_RegToolsThreadParams[0].m_SimilarityMeasureComputationPlan->SimilarityMeasure = NULL;
  }
  return;
}

double RegTools::GPUmemCheck(const char* message, int threadID)
{
  m_RegToolsThreadParams[threadID].m_ProcessingMode = ProcessingMode_MemInfoQuery;
  RunRegToolsThreads(threadID);
  print_and_log("%s(%d): %f MB available (total: %f MB)\n", message, threadID, (float)m_RegToolsThreadParams[threadID].m_FreeMem/1024.0f/1024.0f, (float)m_RegToolsThreadParams[threadID].m_TotalMem/1024.0f/1024.0f)
  return (double)m_RegToolsThreadParams[threadID].m_FreeMem;
}

void RegTools::RunRegToolsThreads(int threadID)
{
	//print_and_log("RegTools::RunRegToolsThreads(%d)\n", threadID);
  if(m_WithGL){
	  //print_and_log("RegTools::RunRegToolsThreads(%d), withGL", threadID);
	  // single-thread run
    RegToolsThread_main(&(m_RegToolsThreadParams[0]));
  } else {
#if _WIN32    // multi-thread run
    if(threadID>=0){
      // run single thread
      ::SetEvent(m_RegToolsThreadParams[threadID].m_ProjectorDataReadyEvent);

      // wait until complete
      ::WaitForSingleObject(m_RegToolsThreadParams[threadID].m_ProjectorCompletedEvent, INFINITE);
      ::ResetEvent(m_RegToolsThreadParams[threadID].m_ProjectorCompletedEvent);
    } else {
		//print_and_log("RegTools::RunRegToolsThreads on all %d devices\n", m_NumRegToolsThreads);
		// run all threads
      for(int i=0;i<m_NumRegToolsThreads;i++) ::SetEvent(m_RegToolsThreadParams[i].m_ProjectorDataReadyEvent);

      // wait until all threads complete
      ::WaitForMultipleObjects(m_NumRegToolsThreads, m_RegToolsThread_AllCompletedEvent, true, INFINITE);
      for(int i=0;i<m_NumRegToolsThreads;i++) ::ResetEvent(m_RegToolsThreadParams[i].m_ProjectorCompletedEvent);
    }
#endif
  }
}

void RegTools::PrepareForRegToolsThread(RegToolsThreadParam *param)
{
  std::map<int, ProjectionParametersSetting*>::iterator it = m_ProjectionParametersSetting.find(m_CurrentProjectionParametersSettingID);
  if(it != m_ProjectionParametersSetting.end()){
    param->m_ProjectionParameters = it->second->m_ProjectionParameters;
    param->m_EnabledProjection = it->second->m_SubSamplingArray;
    param->m_ProjectionWidth = it->second->m_ProjectionWidth;
    param->m_ProjectionHeight = it->second->m_ProjectionHeight;
    param->m_ZeroPixelCount = it->second->m_ZeroPixelCount;
  }

  param->m_NumTotalProjections = GetNumberOfProjections();  // total number of projections
  // determine which projection image is enabled
  param->m_NumEnabledProjections = GetNumberOfEnabledProjections();

  // set up projector parameters
  param->m_ProjectorMode = m_ProjectorMode;
  param->m_TransferBlockSize = m_TransferBlockSize;
  param->m_NumProjectionSets = GetNumberOfProjectionSets();
  param->m_VolumeDim = m_VolumeDim;
  param->m_VoxelSize_mm = m_VoxelSize_mm;
  param->m_NumVolumes = m_NumVolumes;
  param->m_StepSize = m_StepSize;
  param->m_NormalizeMax = m_NormalizeMax;
  param->m_NormalizeMin = m_NormalizeMin;
  param->m_World_Volume_col = m_VolumeTransform_col;
  memcpy(param->m_ProjectionImagePBOResource, m_ProjectionImagePBOResource, sizeof(cudaGraphicsResource*)*m_NumPBO);
  param->m_PBO_index = m_PBO_rendering_start_index;
  param->m_RayCastingThreshold = m_RayCastingThreshold;
  param->m_RayCastingDistanceFalloffCoefficient = m_RayCastingDistanceFalloffCoefficient;
  param->m_RayCastingLOD = m_RayCastingLOD;
  param->m_CountNonIntersectedPixel = m_CountNonIntersectedPixel;
  param->m_DifferentVolumePerProjectionSet = m_DifferentVolumePerProjectionSet;

  param->m_MemoryStoreMode = MemoryStoreMode_Replace;   // 'replace' mode, memory is going to be newly allocated
  param->m_ProjectionsInit = NULL;
  param->d_ProjectionsInit = NULL;

  param->m_Interpolator_num_scattered_pnts = 0;
//  param->m_VolumePlan_cudaArray_warpX = param->m_VolumePlan_cudaArray_warpY = param->m_VolumePlan_cudaArray_warpZ = NULL;
}

int RegTools::CMAESPopulation(int arz_ID, int arx_ID, int arxvalid_ID, int xmean_ID, int diagD_ID, int lbounds_ID, int ubounds_ID)
{
  std::map<int, VolumePlan_cudaArray*>::iterator it = m_VolumePlan_cudaArrays.find(arz_ID);
  if(m_NumRegToolsThreads<1)  return false; // thread is not initialized
  if(it != m_VolumePlan_cudaArrays.end()){
    for(int i=0;i<m_NumRegToolsThreads;i++){
      m_RegToolsThreadParams[i].m_ProcessingMode = ProcessingMode_CMAESPopulation;
      m_RegToolsThreadParams[i].m_VolumeDim = make_cudaExtent(it->second->VolumeDim[0], it->second->VolumeDim[1], it->second->VolumeDim[2]);
      m_RegToolsThreadParams[i].m_VolumePlan_cudaArray = it->second;
      m_RegToolsThreadParams[i].d_arx = FindVolumePlan(arx_ID)->d_raw_volume[i];
      m_RegToolsThreadParams[i].d_arxvalid = FindVolumePlan(arxvalid_ID)->d_raw_volume[i];
      m_RegToolsThreadParams[i].d_CMAES_xmean = FindVolumePlan(xmean_ID)->d_raw_volume[i];
      m_RegToolsThreadParams[i].d_CMAES_diagD = FindVolumePlan(diagD_ID)->d_raw_volume[i];
      m_RegToolsThreadParams[i].d_CMAES_lbounds = FindVolumePlan(lbounds_ID)->d_raw_volume[i];
      m_RegToolsThreadParams[i].d_CMAES_ubounds = FindVolumePlan(ubounds_ID)->d_raw_volume[i];
    }
    RunRegToolsThreads();
    return false;
  } else {
    print_and_log("Volume plan was not found\n");
    return false;
  }
}

// This function returns the best GPU which has maximum GFLOPS. (this function is copied from NVIDIA GPU Computing SDK)
int RegTools::cutGetMaxGflopsDeviceId(void)
{
	int current_device   = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, max_perf_device  = 0;
	int device_count     = 0, best_SM_arch     = 0;
    int arch_cores_sm[3] = { 1, 8, 32 };
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount( &device_count );
	// Find the best major SM Architecture GPU device
	while ( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major > 0 && deviceProp.major < 9999) {
			best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

    // Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
		    sm_per_multiproc = 1;
		} else if (deviceProp.major <= 2) {
			sm_per_multiproc = arch_cores_sm[deviceProp.major];
		} else {
			sm_per_multiproc = arch_cores_sm[2];
		}

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		if( compute_perf  > max_compute_perf ) {
            // If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 ) {
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch) {	
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			} else {
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}

int RegTools::CreateProjectionImagePBO(int num)
{ 
  // we assume single thread for PBO rendering
  if(m_ProjectionImagePBOResource)  delete[] m_ProjectionImagePBOResource;
  if(m_RegToolsThreadParams[0].m_ProjectionImagePBOResource)  delete[] m_RegToolsThreadParams[0].m_ProjectionImagePBOResource;
  m_NumPBO = num;
  m_ProjectionImagePBOResource = new cudaGraphicsResource*[m_NumPBO];
  m_RegToolsThreadParams[0].m_ProjectionImagePBOResource = new cudaGraphicsResource*[m_NumPBO];

  for(int i=0;i<m_NumPBO;i++){
    m_ProjectionImagePBOResource[i] = NULL;
    m_RegToolsThreadParams[0].m_ProjectionImagePBOResource[i] = NULL;
  }
  return true;
}

int RegTools::SetProjectionImagePBO(unsigned int pbo)
{
  unsigned int pbos[1] = {pbo};
  return SetProjectionImagePBOs(pbos, 0, 1);
}

int RegTools::SetProjectionImagePBOs(unsigned int *pbos, int start, int num)
{ 
  // we assume single thread for PBO rendering
  for(int j=start;j<start+num;j++){
    m_RegToolsThreadParams[0].m_ProcessingMode = ProcessingMode_SetPBO;
    m_RegToolsThreadParams[0].m_PBO_index = j;
    if(m_ProjectionImagePBOResource[j]){
      // unregister
      m_RegToolsThreadParams[0].m_ProjectionImagePBOResource[j] = m_ProjectionImagePBOResource[j];
      RunRegToolsThreads(0);
    }
    // register PBO
    m_RegToolsThreadParams[0].m_ProjectionImagePBOResource[j] = NULL;
    m_RegToolsThreadParams[0].m_PBO = pbos[j];
    RunRegToolsThreads(0);
    m_ProjectionImagePBOResource[j] = m_RegToolsThreadParams[0].m_ProjectionImagePBOResource[j]; 
  }
  return true;
}

void RegTools::DeleteProjectionImagePBO(void)
{
  // we assume single thread for PBO rendering
  // unregister all PBOs
  for(int j=0;j<m_NumPBO;j++){
    if(m_ProjectionImagePBOResource[j]){
      m_RegToolsThreadParams[0].m_ProjectionImagePBOResource[j] = m_ProjectionImagePBOResource[j];
      m_RegToolsThreadParams[0].m_ProcessingMode = ProcessingMode_SetPBO;
      m_RegToolsThreadParams[0].m_PBO_index = j;
      RunRegToolsThreads(0);
      m_ProjectionImagePBOResource[j] = NULL;
    }
  }
  // Delete PBO array
  if(m_ProjectionImagePBOResource)  delete[] m_ProjectionImagePBOResource;
  if(m_RegToolsThreadParams[0].m_ProjectionImagePBOResource)  delete[] m_RegToolsThreadParams[0].m_ProjectionImagePBOResource;
  m_ProjectionImagePBOResource = NULL;
  m_RegToolsThreadParams[0].m_ProjectionImagePBOResource = NULL;
  m_NumPBO = 0;
}

void RegTools::SetInitialProjectionOnDevice(const float* h_proj, const size_t len)
{
  for (int dev_idx = 0; dev_idx < m_NumRegToolsThreads; ++dev_idx)
  {
    m_RegToolsThreadParams[dev_idx].m_ProjectionsInit = const_cast<float*>(h_proj);
    m_RegToolsThreadParams[dev_idx].m_ProjectionsInit_len = len;
    m_RegToolsThreadParams[dev_idx].m_ProcessingMode = ProcessingMode_CopyHostInitProjectionToDevice;
  }
  RunRegToolsThreads();
}

void RegTools::ClearInitialProjectionOnDevice()
{
  for (int dev_idx = 0; dev_idx < m_NumRegToolsThreads; ++dev_idx)
  {
    m_RegToolsThreadParams[dev_idx].m_ProcessingMode = ProcessingMode_ClearDeviceInitProjection;
  }
  RunRegToolsThreads();
}

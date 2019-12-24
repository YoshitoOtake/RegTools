# Author(s):  Yoshito Otake, Ali Uneri
# Created on: 2013-03-01

cmake_minimum_required(VERSION 2.6)

project(RegTools)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${SRCROOT})

# define options
# -----------------------------------------------------------------------------
option(RegTools_HAS_MATLAB "Generates Matlab bindings" ON)
option(RegTools_HAS_GLEW "" OFF)
option(RegTools_VERBOSE_MESSAGE "" OFF)
option(RegTools_GI_BACKGROUND_EDGE_CHECK "Zero gradient around the pixel with intensity smaller than threshold" OFF)
set(RegTools_DEFAULT_TRANSFER_BLOCK_SIZE 5000 CACHE STRING "maximum block size when the projection images are transfered from CPU to GPU")
set(RegTools_BLOCK_SIZE_X 8 CACHE STRING "block size in X direction")
set(RegTools_BLOCK_SIZE_Y 8 CACHE STRING "block size in Y direction")
set(RegTools_BLOCK_SIZE_Z 16 CACHE STRING "block size in Z direction")
set(RegTools_MI_NUMBER_OF_BINS 64 CACHE STRING "number of bins in MI computation")
set(RegTools_CUDA_COMPUTE_CAPABILITY "" CACHE STRING "")

# find libraries
# -----------------------------------------------------------------------------
find_package(CUDA REQUIRED)
set(CUDA_VERSION_MAJOR ${CUDA_VERSION_MAJOR} CACHE STRING "Major version of CUDA as computed from nvcc.")
set(CUDA_VERSION_MINOR ${CUDA_VERSION_MINOR} CACHE STRING "Minor version of CUDA as computed from nvcc.")
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})	# we need driver API for cuMemGetInfo()

set(CUDA_COMPUTE_CAPABILITY 1.0)  # minimum supported compute capability
if(RegTools_CUDA_COMPUTE_CAPABILITY)  # defined by user
  set(CUDA_COMPUTE_CAPABILITY ${RegTools_CUDA_COMPUTE_CAPABILITY})
else()  # ty to detect GPU with maximum compute capability
  set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
  try_run(
    RUN_RESULT_VAR COMPILE_RESULT_VAR
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/CMake/CudaMaximumComputeCapability.cxx
    CMAKE_FLAGS
      -DINCLUDE_DIRECTORIES:STRING=${CUDA_INCLUDE_DIRS}
      -DLINK_LIBRARIES:STRING=${CUDA_LIBRARIES}
    COMPILE_OUTPUT_VARIABLE COMPILATION_OUTPUT
    RUN_OUTPUT_VARIABLE RUN_OUTPUT)
  if(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
    message(STATUS "Detected CUDA capable GPU(s) with maximum compute capability ${RUN_OUTPUT}")
    set(CUDA_COMPUTE_CAPABILITY ${RUN_OUTPUT})
	set(RegTools_CUDA_COMPUTE_CAPABILITY ${RUN_OUTPUT} CACHE STRING "" FORCE)
  else()
    message(WARNING "Failed to detect a CUDA capable GPU, please specify CUDA_COMPUTE_CAPABILITY manually")
  endif()
endif()

mark_as_advanced(
  RegTools_HAS_GLEW
)

if(RegTools_HAS_MATLAB)
  set(BUILD_SHARED_LIBS ON)
else()
  install(CODE "message(STATUS \"Matlab bindings are not installed.\")")
endif()

# configure files
# -----------------------------------------------------------------------------
configure_file(
  ${PROJECT_SOURCE_DIR}/config.h.in
  ${PROJECT_BINARY_DIR}/config.h
  @ONLY
  )
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

# additional flags for CUDA
if(CUDA_COMPUTE_CAPABILITY EQUAL 2.0 OR CUDA_COMPUTE_CAPABILITY GREATER 2.0)
  set(RegTools_ENABLE_CUDA20_CAPABILITY_FEATURES ON CACHE BOOL "Enable features that is available on GPUs with compute capability 2.0 or later (3D grid & floating point atomic function)")
endif()

# enable CUDA compiler options based on compute capability
string(REPLACE "." "" GPU_ARCHITECTURE "sm_${CUDA_COMPUTE_CAPABILITY}")
set(CUDA_OPTIONS
  "--ptxas-options=-v"
  "--gpu-architecture=${GPU_ARCHITECTURE}")
if(CUDA_COMPUTE_CAPABILITY LESS 1.0)
  message(FATAL_ERROR "CUDA compute capability less than 1.0 is not supported")
endif()
if(CUDA_COMPUTE_CAPABILITY LESS 3.0)
  list(APPEND CUDA_OPTIONS "--maxrregcount=32")
endif()
set(CUDA_NVCC_FLAGS ${CUDA_OPTIONS} CACHE STRING "" FORCE)

if(RegTools_HAS_GLEW)
  find_library(GLEW_LIBRARY
    NAMES glew64 glew32
    HINTS ${CUDA_SDK_ROOT_DIR}/common/lib
    DOC "glew library"
    )
    SUBDIRS(
	benchmarks/Benchmark_2D3DRegistration
	benchmarks/Benchmark_2D3DRegistration_Parallelization
	samples/Sample01_OpenGL
    )
    set(GLEW_INCLUDE_PATH "" CACHE PATH "include directory for glew library")
endif()

# add libraries
# -----------------------------------------------------------------------------
SET(SIMILARITY_MEASURE_FILES_CUDA
  SimilarityMeasureComputation/SimilarityMeasures.cu
  SimilarityMeasureComputation/SimilarityMeasures.cuh
)
SET(CUDA_FILES
  Projector.cu
  Projector.cuh
  Interpolator.cuh
  )
  
if(RegTools_HAS_MATLAB)
  SET(MATLAB_SOURCE
    RegToolsMatlab.h
    RegToolsMatlab.cpp
  )
endif()

cuda_add_library(
  RegToolsLIB
  ${PROJECT_BINARY_DIR}/config.h
  RegTools.h
  RegTools.cpp
  RegTools_ThreadFunctions.cpp
  RegTools_MathFunctions.cpp
  ${MATLAB_SOURCE}
  ProjectionParameterStructures.h
  multithreading.cpp
  multithreading.h
  my_cutil_math.h
  ${CUDA_FILES}
  ${SIMILARITY_MEASURE_FILES_CUDA}
  ${CUDA_curand_LIBRARY}
  )
CUDA_ADD_CUBLAS_TO_TARGET( RegToolsLIB )
CUDA_ADD_CUFFT_TO_TARGET( RegToolsLIB )

SOURCE_GROUP("Cuda Files" FILES ${CUDA_FILES})
SOURCE_GROUP("Similarity Measure Files" FILES ${SIMILARITY_MEASURE_FILES_CUDA} )
include_directories (SimilarityMeasureComputation)

# if(RegTools_HAS_MATLAB)
  target_link_libraries(RegToolsLIB ${CUDA_curand_LIBRARY})
# endif()

# install bindings
# -----------------------------------------------------------------------------

install(TARGETS
  RegToolsLIB
  DESTINATION ${PROJECT_BINARY_DIR}/bin
  )
configure_file(
  ${PROJECT_SOURCE_DIR}/RegToolsConfig.cmake.in
  ${PROJECT_BINARY_DIR}/RegToolsConfig.cmake
  @ONLY
  )

if(RegTools_HAS_MATLAB)
   if(WIN32)
     set(RegToolsMatlabLIB "RegToolsLIB")
   else()
     set(RegToolsMatlabLIB "libRegToolsLIB")
   endif()
   set(ProjectorMatlabHEADER1 "RegToolsMatlab.h")
   set(ProjectorMatlabHEADER2 "ProjectionParameterStructures.h")
  install(TARGETS
    RegToolsLIB
    DESTINATION ${PROJECT_BINARY_DIR}/Matlab
    )
  install(FILES
    ${PROJECT_SOURCE_DIR}/RegToolsMatlab.h
    ${PROJECT_SOURCE_DIR}/ProjectionParameterStructures.h
    ${PROJECT_BINARY_DIR}/config.h
    ${PROJECT_SOURCE_DIR}/MatlabClasses/RegTools.m
    DESTINATION ${PROJECT_BINARY_DIR}/Matlab
    )
  install(FILES
    ${PROJECT_SOURCE_DIR}/MatlabClasses/Registration2D3D/cmaes_ex.m
    ${PROJECT_SOURCE_DIR}/MatlabClasses/Registration2D3D/cmaes_ex_par.m
    ${PROJECT_SOURCE_DIR}/MatlabClasses/Registration2D3D/fminsearch_ex.m
    ${PROJECT_SOURCE_DIR}/MatlabClasses/Registration2D3D/Registration2D3DInterface.m
    ${PROJECT_SOURCE_DIR}/MatlabClasses/Registration2D3D/Registration2D3D_IntensityBased.m
    ${PROJECT_SOURCE_DIR}/MatlabClasses/Registration2D3D/Registration2D3D_IntensityBased_ObjectiveFunctionParameters.m
    DESTINATION ${PROJECT_BINARY_DIR}/Matlab
    )
endif()

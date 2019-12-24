/*
  Benchmark test for 2D3D registration Parallelization
  Author(s):  Yoshito Otake
  Created on: 2013-02-21
*/

#include "RegTools.h"
#include <iostream>
#include <fstream>
#include <float.h>
#include <time.h>

#define DS_VOLUME 2
#define DS_PROJECTION 8
#define PIXEL_WIDTH_ORIGINAL 0.388
#define PIXEL_HEIGHT_ORIGINAL 0.388
//#define MULTI_GPU

#define VOLUME_DIM 256/DS_VOLUME
#define PROJECTION_DIM 768/DS_PROJECTION

double voxel_size[3] = {1.0*DS_VOLUME, 1.0*DS_VOLUME, 1.0*DS_VOLUME};
double focal_length_mm = 1200, pixel_width_mm = PIXEL_WIDTH_ORIGINAL * DS_PROJECTION, pixel_height_mm = PIXEL_HEIGHT_ORIGINAL * DS_PROJECTION, image_center_pix[2] = {PROJECTION_DIM/2, PROJECTION_DIM/2};
double intrinsic_3x3_column_major[9] = { -focal_length_mm/pixel_width_mm, 0, 0,  0, -focal_length_mm/pixel_width_mm, 0, image_center_pix[0], image_center_pix[1], 1};
#define NUM_REPEAT 50
#define PROJECTION_NUM_STEP 100
#define OUT_FILENAME "./benchmark_log.csv"

#if defined(MULTI_GPU)
#define MAX_PROJECTIONS 2000
int deviceIDs[2] = {0, 1}, num_GPU = 2;
#else
#define MAX_PROJECTIONS 2000
int deviceIDs[1] = {1}, num_GPU = 1;
#endif

int main()
{
  // create random volume and projection
  float *volume = new float[VOLUME_DIM*VOLUME_DIM*VOLUME_DIM];
  float *projection = new float[PROJECTION_DIM*PROJECTION_DIM];
  float *mask = new float[PROJECTION_DIM*PROJECTION_DIM];
  for(int i=0;i<VOLUME_DIM*VOLUME_DIM*VOLUME_DIM;i++) volume[i] = (float)rand()/(float)RAND_MAX * 1;
  for(int i=0;i<PROJECTION_DIM*PROJECTION_DIM;i++){
    projection[i] = (float)rand()/(float)RAND_MAX;
    mask[i] = (float)rand()/(float)RAND_MAX;
  }
  // initialize RegTools library
  RegTools *regTools = new RegTools();
//  regTools->AddLogFile("log_file.txt");  // just for debugging. not required.
  regTools->InitializeRegToolsThread(deviceIDs, num_GPU, NULL, false);

  // projector setting
  regTools->SetStepSize( 1.0f );//voxel_width );

  VolumePlan_cudaArray plan;
  plan.h_volume = volume;
  plan.VolumeDim[0] = plan.VolumeDim[1] = plan.VolumeDim[2] = VOLUME_DIM;
  plan.numVolumes = 1;
  memcpy(plan.VoxelSize, voxel_size, sizeof(double)*3);
  int forwardProjectionPlan = regTools->CreateVolumePlan_cudaArray( &plan );

  // prepare log file
  FILE * pFile;
  pFile = fopen (OUT_FILENAME,"w");
  if(pFile==NULL){ printf("file open error!\n"); exit(0); }
  fprintf(pFile,"number of projections, elapsed time\n");

  LARGE_INTEGER frequency;
  if (::QueryPerformanceFrequency(&frequency) == FALSE)
      throw "foo";

  int num_total_tests = MAX_PROJECTIONS/PROJECTION_NUM_STEP + 5;
  int *benchmark_projection = new int[num_total_tests];
  benchmark_projection[0] = 1;
  for(int i=0;i<5;i++) benchmark_projection[i+1] = 20*(i+1);
  for(int i=0;i<MAX_PROJECTIONS/PROJECTION_NUM_STEP;i++) benchmark_projection[i+5] = (i+1)*PROJECTION_NUM_STEP;
  for(int i=0;i<num_total_tests;i++)  printf("tests[%d]: %d projections\n", i, benchmark_projection[i]);
//exit(0);  // for debugging
  for(int i=0;i<num_total_tests;i++){
    int num_projection = benchmark_projection[i];
    printf("Volume dimension:(%.0f x %.0f x %.0f), Projection:(%d x %d) x %d\n", VOLUME_DIM, VOLUME_DIM, VOLUME_DIM, PROJECTION_DIM, PROJECTION_DIM, num_projection);

    // initialize geometry for RegTools
    int geom_id  = regTools->InitializeProjectionParametersArray(num_projection); // initialize memory to store geometry data for one projection
    regTools->SetProjectionDim(PROJECTION_DIM, PROJECTION_DIM);
    for(int proj=0;proj<num_projection;proj++){
      double ext[16];
      regTools->LoadIdentity_4x4d(ext);
      regTools->Translate_col(ext, 0, 0, 600);

      double pm_3x4_row_major[12], *in = intrinsic_3x3_column_major;
      for(int i=0;i<3;i++) for(int j=0;j<4;j++) pm_3x4_row_major[i*4+j] = in[i+0]*ext[j*4+0]+in[i+3]*ext[j*4+1]+in[i+6]*ext[j*4+2]; // column major -> row major
  //    printf("pm:\n");
  //    for(int i=0;i<3;i++) printf("%f %f %f %f\n", pm_3x4_row_major[i*4+0], pm_3x4_row_major[i*4+1], pm_3x4_row_major[i*4+2], pm_3x4_row_major[i*4+3]);

      regTools->SetProjectionParameter_3x4PM(proj, pm_3x4_row_major, pixel_width_mm, pixel_height_mm, PROJECTION_DIM, PROJECTION_DIM);
    }
    regTools->SetTransferBlockSize( num_projection );

    SimilarityMeasureComputationPlan SM_plan;
    regTools->InitializeSimilarityMeasureComputationPlan( &SM_plan );
    SM_plan.h_fixed_images = projection;
    SM_plan.ImageDim[0] = SM_plan.ImageDim[1] = PROJECTION_DIM;
    SM_plan.ImageDim[2] = 1;
    SM_plan.MaxNumImageSets = num_projection;
    SM_plan.Sigma = 1.0;
    SM_plan.h_mask_weight = mask;
    int similarityMeasurePlan = regTools->CreateSimilarityMeasureComputationPlan( &SM_plan );

    // prepare memory to store the projection image
    struct ProjectionResult projectionResult;
    projectionResult.Data = NULL;
    projectionResult.projectionTime = NULL;
    projectionResult.minValue = projectionResult.maxValue = NULL;
    projectionResult.dDataID = projectionResult.numGPU = -1;

    // check max/min by executing one projection using the default viewpoint
    double localTrans[16], *similarityMeasures = new double[num_projection];
    regTools->LoadIdentity_4x4d(localTrans);

    // start benchmark
    double *globalTrans = new double[16*num_projection];
    for(int i=0;i<num_projection;i++){
      // randomly perturb +-10 degrees in X and Y rotation
      regTools->LoadIdentity_4x4d(globalTrans+i*16);
      double angleX = ((double)rand()/(double)RAND_MAX - 0.5) * 20 * 3.1415926535/180;
      double angleY = ((double)rand()/(double)RAND_MAX - 0.5) * 20 * 3.1415926535/180;
      regTools->RotateX_col(globalTrans+i*16, angleX);
      regTools->RotateY_col(globalTrans+i*16, angleY);
    }
    for(int repeat=0;repeat<NUM_REPEAT;repeat++){
      LARGE_INTEGER start;
      if (::QueryPerformanceCounter(&start) == FALSE)
          throw "foo";
      regTools->ForwardProjection_withPlan(projectionResult, forwardProjectionPlan, num_projection, globalTrans, 1, 1, localTrans);
      regTools->ComputeSimilarityMeasure(similarityMeasurePlan, SIMILARITY_MEASURE_GI_SINGLE, num_projection, similarityMeasures);
      LARGE_INTEGER end;
      if (::QueryPerformanceCounter(&end) == FALSE)
          throw "foo";
      double elapsed_time = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
      printf("repeat: %d, %d proj, elapsed time: %f sec (%f iter/sec)\n", repeat, num_projection, elapsed_time, (float)num_projection/elapsed_time);
      fprintf(pFile, "%d, %f\n", num_projection, elapsed_time);
    }
    delete[] globalTrans;

//      for(int i=0;i<5 && i<num_projection;i++) printf("similarity measure(%d/%d): %.6f\n",i+1,num_projections[trials],similarityMeasures[i]);
    regTools->DeleteProjectionParametersArray(geom_id);
    regTools->DeleteSimilarityMeasureComputationPlan(similarityMeasurePlan);
    delete[] similarityMeasures;
  }
  delete[] benchmark_projection;
  fclose (pFile);
  delete regTools;
  delete[] volume;
  delete[] projection;
  delete[] mask;
}

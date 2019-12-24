/*
  DRR generation using RegTools, example 1
  Author(s):  Yoshito Otake
  Created on: 2013-09-03
*/

#include <GL/glew.h>
#include "RegTools.h"
#include <iostream>
#include <fstream>
#include <string>
#include <float.h>
#include <time.h>
#include "Vector3.h"
#include "zpr.h"
#define _USE_MATH_DEFINES
#include <math.h>  // for M_PI_2, cos, sin

#define MAX_KEYS 256
//int WINDOW_WIDTH = 1000, WINDOW_HEIGHT = 1000;
//int WINDOW_WIDTH = 1024, WINDOW_HEIGHT = 768;
//int WINDOW_WIDTH = 1280, WINDOW_HEIGHT = 1024;
int WINDOW_WIDTH = 512, WINDOW_HEIGHT = 512;
//float PIXEL_WIDTH = 0.291, PIXEL_HEIGHT = 0.291;
float PIXEL_WIDTH = 0.287, PIXEL_HEIGHT = 0.287;

#define DATA4
//#define DATA_DIR "../../../data/"
#define DATA_DIR "D:/Collaboration/Nara Medical University/20141015_Biplane_Ankle/Mhd_format/"
//#define DATA_DIR "C:/Users/yoshi/Documents/Video-CT registration/20130918_OR_Dataset/Case-2012-12-19/CT data/"
//#define DATA_DIR "K:/Projects/Video-CT registration/20130918_OR_Dataset/Case-2013-02-20-2/CT_data/"
//#define DATA_DIR "C:/Users/yoshi/Documents/Collaboration/DrAshikaga/20140325_PreliminaryStudy/20140327_BM_Anonymized/"
//#define DATA_DIR "D:/Projects/FaceTransplant/20140409_AtlasData_from_Rob/original_volume/"

#if defined(DATA1)
  #define VOLUME_FILE_NAME DATA_DIR "Jeeves_BoneB60f_456_263_494_pix_1_1_1_mm.img"
  int vol_w_pix = 456, vol_h_pix = 263, vol_d_pix = 494;
  float voxel_width = 1.0, voxel_height = 1.0, voxel_depth = 1.0;
  #define DEFAULT_Z_TRANSLATION    -1500
  #define DEFAULT_ROTATION        glRotatef(90,1,0,0)
  float intensity_addjustment_scaling = 1;
  float HU_offset = 1000.0f;
#elif defined(DATA2)
  #define VOLUME_FILE_NAME DATA_DIR "JohnyHopkins_Chest.raw"
  int vol_w_pix = 512, vol_h_pix = 512, vol_d_pix = 659;
  float voxel_width = 0.97656, voxel_height = 0.97656, voxel_depth = 0.5;
  #define DEFAULT_Z_TRANSLATION    -1500
  #define DEFAULT_ROTATION        glRotatef(90,1,0,0)
  float intensity_addjustment_scaling = 1;
#elif defined(DATA3)
  #define VOLUME_FILE_NAME DATA_DIR "20130711_C_Spine_Axial_0.75_B70s_medfilt.raw"
  int vol_w_pix = 512, vol_h_pix = 512, vol_d_pix = 1081;
  float voxel_width = 0.683594, voxel_height = 0.683594, voxel_depth = 0.700012;
  #define DEFAULT_Z_TRANSLATION    -1500
  #define DEFAULT_ROTATION        glRotatef(-90,1,0,0)
  float intensity_addjustment_scaling = 1;
  float HU_offset = 0.0f;
#elif defined(DATA4)
//  #define VOLUME_FILE_NAME DATA_DIR "GALLIA 01092010_preopCT_512_512_394_pix_0468_0468_0500_mm.raw"
//  int vol_w_pix = 512, vol_h_pix = 512, vol_d_pix = 394;
//  #define VOLUME_FILE_NAME DATA_DIR "SE2_ConvolutionKernel_FC81.raw"
//  #define VOLUME_FILE_NAME DATA_DIR "SE2_ConvolutionKernel_FC81_tarsus.raw"
  #define VOLUME_FILE_NAME DATA_DIR "SE2_ConvolutionKernel_FC81_tibia_fibula.raw"
//  #define VOLUME_FILE_NAME DATA_DIR "SE3_ConvolutionKernel_FC05.raw"
  //int vol_w_pix = 512, vol_h_pix = 512, vol_d_pix = 376;
  //float voxel_width = 0.468, voxel_height = 0.468, voxel_depth = 0.5;
  int vol_w_pix = 512, vol_h_pix = 512, vol_d_pix = 431;
  float voxel_width = 0.266, voxel_height = 0.266, voxel_depth = 0.299896;
  #define VOLUME_MASK_FILE_NAME ""
///  #define VOLUME_FILE_NAME DATA_DIR "Case-2012-12-19_CT_Series_DentalThin.raw"
///  int vol_w_pix = 512, vol_h_pix = 512, vol_d_pix = 219;
///  float voxel_width = 0.406, voxel_height = 0.406, voxel_depth = 0.6;
//  #define FIDUCIAL_MARKUPS_FILE_NAME DATA_DIR "AnatomicalLandmark.fcsv"
  //#define FIDUCIAL_MARKUPS_FILE_NAME DATA_DIR "uncinate process left.fcsv"
  //#define VOLUME_FILE_NAME DATA_DIR "Case-2013-02-20-2_Series6_H30s.raw"
  //#define VOLUME_MASK_FILE_NAME DATA_DIR "Case-2013-02-20-2_Series6_H30s_decongestion_mask_uncomp.raw"
  //int vol_w_pix = 512, vol_h_pix = 512, vol_d_pix = 226;
  //float voxel_width = 0.443, voxel_height = 0.443, voxel_depth = 0.6;
  #define DEFAULT_Z_TRANSLATION    -732
#define DEFAULT_ROTATION    { glRotatef(-90,1,0,0); }
  #define DEFAULT_X_OFFSET    0
  #define DEFAULT_Y_OFFSET    0
  #define DEFAULT_Z_OFFSET    0
  #define NO_Y_FLIP
  #define NO_X_FLIP
  float intensity_addjustment_scaling = 1;
  float HU_offset = 0.0f;
#elif defined(DATA5)
  #define VOLUME_FILE_NAME DATA_DIR "CT_cardiac_AFAIDR.raw"
  #define VOLUME_MASK_FILE_NAME DATA_DIR ""
  int vol_w_pix = 512, vol_h_pix = 512, vol_d_pix = 256;
  float voxel_width = 0.625, voxel_height = 0.625, voxel_depth = 0.5;
  #define DEFAULT_Z_TRANSLATION    -500
  #define DEFAULT_ROTATION        glRotatef(-90,1,0,0)
  #define DEFAULT_X_OFFSET    0
  #define DEFAULT_Y_OFFSET    0
  #define DEFAULT_Z_OFFSET    0
  #define NO_Y_FLIP
  #define NO_X_FLIP
  float intensity_addjustment_scaling = 1;
  float HU_offset = 0.0f;
#elif defined(DATA6)
  #define VOLUME_FILE_NAME DATA_DIR "QIN-HEADNECK-01-0584_01.raw"
  #define VOLUME_MASK_FILE_NAME DATA_DIR ""
  int vol_w_pix = 106, vol_h_pix = 136, vol_d_pix = 154;
  float voxel_width = 2.0, voxel_height = 2.0, voxel_depth = 2.0;
  #define DEFAULT_Z_TRANSLATION    -500
  #define DEFAULT_ROTATION        glRotatef(-90,1,0,0)
  #define DEFAULT_X_OFFSET    0
  #define DEFAULT_Y_OFFSET    0
  #define DEFAULT_Z_OFFSET    0
  #define NO_X_FLIP
  float intensity_addjustment_scaling = 1;
  float HU_offset = 0.0f;
#endif

bool gKeys[MAX_KEYS];

GLuint projectionImageTextureBuffer;
GLuint projectionImagePBO;

RegTools *regTools;
int DRR_planID;
struct ProjectionResult projectionResult;
float intensity_window_default, intensity_level_default;
float intensity_window_current, intensity_level_current;
float ray_casting_thresh = -300;
int isSiddon = false, isRayCasting = false;
bool DRR_inverse = false;
float stepsize_unit = 1.0;

int numFiducials = 0;
#define MAX_NUM_FIDUCIALS 50
#define FIDUCIAL_SIZE 0.015
double fiducials[MAX_NUM_FIDUCIALS][3];
char fiducial_ID[5];

// intrinsic parameter matrix
int uv_dim_pix[2] = {WINDOW_WIDTH, WINDOW_HEIGHT};
double focal_length_mm = 1098, pixel_width_mm = PIXEL_WIDTH, pixel_height_mm = PIXEL_HEIGHT, image_center_pix[2] = {uv_dim_pix[0]/2, uv_dim_pix[1]/2};
double intrinsic_3x3_column_major[9] = { -focal_length_mm/pixel_width_mm, 0, 0,  0, -focal_length_mm/(pixel_width_mm*((float)WINDOW_HEIGHT/(float)WINDOW_WIDTH)), 0, image_center_pix[0], image_center_pix[1], 1};

// typical endoscope
//double intrinsic_3x3_column_major[9] = { -780, 0, 0,  0, -780, 0, image_center_pix[0], image_center_pix[1], 1};
//double intrinsic_3x3_column_major[9] = { -871.305, 0, 0,  0, -873.698, 0, 491.012, 390.311, 1};
//double intrinsic_3x3_column_major[9] = { -842.808, 0, 0,  0, -838.348, 0, 594.889, 537.299, 1}; // Case-2012-12-19
//double intrinsic_3x3_column_major[9] = { -845.949, 0, 0,  0, -839.068, 0, 731.721, 468.514, 1}; // Case-2013-02-20-2

void checkGLError(char *str)
{
  GLenum error;
  if ((error = glGetError()) != GL_NO_ERROR)
    printf("GL Error: %s (%s)\n", gluErrorString(error), str);
}

void UpdateWindowLevel()
{
  regTools->SetNormalizeMaxMin(intensity_level_current+intensity_window_current/2, intensity_level_current-intensity_window_current/2);
  regTools->SetRayCastingThreshold(ray_casting_thresh);
  printf("intensity_window = %f, intensity_level = %f, raycasting_threshold = %f\n", intensity_window_current, intensity_level_current, ray_casting_thresh);
}

// create a test volume texture, here you could load your own volume
void load_data()
{
	int size = vol_w_pix * vol_h_pix * vol_d_pix;
  short *loaded = new short[size];
  unsigned char *mask = new unsigned char[size];
  std::ifstream in(VOLUME_FILE_NAME, ios::binary);
  if(!in){
    printf("volume file: %s cannot be loaded\n", VOLUME_FILE_NAME);
    exit(0);
  }
  in.read( reinterpret_cast<char*>(loaded), size * sizeof(short) );

  std::ifstream in_mask(VOLUME_MASK_FILE_NAME, ios::binary);
  if(!in_mask){
    printf("volume mask file: %s cannot be loaded\n", VOLUME_MASK_FILE_NAME);
    memset(mask, 0, size * sizeof(unsigned char));
  } else {
    in_mask.read( reinterpret_cast<char*>(mask), size * sizeof(unsigned char) );
  }
#if !defined NO_Y_FLIP
  // flip in y axis
  int temp, dest, source;
  for(int z=0;z<vol_d_pix;z++){
    for(int y=0;y<vol_h_pix/2;y++){
      for(int x=0;x<vol_w_pix;x++){
        dest = z*vol_h_pix*vol_w_pix + (vol_h_pix-1-y)*vol_w_pix + x;
        source = z*vol_h_pix*vol_w_pix + y*vol_w_pix + x;
        temp = loaded[dest];  loaded[dest] = loaded[source];  loaded[source] = temp;    // swap dest<->source
      }
    }
  }
#endif
#if !defined NO_X_FLIP
  // flip in y axis
  int temp, dest, source;
  for(int z=0;z<vol_d_pix;z++){
    for(int y=0;y<vol_h_pix;y++){
      for(int x=0;x<vol_w_pix/2;x++){
        dest = z*vol_h_pix*vol_w_pix + y*vol_w_pix + (vol_w_pix-1-x);
        source = z*vol_h_pix*vol_w_pix + y*vol_w_pix + x;
        temp = loaded[dest];  loaded[dest] = loaded[source];  loaded[source] = temp;    // swap dest<->source
      }
    }
  }
#endif
/*
  int dest;
  // crop nose tip if needed
  for(int z=0;z<vol_d_pix;z++){
    for(int y=0;y<80;y++){
      for(int x=0;x<vol_w_pix;x++){
        dest = z*vol_h_pix*vol_w_pix + y*vol_w_pix + x;
        loaded[dest] = -1000.0;
      }
    }
  }
*/
  float maxVal = -FLT_MAX, minVal = FLT_MAX;
  float myu_water = 0.02683; // linear attenuation coefficient of water at the effective energy (~120kVp*~30-40%) (see RegTools.m, HU2Myu() for detail)
  float *loaded_f = new float[size];
  for(int xyz = 0; xyz<size;xyz++){
    loaded_f[xyz] = ((float)loaded[xyz]+HU_offset) * (myu_water/1000);  // convert HU -> myu
    if(loaded_f[xyz]<0) loaded_f[xyz] = 0;
    if(mask[xyz] == 
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      

























































































































































































































































































1) loaded[xyz] = -1000;
//    loaded_f[xyz] = (float)loaded[xyz];
    if(loaded_f[xyz]>maxVal) maxVal = loaded_f[xyz];
    if(loaded_f[xyz]<minVal) minVal = loaded_f[xyz];
  }
  delete []loaded;
  printf("maximum value in volume: %f, minimum value in volume: %f\n", maxVal, minVal);

  // initialize DRR rendering plan
  VolumePlan_cudaArray plan;
  plan.h_volume = loaded_f;
  plan.VolumeDim[0] = vol_w_pix; plan.VolumeDim[1] = vol_h_pix; plan.VolumeDim[2] = vol_d_pix;
  plan.VoxelSize[0] = voxel_width; plan.VoxelSize[1] = voxel_height; plan.VoxelSize[2] = voxel_depth;
  plan.numVolumes = 1;
  DRR_planID = regTools->CreateVolumePlan_cudaArray( &plan );
  delete []loaded_f;
}

void load_fiducials()
{
#if defined FIDUCIAL_MARKUPS_FILE_NAME
  std::ifstream in(FIDUCIAL_MARKUPS_FILE_NAME);
  if(!in){
    printf("fiducial markup file: %s cannot be loaded\n", FIDUCIAL_MARKUPS_FILE_NAME);
    exit(0);
  }
  std::string dummy;
  std::getline(in,dummy);
  std::getline(in,dummy);
  std::getline(in,dummy);
  printf("start loading fiducial file\n");
  while(!in.eof() || numFiducials<MAX_NUM_FIDUCIALS){
    std::getline(in, dummy, ',');
    std::getline(in, dummy, ','); fiducials[numFiducials][0] = -atof( dummy.c_str() );  // slicer coordinate -> our coordinate
    std::getline(in, dummy, ','); fiducials[numFiducials][1] = -atof( dummy.c_str() );  // slicer coordinate -> our coordinate
    std::getline(in, dummy, ','); fiducials[numFiducials][2] = atof( dummy.c_str() );
    std::getline(in,dummy,'\n');
    if(in.eof()) break;
    numFiducials++;
  }
  for(int i=0;i<numFiducials && i<MAX_NUM_FIDUCIALS;i++){
    printf("%d: %f, %f, %f\n", i+1, fiducials[i][0], fiducials[i][1], fiducials[i][2]);
  }
#endif
}


void _setStepSize(float step_size)
{
  regTools->SetStepSize(step_size);
  printf("stepsize = %f mm\n", step_size);
  glutPostRedisplay();
}

void SetDefaultPosition()
{
  glLoadIdentity();
  glTranslatef(0,0,DEFAULT_Z_TRANSLATION);
  DEFAULT_ROTATION;
}

// for contiunes keypresses
void ProcessKeys()
{
}

void ResetMaxMin()
{
  float DRR_min, DRR_max;
  projectionResult.minValue = &DRR_min;
  projectionResult.maxValue = &DRR_max;

  // check max/min by executing one projection using the default viewpoint
  regTools->SetNormalizeMaxMin(-1, 1);  // disable normalization by setting min>max
  regTools->ForwardProjection_withPlan(projectionResult, DRR_planID);
  printf("ResetMaxMin(), max: %f, min: %f\n", DRR_max, DRR_min);
  regTools->SetNormalizeMaxMin(DRR_max, DRR_min);
  intensity_window_default = intensity_window_current = DRR_max-DRR_min;
  intensity_level_default = intensity_level_current = (DRR_max+DRR_min)/2;
  projectionResult.minValue = projectionResult.maxValue = NULL; // we don't need to compute max an min anymore (disable it because it is a time consuming process)
}

void ComputeProjectionMatrix(double extrinsic_4x4_column_major[16], double pm_3x4_row_major[12])
{
  double extrinsic_offset_4x4_column_major[16], extrinsic_offsetted_4x4_column_major[16];
  regTools->LoadIdentity_4x4d(extrinsic_offset_4x4_column_major);
  regTools->Translate_col(extrinsic_offset_4x4_column_major, DEFAULT_X_OFFSET, DEFAULT_Y_OFFSET, DEFAULT_Z_OFFSET);
  regTools->MultMatrixd_col(extrinsic_4x4_column_major, extrinsic_offset_4x4_column_major, extrinsic_offsetted_4x4_column_major);

  // compute 3x4 projection matrix based on the current modelview matrix
  double *in = intrinsic_3x3_column_major, *ext = extrinsic_offsetted_4x4_column_major;
  for(int i=0;i<3;i++) for(int j=0;j<4;j++) pm_3x4_row_major[i*4+j] = in[i+0]*ext[j*4+0]+in[i+3]*ext[j*4+1]+in[i+6]*ext[j*4+2]; // note: column major -> row major

  // return offsetted matrix
  for(int i=0;i<16;i++) extrinsic_4x4_column_major[i] = extrinsic_offsetted_4x4_column_major[i];
}

void key(unsigned char k, int x, int y)
{
	gKeys[k] = true;
	// Process keys
	for (int i = 0; i < 256; i++)
	{
		if (!gKeys[i])  { continue; }
    bool isSiddon;
    switch (i)
		{
		case '=': _setStepSize(0.1*stepsize_unit);	break;
		case '-': _setStepSize(0.5*stepsize_unit);	break;
		case '1': _setStepSize(1.0*stepsize_unit);	break;
		case '2': _setStepSize(2.0*stepsize_unit);	break;
		case '3': _setStepSize(3.0*stepsize_unit);	break;
		case '4': _setStepSize(4.0*stepsize_unit);	break;
		case '5': _setStepSize(5.0*stepsize_unit);	break;
		case '6': _setStepSize(6.0*stepsize_unit);	break;
		case '7': _setStepSize(7.0*stepsize_unit);	break;
		case '8': _setStepSize(8.0*stepsize_unit);	break;
		case '9': _setStepSize(9.0*stepsize_unit);	break;
		case '0': _setStepSize(10.0*stepsize_unit);	break;
		case 'r':
      ResetMaxMin();
	    glutPostRedisplay();
			break;
		case 'R':
      _setStepSize(2.0*stepsize_unit);
      intensity_window_current = intensity_window_default;
      intensity_level_current = intensity_level_default;
      UpdateWindowLevel();
      SetDefaultPosition();
	    glutPostRedisplay();
			break;
		case 's':
      isSiddon = !(regTools->GetProjectorMode()==ProjectorMode_Siddon);
      regTools->SetProjectorMode( (isSiddon) ? ProjectorMode_Siddon : ProjectorMode_LinearInterpolation );
      if(isSiddon)   printf("Siddon mode\n");
      else           printf("Linear interpolation mode\n");
      glutPostRedisplay();
      break;
		case 'c':
      isRayCasting = !(regTools->GetProjectorMode()==ProjectorMode_RayCasting);
      regTools->SetProjectorMode( (isRayCasting) ? ProjectorMode_RayCasting : ProjectorMode_LinearInterpolation );
      ResetMaxMin();
      if(isRayCasting)  printf("RayCasting mode\n");
      else              printf("Linear interpolation mode\n");
      glutPostRedisplay();
      break;
    case 'i': 
      DRR_inverse = !DRR_inverse; 
      glutPostRedisplay();
    case 'p': 
      double extrinsic_4x4_column_major[16], pm_3x4_row_major[12], extrinsic_1x6[6];
      glGetDoublev(GL_MODELVIEW_MATRIX, extrinsic_4x4_column_major);
      ComputeProjectionMatrix(extrinsic_4x4_column_major, pm_3x4_row_major);
      regTools->convert4x4ToTransRot(extrinsic_4x4_column_major, extrinsic_1x6);
      printf("extrinsic:\n");
      printf("translation: (%f, %f, %f), rotation: (%f, %f, %f)\n"
        , extrinsic_1x6[0], extrinsic_1x6[1], extrinsic_1x6[2], extrinsic_1x6[3]*180/M_PI, extrinsic_1x6[4]*180/M_PI, extrinsic_1x6[5]*180/M_PI);
      for(int i=0;i<4;i++) printf("%f, %f, %f, %f\n", extrinsic_4x4_column_major[i+0], extrinsic_4x4_column_major[i+4], extrinsic_4x4_column_major[i+8], extrinsic_4x4_column_major[i+12]);
      printf("projection matrix:\n");
      for(int i=0;i<3;i++) printf("%f, %f, %f, %f\n", pm_3x4_row_major[i*4+0], pm_3x4_row_major[i*4+1], pm_3x4_row_major[i*4+2], pm_3x4_row_major[i*4+3]);
      break;
		}
	}
}

void cleanup();

void KeyboardUpCallback(unsigned char key, int x, int y)
{
	gKeys[key] = false;

	switch (key)
	{
	case 27 :
		{
      cleanup(); exit(0); break;
		}
	case ' ':
		break;
	}
}

// glut idle function
void idle_func()
{
	ProcessKeys();
}

////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void createPBO(GLuint* pbo, unsigned int size)
{
  // create buffer object
  glGenBuffers(1, pbo);
  glBindBuffer(GL_ARRAY_BUFFER, *pbo);
  glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete PBO
////////////////////////////////////////////////////////////////////////////////
void deletePBO(GLuint* pbo)
{
  glBindBuffer(GL_ARRAY_BUFFER, *pbo);
  glDeleteBuffers(1, pbo);
  CUT_CHECK_ERROR_GL();

  *pbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Create Texture
////////////////////////////////////////////////////////////////////////////////
void createTexture(GLuint* tex_name, unsigned int size_x, unsigned int size_y, GLint internalFormat, GLenum format, GLenum type)
{
  // create a texture
  glGenTextures(1, tex_name);
  glBindTexture(GL_TEXTURE_2D, *tex_name);

  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  // set basic parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // buffer data
  glTexImage2D(GL_TEXTURE_2D, 0,internalFormat, size_x, size_y, 0, format, type, NULL);
  glBindTexture(GL_TEXTURE_2D, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Delete Texture
////////////////////////////////////////////////////////////////////////////////
void deleteTexture(GLuint* tex)
{
  glDeleteTextures(1, tex);
  *tex = 0;
}

void reshape_ortho(int w, int h)
{
	if (h == 0) h = 1;
//	glViewport(0, 0,w,h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, 1, 0, 1);
	glMatrixMode(GL_MODELVIEW);
}

void draw_fullscreen_quad()
{
	glDisable(GL_DEPTH_TEST);
	glBegin(GL_QUADS);
   
	glTexCoord2f(0,0); 
	glVertex2f(0,0);

	glTexCoord2f(1,0); 
	glVertex2f(1,0);

	glTexCoord2f(1, 1); 
	glVertex2f(1, 1);

  glTexCoord2f(0, 1); 
	glVertex2f(0, 1);

	glEnd();
	glEnable(GL_DEPTH_TEST);
}

// This display function is called once per frame 
void display()
{
  clock_t t_start = clock();
  glPushMatrix();
    double extrinsic_4x4_column_major[16], pm_3x4_row_major[12];
    glGetDoublev(GL_MODELVIEW_MATRIX, extrinsic_4x4_column_major);
    ComputeProjectionMatrix(extrinsic_4x4_column_major, pm_3x4_row_major);
    regTools->SetProjectionParameter_3x4PM(0, pm_3x4_row_major, pixel_width_mm, pixel_height_mm, uv_dim_pix[0], uv_dim_pix[1]);
//    for(int i=0;i<3;i++)
//      printf("%f %f %f %f\n", pm_3x4_row_major[i*4], pm_3x4_row_major[i*4+1], pm_3x4_row_major[i*4+2], pm_3x4_row_major[i*4+3]);

//    if(DRR_inverse) regTools->SetProjectionOption(FORWARD_PROJECTION_INVERSE);
//    else            regTools->SetProjectionOption(0);
    regTools->ForwardProjection_withPlan(projectionResult, DRR_planID);
   
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glLoadIdentity();

    // download texture from PBO (projectionImagePBO)
    glEnable(GL_TEXTURE_2D);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, projectionImagePBO);
    glBindTexture(GL_TEXTURE_2D, projectionImageTextureBuffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_LUMINANCE, GL_FLOAT, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // render a fullscreen plane with texture
    reshape_ortho(WINDOW_WIDTH, WINDOW_HEIGHT);
    draw_fullscreen_quad();
    glDisable(GL_TEXTURE_2D);

    for(int i=0;i<numFiducials;i++){
      double x = pm_3x4_row_major[0]*fiducials[i][0] + pm_3x4_row_major[1]*fiducials[i][1] + pm_3x4_row_major[2]*fiducials[i][2] + pm_3x4_row_major[3];
      double y = pm_3x4_row_major[4]*fiducials[i][0] + pm_3x4_row_major[5]*fiducials[i][1] + pm_3x4_row_major[6]*fiducials[i][2] + pm_3x4_row_major[7];
      double z = pm_3x4_row_major[8]*fiducials[i][0] + pm_3x4_row_major[9]*fiducials[i][1] + pm_3x4_row_major[10]*fiducials[i][2] + pm_3x4_row_major[11];
	    x = (x/z - image_center_pix[0])/WINDOW_WIDTH + 0.5; y = (y/z - image_center_pix[1])/WINDOW_HEIGHT + 0.5;
      //printf("fid[%d]: %f, %f\n", i+1, x, y);
      glBegin(GL_LINES);
        glColor3f(1.0, 0.0, 0.0);
	      glVertex2f(x-FIDUCIAL_SIZE*WINDOW_HEIGHT/WINDOW_WIDTH,y);
	      glVertex2f(x+FIDUCIAL_SIZE*WINDOW_HEIGHT/WINDOW_WIDTH,y);
	      glVertex2f(x,y-FIDUCIAL_SIZE);
	      glVertex2f(x,y+FIDUCIAL_SIZE);
	    glEnd();
      sprintf(fiducial_ID, "%d\0", i+1);
      glRasterPos2f(x+FIDUCIAL_SIZE*WINDOW_HEIGHT/WINDOW_WIDTH/2,y+FIDUCIAL_SIZE/2);
      for(char *c=fiducial_ID; *c!='\0'; c++)
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *c);
    }
  glPopMatrix();
  glutSwapBuffers();

  float time = (float)(clock()-t_start)/CLOCKS_PER_SEC*1000.0f;
  cout << "elapsed time: " << time << " msec" << std::endl;
}

int main(int argc, char* argv[])
{
  if(argc >= 2){
    WINDOW_WIDTH = WINDOW_HEIGHT = atoi(argv[1]);
  }
	glutInit(&argc,argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH);
  char title[256];
#if _WIN32
  sprintf_s(title, 256, "RegTools DRR Sample 1 (%d x %d x %d), window: (%d x %d)", vol_w_pix, vol_h_pix, vol_d_pix, WINDOW_WIDTH, WINDOW_HEIGHT);
#else
  snprintf(title, 256, "RegTools DRR Sample 1 (%d x %d x %d), window: (%d x %d)", vol_w_pix, vol_h_pix, vol_d_pix, WINDOW_WIDTH, WINDOW_HEIGHT);
#endif
  glutCreateWindow(title);
	glutReshapeWindow(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutKeyboardFunc(key);
	glutKeyboardUpFunc(KeyboardUpCallback);
  GLenum err = glewInit();

	glMatrixMode(GL_MODELVIEW);
  SetDefaultPosition();

  // initialize RegTools library
  regTools = new RegTools();
  regTools->AddLogFile("log_file.txt");  // just for debugging. not required.
  regTools->InitializeRegToolsThread_withGL();

  // initialize Pixel Buffer Object
  createTexture(&projectionImageTextureBuffer, WINDOW_WIDTH, WINDOW_HEIGHT, GL_INTENSITY32F_ARB, GL_LUMINANCE, GL_FLOAT);
  createPBO(&projectionImagePBO, WINDOW_WIDTH*WINDOW_HEIGHT*sizeof(float));   // pixel buffer object for the projection image
  regTools->CreateProjectionImagePBO(1);
  regTools->SetProjectionImagePBO(projectionImagePBO);
  regTools->SetRayCastingThreshold(ray_casting_thresh);

  // initialize geometry for RegTools
  int geom_id  = regTools->InitializeProjectionParametersArray(1); // initialize memory to store geometry data for one projection
  regTools->SetProjectionDim(uv_dim_pix[0], uv_dim_pix[1]);
  double extrinsic_4x4_column_major[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, extrinsic_4x4_column_major);
  double pm_3x4_row_major[12], *in = intrinsic_3x3_column_major, *ext = extrinsic_4x4_column_major;
  for(int i=0;i<3;i++) for(int j=0;j<4;j++) pm_3x4_row_major[i*4+j] = in[i+0]*ext[j*4+0]+in[i+3]*ext[j*4+1]+in[i+6]*ext[j*4+2]; // column major -> row major
  printf("pm:\n");
  for(int i=0;i<3;i++) printf("%f %f %f %f\n", pm_3x4_row_major[i*4+0], pm_3x4_row_major[i*4+1], pm_3x4_row_major[i*4+2], pm_3x4_row_major[i*4+3]);

  regTools->SetProjectionParameter_3x4PM(0, pm_3x4_row_major, pixel_width_mm, pixel_height_mm, uv_dim_pix[0], uv_dim_pix[1]);

  // load data
  load_data();
  load_fiducials();

  // projector setting
  regTools->SetProjectorMode( (isSiddon) ? ProjectorMode_Siddon : ProjectorMode_LinearInterpolation );
  regTools->SetStepSize( 1.0f );//voxel_width );

  // prepare memory to store the projection image
//  projectionResult.Data = new float[WINDOW_WIDTH*WINDOW_HEIGHT];  // use this if you need the projection images to be returned to host memory
  projectionResult.Data = NULL;
  projectionResult.projectionTime = NULL;

  ResetMaxMin();

	glutDisplayFunc(display);
	glutIdleFunc(idle_func);
  glLineWidth(2.0);

  /* Configure ZPR module */
  zprInit();

	glutMainLoop();
}

void cleanup()
{
  deletePBO(&projectionImagePBO);
  if(projectionResult.Data) delete[] projectionResult.Data;
  if(projectionResult.minValue) delete projectionResult.minValue;
  if(projectionResult.maxValue) delete projectionResult.maxValue;
  delete regTools;  // need to call destructor before exit()
}

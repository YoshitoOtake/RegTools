#include "RegTools.h"
#include <string.h>     // for memcpy
#define _USE_MATH_DEFINES
#include "math.h"   // for M_PI_2, cos, sin

extern FILE *m_LogFile;

bool RegTools::Inverse4x4d(const double *in, double *out)
{
  // 'in' should be column-major
  double determinant = Determinant4x4d(in);
  const double epsilon = 1e-10;
  if(fabs(determinant)<epsilon){
    print_and_log("RegTools::Inverse4x4d(), determinant is zero\n");
    return false;
  } else {
    out[0]  = (in[9]*in[14]*in[7] - in[13]*in[10]*in[7] + in[13]*in[6]*in[11] - in[5]*in[14]*in[11] - in[9]*in[6]*in[15] + in[5]*in[10]*in[15]) / determinant;
    out[4]  = (in[12]*in[10]*in[7] - in[8]*in[14]*in[7] - in[12]*in[6]*in[11] + in[4]*in[14]*in[11] + in[8]*in[6]*in[15] - in[4]*in[10]*in[15]) / determinant;
    out[8]  = (in[8]*in[13]*in[7] - in[12]*in[9]*in[7] + in[12]*in[5]*in[11] - in[4]*in[13]*in[11] - in[8]*in[5]*in[15] + in[4]*in[9]*in[15]) / determinant;
    out[12] = (in[12]*in[9]*in[6] - in[8]*in[13]*in[6] - in[12]*in[5]*in[10] + in[4]*in[13]*in[10] + in[8]*in[5]*in[14] - in[4]*in[9]*in[14]) / determinant;
    out[1]  = (in[13]*in[10]*in[3] - in[9]*in[14]*in[3] - in[13]*in[2]*in[11] + in[1]*in[14]*in[11] + in[9]*in[2]*in[15] - in[1]*in[10]*in[15]) / determinant;
    out[5]  = (in[8]*in[14]*in[3] - in[12]*in[10]*in[3] + in[12]*in[2]*in[11] - in[0]*in[14]*in[11] - in[8]*in[2]*in[15] + in[0]*in[10]*in[15]) / determinant;
    out[9]  = (in[12]*in[9]*in[3] - in[8]*in[13]*in[3] - in[12]*in[1]*in[11] + in[0]*in[13]*in[11] + in[8]*in[1]*in[15] - in[0]*in[9]*in[15]) / determinant;
    out[13] = (in[8]*in[13]*in[2] - in[12]*in[9]*in[2] + in[12]*in[1]*in[10] - in[0]*in[13]*in[10] - in[8]*in[1]*in[14] + in[0]*in[9]*in[14]) / determinant;
    out[2]  = (in[5]*in[14]*in[3] - in[13]*in[6]*in[3] + in[13]*in[2]*in[7] - in[1]*in[14]*in[7] - in[5]*in[2]*in[15] + in[1]*in[6]*in[15]) / determinant;
    out[6]  = (in[12]*in[6]*in[3] - in[4]*in[14]*in[3] - in[12]*in[2]*in[7] + in[0]*in[14]*in[7] + in[4]*in[2]*in[15] - in[0]*in[6]*in[15]) / determinant;
    out[10] = (in[4]*in[13]*in[3] - in[12]*in[5]*in[3] + in[12]*in[1]*in[7] - in[0]*in[13]*in[7] - in[4]*in[1]*in[15] + in[0]*in[5]*in[15]) / determinant;
    out[14] = (in[12]*in[5]*in[2] - in[4]*in[13]*in[2] - in[12]*in[1]*in[6] + in[0]*in[13]*in[6] + in[4]*in[1]*in[14] - in[0]*in[5]*in[14]) / determinant;
    out[3]  = (in[9]*in[6]*in[3] - in[5]*in[10]*in[3] - in[9]*in[2]*in[7] + in[1]*in[10]*in[7] + in[5]*in[2]*in[11] - in[1]*in[6]*in[11]) / determinant;
    out[7]  = (in[4]*in[10]*in[3] - in[8]*in[6]*in[3] + in[8]*in[2]*in[7] - in[0]*in[10]*in[7] - in[4]*in[2]*in[11] + in[0]*in[6]*in[11]) / determinant;
    out[11] = (in[8]*in[5]*in[3] - in[4]*in[9]*in[3] - in[8]*in[1]*in[7] + in[0]*in[9]*in[7] + in[4]*in[1]*in[11] - in[0]*in[5]*in[11]) / determinant;
    out[15] = (in[4]*in[9]*in[2] - in[8]*in[5]*in[2] + in[8]*in[1]*in[6] - in[0]*in[9]*in[6] - in[4]*in[1]*in[10] + in[0]*in[5]*in[10]) / determinant;
  }
/*
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("Output matrix: ");
  for(int i=0;i<4;i++)  print_and_log("%f %f %f %f\n", out[i], out[i+4], out[i+8], out[i+12]);
#endif
*/
  return true;
}

bool RegTools::Inverse3x3d(const double *in, double *out)
{
  // 'in' should be column-major
  double determinant = Determinant3x3d(in);
  const double epsilon = 1e-10;
  if(fabs(determinant)<epsilon){
    print_and_log("RegTools::Inverse3x3f(), determinant is zero\n");
    return false;
  } else {
    out[0]  = (in[4]*in[8] - in[5]*in[7]) / determinant;
    out[3]  = (in[5]*in[6] - in[3]*in[8]) / determinant;
    out[6]  = (in[3]*in[7] - in[4]*in[6]) / determinant;
    out[1]  = (in[2]*in[7] - in[1]*in[8]) / determinant;
    out[4]  = (in[0]*in[8] - in[2]*in[6]) / determinant;
    out[7]  = (in[1]*in[6] - in[0]*in[7]) / determinant;
    out[2]  = (in[1]*in[5] - in[2]*in[4]) / determinant;
    out[5]  = (in[2]*in[3] - in[0]*in[5]) / determinant;
    out[8]  = (in[0]*in[4] - in[1]*in[3]) / determinant;
  }
  return true;
}

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define SQR(x)  ((x)*(x))
#define FMAX(a,b) (a > b ? a : b)

bool RegTools::QRDecomposition_square(const double *in_column_major, const int n, double *Q, double *R)
{
// This function is copied from Numerical recipes and modified by YO. See the following for detail
// http://books.google.com/books?id=1aAOdzK3FegC&pg=PA103&lpg=PA103&dq=QR+decomposition+numerical+recipes+get+Q&source=bl&ots=3hVgCaGrpd&sig=IFURTqoIR4um7kyyghIOY16h5xA&hl=en&ei=W2XkTNHmAoPGlQeelci2DQ&sa=X&oi=book_result&ct=result&resnum=3&ved=0CCYQ6AEwAg#v=onepage&q&f=false
//
//  void qrdcmp(float **a, int n, float *c, float *d, int *sing)
//  Construct the QR decomposition of a[0..n-1][0..n-1]. The upper triangular matrix R and the orthogonal matrix Q are stored.
//  sing is set to true if a singularity is encountered during the decomposition, but the decomposition is still completed in this case
//  otherwise it is set to false
  int i,j,k;
  for(i=0;i<n;i++)  for(int j=0;j<n;j++)  R[i*n+j] = in_column_major[j*n+i];  // transpose and copy (column_major -> row_major)

  double *c = new double[n], *d = new double[n];
  double scale = 0.0f, sigma = 0.0f, sum = 0.0f, tau = 0.0f;
  int sing = 0;
  for (k=0;k<n-1;k++) {
    scale = 0.0f;
    for (i=k;i<n;i++) scale = FMAX(scale,fabs(R[i*n+k]));
    if (scale == 0.0f){   // Singular case.
      sing = 1;
      c[k] = d[k] = 0.0f;
    } else {              // Form Qk and Qk * A.
      for (i=k;i<n;i++) R[i*n+k] /= scale;
      for (sum=0.0,i=k;i<n;i++) sum += SQR(R[i*n+k]);
      sigma = SIGN(sqrt(sum),(float)(R[k*n+k]));
      R[k*n+k] += sigma;
      c[k] = sigma*R[k*n+k];
      d[k] = -scale*sigma;
      for (j=k+1;j<n;j++) {
        for (sum = 0.0f,i=k;i<n;i++) sum += R[i*n+k]*R[i*n+j];
        tau = sum/c[k];
        for (i=k;i<n;i++) R[i*n+j] -= tau*R[i*n+k];
      }
    }
  }
  d[n-1] = R[(n-1)*n+(n-1)];
  if (d[n-1] == 0.0)  sing = 1;

  // form Q explicitly in column major
  for(i=0;i<n;i++){     
    for(j=0;j<n;j++)  Q[i*n+j] = 0.0;
    Q[i*n+i] = 1.0;
  }
  for(k=0;k<n-1;k++){
    if(c[k] != 0.0f){
      for(j=0;j<n;j++){
        sum = 0.0;
        for(i=k;i<n;i++)  sum += R[i*n+k]*Q[i*n+j];
        sum /= c[k];
        for(i=k;i<n;i++)  Q[i*n+j] -= sum*R[i*n+k];
      }
    }
  }

  // form R explicitly in column major
  for(i=0;i<n;i++) for(j=0;j<n;j++)
    // copy upper triangular matrix R except for the diagonal elements (convert row major -> column major at the same time), diagonal elements come from d[]
    R[j*n+i] = (j<i) ? 0.0 : ((j==i) ? d[i] : R[i*n+j]);  

  delete[] c, d;

  if(sing==1){
    print_and_log("RegTools::QRDecomposition_square(), singularity is encountered\n");
    return false;
  } else 
    return true;
}

bool RegTools::CrossProduct3(const double *in1, const double *in2, double *out)
{
  // compute cross product of 3x1 vectors
  out[0] = in1[1]*in2[2]-in1[2]*in2[1];
  out[1] = in1[2]*in2[0]-in1[0]*in2[2];
  out[2] = in1[0]*in2[1]-in1[1]*in2[0];
  return true;
}

double RegTools::DotProduct3(const double *in1, const double *in2)
{
  return in1[0]*in2[0]+in1[1]*in2[1]+in1[2]*in2[2];
}

double RegTools::Determinant3x3d(const double *in)
{
  // this works both for row-major and column-major (since matrix transpose doesn't change its determinant)
  return
    in[0] * in[4]* in[8] + in[1] * in[5]* in[6] + in[2] * in[3]* in[7] - in[0] * in[5]* in[7] - in[1] * in[3]* in[8] - in[2] * in[4]* in[6];
}

double RegTools::Determinant4x4d(const double *in)
{
  // this works both for row-major and column-major (since matrix transpose doesn't change its determinant)
  return
    in[12] * in[9]* in[6]  * in[3]-in[8]  * in[13] * in[6]  * in[3]-in[12]  * in[5] * in[10] * in[3]+in[4]  * in[13] * in[10] * in[3]  +
    in[8] * in[5] * in[14] * in[3]-in[4]  * in[9]  * in[14] * in[3]-in[12]  * in[9] * in[2]  * in[7]+in[8]  * in[13] * in[2]  * in[7]  +
    in[12] * in[1]* in[10] * in[7]-in[0]  * in[13] * in[10] * in[7]-in[8]   * in[1] * in[14] * in[7]+in[0]  * in[9]  * in[14] * in[7]  +
    in[12] * in[5]* in[2]  * in[11]-in[4] * in[13] * in[2]  * in[11]-in[12] * in[1] * in[6]  * in[11]+in[0] * in[13] * in[6]  * in[11] +
    in[4] * in[1] * in[14] * in[11]-in[0] * in[5]  * in[14] * in[11]-in[8]  * in[5] * in[2]  * in[15]+in[4] * in[9]  * in[2]  * in[15] +
    in[8] * in[1] * in[6]  * in[15]-in[0] * in[9]  * in[6]  * in[15]-in[4]  * in[1] * in[10] * in[15]+in[0] * in[5]  * in[10] * in[15];
}

void RegTools::MultMatrixd_col(const double *mat, const double *mul, double *out)
{
  // (column-major) * (colum-major) -> (column_major)
  out[0]  = mat[0]*mul[0] +mat[4]*mul[1] +mat[8] *mul[2] +mat[12]*mul[3];
  out[1]  = mat[1]*mul[0] +mat[5]*mul[1] +mat[9] *mul[2] +mat[13]*mul[3];
  out[2]  = mat[2]*mul[0] +mat[6]*mul[1] +mat[10]*mul[2] +mat[14]*mul[3];
  out[3]  = mat[3]*mul[0] +mat[7]*mul[1] +mat[11]*mul[2] +mat[15]*mul[3];
  out[4]  = mat[0]*mul[4] +mat[4]*mul[5] +mat[8] *mul[6] +mat[12]*mul[7];
  out[5]  = mat[1]*mul[4] +mat[5]*mul[5] +mat[9] *mul[6] +mat[13]*mul[7];
  out[6]  = mat[2]*mul[4] +mat[6]*mul[5] +mat[10]*mul[6] +mat[14]*mul[7];
  out[7]  = mat[3]*mul[4] +mat[7]*mul[5] +mat[11]*mul[6] +mat[15]*mul[7];
  out[8]  = mat[0]*mul[8] +mat[4]*mul[9] +mat[8] *mul[10]+mat[12]*mul[11];
  out[9]  = mat[1]*mul[8] +mat[5]*mul[9] +mat[9] *mul[10]+mat[13]*mul[11];
  out[10] = mat[2]*mul[8] +mat[6]*mul[9] +mat[10]*mul[10]+mat[14]*mul[11];
  out[11] = mat[3]*mul[8] +mat[7]*mul[9] +mat[11]*mul[10]+mat[15]*mul[11];
  out[12] = mat[0]*mul[12]+mat[4]*mul[13]+mat[8] *mul[14]+mat[12]*mul[15];
  out[13] = mat[1]*mul[12]+mat[5]*mul[13]+mat[9] *mul[14]+mat[13]*mul[15];
  out[14] = mat[2]*mul[12]+mat[6]*mul[13]+mat[10]*mul[14]+mat[14]*mul[15];
  out[15] = mat[3]*mul[12]+mat[7]*mul[13]+mat[11]*mul[14]+mat[15]*mul[15];
}

void RegTools::TransposeMatrixd(double *mat)
{
  // transpose 4x4 matrix
  // [1]<->[4], [2]<->[8], [3]<->[12], [6]<->[9], [7]<->[13], [11]<->[14]
  double temp;
  temp = mat[1];  mat[1] = mat[4];   mat[4] = temp;
  temp = mat[2];  mat[2] = mat[8];   mat[8] = temp;
  temp = mat[3];  mat[3] = mat[12];  mat[12] = temp;
  temp = mat[6];  mat[6] = mat[9];   mat[9] = temp;
  temp = mat[7];  mat[7] = mat[13];  mat[13] = temp;
  temp = mat[11]; mat[11] = mat[14]; mat[14] = temp;
}

void RegTools::LoadIdentity_4x4d(double *mat)
{
  mat[0] = 1.0; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
  mat[1] = 0.0; mat[5] = 1.0; mat[9] = 0.0; mat[13] = 0.0;
  mat[2] = 0.0; mat[6] = 0.0; mat[10]= 1.0; mat[14] = 0.0;
  mat[3] = 0.0; mat[7] = 0.0; mat[11]= 0.0; mat[15] = 1.0;
}

void RegTools::RotateX_col(double *mat, double angle_rad)
{
  double m[16], out[16];
  m[0] = 1.0; m[4] = 0.0;             m[8] = 0.0;             m[12] = 0.0;
  m[1] = 0.0; m[5] = cos(angle_rad);  m[9] = -sin(angle_rad); m[13] = 0.0;
  m[2] = 0.0; m[6] = sin(angle_rad);  m[10]= cos(angle_rad);  m[14] = 0.0;
  m[3] = 0.0; m[7] = 0.0;             m[11]= 0.0;             m[15] = 1.0;
  MultMatrixd_col(mat, m, out);
  memcpy(mat, out, sizeof(double)*16);
}

void RegTools::RotateY_col(double *mat, double angle_rad)
{
  double m[16], out[16];
  m[0] = cos(angle_rad);  m[4] = 0.0;   m[8] = sin(angle_rad);  m[12] = 0.0;
  m[1] = 0.0;             m[5] = 1.0;   m[9] = 0.0;             m[13] = 0.0;
  m[2] = -sin(angle_rad); m[6] = 0.0;   m[10]= cos(angle_rad);  m[14] = 0.0;
  m[3] = 0.0;             m[7] = 0.0;   m[11]= 0.0;             m[15] = 1.0;
  MultMatrixd_col(mat, m, out);
  memcpy(mat, out, sizeof(double)*16);
}

void RegTools::RotateZ_col(double *mat, double angle_rad)
{
  double m[16], out[16];
  m[0] = cos(angle_rad);  m[4] = -sin(angle_rad); m[8] = 0.0;  m[12] = 0.0;
  m[1] = sin(angle_rad);  m[5] = cos(angle_rad);  m[9] = 0.0;  m[13] = 0.0;
  m[2] = 0.0;             m[6] = 0.0;             m[10]= 1.0;  m[14] = 0.0;
  m[3] = 0.0;             m[7] = 0.0;             m[11]= 0.0;  m[15] = 1.0;
  MultMatrixd_col(mat, m, out);
  memcpy(mat, out, sizeof(double)*16);
}

void RegTools::Translate_col(double *mat, double x, double y, double z)
{
  double m[16], out[16];
  m[0] = 1.0;  m[4] = 0.0;  m[8] = 0.0;  m[12] = x;
  m[1] = 0.0;  m[5] = 1.0;  m[9] = 0.0;  m[13] = y;
  m[2] = 0.0;  m[6] = 0.0;  m[10]= 1.0;  m[14] = z;
  m[3] = 0.0;  m[7] = 0.0;  m[11]= 0.0;  m[15] = 1.0;
  MultMatrixd_col(mat, m, out);
  memcpy(mat, out, sizeof(double)*16);
}

void RegTools::ApplyTransformation_col(double *mat_4x4, double *vec)
{
  // mat (4x4 matrix): column major
  // vec (3x1 vector)
  double out[3];
  out[0] = mat_4x4[0]*vec[0]+mat_4x4[4]*vec[1]+mat_4x4[8]*vec[2]+mat_4x4[12];
  out[1] = mat_4x4[1]*vec[0]+mat_4x4[5]*vec[1]+mat_4x4[9]*vec[2]+mat_4x4[13];
  out[2] = mat_4x4[2]*vec[0]+mat_4x4[6]*vec[1]+mat_4x4[10]*vec[2]+mat_4x4[14];
  memcpy(vec, out, sizeof(double)*3);
}

float3 RegTools::MultMatrixf_3x3_col(const double *R_3x3_col, const double x, const double y, const double z)
{
  return make_float3( static_cast<float>(R_3x3_col[0]*x  + R_3x3_col[3]*y  + R_3x3_col[6]*z),
                      static_cast<float>(R_3x3_col[1]*x  + R_3x3_col[4]*y  + R_3x3_col[7]*z),
                      static_cast<float>(R_3x3_col[2]*x  + R_3x3_col[5]*y  + R_3x3_col[8]*z) );
}

void RegTools::MultMatrix_3x3_col(const double *R_3x3_col, const double in[3], double *out)
{
  out[0] = R_3x3_col[0]*in[0] + R_3x3_col[3]*in[1] + R_3x3_col[6]*in[2];
  out[1] = R_3x3_col[1]*in[0] + R_3x3_col[4]*in[1] + R_3x3_col[7]*in[2];
  out[2] = R_3x3_col[2]*in[0] + R_3x3_col[5]*in[1] + R_3x3_col[8]*in[2];
}

double RegTools::ComputeFocalLength(const double *pm_3x4_row, double *pixel_size)
{
  double pm[12], r1[3], r2[3];
  double scale = sqrt( pm_3x4_row[8]*pm_3x4_row[8]+pm_3x4_row[9]*pm_3x4_row[9]+pm_3x4_row[10]*pm_3x4_row[10] );
  for(int i=0;i<12;i++) pm[i] = pm_3x4_row[i]/scale;
  CrossProduct3(&pm[4], &pm[8], r1);
  Normalize3(r1);
  CrossProduct3(&pm[8], r1, r2);
  double fx_mm = (pm[0]*r1[0] + pm[1]*r1[1] + pm[2]*r1[2])*pixel_size[0];
  double fy_mm = (pm[4]*r2[0] + pm[5]*r2[1] + pm[6]*r2[2])*pixel_size[1];
  return (abs(fx_mm)+abs(fy_mm))/2;
}

void RegTools::ComputeSourcePosition(const double *pm_3x4_row, double source_position[3])
{
  double in_col[9], inv_in_col[9]; // intrinsic parameter matrix (3x3) in column-major
  in_col[0] = pm_3x4_row[0]; in_col[3] = pm_3x4_row[1]; in_col[6] = pm_3x4_row[2];
  in_col[1] = pm_3x4_row[4]; in_col[4] = pm_3x4_row[5]; in_col[7] = pm_3x4_row[6];
  in_col[2] = pm_3x4_row[8]; in_col[5] = pm_3x4_row[9]; in_col[8] = pm_3x4_row[10];
  Inverse3x3d(in_col, inv_in_col);
  double p[3] = {pm_3x4_row[3], pm_3x4_row[7], pm_3x4_row[11]};
  MultMatrix_3x3_col(inv_in_col, p, source_position);
  source_position[0]*=-1; source_position[1]*=-1; source_position[2]*=-1;
}

void RegTools::Normalize3(double *in_out)
{
  double norm = sqrt(in_out[0]*in_out[0] + in_out[1]*in_out[1] + in_out[2]*in_out[2]);
  in_out[0]/=norm; in_out[1]/=norm; in_out[2]/=norm;
}

int RegTools::convertTransRotTo4x4(double *in_1x6vec, double *out_4x4_col)
{
  LoadIdentity_4x4d(out_4x4_col);
  Translate_col(out_4x4_col, in_1x6vec[0], in_1x6vec[1], in_1x6vec[2]);
  RotateZ_col(out_4x4_col, in_1x6vec[5]);
  RotateY_col(out_4x4_col, in_1x6vec[4]);
  RotateX_col(out_4x4_col, in_1x6vec[3]);
  return true;
}

int RegTools::convertRotTransTo4x4(double *in_1x6vec, double *out_4x4_col)
{
  LoadIdentity_4x4d(out_4x4_col);
  RotateZ_col(out_4x4_col, in_1x6vec[5]);
  RotateY_col(out_4x4_col, in_1x6vec[4]);
  RotateX_col(out_4x4_col, in_1x6vec[3]);
  Translate_col(out_4x4_col, in_1x6vec[0], in_1x6vec[1], in_1x6vec[2]);
  return true;
}

int RegTools::convert4x4ToRotTrans(double *in_4x4_col, double *out_1x6vec)
{
  // 4x4 matrix -> Rotation-Translation representation
  double t[16];
  const double epsilon = 1e-10;
  Inverse4x4d(in_4x4_col, t);
  out_1x6vec[0] = -t[12]; out_1x6vec[1] = -t[13]; out_1x6vec[2] = -t[14];

  double psi = atan2(t[8], sqrt(t[0]*t[0] + t[4]*t[4])), theta, fai;
  if(abs(psi - M_PI/2) < epsilon){
    theta = 0; fai = -atan2(t[1], t[2]);
  } else if (abs(psi + M_PI/2) < epsilon) {
    theta = 0; fai = atan2(t[4], t[5]);
  } else {
    theta = -atan2(t[9]/cos(psi), t[10]/cos(psi));
    fai = -atan2(t[4]/cos(psi), t[0]/cos(psi));
  }

  out_1x6vec[3] = -theta; out_1x6vec[4] = -psi; out_1x6vec[5] = -fai;
  return true;
}

int RegTools::convert4x4ToTransRot(double *in_4x4_col, double *out_1x6vec)
{
  out_1x6vec[0] = in_4x4_col[12]; out_1x6vec[1] = in_4x4_col[13]; out_1x6vec[2] = in_4x4_col[14];

  double pitch = atan2(-in_4x4_col[2], sqrt(in_4x4_col[0]*in_4x4_col[0] + in_4x4_col[1]*in_4x4_col[1])), yaw, roll;
  const double epsilon = 1e-10;

  if(abs(pitch - M_PI/2) < epsilon){
    yaw = 0;  roll = atan2(in_4x4_col[4], in_4x4_col[5]);
  } else if(abs(pitch + M_PI/2) < epsilon){
    yaw = 0;  roll = -atan2(in_4x4_col[4], in_4x4_col[5]);
  } else {
    yaw  = atan2(in_4x4_col[1]/cos(pitch), in_4x4_col[0]/cos(pitch));
    roll = atan2(in_4x4_col[6]/cos(pitch), in_4x4_col[10]/cos(pitch));
  }
  out_1x6vec[3] = roll; out_1x6vec[4] = pitch; out_1x6vec[5] = yaw;
  return true;
}

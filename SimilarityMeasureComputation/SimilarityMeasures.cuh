#ifndef _SIMILARITY_MEASURES_CUH_
#define _SIMILARITY_MEASURES_CUH_

#include <cuda.h>
#include <cublas_v2.h>
#include <vector_types.h>
#include <iostream>
#include <fstream>

typedef float value_type;

extern "C" void padKernel(float *d_Dst, float *d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX);
extern "C" void unPadDataClampToBorder(float *d_Dst, float *d_Src, int fftH, int fftW, int dataH, int dataW, int kernelY, int kernelX, int imageRepeat);
__global__ void unPadDataClampToBorder_kernel(float *d_Dst, float *d_Src, int fftH, int fftW, int dataH, int dataW, int kernelY, int kernelX, int imageRepeat);
extern "C" void padDataClampToBorder(float *d_Dst, float *d_Src, int fftH, int fftW, int dataH, int dataW, int kernelW, int kernelH, int kernelY, int kernelX, int imageRepeat);
__global__ void padDataClampToBorder_kernel(float *d_Dst, float *d_Src, int fftH, int fftW, int dataH, int dataW, int kernelH, int kernelW, int kernelY, int kernelX, int imageRepeat);
extern "C" void fillEdges(float *d_Dst, float *d_Ref, int fftH, int fftW, int dataH, int dataW, int kernelW, int kernelH, int kernelY, int kernelX, int imageRepeat, float threshold);
__global__ void fillEdges_kernel(float *d_Dst, float *d_Ref, int fftH, int fftW, int dataH, int dataW, int kernelH, int kernelW, int kernelY, int kernelX, int imageRepeat, float threshold);
extern "C" void modulateAndNormalize(fComplex *d_Dst, fComplex *d_Src, int fftH, int fftW, int imageRepeat, int padding);
extern "C" int snapTransformSize(int dataSize);
extern "C" void computeKernelSpectrumGPU(float *h_Kernel, int dataH, int dataW, int kernelH, int kernelW, int kernelY, int kernelX
                                         , fComplex *d_KernelSpectrum, cufftHandle fftPlanFwd);
extern "C" value_type gauss(value_type x, value_type sigma);
extern "C" value_type dgauss(value_type x, value_type sigma);
extern "C" unsigned int getKernelSize(float sigma);

extern "C" bool zeromean_stddivide_Images(float *d_images, float *d_temp, int image_size, int num_image_sets, cublasHandle_t cublasHandle, float *d_mean, float *d_OneVector, float h_NormalizationFactor, float *d_mask_weight);
extern "C" bool computeNormalizedCrossCorrelation_Pixels(float *d_images1_zeromean_multi, float *d_images2_zeromean_single, int image_size, int num_image_sets, cublasHandle_t cublasHandle);
extern "C" bool computeNormalizedCrossCorrelation_Sum(float *d_images1_multi, float *d_images2_multi, float *d_output_multi, double *NCC, int image_size, int num_image_sets
                                                      , cublasHandle_t cublasHandle, float *d_temp_NCC, float *h_temp_NCC, float *d_OneVector);

extern "C" bool computeJointPDF(float *d_data1, float *d_data2, int length, int *d_joint_hist, float *d_joint_pdf);
extern "C" float computeJointEntropy(float *d_data1, float *d_data2, int length, int *d_joint_hist, float *d_joint_pdf);
extern "C" bool computePDF(float *d_data, int length, int *d_hist, float *d_pdf);
extern "C" float computeEntropy(float *d_data, int length, int *d_hist, float *d_pdf);
__global__ void computeJointHistogramKernel(float *d_data1, float *d_data2, int length, int *d_joint_hist, int num_bins);
__global__ void normalizeHistogramKernel(int *d_hist, float *d_pdf, int size, float denominator);
__global__ void computeHistogramKernel(float *d_data, int length, int *d_hist, int num_bins);

extern "C" int getGaussianGradientKernel(const value_type sigma, const int kernel_size, value_type *h_kernel);
extern "C" int getGaussianKernel(const value_type sigma, const int kernel_size, value_type *h_kernel);
extern "C" bool computeGaussianGradientKernelSpectrum(float sigma, int x_dim, int y_dim, int z_dim, fComplex *x_kernel_spectrum, fComplex *y_kernel_spectrum
                                                      , cufftHandle fftPlanFwd);
extern "C" bool computeGaussianKernelSpectrum(float sigma, int x_dim, int y_dim, int z_dim, fComplex *kernel_spectrum, cufftHandle fftPlanFwd);
extern "C" bool computeGaussianGradientGPUMulti(float *d_input_images, int *dim, int num_image_sets, float *d_output_imagesX, float *d_output_imagesY, float sigma
                                           , fComplex *d_x_kernel_spectrum, fComplex *d_y_kernel_spectrum, float *temp_padded, fComplex *temp_spectrum
                                           , cufftHandle fftPlanManyFwd, cufftHandle fftPlanManyInv, float GI_threshold);
extern "C" bool computeGaussianGPUMulti(float *d_input_images, int *dim, int num_image_sets, float *d_output_images1, float *d_output_images2, float sigma
                                           , fComplex *d_x_kernel_spectrum,float *temp_padded, fComplex *temp_spectrum
                                           , cufftHandle fftPlanManyFwd, cufftHandle fftPlanManyInv);
extern "C" bool computeCovarianceGPUMulti(float *d_input_images1, float *d_mu1, float *d_input_images2, float *d_mu2, int *dim, int num_image_sets, float *d_output_images, float sigma
                                           , fComplex *d_kernel_spectrum, float *temp_padded, fComplex *temp_spectrum
                                           , cufftHandle fftPlanManyFwd, cufftHandle fftPlanManyInv );
extern "C" void computeGradientInformation(float *d_grad1X, float *d_grad1Y, float *d_grad2X, float *d_grad2Y, float *d_output, int image_size
                                           , float *d_mask_weight, bool isSingleModality, double *gradient_information, int num_image_sets
                                           , cublasHandle_t cublasHandle, float *d_temp_SM, float *h_temp_SM, float *d_OneVector, float h_NormalizationFactor, int exclusive_norm);
extern "C" void computeGradientInformation_StdNorm(float *d_grad1X, float *d_grad1Y, float *d_grad2X, float *d_grad2Y, float *d_output, int image_size
                                           , float *d_mask_weight, bool isSingleModality, double *gradient_information, int num_image_sets
                                           , cublasHandle_t cublasHandle, float *d_temp_SM, float *h_temp_SM, float *d_OneVector, float h_NormalizationFactor);
extern "C" void computeSSIM(float *d_grad1X, float *d_grad1Y, float *d_grad2X, float *d_grad2Y, float *d_output, int image_size
                            , float *d_mask_weight, double *SSIM, int num_image_sets
                            , cublasHandle_t cublasHandle, float *d_temp_SM, float *h_temp_SM, float *d_OneVector, float h_DynamicRange);
extern "C" bool computeMeanSquaredError(float *d_images1_multi, float *d_images2_single, float *d_OneVector, double *MSE, int image_size, int num_image_sets
                                        , float *d_temp_MSE, float *h_temp_MSE, cublasHandle_t cublasHandle);
__global__ void computeGradientInformation_kernel(float *d_grad1X, float *d_grad1Y, float *d_grad2X, float *d_grad2Y, float *d_output, int image_size
                                                  , bool isSingleModality, float *d_mask_weight, int num_image_sets, int exclusive_norm);
__global__ void computeSSIM_kernel(float *d_grad1X, float *d_grad1Y, float *d_grad2X, float *d_grad2Y, float *d_output, int image_size
                                   , float *d_mask_weight, int num_image_sets, float C1, float C2);
__global__ void computeCovariance_kernel(float *d_img1, float *d_mu1, float *d_img2, float *d_mu2, float *d_output, int image_size, int num_image_sets);
__global__ void computeImageSum_kernel(float *d_image, int image_size, int num_image_sets, float *d_weight, float *d_sum);
extern "C" void normalizeImages(float *d_images, int size, float norm_max, float norm_min);
extern "C" void maskImages( float *d_images, float *d_mask_weight, int size, int num_image_sets, cublasHandle_t cublasHandle );
extern "C" int countZeroPixels(float *d_image, int image_size);

__global__ void padKernel_kernel(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
);


inline __device__ void mulAndScale(fComplex& a, const fComplex& b, const float& c){
    fComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    a = t;
}

__global__ void modulateAndNormalize_kernel(
    fComplex *d_Dst,
    fComplex *d_Src,
    int dataSize,
    int imageRepeat,
    float c
);

#endif // _SIMILARITY_MEASIRES_CUH_

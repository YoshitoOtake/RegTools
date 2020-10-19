#include <driver_types.h>
#include "RegTools.h"
#include <cufft.h>
#define _USE_MATH_DEFINES
#include "math.h"   // for M_PI_2, cos, sin
#include <float.h>

// thrust functions
// important note: thrust is not compatible with multi-thread environment, thus not compativle with multi-GPU in this library
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/count.h>

#include <assert.h>

#define USE_TEXTURE 0  // When we use texture for small image (less than 256x256 maybe), padDataClampToBorder() (SET_FLOAT_BASE) crashes, for some reason

#if(USE_TEXTURE)
    texture<float, 1, cudaReadModeElementType> texFloat;
    #define   LOAD_FLOAT(i) tex1Dfetch(texFloat, i)
    #define  SET_FLOAT_BASE cutilSafeCall( cudaBindTexture(0, texFloat, d_Src) )
#else
    #define  LOAD_FLOAT(i) d_Src[i]
    #define SET_FLOAT_BASE
#endif

extern FILE *m_LogFile;

////////////////////////////////////////////////////////////////////////////////
// 2D R2C / C2R post/preprocessing kernels
////////////////////////////////////////////////////////////////////////////////
#if(USE_TEXTURE)
    texture<fComplex, 1, cudaReadModeElementType> texComplexA;
    texture<fComplex, 1, cudaReadModeElementType> texComplexB;
    #define    LOAD_FCOMPLEX(i) tex1Dfetch(texComplexA, i)
    #define  LOAD_FCOMPLEX_A(i) tex1Dfetch(texComplexA, i)
    #define  LOAD_FCOMPLEX_B(i) tex1Dfetch(texComplexB, i)

    #define   SET_FCOMPLEX_BASE cutilSafeCall( cudaBindTexture(0, texComplexA,  d_Src) )
    #define SET_FCOMPLEX_BASE_A cutilSafeCall( cudaBindTexture(0, texComplexA, d_SrcA) )
    #define SET_FCOMPLEX_BASE_B cutilSafeCall( cudaBindTexture(0, texComplexB, d_SrcB) )
#else
    #define    LOAD_FCOMPLEX(i)  d_Src[i]
    #define  LOAD_FCOMPLEX_A(i) d_SrcA[i]
    #define  LOAD_FCOMPLEX_B(i) d_SrcB[i]

    #define   SET_FCOMPLEX_BASE
    #define SET_FCOMPLEX_BASE_A
    #define SET_FCOMPLEX_BASE_B
#endif

extern "C" int snapTransformSize(int dataSize){
  int hiBit;
  unsigned int lowPOT, hiPOT;

  dataSize = iAlignUp(dataSize, 16);

  for(hiBit = 31; hiBit >= 0; hiBit--)
    if(dataSize & (1U << hiBit)) break;

  lowPOT = 1U << hiBit;
  if(lowPOT == dataSize)
    return dataSize;

  hiPOT = 1U << (hiBit + 1);
  if(hiPOT <= 1024)
    return hiPOT;
  else 
    return iAlignUp(dataSize, 512);
}

extern "C" void computeKernelSpectrumGPU(float *h_Kernel, int dataH, int dataW, int kernelH, int kernelW, int kernelY, int kernelX
                                            , fComplex *d_KernelSpectrum, cufftHandle fftPlanFwd)
{
  // this function is copied from CUDA SDK (convolutionFFT2D) and modified by Yoshito Otake
  // Note: make sure that there is no 'bad' number in d_Data (e.g., Inf, Nan, etc.)
  //       if there is any 'bad' number, it breaks the entire result.
  const int fftH = snapTransformSize(dataH + kernelH - 1);
  const int fftW = snapTransformSize(dataW + kernelW - 1);

  float *d_Kernel, *d_PaddedKernel;
  cutilSafeCall( cudaMalloc(&(d_Kernel), kernelH*kernelW*sizeof(float)) );
  cutilSafeCall( cudaMalloc(&(d_PaddedKernel), fftH*fftW*sizeof(float)) );
//  fComplex *d_KernelSpectrum = temp_spectrum+ (fftH * (fftW / 2 + 1));
  //print_and_log("kernelH: %d, kernelW: %d, fftH: %d, fftW: %d\n", kernelH, kernelW, fftH, fftW);
  cutilSafeCall( cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice) );
  cutilSafeCall( cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)) );

  #if defined RegTools_VERBOSE_MESSAGE
    print_and_log("h_Kernel:\n");
    for(int i=0;i<kernelW;i++){ for(int j=0;j<kernelH;j++) print_and_log("%f ", h_Kernel[i*kernelH+j]); print_and_log("\n"); }
  #endif

  padKernel( d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, kernelY, kernelX );

  #if defined RegTools_VERBOSE_MESSAGE
    print_and_log("h_PaddedKernel:\n");
    float *h_PaddedKernel = new float[fftH*fftW];
    cudaMemcpy(h_PaddedKernel, d_PaddedKernel, fftH*fftW*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0;i<fftW;i++){ for(int j=0;j<fftH;j++) print_and_log("%f ", h_PaddedKernel[i*fftH+j]); print_and_log("\n"); }
    delete[] h_PaddedKernel;
  #endif

  int error;
  if( (error=cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum)) != CUFFT_SUCCESS) {
    print_and_log("ERROR at cufftExecR2C in computeKernelSpectrumGPU (6 means CUFFT_EXEC_FAILED), error code: %d\n", error);
  }

  cutilSafeCall( cudaFree( d_Kernel ) );
  cutilSafeCall( cudaFree( d_PaddedKernel ) );
}

extern "C" unsigned int getKernelSize(float sigma)
{
  // compute kernel size based on sigma
  const float epsilon = 1e-2;
  return ceil(sigma*sqrt(-2*log(sqrt(2*M_PI)*sigma*epsilon)));
}

extern "C" bool computeGaussianGradientGPUMulti(float *d_input_images, int *dim, int num_image_sets, float *d_output_imagesX, float *d_output_imagesY, float sigma
                                           , fComplex *d_x_kernel_spectrum, fComplex *d_y_kernel_spectrum, float *temp_padded, fComplex *temp_spectrum
                                           , cufftHandle fftPlanManyFwd, cufftHandle fftPlanManyInv, float GI_threshold )
{
  const unsigned int kernelY = getKernelSize(sigma), kernelX = kernelY;
  const unsigned int kernelH = 2*kernelY+1, kernelW = kernelH;
  const int fftH = snapTransformSize(dim[1] + kernelH - 1), dataH = dim[1];
  const int fftW = snapTransformSize(dim[0] + kernelW - 1), dataW = dim[0];
  const int spectrum_size = fftH*(fftW/2+1), padded_size = fftH*fftW;

  const int y_offset = dim[2]*num_image_sets;
  // FFT/iFFT for all images simultaneously (fftPlanMany needs to be created for exactly the same number of images)
  padDataClampToBorder( temp_padded, d_input_images, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX, dim[2]*num_image_sets );

  // Note: the following FFT fails if input image contains NaN (even only one pixel!)
  cufftSafeCall( cufftExecR2C( fftPlanManyFwd, (cufftReal *)temp_padded, (cufftComplex *)temp_spectrum ) );
  cutilSafeCall( cudaMemcpy(temp_spectrum+spectrum_size*y_offset, temp_spectrum, spectrum_size*y_offset*sizeof(fComplex), cudaMemcpyDeviceToDevice) );

  // modulate all spectrums
  modulateAndNormalize(temp_spectrum, d_x_kernel_spectrum, fftH, fftW, dim[2]*num_image_sets, 1);
  modulateAndNormalize(temp_spectrum+spectrum_size*y_offset, d_y_kernel_spectrum, fftH, fftW, dim[2]*num_image_sets, 1);

//  print_and_log("computeGaussianGradientGPUMulti(), dim: %dx%dx%d, fft size: %dx%d, kernelXY: %dx%d, kernelWH: %dx%d, spectrum_size: %d, padded_size: %d, fftPlanManyInv: %d, GI_threshold: %f\n", 
//                  dim[0], dim[1], dim[2], fftW, fftH, kernelX, kernelY, kernelW, kernelH, spectrum_size, padded_size, fftPlanManyInv, GI_threshold);
  // inverse FFT
  cufftSafeCall( cufftExecC2R(fftPlanManyInv, (cufftComplex *)temp_spectrum, (cufftReal *)temp_padded) );

#if defined RegTools_GI_BACKGROUND_EDGE_CHECK
  // fill zero in (kernelW x kernelH) pixels of the gradient image around the pixel which has an intensity lower than threshold value in the original image
  fillEdges( temp_padded, d_input_images, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX, dim[2]*num_image_sets, GI_threshold );
  fillEdges( temp_padded+padded_size*y_offset, d_input_images, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX, dim[2]*num_image_sets, GI_threshold );
#endif

  // unPad all images
  unPadDataClampToBorder(d_output_imagesX, temp_padded, fftH, fftW, dataH, dataW, kernelY, kernelX, dim[2]*num_image_sets);
  unPadDataClampToBorder(d_output_imagesY, temp_padded+padded_size*y_offset, fftH, fftW, dataH, dataW, kernelY, kernelX, dim[2]*num_image_sets);
  return true;
}

struct sqrt_inverse_functor
{
  __host__ __device__ float operator()(const float &x) const { return 1/sqrt(x); }
};

struct square_functor
{
  __host__ __device__ float operator()(const float &x) const { return x*x; }
};

__global__ void apply_floating_mask_kernel(float *d_images, float *d_floating_mask, int size)
{
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= size) return;
	if(d_floating_mask[index]<1e-6) d_images[index] = 0;
}

__global__ void square_functor_kernel(float *d_input, float *d_output, int size)
{
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= size) return;
	d_output[index] = d_input[index] * d_input[index];
}

__global__ void sqrt_inverse_functor_kernel(float *d_input, float *d_output, int size)
{
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= size) return;
	d_output[index] = 1/sqrt(d_input[index]);
}

extern "C" bool zeromean_stddivide_Images(float *d_images, float *d_temp, int image_size, int num_image_sets, cublasHandle_t cublasHandle, float *d_mean_std, float *d_OneVector, float h_NormalizationFactor, float *d_mask_weight, float *d_floating_mask)
{
  float alpha, beta;

  // (OPTIONAL) apply mask to each image (column). Note that mask_weight is single image set (single column vector)
  //print_and_log("zeromean_stddivide_Images(), pass0, d_images: %d, d_temp: %d, image_size: %d, num_image_sets: %d, cublasHandle: %d\n", d_images, d_temp, image_size, num_image_sets, cublasHandle);
  if(d_mask_weight)  
     cublasSdgmm(cublasHandle, CUBLAS_SIDE_LEFT, image_size, num_image_sets, d_images, image_size, d_mask_weight, 1, d_images, image_size);

  // compute mean of each image set (dim[2] images are considered as one set)
  beta = 0.0;
  cublasSgemv(cublasHandle, CUBLAS_OP_T, image_size, num_image_sets, &h_NormalizationFactor, d_images, image_size, d_OneVector, 1, &beta, d_mean_std, 1);
  //print_and_log("zeromean_stddivide_Images(), pass1\n");

  // subtract mean from each image set (column-wise subtraction)
  alpha = -1.0f; beta = 1.0;
  cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, image_size, num_image_sets, 1, &alpha, d_OneVector, image_size, d_mean_std, 1, &beta, d_images, image_size);
  // is this correct? maybe below? (d_mean_std and d_OneVector should be opposite order)
  //cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, image_size, num_image_sets, 1, &alpha, d_mean_std, image_size, d_OneVector, 1, &beta, d_images, image_size);
  //print_and_log("zeromean_stddivide_Images(), pass2, image_size: %d, num_image_sets: %d\n", image_size, num_image_sets);

  const int b = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;
  // apply floating mask if needed (set zero for the pixels with d_floating_mask>0)
  if(d_floating_mask)
	apply_floating_mask_kernel << <iDivUp(image_size*num_image_sets, b), b >> > (d_images, d_floating_mask, image_size*num_image_sets);

  // compute std of each image set (column-wise)
  //thrust::transform(thrust::device_ptr<float>(d_images), thrust::device_ptr<float>(d_images)+image_size*num_image_sets, thrust::device_ptr<float>(d_temp), square_functor());
  square_functor_kernel << <iDivUp(image_size*num_image_sets, b), b >> > (d_images, d_temp, image_size*num_image_sets);
  //print_and_log("zeromean_stddivide_Images(), pass3\n");
  alpha = 1.0f; beta = 0.0;
  cublasSgemv(cublasHandle, CUBLAS_OP_T, image_size, num_image_sets, &alpha, d_temp, image_size, d_OneVector, 1, &beta, d_mean_std, 1);
  //print_and_log("zeromean_stddivide_Images(), pass4\n");
  //thrust::transform(thrust::device_ptr<float>(d_mean_std), thrust::device_ptr<float>(d_mean_std)+num_image_sets, thrust::device_ptr<float>(d_mean_std), sqrt_inverse_functor());
  sqrt_inverse_functor_kernel << <iDivUp(num_image_sets, b), b >> > (d_mean_std, d_mean_std, num_image_sets);

  // devide each image set by std (column-wise division)
  cublasSdgmm(cublasHandle, CUBLAS_SIDE_RIGHT, image_size, num_image_sets, d_images, image_size, d_mean_std, 1, d_images, image_size);

/*
  float *h_mean_std = new float[num_image_sets];
  cutilSafeCall( cudaMemcpy(h_mean_std, d_mean_std, num_image_sets*sizeof(float), cudaMemcpyDeviceToHost) );
  for(int i=0;i<num_image_sets;i++){ print_and_log("d_images[%d] mean: %f\n", i, h_mean_std[i]); }
  delete[] h_mean_std;
  */
  return true;
}

extern "C" bool computeNormalizedCrossCorrelation_Pixels(float *d_images1_zeromean_multi, float *d_images2_zeromean_single, int image_size, int num_image_sets, cublasHandle_t cublasHandle)
{
  // (numerator) compute normalized covariance (pixel-wise multiplication between images1 and images2)
  // overwrite onto images1
  cublasSdgmm(cublasHandle, CUBLAS_SIDE_LEFT, image_size, num_image_sets, d_images1_zeromean_multi, image_size, d_images2_zeromean_single, 1, d_images1_zeromean_multi, image_size);
  return true;
}

extern "C" bool computeMeanSquaredError(float *d_images1_multi, float *d_images2_single, float *d_OneVector, double *MSE, int image_size, int num_image_sets, float *d_temp_MSE, float *h_temp_MSE, cublasHandle_t cublasHandle)
{
  // subtract images2_single from each image set in d_images1_multi (column-wise subtraction)
  float alpha = -1.0f, beta = 1.0f;
//  cudaMemcpy(d_images1_multi, d_images2_single, image_size*sizeof(float), cudaMemcpyDeviceToDevice);
  cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, image_size, num_image_sets, 1, &alpha, d_images2_single, image_size, d_OneVector, 1, &beta, d_images1_multi, image_size);

  // square d_images1_multi
  //thrust::transform(thrust::device_ptr<float>(d_images1_multi), thrust::device_ptr<float>(d_images1_multi)+image_size*num_image_sets, thrust::device_ptr<float>(d_images1_multi), square_functor());
  const int b = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;
  square_functor_kernel << <iDivUp(image_size*num_image_sets, b), b >> > (d_images1_multi, d_images1_multi, image_size*num_image_sets);

  // average over each image
  alpha = 1.0f/(float)image_size; beta = 0.0;
  cublasSgemv(cublasHandle, CUBLAS_OP_T, image_size, num_image_sets, &alpha, d_images1_multi, image_size, d_OneVector, 1, &beta, d_temp_MSE, 1);

  // copy the results back to host
  cutilSafeCall( cudaMemcpy(h_temp_MSE, d_temp_MSE, num_image_sets*sizeof(float), cudaMemcpyDeviceToHost) );
  for(int i=0;i<num_image_sets;i++) MSE[i] += (double)(h_temp_MSE[i]); // float -> double conversion (result is accumulated. make sure zeroing the initial array)

  return true;
}

struct absolute_value_functor
{
  __host__ __device__
    float operator()(const float &x) const { return fabs(x); }
};

extern "C" bool computeNormalizedCrossCorrelation_Sum(float *d_images1_multi, float *d_images2_multi, float *d_output_multi, double *NCC, int image_size, int num_image_sets
                                                      , cublasHandle_t cublasHandle, float *d_temp_NCC, float *h_temp_NCC, float *d_OneVector)
{
  if(d_images2_multi){
    float alpha = 0.5f;
    // pixel-wise mean (for Gradient Correlation)
    cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, image_size, num_image_sets, &alpha, d_images1_multi, image_size, &alpha, d_images2_multi, image_size, d_output_multi, image_size);
    
    // compute absolute value
//    thrust::transform( thrust::device_ptr<float>(d_output_multi), thrust::device_ptr<float>(d_output_multi)+image_size*num_image_sets, 
//                       thrust::device_ptr<float>(d_output_multi), absolute_value_functor() );
  } else if(d_images1_multi){
    // just memory copy (for NCC)
    cutilSafeCall( cudaMemcpy(d_output_multi, d_images1_multi, num_image_sets*image_size*sizeof(float), cudaMemcpyDeviceToDevice) );
  }

  // summation (column-wise)
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemv(cublasHandle, CUBLAS_OP_T, image_size, num_image_sets, &alpha, d_output_multi, image_size, d_OneVector, 1, &beta, d_temp_NCC, 1);

  // copy the results back to host
//  print_and_log("computeNormalizedCrossCorrelation_Sum(), d_temp_NCC: %d\n", d_temp_NCC);
//  for (int i = 0; i < num_image_sets; i++) print_and_log("%f, ", h_temp_NCC[i]);
//  print_and_log("\n");
  //cutilSafeCall(cudaMemcpy(d_temp_NCC, h_temp_NCC, num_image_sets * sizeof(float), cudaMemcpyHostToDevice));
  cutilSafeCall( cudaMemcpy(h_temp_NCC, d_temp_NCC, num_image_sets*sizeof(float), cudaMemcpyDeviceToHost) );
  for(int i=0;i<num_image_sets;i++) NCC[i] += (double)(h_temp_NCC[i]); // float -> double conversion (result is accumulated. make sure zeroing the initial array)
//  for(int i=0;i<num_image_sets;i++){ print_and_log("NCC[%d]: %f\n", i, NCC[i]); }
  return true;
}

struct subtract_square_functor
{
  __host__ __device__
  float operator()(const float &x, const float &y) const {
    return (x-y)*(x-y);
  }
};

__global__ void subtract_square_functor_kernel(float *d_input1, float *d_input2, float *d_output, int size)
{
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= size) return;
	d_output[index] = (d_input1[index]-d_input2[index]) * (d_input1[index] - d_input2[index]);
}

extern "C" bool computeGaussianGPUMulti(float *d_input_images, int *dim, int num_image_sets, float *d_output_mu, float *d_output_sigma_sq, float sigma
                                           , fComplex *d_kernel_spectrum, float *temp_padded, fComplex *temp_spectrum
                                           , cufftHandle fftPlanManyFwd, cufftHandle fftPlanManyInv )
{
  const unsigned int kernelY = getKernelSize(sigma), kernelX = kernelY;
  const unsigned int kernelH = 2*kernelY+1, kernelW = kernelH;
  const int fftH = snapTransformSize(dim[1] + kernelH - 1), dataH = dim[1];
  const int fftW = snapTransformSize(dim[0] + kernelW - 1), dataW = dim[0];
  const int spectrum_size = fftH*(fftW/2+1), padded_size = fftH*fftW;

  const int total_size = dim[0]*dim[1]*dim[2]*num_image_sets;

  // compute mean images
  // FFT/iFFT for all images simultaneously (fftPlanMany needs to be created for exactly the same number of images)
  // Note: the following FFT fails if input image contains NaN (even only one pixel!)
  // compute mu images
  padDataClampToBorder( temp_padded, d_input_images, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX, dim[2]*num_image_sets );
  cufftSafeCall( cufftExecR2C( fftPlanManyFwd, (cufftReal *)temp_padded, (cufftComplex *)temp_spectrum ) );
  modulateAndNormalize(temp_spectrum, d_kernel_spectrum, fftH, fftW, dim[2]*num_image_sets, 1);
  cufftSafeCall( cufftExecC2R(fftPlanManyInv, (cufftComplex *)temp_spectrum, (cufftReal *)temp_padded) );
  unPadDataClampToBorder(d_output_mu, temp_padded, fftH, fftW, dataH, dataW, kernelY, kernelX, dim[2]*num_image_sets);

  // compute sigma images (gaussian filter after mean subtraction and square.
//  thrust::transform( thrust::device_ptr<float>(d_input_images), thrust::device_ptr<float>(d_input_images)+total_size,
//                     thrust::device_ptr<float>(d_output_mu), thrust::device_ptr<float>(d_output_sigma_sq), subtract_square_functor() );
  const int b = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;
  subtract_square_functor_kernel << <iDivUp(total_size, b), b >> > (d_input_images, d_output_mu, d_output_sigma_sq, total_size);
  padDataClampToBorder( temp_padded, d_output_sigma_sq, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX, dim[2]*num_image_sets );
  cufftSafeCall( cufftExecR2C( fftPlanManyFwd, (cufftReal *)temp_padded, (cufftComplex *)temp_spectrum ) ); // fft
  modulateAndNormalize(temp_spectrum, d_kernel_spectrum, fftH, fftW, dim[2]*num_image_sets, 1);
  cufftSafeCall( cufftExecC2R(fftPlanManyInv, (cufftComplex *)temp_spectrum, (cufftReal *)temp_padded) );   // inverse fft
  unPadDataClampToBorder(d_output_sigma_sq, temp_padded, fftH, fftW, dataH, dataW, kernelY, kernelX, dim[2]*num_image_sets);

//  print_and_log("computeGaussianGPUMulti(), dim: %dx%dx%d, fft size: %dx%d, kernelXY: %dx%d, kernelWH: %dx%d, spectrum_size: %d, padded_size: %d, fftPlanManyInv: %d, total_size: %d\n", 
//                  dim[0], dim[1], dim[2], fftW, fftH, kernelX, kernelY, kernelW, kernelH, spectrum_size, padded_size, fftPlanManyInv, total_size);
  return true;
}

struct sqrt_op
{
	__host__ __device__
		float operator()(const float& x) const {
		return x < 1e-8f ? 0.0f : sqrt(x);
	}
};

struct checked_div
{
	__host__ __device__
		float operator()(const float& numerator, const float& denominator) const {
		return abs(denominator) < 1e-8f ? 0.0f : numerator / denominator;
	}
};

extern "C" bool computeLocalContrastNormalizationGPUMulti(float *d_input_images, int *dim, int num_image_sets, float *d_output_centered, float *d_output_std, float *d_output, float sigma
														, fComplex *d_kernel_spectrum, float *temp_padded, fComplex *temp_spectrum
														, cufftHandle fftPlanManyFwd, cufftHandle fftPlanManyInv)
{
	float *d_temp_std = d_output_std ? d_output_std : d_output;	// d_output_std is just for debugging. set to NULL if no std image output is necessary

	const unsigned int kernelY = getKernelSize(sigma), kernelX = kernelY;
	const unsigned int kernelH = 2 * kernelY + 1, kernelW = kernelH;
	const int fftH = snapTransformSize(dim[1] + kernelH - 1), dataH = dim[1];
	const int fftW = snapTransformSize(dim[0] + kernelW - 1), dataW = dim[0];
	const int spectrum_size = fftH*(fftW / 2 + 1), padded_size = fftH*fftW;

	const int total_size = dim[0] * dim[1] * dim[2] * num_image_sets;

	// compute centered images
	// FFT/iFFT for all images simultaneously (fftPlanMany needs to be created for exactly the same number of images)
	// Note: the following FFT fails if input image contains NaN (even only one pixel!)
	// compute mu images
	padDataClampToBorder(temp_padded, d_input_images, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX, dim[2] * num_image_sets);
	cufftSafeCall(cufftExecR2C(fftPlanManyFwd, (cufftReal *)temp_padded, (cufftComplex *)temp_spectrum));
	modulateAndNormalize(temp_spectrum, d_kernel_spectrum, fftH, fftW, dim[2] * num_image_sets, 1);
	cufftSafeCall(cufftExecC2R(fftPlanManyInv, (cufftComplex *)temp_spectrum, (cufftReal *)temp_padded));
	unPadDataClampToBorder(d_output_centered, temp_padded, fftH, fftW, dataH, dataW, kernelY, kernelX, dim[2] * num_image_sets);
	thrust::transform(thrust::device_ptr<float>(d_input_images), thrust::device_ptr<float>(d_input_images) + total_size,
		thrust::device_ptr<float>(d_output_centered), thrust::device_ptr<float>(d_output_centered), thrust::minus<float>());

	// compute std images (gaussian filtering on squared centered images).
	thrust::transform(thrust::device_ptr<float>(d_output_centered), thrust::device_ptr<float>(d_output_centered) + total_size,
		thrust::device_ptr<float>(d_temp_std), thrust::square<float>());

	padDataClampToBorder(temp_padded, d_temp_std, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX, dim[2] * num_image_sets);
	cufftSafeCall(cufftExecR2C(fftPlanManyFwd, (cufftReal *)temp_padded, (cufftComplex *)temp_spectrum)); // fft
	modulateAndNormalize(temp_spectrum, d_kernel_spectrum, fftH, fftW, dim[2] * num_image_sets, 1);
	cufftSafeCall(cufftExecC2R(fftPlanManyInv, (cufftComplex *)temp_spectrum, (cufftReal *)temp_padded));   // inverse fft
	unPadDataClampToBorder(d_temp_std, temp_padded, fftH, fftW, dataH, dataW, kernelY, kernelX, dim[2] * num_image_sets);
	thrust::transform(thrust::device_ptr<float>(d_temp_std), thrust::device_ptr<float>(d_temp_std) + total_size,
		thrust::device_ptr<float>(d_temp_std), sqrt_op());
	thrust::transform(thrust::device_ptr<float>(d_output_centered), thrust::device_ptr<float>(d_output_centered) + total_size,
		thrust::device_ptr<float>(d_temp_std), thrust::device_ptr<float>(d_output), checked_div());

//	thrust::transform(thrust::device_ptr<float>(d_input_images), thrust::device_ptr<float>(d_input_images) + total_size,
//		thrust::device_ptr<float>(d_input_images), thrust::device_ptr<float>(d_output), checked_div());

	//  print_and_log("computeLocalContrastNormalizationGPUMulti(), dim: %dx%dx%d, fft size: %dx%d, kernelXY: %dx%d, kernelWH: %dx%d, spectrum_size: %d, padded_size: %d, fftPlanManyInv: %d, total_size: %d\n", 
	//                  dim[0], dim[1], dim[2], fftW, fftH, kernelX, kernelY, kernelW, kernelH, spectrum_size, padded_size, fftPlanManyInv, total_size);
	return true;
}

extern "C" bool computeCovarianceGPUMulti(float *d_input_images1_multi, float *d_mu1_multi, float *d_input_images2_single, float *d_mu2_single, int *dim, int num_image_sets
                                           , float *d_output_images_multi, float sigma, fComplex *d_kernel_spectrum, float *temp_padded, fComplex *temp_spectrum
                                           , cufftHandle fftPlanManyFwd, cufftHandle fftPlanManyInv )
{
  const int b = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;
  const int image_size = dim[0]*dim[1]*dim[2];
  computeCovariance_kernel<<<iDivUp(image_size*num_image_sets, b), b>>>
    (d_input_images1_multi, d_mu1_multi, d_input_images2_single, d_mu2_single, d_output_images_multi, image_size, num_image_sets);

  const unsigned int kernelY = getKernelSize(sigma), kernelX = kernelY;
  const unsigned int kernelH = 2*kernelY+1, kernelW = kernelH;
  const int fftH = snapTransformSize(dim[1] + kernelH - 1), dataH = dim[1];
  const int fftW = snapTransformSize(dim[0] + kernelW - 1), dataW = dim[0];
  const int spectrum_size = fftH*(fftW/2+1), padded_size = fftH*fftW;

  // gaussian filter (local sum)
  padDataClampToBorder( temp_padded, d_output_images_multi, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX, dim[2]*num_image_sets );
  cufftSafeCall( cufftExecR2C( fftPlanManyFwd, (cufftReal *)temp_padded, (cufftComplex *)temp_spectrum ) ); // fft
  modulateAndNormalize(temp_spectrum, d_kernel_spectrum, fftH, fftW, dim[2]*num_image_sets, 1); // one kernel spectrum is repeatedly mulplied to a set of images
  cufftSafeCall( cufftExecC2R(fftPlanManyInv, (cufftComplex *)temp_spectrum, (cufftReal *)temp_padded) );   // inverse fft
  unPadDataClampToBorder(d_output_images_multi, temp_padded, fftH, fftW, dataH, dataW, kernelY, kernelX, dim[2]*num_image_sets);

//  print_and_log("computeCovarianceGPUMulti(), dim: %dx%dx%d, fft size: %dx%d, kernelXY: %dx%d, kernelWH: %dx%d, spectrum_size: %d, padded_size: %d, fftPlanManyInv: %d\n", 
//                  dim[0], dim[1], dim[2], fftW, fftH, kernelX, kernelY, kernelW, kernelH, spectrum_size, padded_size, fftPlanManyInv);
  return true;
}

__global__ void computeCovariance_kernel(float *d_img1_multi, float *d_mu1_multi, float *d_img2_single, float *d_mu2_single, float *d_output_multi, int image_size, int num_image_sets)
{
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  if(index >= image_size*num_image_sets) return;
  const int index2 = index % image_size;
  d_output_multi[index] = (d_img2_single[index2]-d_mu2_single[index2])*(d_img1_multi[index]-d_mu1_multi[index]);
}

extern "C" value_type gauss(value_type x, value_type sigma)
{
  // Gaussian
  return exp(-x*x/(2*sigma*sigma)) / (sigma*sqrt(2*M_PI));
}

extern "C" value_type dgauss(value_type x, value_type sigma)
{
  // first order derivative of Gaussian
  return -x * gauss(x,sigma) / (sigma*sigma);
}

extern "C" int getGaussianGradientKernel(const value_type sigma, const int kernel_size, value_type *h_kernel)
{
  // we assume kernel_size is odd number, which is computed by something like the following
  // epsilon=1e-2;
  // halfsize=ceil(sigma*sqrt(-2*log(sqrt(2*pi)*sigma*epsilon)));
  // size=2*halfsize+1;

  int half_size = (kernel_size-1)/2;
  value_type sum = 0.0;
  for(int i=0;i<kernel_size;i++){
    for(int j=0;j<kernel_size;j++){
      int index = j*kernel_size+i;
      h_kernel[index] = gauss(i-half_size, sigma)*dgauss(j-half_size, sigma);
      sum += abs(h_kernel[index])*abs(h_kernel[index]);
    }
  }
  sum = sqrt(sum);
  value_type* end_ptr = &(h_kernel[kernel_size*kernel_size]);
  for(value_type *ptr = h_kernel;ptr != end_ptr;ptr++)  *ptr /= sum;

  return true;
}

extern "C" int getGaussianKernel(const value_type sigma, const int kernel_size, value_type *h_kernel)
{
  // we assume kernel_size is odd number, which is computed by something like the following

  value_type half_size = (kernel_size-1)/2;
  value_type sum = 0.0;
  for(int i=0;i<kernel_size;i++){
    for(int j=0;j<kernel_size;j++){
      int index = j*kernel_size+i;
      h_kernel[index] = gauss(sqrt((i-half_size)*(i-half_size)+(j-half_size)*(j-half_size)), sigma);
      sum += h_kernel[index];
    }
  }
  value_type* end_ptr = &(h_kernel[kernel_size*kernel_size]);
  for(value_type *ptr = h_kernel;ptr != end_ptr;ptr++)  *ptr /= sum;

  return true;
}

extern "C" bool computeGaussianGradientKernelSpectrum(float sigma, int x_dim, int y_dim, int z_dim, fComplex *x_kernel_spectrum, fComplex *y_kernel_spectrum, cufftHandle fftPlanFwd)
{
  unsigned int halfsize = getKernelSize(sigma);
  unsigned int kernel_size=2*halfsize+1;
//  print_and_log("computeGaussianGradientKernel(), kernel_size: %d\n", kernel_size);
  float *x_kernel = new float[kernel_size*kernel_size], *y_kernel = new float[kernel_size*kernel_size];
  // compute gaussian gradient kernel for x direction
  getGaussianGradientKernel(sigma, kernel_size, x_kernel);
  // compute gaussian gradient kernel for y direction (just transpose x kernel)
  for(unsigned int i=0;i<kernel_size;i++)  for(unsigned int j=0;j<kernel_size;j++)  y_kernel[j*kernel_size+i] = x_kernel[i*kernel_size+j];

  computeKernelSpectrumGPU( x_kernel, y_dim, x_dim, kernel_size, kernel_size, halfsize, halfsize, x_kernel_spectrum, fftPlanFwd );
  computeKernelSpectrumGPU( y_kernel, y_dim, x_dim, kernel_size, kernel_size, halfsize, halfsize, y_kernel_spectrum, fftPlanFwd );

  delete[] x_kernel;
  delete[] y_kernel;
  return true;
}

extern "C" bool computeGaussianKernelSpectrum(float sigma, int x_dim, int y_dim, int z_dim, fComplex *kernel_spectrum, cufftHandle fftPlanFwd)
{
  unsigned int halfsize = getKernelSize(sigma);
  unsigned int kernel_size=2*halfsize+1;
  //print_and_log("computeGaussianKernel(), kernel_size: %d, sigma: %f\n", kernel_size, sigma);
  float *kernel = new float[kernel_size*kernel_size];
  // compute gaussian gradient kernel
  getGaussianKernel(sigma, kernel_size, kernel);
  //for (int i = 0; i < kernel_size*kernel_size; i++) print_and_log("%f,", kernel[i]);
  //print_and_log("\n");

  computeKernelSpectrumGPU( kernel, y_dim, x_dim, kernel_size, kernel_size, halfsize, halfsize, kernel_spectrum, fftPlanFwd );

  delete[] kernel;
  return true;
}

extern "C" void computeGradientInformation(float *d_grad1X, float *d_grad1Y, float *d_grad2X, float *d_grad2Y, float *d_output, int image_size
                                             , float *d_mask_weight, bool isSingleModality, double *gradient_information, int num_image_sets
                                             , cublasHandle_t cublasHandle, float *d_temp_SM, float *h_temp_SM, float *d_OneVector, float h_NormalizationFactor, int exclusive_norm)
{
  const int b = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;
  computeGradientInformation_kernel<<<iDivUp(image_size*num_image_sets, b), b>>>
    (d_grad1X, d_grad1Y, d_grad2X, d_grad2Y, d_output, image_size, isSingleModality, d_mask_weight, num_image_sets, exclusive_norm);
  // compute sum of each gradient information image
  // to parallelize the computation, we consider d_output as a matrix of (image_size x num_image_sets)' and multiply one vector of image_size element
//  computeImageSum_kernel<<<iDivUp(num_image_sets, b), b>>>(d_output, image_size, num_image_sets, d_OneVector, d_temp_SM);
//  cutilSafeCall( cudaMemcpy(h_temp_SM, d_output, num_image_sets*sizeof(float), cudaMemcpyDeviceToHost) );
//  for(int i=0;i<num_image_sets;i++){ print_and_log("d_output[%d] = %f\n", i, h_temp_SM[i]); }

  float beta = 0.0f;
  cublasStatus_t status = cublasSgemv(cublasHandle, CUBLAS_OP_T, image_size, num_image_sets, &h_NormalizationFactor, d_output, image_size, d_OneVector, 1, &beta, d_temp_SM, 1);
  if (status != CUBLAS_STATUS_SUCCESS){ print_and_log("!!!! cublasSgemv execution error\n"); }
  cutilSafeCall( cudaMemcpy(h_temp_SM, d_temp_SM, num_image_sets*sizeof(float), cudaMemcpyDeviceToHost) );
  for(int i=0;i<num_image_sets;i++) gradient_information[i] = (double)(h_temp_SM[i]); // float -> double conversion
//  for(int i=0;i<num_image_sets;i++){ print_and_log("GI[%d] = %f\n", i, gradient_information[i]); }
}

extern "C" void computeGradientInformation_StdNorm(float *d_grad1X, float *d_grad1Y, float *d_grad2X, float *d_grad2Y, float *d_output, int image_size
                                             , float *d_mask_weight, bool isSingleModality, double *gradient_information, int num_image_sets
                                             , cublasHandle_t cublasHandle, float *d_temp_SM, float *h_temp_SM, float *d_OneVector, float h_NormalizationFactor)
{
/*  // Note: d_grad1X, d_grad1Y and d_output have multiple image sets, while d_grad2X and d_grad2Y have a single image set.
  cublasSdgmm(cublasHandle, CUBLAS_SIDE_LEFT, image_size, num_image_sets, d_grad1X, image_size, d_mask_weight, 1, d_images, image_size);

  const int b = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;
  computeGradientInformation_kernel<<<iDivUp(image_size*num_image_sets, b), b>>>
    (d_grad1X, d_grad1Y, d_grad2X, d_grad2Y, d_output, image_size, isSingleModality, d_mask_weight, num_image_sets);

  float beta = 0.0f;
  cublasStatus_t status = cublasSgemv(cublasHandle, CUBLAS_OP_T, image_size, num_image_sets, &h_NormalizationFactor, d_output, image_size, d_OneVector, 1, &beta, d_temp_SM, 1);
  if (status != CUBLAS_STATUS_SUCCESS){ print_and_log("!!!! cublasSgemv execution error\n"); }
  cutilSafeCall( cudaMemcpy(h_temp_SM, d_temp_SM, num_image_sets*sizeof(float), cudaMemcpyDeviceToHost) );
  for(int i=0;i<num_image_sets;i++) gradient_information[i] = (double)(h_temp_SM[i]); // float -> double conversion
//  for(int i=0;i<num_image_sets;i++){ print_and_log("GI[%d] = %f\n", i, gradient_information[i]); }
*/
}

__global__ void computeGradientInformation_kernel(float *d_grad1X, float *d_grad1Y, float *d_grad2X, float *d_grad2Y, float *d_output, int image_size, bool isSingleModality
                                                  , float *d_mask_weight, int num_image_sets, int exclusive_norm)
{
  // image1 has multiple image sets, image2 has only one image set
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  if(index >= image_size*num_image_sets) return;
  const int index2 = index % image_size;
  const float epsilon = 1e-6;

  // compute inner product of gradient vectors
  float inner_product = (d_grad1X[index]*d_grad2X[index2]) + (d_grad1Y[index]*d_grad2Y[index2]);
  // compute product of norm
  float norm[2] = {sqrt( d_grad1X[index]*d_grad1X[index]+d_grad1Y[index]*d_grad1Y[index] ), sqrt( d_grad2X[index2]*d_grad2X[index2]+d_grad2Y[index2]*d_grad2Y[index2] )};
  float norm_prod = norm[0] * norm[1];
  float norm_angle_cos = inner_product / norm_prod;

  // compute angle between two gradient vector and compute weight
  // if norm_prod is small, weight is 0
  float weight;
  if(norm_prod<epsilon)                               weight = 0;
  else if(norm_angle_cos>1.0 || norm_angle_cos<-1.0)  weight = 1;
  else if(isSingleModality)                           weight = ( cos( acos(norm_angle_cos) ) + 1 ) / 2;   // for single-modality registration (zero weight on 180 degrees)
  else                                                weight = ( cos( 2*acos(norm_angle_cos) ) + 1 ) / 2; // for multi-modality registration (high weight on 180 degrees)
  // compute gradient angle
  if(d_mask_weight) weight *= d_mask_weight[index2]; // multiply mask (for the case when we exclude part of the image from contribution to the similarity metric)

  if(exclusive_norm >= 0)   d_output[index] = weight * norm[exclusive_norm];  // if specified, we always use the norm
  else                      d_output[index] = weight * MIN( norm[0], norm[1] );
}

extern "C" void computeSSIM(float *d_mu1_multi, float *d_sigma_sq1_multi, float *d_mu2_single, float *d_sigma_sq2_single, float *d_output_multi, int image_size
                            , float *d_mask_weight, double *SSIM, int num_image_sets
                            , cublasHandle_t cublasHandle, float *d_temp_SSIM, float *h_temp_SSIM, float *d_OneVector, float h_DynamicRange)
{
  // image1 has multiple image sets, image2 has only one image set
  const int b = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;
  // d_output needs to be set as the covariance image prior to this function call
  float K1 = 0.01, K2 = 0.03;
  float C1 = (K1*h_DynamicRange)*(K1*h_DynamicRange), C2 = (K2*h_DynamicRange)*(K2*h_DynamicRange);
//  print_and_log("computeSSIM(), dynamic range: %f, C1: %f, C2: %f\n", h_DynamicRange, C1, C2);
  computeSSIM_kernel<<<iDivUp(image_size*num_image_sets, b), b>>>
    (d_mu1_multi, d_sigma_sq1_multi, d_mu2_single, d_sigma_sq2_single, d_output_multi, image_size, d_mask_weight, num_image_sets, C1, C2);

  // compute sum of each SSIM image
  // to parallelize the computation, we consider d_output as a matrix of (image_size x num_image_sets)' and multiply one vector of image_size element
//  computeImageSum_kernel<<<iDivUp(num_image_sets, b), b>>>(d_output, image_size, num_image_sets, d_temp_SSIM);

  float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t status = cublasSgemv(cublasHandle, CUBLAS_OP_T, image_size, num_image_sets, &alpha, d_output_multi, image_size, d_OneVector, 1, &beta, d_temp_SSIM, 1);
  if (status != CUBLAS_STATUS_SUCCESS){ print_and_log("!!!! cublasSgemv execution error\n"); }
  cutilSafeCall( cudaMemcpy(h_temp_SSIM, d_temp_SSIM, num_image_sets*sizeof(float), cudaMemcpyDeviceToHost) );
  for(int i=0;i<num_image_sets;i++) SSIM[i] = (double)(h_temp_SSIM[i]); // float -> double conversion
//  for(int i=0;i<num_image_sets;i++){ print_and_log("SSIM[%d] = %f\n", i, SSIM[i]); }
}

__global__ void computeSSIM_kernel(float *d_mu1_multi, float *d_sigma_sq1_multi, float *d_mu2_single, float *d_sigma_sq2_single, float *d_output_multi, int image_size
                                   , float *d_mask_weight, int num_image_sets, float C1, float C2)
{
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  if(index >= image_size*num_image_sets) return;
  const int index2 = index % image_size;
  d_output_multi[index] = ( (2*d_mu1_multi[index]*d_mu2_single[index2]+C1) * (2*d_output_multi[index] + C2) )/ 
                          ( (d_mu1_multi[index]*d_mu1_multi[index]+d_mu2_single[index2]*d_mu2_single[index2] + C1) * (d_sigma_sq1_multi[index] + d_sigma_sq2_single[index2] + C2) );
  if(d_mask_weight) d_output_multi[index] *= d_mask_weight[index2];
//  d_output[index] = (2*d_mu1[index]*d_mu2[index]+C1)/ (d_mu1[index]*d_mu1[index]+d_mu2[index]*d_mu2[index] + C1);
//  d_output[index] = (2*d_output[index] + C2)/ (d_sigma_sq1[index] + d_sigma_sq2[index] + C2);
}

__global__ void computeImageSum_kernel(float *d_image, int image_size, int num_image_sets, float *d_weight, float *d_sum)
{
  const int image_indx = blockDim.x * blockIdx.x + threadIdx.x;
  if(image_indx >= num_image_sets) return;
  float sum = 0, *start_ptr = d_image+image_indx*image_size, *end_ptr = start_ptr+image_size;
  for(float *ptr=start_ptr;ptr!=end_ptr;ptr++) sum += (*ptr);
  d_sum[image_indx] = sum*d_weight[image_indx];
}

struct p_log_p : public thrust::unary_function<float, float> {  
  __host__ __device__     
  float operator()(float a){
    return (a == 0) ? 0 : -a*logf(a);
  }  
};  

extern "C" float computeJointEntropy(float *d_data1, float *d_data2, int length, int *d_joint_hist, float *d_joint_pdf)
{
  computeJointPDF(d_data1, d_data2, length, d_joint_hist, d_joint_pdf);
  thrust::device_vector<float> pdf_vec(MI_NUMBER_OF_BINS*MI_NUMBER_OF_BINS);
  thrust::copy_n( thrust::device_ptr<float>(d_joint_pdf), MI_NUMBER_OF_BINS*MI_NUMBER_OF_BINS, pdf_vec.begin() );
  // make_transform_iterator could not be applied directly to device_ptr<float>, so we need to make a temporary copy...
  return thrust::reduce( make_transform_iterator( pdf_vec.begin(), p_log_p() ), 
                         make_transform_iterator( pdf_vec.end(), p_log_p() ), 
                         0.0f,
                         thrust::plus<float>() );
}

extern "C" bool computeJointPDF(float *d_data1, float *d_data2, int length, int *d_joint_hist, float *d_joint_pdf)
{
  const int block_size = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;
	dim3 dimBlock(block_size, 1);
  int block_num = iDivUp(length, block_size);
  int size_sq = static_cast<int>(ceil( sqrtf(static_cast<float>(block_num)) ));
	dim3 dimDataGrid(size_sq, iDivUp(block_num, size_sq), 1);
//  print_and_log("computeJointPDF(), length of data: %d, size of joint_pdf: %d, sqrt of size: %f(%d), grid = (%d, %d), block = (%d, %d)\n"
//    , length, MI_NUMBER_OF_BINS*MI_NUMBER_OF_BINS, sqrtf(block_num), size_sq, dimDataGrid.x, dimDataGrid.y, dimBlock.x, dimBlock.y);
  cutilSafeCall( cudaMemset(d_joint_hist, 0, MI_NUMBER_OF_BINS*MI_NUMBER_OF_BINS*sizeof(int)) );
  computeJointHistogramKernel<<< dimDataGrid, dimBlock, 0 >>> (d_data1, d_data2, length, d_joint_hist, MI_NUMBER_OF_BINS);

  // normalize histogram (divide each element by length)
  thrust::transform( thrust::device_ptr<int>(d_joint_hist), thrust::device_ptr<int>(d_joint_hist) + MI_NUMBER_OF_BINS*MI_NUMBER_OF_BINS, 
    thrust::make_constant_iterator<float>(length), thrust::device_ptr<float>(d_joint_pdf), thrust::divides<float>());
  return true;
}

extern "C" float computeEntropy(float *d_data, int length, int *d_hist, float *d_pdf)
{
  computePDF(d_data, length, d_hist, d_pdf);
  thrust::device_vector<float> pdf_vec(MI_NUMBER_OF_BINS);
  thrust::copy_n( thrust::device_ptr<float>(d_pdf), MI_NUMBER_OF_BINS, pdf_vec.begin() );
  // make_transform_iterator could not be applied directly to device_ptr<float>, so we need to make a temporary copy...
  return thrust::reduce( make_transform_iterator( pdf_vec.begin(), p_log_p() ), 
                         make_transform_iterator( pdf_vec.end(), p_log_p() ), 
                         0.0f,
                         thrust::plus<float>() );
}

extern "C" bool computePDF(float *d_data, int length, int *d_hist, float *d_pdf)
{
  const int block_size = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;
	dim3 dimBlock(block_size, 1);
  int block_num = iDivUp(length, block_size);
  int size_sq = static_cast<int>(ceil( sqrtf(static_cast<float>(block_num)) ));
	dim3 dimDataGrid(size_sq, iDivUp(block_num, size_sq), 1);
  cutilSafeCall( cudaMemset(d_hist, 0, MI_NUMBER_OF_BINS*sizeof(int)) );
  computeHistogramKernel<<< dimDataGrid, dimBlock, 0 >>> (d_data, length, d_hist, MI_NUMBER_OF_BINS);

  // normalize histogram (divide each element by length)
  thrust::transform( thrust::device_ptr<int>(d_hist), thrust::device_ptr<int>(d_hist) + MI_NUMBER_OF_BINS, 
    thrust::make_constant_iterator<float>(length), thrust::device_ptr<float>(d_pdf), thrust::divides<float>());
  return true;
}

__global__ void computeJointHistogramKernel(float *d_data1, float *d_data2, int length, int *d_joint_hist, int num_bins)
{
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;   // blockDim.y = 1
  if(index < length && index>=0){
    int accumulate_index = floor(d_data1[index]*num_bins)*num_bins+floor(d_data2[index]*num_bins);  // d_data has to be normalized to [0 1] in advance
    if(accumulate_index < num_bins*num_bins)  atomicAdd(d_joint_hist+accumulate_index, 1);
  }
}

__global__ void computeHistogramKernel(float *d_data, int length, int *d_hist, int num_bins)
{
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;   // blockDim.y = 1
  if(index < length && index>=0){
    int accumulate_index = floor(d_data[index]*num_bins);  // d_data has to be normalized to [0 1] in advance
    if(accumulate_index < num_bins)  atomicAdd(d_hist+accumulate_index, 1);
  }
}

__global__ void normalizeHistogramKernel(int *d_hist, float *d_pdf, int size, float denominator)
{
	int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;   // blockDim.y = 1
  if(index < size)  d_pdf[index] = (float)d_hist[index]/denominator;
}


////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
__global__ void padKernel_kernel(float *d_Dst, float *d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX)
{
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(y < kernelH && x < kernelW){
        int ky = y - kernelY; if(ky < 0) ky += fftH;
        int kx = x - kernelX; if(kx < 0) kx += fftW;
        d_Dst[ky * fftW + kx] = LOAD_FLOAT(y * kernelW + x);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataClampToBorder_kernel(float *d_Dst, float *d_Src, int fftH, int fftW, int dataH, int dataW, int kernelH, int kernelW, int kernelY, int kernelX, int imageRepeat)
{
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int total_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int borderH = dataH + kernelY;
  const int borderW = dataW + kernelX;
  const int x = total_x % fftW;
  const int imageID = (total_x-x)/fftW;    // division in 'int'
  if(imageID>=imageRepeat || imageID<0) return;

  if(y < fftH && x < fftW){
    const int dy = (y<dataH) ? y : ( (y>=dataH && y<borderH) ? (dataH-1) : 0 );
    const int dx = (x<dataW) ? x : ( (x>=dataW && x<borderW) ? (dataW-1) : 0 );
    d_Dst[y * fftW + x + fftH*fftW*imageID] = LOAD_FLOAT(dy * dataW + dx + dataH*dataW*imageID);
	//d_Dst[y * fftW + x + fftH*fftW*imageID] = 0; // LOAD_FLOAT(dy * dataW + dx + dataH*dataW*imageID);
  }
}

__global__ void fillEdges_kernel(float *d_Dst, float *d_Ref, int fftH, int fftW, int dataH, int dataW, int kernelH, int kernelW, int kernelY, int kernelX, int imageRepeat, float threshold)
{
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int total_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int borderH = dataH + kernelY;
  const int borderW = dataW + kernelX;
  const int x = total_x % fftW;
  const int imageID = (total_x-x)/fftW;    // division in 'int'
  if(imageID>=imageRepeat || imageID<0) return;

  if(y < fftH && x < fftW){
    const int dy = (y<dataH) ? y : ( (y>=dataH && y<borderH) ? (dataH-1) : 0 );
    const int dx = (x<dataW) ? x : ( (x>=dataW && x<borderW) ? (dataW-1) : 0 );
    if(d_Ref[dy * dataW + dx + dataH*dataW*imageID]<threshold){
      // if the pixel in the reference image is smaller than threshold, fill (kernelH x kernenW) pixels around the pixel
      for(int i=-kernelX;i<=kernelX;i++){
        for(int j=-kernelY;j<=kernelY;j++){
          if((y+j>=0) && (y+j<fftH) && (x+i>=0) && (x+i<fftW))
            d_Dst[(y+j) * fftW + (x+i) + fftH*fftW*imageID] = 0;
        }
      }
//      for(int i=-0;i<=0;i++) for(int j=-0;j<=0;j++) d_Dst[(y+j) * fftW + (x+i) + fftH*fftW*imageID] = 0;
    }
  }
}

__global__ void unPadDataClampToBorder_kernel(float *d_Dst, float *d_Src, int fftH, int fftW, int dataH, int dataW, int kernelY, int kernelX, int imageRepeat)
{
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int total_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int borderH = dataH + kernelY;
  const int borderW = dataW + kernelX;
  const int x = total_x % dataW;
  const int imageID = (total_x-x)/dataW;    // division in 'int'
  if(imageID>=imageRepeat || imageID<0) return;

  if(y < dataH && x < dataW){
    const int dy = (y<dataH) ? y : ( (y>=dataH && y<borderH) ? (dataH-1) : 0 );
    const int dx = (x<dataW) ? x : ( (x>=dataW && x<borderW) ? (dataW-1) : 0 );
    d_Dst[dy*dataW+dx + dataH*dataW*imageID] = LOAD_FLOAT(y*fftW+x + fftH*fftW*imageID);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
extern "C" void padKernel(float *d_Dst, float *d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y));

    SET_FLOAT_BASE;
    padKernel_kernel<<<grid, threads>>>(d_Dst, d_Src, fftH, fftW, kernelH, kernelW, kernelY, kernelX);
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataClampToBorder(float *d_Dst, float *d_Src, int fftH, int fftW, int dataH, int dataW, int kernelW, int kernelH, int kernelY, int kernelX, int imageRepeat)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(fftW*imageRepeat, threads.x), iDivUp(fftH, threads.y));
//    print_and_log("padDataClampToBorder, grid size: (%dx%d), block size: (%dx%d), fftH: %d, fftW: %d\n", grid.x, grid.y, threads.x, threads.y, fftH, fftW);
//	print_and_log("dataH: %d, dataW: %d, kernelH: %d, kernelW: %d, kernelY: %d, kernelX: %d, imageRepeat: %d\n", dataH, dataW, kernelH, kernelW, kernelY, kernelX, imageRepeat);
    SET_FLOAT_BASE;   // When we use texture for small image (less than 256x256 maybe), this line causes crashes, for some reason...
    padDataClampToBorder_kernel<<<grid, threads>>>(d_Dst, d_Src, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX, imageRepeat);
}

extern "C" void fillEdges(float *d_Dst, float *d_Ref, int fftH, int fftW, int dataH, int dataW, int kernelW, int kernelH, int kernelY, int kernelX, int imageRepeat, float threshold)
{
    assert(d_Ref != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(fftW*imageRepeat, threads.x), iDivUp(fftH, threads.y));
//    print_and_log("fillEdges, grid size: (%dx%d), block size: (%dx%d)\n", grid.x, grid.y, threads.x, threads.y);
    SET_FLOAT_BASE;   // When we use texture for small image (less than 256x256 maybe), this line causes crashes, for some reason...
    fillEdges_kernel<<<grid, threads>>>(d_Dst, d_Ref, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX, imageRepeat, threshold);
}

extern "C" void unPadDataClampToBorder(float *d_Dst, float *d_Src, int fftH, int fftW, int dataH, int dataW, int kernelY, int kernelX, int imageRepeat)
{
    dim3 threads(32, 8);
    dim3 grid(iDivUp(dataW*imageRepeat, threads.x), iDivUp(dataH, threads.y));
//    print_and_log("unPadDataClampToBorder, grid size: (%dx%d), block size: (%dx%d)\n", grid.x, grid.y, threads.x, threads.y);
    SET_FLOAT_BASE;
    unPadDataClampToBorder_kernel<<<grid, threads>>>(d_Dst, d_Src, fftH, fftW, dataH, dataW, kernelY, kernelX, imageRepeat);
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
__global__ void modulateAndNormalize_kernel(fComplex *d_Dst, fComplex *d_Src, int dataSize, int imageRepeat, float c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= (dataSize*imageRepeat)) return;

    fComplex a = d_Src[i%dataSize];   // the same d_Src is repeatedly multiplied to d_Dst
    fComplex b = d_Dst[i];

    mulAndScale(a, b, c);

    d_Dst[i] = a;
}

extern "C" void modulateAndNormalize(fComplex *d_Dst, fComplex *d_Src, int fftH, int fftW, int imageRepeat, int padding)
{
    assert( fftW % 2 == 0 );
    const int dataSize = fftH * (fftW / 2 + padding);

    const int b = 256; //BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;
    modulateAndNormalize_kernel<<<iDivUp(dataSize*imageRepeat, b), b>>>(
        d_Dst,
        d_Src,
        dataSize,
        imageRepeat,
        1.0f / (float)(fftW * fftH)
    );
}

extern "C" void normalizeImages(float *d_images, int size, float norm_max, float norm_min)
{
  if(norm_max == 0 && norm_min == 0){
  // compute maximum and minimum if needed
    ComputeMaxMin(d_images, size, &norm_max, &norm_min);
//    print_and_log("(min,max) = (%f, %f)\n", norm_min, norm_max); 
  }

  // normalize images
  NormalizeData(d_images, size, norm_max, norm_min);
}

extern "C" void maskImages( float *d_images, float *d_mask_weight, int image_size, int num_image_sets, cublasHandle_t cublasHandle )
{
  cublasSdgmm(cublasHandle, CUBLAS_SIDE_LEFT, image_size, num_image_sets, d_images, image_size, d_mask_weight, 1, d_images, image_size);
}

extern "C" int countZeroPixels(float *d_image, int image_size)
{
  return thrust::count( thrust::device_ptr<float>(d_image), thrust::device_ptr<float>(d_image)+image_size, 0.0f);
}
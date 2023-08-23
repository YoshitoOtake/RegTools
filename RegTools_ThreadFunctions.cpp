#include "RegTools.h"
#define _USE_MATH_DEFINES
#include "math.h"   // for M_PI_2, cos, sin

extern FILE *m_LogFile;

int RegTools::RegToolsThread_MemGetInfo(size_t &free_mem, size_t &total_mem)
{
  // Check available GPU memory
#if ((CUDA_VERSION_MAJOR>=3) && (CUDA_VERSION_MINOR>=2)) | (CUDA_VERSION_MAJOR>=4)
  cutilSafeCall( cudaMemGetInfo(&free_mem, &total_mem) );
#else
  unsigned int free, total;
  cutilSafeCall( cudaMemGetInfo(&free, &total) );
  free_mem = free; total_mem = total;
#endif
  return true;
}

void RegTools::RegToolsThread_SimilarityMeasureComputation(RegToolsThreadParam *in_param, float *d_Projections, float *d_Volume)
{
#if defined(RegTools_ENABLE_SIMILARITY_MEASURE_BENCHMARK)
  if(in_param->m_ElapsedTime) *(in_param->m_ElapsedTime) = 0;
  // start kernel running time counter
  cudaEventRecord(in_param->h_timer_start, 0);
#endif

  SimilarityMeasureComputationPlan *plan = in_param->m_SimilarityMeasureComputationPlan;
  SimilarityMeasureComputationPlan *plan2 = in_param->m_SimilarityMeasureComputationPlan2;
  size_t image_size = plan->ImageDim[0]*plan->ImageDim[1];
  size_t fixedImageDim = plan->ImageDim[2]*image_size;

//  print_and_log("fixedImageDim: %d, I0: %f\n", fixedImageDim, plan->I0);
  if(plan->SimilarityMeasure){
    if(in_param->m_SimilarityMeasure_NumberOfImageSet == 0) return;
    memset(plan->SimilarityMeasure, 0, in_param->m_SimilarityMeasure_NumberOfImageSet * sizeof(float));
    float *d_floating_images = NULL, *d_floating_Xgrad = NULL, *d_floating_Ygrad = NULL, *d_floating_normalized = NULL;
    if(plan2){
      // computation of the similarity measure between plan1 and plan2 (we consider plan2 as floating images)
      d_floating_images = plan2->d_fixed_images;
      d_floating_Xgrad = plan2->d_fixed_Xgrad;  d_floating_Ygrad = plan2->d_fixed_Ygrad;  d_floating_normalized = plan2->d_fixed_normalized;
    } else {
      if(d_Projections){ d_floating_images = d_Projections;/*   print_and_log("RegToolsThread_SimilarityMeasureComputation(), compute similarity with projections\n");*/ }
      else if(d_Volume){ d_floating_images = d_Volume;      /*  print_and_log("RegToolsThread_SimilarityMeasureComputation(), compute similarity with volume\n");*/ }
      else { print_and_log("Error at SimilarityMeasureComputation. Call ForwardProject(), BackProject() or Interpolation() before computing similarity measrue\n");  return; }
//        if(in_param->m_SimilarityMeasureComputationImageOffset<in_param->m_TransferBlockSize) // make sure if the following address is not out of range
      d_floating_Xgrad = plan->d_floating_Xgrad;   d_floating_Ygrad = plan->d_floating_Ygrad;   d_floating_normalized = plan->d_floating_normalized;
//	  print_and_log("d_floating_Xgrad: %d, d_floating_Ygrad: %d, d_floating_normalized: %d\n", d_floating_Xgrad, d_floating_Ygrad, d_floating_normalized);
    }

    // computation of the similarity measure between plan and forward projected images (forward projected images "d_Projections" are the floating images)
    if( (plan->SimilarityMeasureType == SIMILARITY_MEASURE_GI || plan->SimilarityMeasureType == SIMILARITY_MEASURE_GI_SINGLE
      || plan->SimilarityMeasureType == SIMILARITY_MEASURE_GI_SINGLE_ALWAYS_FLOATING_NORM) && plan->Sigma>0){
      if(!plan->d_fixed_Xgrad || !plan->d_fixed_Ygrad || !plan->d_intermediate_images || !d_floating_Xgrad || !d_floating_Ygrad){ print_and_log("Error:: Similarity computation plan is not created for GI. Check Sigma value.\n"); return; }
      bool isSingleModality = (plan->SimilarityMeasureType == SIMILARITY_MEASURE_GI_SINGLE);
      int exclusive_norm = (plan->SimilarityMeasureType == SIMILARITY_MEASURE_GI_SINGLE_ALWAYS_FLOATING_NORM) ? 0 : -1;
//      RegToolsThread_MemGetInfo( in_param->m_FreeMem, in_param->m_TotalMem );
      computeGaussianGradientGPUMulti(d_floating_images, plan->ImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, d_floating_Xgrad, d_floating_Ygrad, plan->Sigma
                                  , plan->d_X_kernel_spectrum, plan->d_Y_kernel_spectrum, plan->d_temp_padded, plan->d_temp_spectrum
                                  , *(plan->fftPlanManyFwd), *(plan->fftPlanManyInv), plan->h_GI_threshold);
      computeGradientInformation(d_floating_Xgrad, d_floating_Ygrad, plan->d_fixed_Xgrad, plan->d_fixed_Ygrad, plan->d_intermediate_images
              , static_cast<int>(fixedImageDim), plan->d_mask_weight, isSingleModality, plan->SimilarityMeasure, in_param->m_SimilarityMeasure_NumberOfImageSet
              , in_param->m_CublasHandle, plan->d_temp_SM, plan->h_temp_SM, plan->d_WeightVector, 1.0f/plan->NormalizationFactor, exclusive_norm);
      if(in_param->m_CountNonIntersectedPixel){
//        cutilSafeCall( cudaMemcpy(zero_pixel_count, plan->d_temp_ZeroPixelCount, in_param->m_SimilarityMeasure_NumberOfImageSet*sizeof(int), cudaMemcpyDeviceToHost) );
        // normalization by the number of non-zero pixels
        for(int i=0;i<in_param->m_SimilarityMeasure_NumberOfImageSet;i++){
//          size_t zero_pixels = countZeroPixels(d_floating_images+i*image_size, image_size);
          plan->SimilarityMeasure[i] *= (double)(image_size) / (double)(image_size - in_param->m_ZeroPixelCount[i]);
//          print_and_log("similarity computation: image #%d, number of zero pixels = %d/%d, weight = %f\n", i, in_param->m_ZeroPixelCount[i], image_size, (double)(image_size) / (double)(image_size - in_param->m_ZeroPixelCount[i]));
        }
      }
    } else if(plan->SimilarityMeasureType == SIMILARITY_MEASURE_SSIM){
      // normalize DRRs (compute max/min for each image)
      for(int i=0;i<in_param->m_SimilarityMeasure_NumberOfImageSet;i++)
        normalizeImages( d_floating_images+fixedImageDim*i, static_cast<int>(fixedImageDim), 0, 0);
      computeGaussianGPUMulti(d_floating_images, plan->ImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, d_floating_Xgrad, d_floating_Ygrad, plan->Sigma
                                  , plan->d_X_kernel_spectrum, plan->d_temp_padded, plan->d_temp_spectrum
                                  , *(plan->fftPlanManyFwd), *(plan->fftPlanManyInv));
      computeCovarianceGPUMulti(d_floating_images, d_floating_Xgrad, plan->d_fixed_images, plan->d_fixed_Xgrad
                                  , plan->ImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, plan->d_intermediate_images, plan->Sigma
                                  , plan->d_X_kernel_spectrum, plan->d_temp_padded, plan->d_temp_spectrum
                                  , *(plan->fftPlanManyFwd), *(plan->fftPlanManyInv));
      computeSSIM(d_floating_Xgrad, d_floating_Ygrad, plan->d_fixed_Xgrad, plan->d_fixed_Ygrad, plan->d_intermediate_images
              , static_cast<int>(fixedImageDim), plan->d_mask_weight, plan->SimilarityMeasure, in_param->m_SimilarityMeasure_NumberOfImageSet
              , in_param->m_CublasHandle, plan->d_temp_SM, plan->h_temp_SM, plan->d_WeightVector, plan->h_SSIM_DynamicRange);
    } else if(plan->SimilarityMeasureType == SIMILARITY_MEASURE_MI || plan->SimilarityMeasureType == SIMILARITY_MEASURE_NMI){
      if(!plan->d_fixed_normalized || !d_floating_normalized){ print_and_log("Error:: Similarity computation plan is not created for MI/NMI. Check NoralizeMax and NormalizeMin.\n"); return; }
      // normalize all image sets to the same dynamic range at once for speed up
      normalizeImages( d_floating_images, static_cast<int>(in_param->m_SimilarityMeasure_NumberOfImageSet*fixedImageDim), plan->NormalizeMax_floating, plan->NormalizeMin_floating);
      if(plan->d_mask_weight) maskImages( d_floating_images, plan->d_mask_weight, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, in_param->m_CublasHandle );
      for(int image_set_id = 0; image_set_id<in_param->m_SimilarityMeasure_NumberOfImageSet; image_set_id++){
        // normalize each image separately if needed
//        normalizeImages( d_floating_images+image_set_id*fixedImageDim, static_cast<int>(fixedImageDim), plan->NormalizeMax_floating, plan->NormalizeMin_floating);
        float h1 = computeEntropy( d_floating_images+image_set_id*fixedImageDim, static_cast<int>(fixedImageDim), plan->d_hist_buf, plan->d_pdf_buf);
        float h12 = computeJointEntropy(d_floating_images+image_set_id*fixedImageDim, plan->d_fixed_normalized, (int)fixedImageDim, plan->d_joint_hist, plan->d_joint_pdf);
        plan->SimilarityMeasure[image_set_id] = (plan->SimilarityMeasureType == SIMILARITY_MEASURE_NMI) ? (h1+plan->h_fixed_entropy)/h12 : h1+plan->h_fixed_entropy-h12;
        //      print_and_log("GPU, Entropies: (%.3f, %.3f), JointEntropy: %.3f, MI: %.3f, NMI: %.3f\n", plan->h_fixed_entropy, h1, h12, h1+plan->h_fixed_entropy-h12, (h1+plan->h_fixed_entropy)/h12 );
      }
    } else if(plan->SimilarityMeasureType == SIMILARITY_MEASURE_NCC){
      memset(plan->SimilarityMeasure, 0, sizeof(double)*in_param->m_SimilarityMeasure_NumberOfImageSet);
      zeromean_stddivide_Images(d_floating_images, plan->d_intermediate_images, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, in_param->m_CublasHandle, plan->d_temp_SM
                      , plan->d_WeightVector, 1.0f/plan->NormalizationFactor, plan->d_mask_weight);
      computeNormalizedCrossCorrelation_Pixels(d_floating_images, plan->d_fixed_images, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, in_param->m_CublasHandle);
      computeNormalizedCrossCorrelation_Sum(NULL, NULL, d_floating_images, plan->SimilarityMeasure, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet
                                                      , in_param->m_CublasHandle, plan->d_temp_SM, plan->h_temp_SM, plan->d_WeightVector);
    } else if(plan->SimilarityMeasureType == SIMILARITY_MEASURE_MSE){
//      print_and_log("RegTools::RegToolsThread_SimilarityMeasureComputation(), compute MSE\n");
      cutilSafeCall( cudaMemcpy(plan->d_intermediate_images, d_floating_images, fixedImageDim*in_param->m_SimilarityMeasure_NumberOfImageSet*sizeof(float), cudaMemcpyDeviceToDevice) );
      computeMeanSquaredError(plan->d_intermediate_images, plan->d_fixed_images, plan->d_WeightVector, plan->SimilarityMeasure, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet
                                , plan->d_temp_SM, plan->h_temp_SM, in_param->m_CublasHandle);
    } else if(plan->SimilarityMeasureType == SIMILARITY_MEASURE_GC){
      computeGaussianGradientGPUMulti(d_floating_images, plan->ImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, d_floating_Xgrad, d_floating_Ygrad, plan->Sigma
                                  , plan->d_X_kernel_spectrum, plan->d_Y_kernel_spectrum, plan->d_temp_padded, plan->d_temp_spectrum
                                  , *(plan->fftPlanManyFwd), *(plan->fftPlanManyInv), plan->h_GI_threshold);
	  //print_and_log("RegTools::RegToolsThread_SimilarityMeasureComputation(), computing GC1\n");
	  memset(plan->SimilarityMeasure, 0, sizeof(double)*in_param->m_SimilarityMeasure_NumberOfImageSet);
	  //print_and_log("RegTools::RegToolsThread_SimilarityMeasureComputation(), in_param->m_CudaDeviceID: %d, computing GC2\n", in_param->m_CudaDeviceID);

	  zeromean_stddivide_Images(d_floating_Xgrad, plan->d_intermediate_images, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, in_param->m_CublasHandle, plan->d_temp_SM, plan->d_WeightVector
			  , 1.0f/plan->NormalizationFactor, plan->d_mask_weight, d_floating_images);
	  //print_and_log("RegTools::RegToolsThread_SimilarityMeasureComputation(), computing GC3\n");
	  zeromean_stddivide_Images(d_floating_Ygrad, plan->d_intermediate_images, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, in_param->m_CublasHandle, plan->d_temp_SM, plan->d_WeightVector
                      , 1.0f/plan->NormalizationFactor, plan->d_mask_weight, d_floating_images);
	  //print_and_log("RegTools::RegToolsThread_SimilarityMeasureComputation(), computing GC4\n");
	  computeNormalizedCrossCorrelation_Pixels(d_floating_Xgrad, plan->d_fixed_Xgrad, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, in_param->m_CublasHandle);
      computeNormalizedCrossCorrelation_Pixels(d_floating_Ygrad, plan->d_fixed_Ygrad, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, in_param->m_CublasHandle);
	  //print_and_log("RegTools::RegToolsThread_SimilarityMeasureComputation(), computing GC5\n");
	  //cudaSetDevice(in_param->m_CudaDeviceID);
	  computeNormalizedCrossCorrelation_Sum(d_floating_Xgrad, d_floating_Ygrad, plan->d_intermediate_images, plan->SimilarityMeasure, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet
                                                      , in_param->m_CublasHandle, plan->d_temp_SM, plan->h_temp_SM, plan->d_WeightVector);
	  //print_and_log("RegTools::RegToolsThread_SimilarityMeasureComputation(), computed GC\n");
/*	  if (in_param->m_CountNonIntersectedPixel) {
		  // normalization by the number of non-zero pixels
		  for (int i = 0; i<in_param->m_SimilarityMeasure_NumberOfImageSet; i++) {
			  plan->SimilarityMeasure[i] *= (double)(image_size) / (double)(image_size - in_param->m_ZeroPixelCount[i]);
			  //print_and_log("similarity computation: image #%d, number of zero pixels = %d/%d, weight = %f\n", i, in_param->m_ZeroPixelCount[i], image_size, (double)(image_size) / (double)(image_size - in_param->m_ZeroPixelCount[i]));
		  }
	  }
*/
	} else if(plan->SimilarityMeasureType == SIMILARITY_MEASURE_GI_STD_NORM){
/*
      computeGaussianGradientGPUMulti(d_floating_images, plan->ImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, d_floating_Xgrad, d_floating_Ygrad, plan->Sigma
                                  , plan->d_X_kernel_spectrum, plan->d_Y_kernel_spectrum, plan->d_temp_padded, plan->d_temp_spectrum
                                  , *(plan->fftPlanManyFwd), *(plan->fftPlanManyInv), plan->h_GI_threshold);
      computeGradientInformation_StdNorm(d_floating_Xgrad, d_floating_Ygrad, plan->d_fixed_Xgrad, plan->d_fixed_Ygrad, plan->d_intermediate_images
              , static_cast<int>(fixedImageDim), plan->d_mask_weight, isSingleModality, plan->SimilarityMeasure, in_param->m_SimilarityMeasure_NumberOfImageSet
              , in_param->m_CublasHandle, plan->d_temp_SM, plan->h_temp_SM, plan->d_WeightVector, 1.0f/plan->NormalizationFactor, exclusive_norm);
*/
      //memset(plan->SimilarityMeasure, 0, sizeof(double)*in_param->m_SimilarityMeasure_NumberOfImageSet);
      //zeromean_stddivide_Images(d_floating_Xgrad, plan->d_intermediate_images, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, in_param->m_CublasHandle, plan->d_temp_SM, plan->d_WeightVector
      //                , 1.0f/plan->NormalizationFactor, plan->d_mask_weight);
      //zeromean_stddivide_Images(d_floating_Ygrad, plan->d_intermediate_images, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, in_param->m_CublasHandle, plan->d_temp_SM, plan->d_WeightVector
      //                , 1.0f/plan->NormalizationFactor, plan->d_mask_weight);
      //computeNormalizedCrossCorrelation_Pixels(d_floating_Xgrad, plan->d_fixed_Xgrad, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, in_param->m_CublasHandle);
      //computeNormalizedCrossCorrelation_Pixels(d_floating_Ygrad, plan->d_fixed_Ygrad, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet, in_param->m_CublasHandle);
      //computeNormalizedCrossCorrelation_Sum(d_floating_Xgrad, d_floating_Ygrad, plan->d_intermediate_images, plan->SimilarityMeasure, fixedImageDim, in_param->m_SimilarityMeasure_NumberOfImageSet
      //                                                , in_param->m_CublasHandle, plan->d_temp_SM, plan->h_temp_SM, plan->d_WeightVector);
    }
  } else if(plan->h_fixed_images && plan->d_fixed_images){
    // get data from cudaArray to host memory
    cutilSafeCall( cudaMemcpy(plan->h_fixed_images, plan->d_fixed_images, fixedImageDim*sizeof(float), cudaMemcpyDeviceToHost) );
  } else if(plan->h_intermediate_images && plan->d_intermediate_images){
    cutilSafeCall( cudaMemcpy(plan->h_intermediate_images, plan->d_intermediate_images+fixedImageDim*plan->h_get_frame_no, fixedImageDim*sizeof(float), cudaMemcpyDeviceToHost) );
  } else if(plan->h_fixed_Xgrad && plan->d_fixed_Xgrad){
    cutilSafeCall( cudaMemcpy(plan->h_fixed_Xgrad, plan->d_fixed_Xgrad+fixedImageDim*plan->h_get_frame_no, fixedImageDim*sizeof(float), cudaMemcpyDeviceToHost) );
  } else if(plan->h_fixed_Ygrad && plan->d_fixed_Ygrad){
    cutilSafeCall( cudaMemcpy(plan->h_fixed_Ygrad, plan->d_fixed_Ygrad+fixedImageDim*plan->h_get_frame_no, fixedImageDim*sizeof(float), cudaMemcpyDeviceToHost) );
  } else if(plan->h_floating_Xgrad && plan->d_floating_Xgrad){
    cutilSafeCall( cudaMemcpy(plan->h_floating_Xgrad, plan->d_floating_Xgrad+fixedImageDim*plan->h_get_frame_no, fixedImageDim*sizeof(float), cudaMemcpyDeviceToHost) );
  } else if(plan->h_floating_Ygrad && plan->d_floating_Ygrad){
    cutilSafeCall( cudaMemcpy(plan->h_floating_Ygrad, plan->d_floating_Ygrad+fixedImageDim*plan->h_get_frame_no, fixedImageDim*sizeof(float), cudaMemcpyDeviceToHost) );
  } else if(plan->h_fixed_normalized && plan->d_fixed_normalized){
    cutilSafeCall( cudaMemcpy(plan->h_fixed_normalized, plan->d_fixed_normalized+fixedImageDim*plan->h_get_frame_no, fixedImageDim*sizeof(float), cudaMemcpyDeviceToHost) );
  } else if(plan->h_floating_normalized && plan->d_floating_normalized){
    cutilSafeCall( cudaMemcpy(plan->h_floating_normalized, plan->d_floating_normalized+fixedImageDim*plan->h_get_frame_no, fixedImageDim*sizeof(float), cudaMemcpyDeviceToHost) );
  } else if(plan->h_joint_pdf && plan->d_joint_pdf){
    cutilSafeCall( cudaMemcpy(plan->h_joint_pdf, plan->d_joint_pdf+MI_NUMBER_OF_BINS*MI_NUMBER_OF_BINS*plan->h_get_frame_no, MI_NUMBER_OF_BINS*MI_NUMBER_OF_BINS*sizeof(float), cudaMemcpyDeviceToHost) );
  }  else if(plan->h_mask_weight && plan->d_mask_weight){
    cutilSafeCall( cudaMemcpy(plan->h_mask_weight, plan->d_mask_weight, fixedImageDim*sizeof(float), cudaMemcpyDeviceToHost) );
  } else if(plan->d_fixed_images){
    // delete similarity measure computation plan
    if(plan->d_fixed_images)        cutilSafeCall( cudaFree( plan->d_fixed_images ) );
    if(plan->d_fixed_normalized)    cutilSafeCall( cudaFree( plan->d_fixed_normalized ) );
    if(plan->d_floating_normalized) cutilSafeCall( cudaFree( plan->d_floating_normalized ) );
    if(plan->d_pdf_buf)             cutilSafeCall( cudaFree( plan->d_pdf_buf ) );
    if(plan->d_hist_buf)            cutilSafeCall( cudaFree( plan->d_hist_buf ) );
    if(plan->d_joint_pdf)           cutilSafeCall( cudaFree( plan->d_joint_pdf ) );
    if(plan->d_joint_hist)          cutilSafeCall( cudaFree( plan->d_joint_hist ) );
    if(plan->d_fixed_measurement)   cutilSafeCall( cudaFree( plan->d_fixed_measurement ) );
    if(plan->d_intermediate_images) cutilSafeCall( cudaFree( plan->d_intermediate_images ) );
    if(plan->d_fixed_Xgrad)         cutilSafeCall( cudaFree( plan->d_fixed_Xgrad ) );
    if(plan->d_fixed_Ygrad)         cutilSafeCall( cudaFree( plan->d_fixed_Ygrad ) );
    if(plan->d_floating_Xgrad)      cutilSafeCall( cudaFree( plan->d_floating_Xgrad ) );
    if(plan->d_floating_Ygrad)      cutilSafeCall( cudaFree( plan->d_floating_Ygrad ) );
    if(plan->d_temp_images)         cutilSafeCall( cudaFree( plan->d_temp_images ) );
    if(plan->d_temp_padded)         cutilSafeCall( cudaFree( plan->d_temp_padded ) );
    if(plan->d_temp_spectrum)       cutilSafeCall( cudaFree( plan->d_temp_spectrum ) );
    if(plan->d_temp_SM)             cutilSafeCall( cudaFree( plan->d_temp_SM ) );
    if(plan->h_temp_SM)             delete[] plan->h_temp_SM;
    if(plan->d_WeightVector)        cutilSafeCall( cudaFree( plan->d_WeightVector ) );
    if(plan->d_X_kernel_spectrum)   cutilSafeCall( cudaFree( plan->d_X_kernel_spectrum ) );
    if(plan->d_Y_kernel_spectrum)   cutilSafeCall( cudaFree( plan->d_Y_kernel_spectrum ) );
    if(plan->d_mask_weight)         cutilSafeCall( cudaFree( plan->d_mask_weight ) );
    if(plan->fftPlanFwd){
      cufftSafeCall( cufftDestroy(*(plan->fftPlanFwd)) ); delete plan->fftPlanFwd; plan->fftPlanFwd = NULL;
      cufftSafeCall( cufftDestroy(*(plan->fftPlanInv)) ); delete plan->fftPlanInv; plan->fftPlanInv = NULL;
      cufftSafeCall( cufftDestroy(*(plan->fftPlanManyFwd)) ); delete plan->fftPlanManyFwd; plan->fftPlanManyFwd = NULL;
      cufftSafeCall( cufftDestroy(*(plan->fftPlanManyInv)) ); delete plan->fftPlanManyInv; plan->fftPlanManyInv = NULL;
    }
  } else if(plan->h_fixed_images){
    // create similarity measure computation plan
//    print_and_log("create similarity measure computation plan, image dim: (%d, %d, %d), sigma: %f\n", plan->ImageDim[0], plan->ImageDim[1], plan->ImageDim[2], plan->Sigma);
    plan->d_fixed_normalized = plan->d_floating_normalized = NULL;
    plan->d_intermediate_images = plan->d_fixed_Xgrad = plan->d_fixed_Ygrad = plan->d_floating_Xgrad = plan->d_floating_Ygrad = NULL;
    if(!plan->d_fixed_images)    cutilSafeCall( cudaMalloc(&(plan->d_fixed_images), fixedImageDim*sizeof(float)) );
    cutilSafeCall( cudaMemcpy( plan->d_fixed_images, plan->h_fixed_images, fixedImageDim*sizeof(float), cudaMemcpyHostToDevice) );
    
    if((plan->NormalizeMax_fixed==0 && plan->NormalizeMin_fixed==0) || (plan->NormalizeMax_fixed != plan->NormalizeMin_fixed)){
//      print_and_log("prepare for MI/NMI normalization, max = %f, min = %f\n", plan->NormalizeMax_fixed, plan->NormalizeMin_fixed);
      // if both Min and Max are non-zero and equal, we do not prepare for normalization (MI/NMI cannot be computed)
      if(!plan->d_fixed_normalized)    cutilSafeCall( cudaMalloc(&(plan->d_fixed_normalized), fixedImageDim*sizeof(float)) );
      if(!plan->d_floating_normalized) cutilSafeCall( cudaMalloc(&(plan->d_floating_normalized), fixedImageDim*sizeof(float)) );
      if(!plan->d_pdf_buf)             cutilSafeCall( cudaMalloc(&(plan->d_pdf_buf), MI_NUMBER_OF_BINS*sizeof(float)) );
      if(!plan->d_hist_buf)            cutilSafeCall( cudaMalloc(&(plan->d_hist_buf), MI_NUMBER_OF_BINS*sizeof(int)) );
      if(!plan->d_joint_pdf)           cutilSafeCall( cudaMalloc(&(plan->d_joint_pdf), MI_NUMBER_OF_BINS*MI_NUMBER_OF_BINS*sizeof(float)) );
      if(!plan->d_joint_hist)          cutilSafeCall( cudaMalloc(&(plan->d_joint_hist), MI_NUMBER_OF_BINS*MI_NUMBER_OF_BINS*sizeof(int)) );
    }

    if(plan->Sigma>0){
      // if Sigma is negative, we do not prepare for gradient based similarity measure (e.g. GI)
      if(!plan->d_fixed_Xgrad)          cutilSafeCall( cudaMalloc(&(plan->d_fixed_Xgrad), fixedImageDim*sizeof(float)) );
      if(!plan->d_fixed_Ygrad)          cutilSafeCall( cudaMalloc(&(plan->d_fixed_Ygrad), fixedImageDim*sizeof(float)) );
      if(!plan->d_floating_Xgrad)       cutilSafeCall( cudaMalloc(&(plan->d_floating_Xgrad), fixedImageDim*plan->MaxNumImageSets*sizeof(float)) );
      if(!plan->d_floating_Ygrad)       cutilSafeCall( cudaMalloc(&(plan->d_floating_Ygrad), fixedImageDim*plan->MaxNumImageSets*sizeof(float)) );
      unsigned int halfsize = getKernelSize(plan->Sigma);
      unsigned int kernel_size=2*halfsize+1;
      const int fftH = snapTransformSize(plan->ImageDim[1] + kernel_size - 1), fftW = snapTransformSize(plan->ImageDim[0] + kernel_size - 1);
//	  print_and_log("RegToolsThread_SimilarityMeasureComputation(), fftH: %d, fftW: %d, plan->MaxNumImageSets: %d, plan->ImageDim[2]: %d\n", fftH, fftW, plan->MaxNumImageSets, plan->ImageDim[2]);
      if(!plan->d_temp_padded)          cutilSafeCall( cudaMalloc(&(plan->d_temp_padded), fftH*fftW*plan->MaxNumImageSets*plan->ImageDim[2]*2*sizeof(float)) ); // for X and Y gradient
//	  if(!plan->d_temp_padded)          cutilSafeCall( cudaMalloc(&(plan->d_temp_padded), fftH*fftW*plan->MaxNumImageSets*plan->ImageDim[2]*sizeof(float)) ); // for X and Y gradient (removed "*2" on 2020/1/18, I believe this is a redundant malloc)
	  if(!plan->d_temp_spectrum)        cutilSafeCall( cudaMalloc(&(plan->d_temp_spectrum), fftH *(fftW/2+1)*plan->MaxNumImageSets*plan->ImageDim[2]*2 * sizeof(fComplex)) ); // for X and Y gradient
      if(!plan->d_X_kernel_spectrum)    cutilSafeCall( cudaMalloc(&(plan->d_X_kernel_spectrum), fftH * (fftW / 2 + 1) * sizeof(fComplex)) );
      if(!plan->d_Y_kernel_spectrum)    cutilSafeCall( cudaMalloc(&(plan->d_Y_kernel_spectrum), fftH * (fftW / 2 + 1) * sizeof(fComplex)) );
    }
    
    // these are used for NCC, GI, GC
    if(!plan->d_intermediate_images)  cutilSafeCall( cudaMalloc(&(plan->d_intermediate_images), fixedImageDim*plan->MaxNumImageSets*sizeof(float)) );
    if(!plan->d_WeightVector)         cutilSafeCall( cudaMalloc(&(plan->d_WeightVector), fixedImageDim*sizeof(float)) ); // for cublasSgemv

    if(plan->h_mask_weight){
      if(!plan->d_mask_weight)    cutilSafeCall( cudaMalloc(&(plan->d_mask_weight), fixedImageDim*sizeof(float)) );
      cutilSafeCall( cudaMemcpy( plan->d_mask_weight, plan->h_mask_weight, fixedImageDim*sizeof(float), cudaMemcpyHostToDevice) );
    }
    if(!plan->d_temp_SM)              cutilSafeCall( cudaMalloc(&(plan->d_temp_SM), plan->MaxNumImageSets*plan->ImageDim[2]*sizeof(float)) ); // for GI of each image (temporary device memory)
    if(!plan->h_temp_SM)              plan->h_temp_SM = new float[plan->MaxNumImageSets*plan->ImageDim[2]];
    RegToolsThread_UpdateSimilarityMeasureComputationPlan(plan, in_param);

    //RegToolsThread_MemGetInfo( in_param->m_FreeMem, in_param->m_TotalMem );
    //print_and_log("SimilarityMeasure malloc, available memory: %f MB\n", (float)in_param->m_FreeMem/1024.0f/1024.0f);
  }

#if defined(RegTools_ENABLE_SIMILARITY_MEASURE_BENCHMARK)
  // stop timer
  cudaEventRecord(in_param->h_timer_stop, 0);
  cudaEventSynchronize(in_param->h_timer_stop);
  float kernel_running_time = 0;
  cudaEventElapsedTime(&kernel_running_time, in_param->h_timer_start, in_param->h_timer_stop);
  if(in_param->m_ElapsedTime) *(in_param->m_ElapsedTime) += kernel_running_time;
#endif
}

void RegTools::RegToolsThread_UpdateSimilarityMeasureComputationPlan(SimilarityMeasureComputationPlan *plan, RegToolsThreadParam *in_param)
{
  size_t fixedImageDim = plan->ImageDim[0]*plan->ImageDim[1]*plan->ImageDim[2];
  if((plan->NormalizeMax_fixed==0 && plan->NormalizeMin_fixed==0) || (plan->NormalizeMax_fixed != plan->NormalizeMin_fixed)){
    //print_and_log("normalize fixed image for MI/NMI computation, max: %f, min: %f\n", plan->NormalizeMax_fixed, plan->NormalizeMin_fixed)
    // if both Min and Max are non-zero and equal, we do not prepare for normalization (MI/NMI cannot be computed)
    cutilSafeCall( cudaMemcpy( plan->d_fixed_normalized, plan->d_fixed_images, fixedImageDim*sizeof(float), cudaMemcpyDeviceToDevice) );
    normalizeImages(plan->d_fixed_normalized, static_cast<int>(fixedImageDim), plan->NormalizeMax_fixed, plan->NormalizeMin_fixed);
    if(plan->d_mask_weight) maskImages( plan->d_fixed_normalized, plan->d_mask_weight, fixedImageDim, 1, in_param->m_CublasHandle );
    plan->h_fixed_entropy = computeEntropy( plan->d_fixed_normalized, static_cast<int>(fixedImageDim), plan->d_hist_buf, plan->d_pdf_buf);
  }
  //print_and_log("plan->SimilarityMeasureType: %d, plan->MaxNumImageSets: %d\n", plan->SimilarityMeasureType, plan->MaxNumImageSets);
  FillData(plan->d_WeightVector, fixedImageDim, 1.0f); // this initialization is important for some similarity metrics
  if(plan->Sigma>0){
    // if Sigma is negative, we do not prepare for gradient based similarity measure (e.g. GI)
    unsigned int halfsize = getKernelSize(plan->Sigma);
    unsigned int kernel_size=2*halfsize+1;
    const int fftH = snapTransformSize(plan->ImageDim[1] + kernel_size - 1), fftW = snapTransformSize(plan->ImageDim[0] + kernel_size - 1);
    if(plan->fftPlanFwd){
      cufftSafeCall( cufftDestroy(*(plan->fftPlanFwd)) ); delete plan->fftPlanFwd; plan->fftPlanFwd = NULL;
      cufftSafeCall( cufftDestroy(*(plan->fftPlanInv)) ); delete plan->fftPlanInv; plan->fftPlanInv = NULL;
      cufftSafeCall( cufftDestroy(*(plan->fftPlanManyFwd)) ); delete plan->fftPlanManyFwd; plan->fftPlanManyFwd = NULL;
      cufftSafeCall( cufftDestroy(*(plan->fftPlanManyInv)) ); delete plan->fftPlanManyInv; plan->fftPlanManyInv = NULL;
    }
    if(plan->MaxNumImageSets>0){
      plan->fftPlanFwd = new cufftHandle;
      plan->fftPlanInv = new cufftHandle;
      plan->fftPlanManyFwd = new cufftHandle;
      plan->fftPlanManyInv = new cufftHandle;
      cufftSafeCall( cufftPlan2d(plan->fftPlanFwd, fftH, fftW, CUFFT_R2C) );
      cufftSafeCall( cufftPlan2d(plan->fftPlanInv, fftH, fftW, CUFFT_C2R) );
      int n[2] = {fftH, fftW};
//	  print_and_log("fftH: %d, fftW: %d, plan->ImageDim[2]: %d, plan->MaxNumImageSets: %d, n: (%d, %d), plan->m_CudaDeviceID_Sequential: %d, in_param->m_CudaDeviceID: %d, in_param->m_CublasHandle: %d, plan->d_temp_SM: %d\n", fftH, fftW, plan->ImageDim[2], plan->MaxNumImageSets, n[0], n[1], plan->m_CudaDeviceID_Sequential, in_param->m_CudaDeviceID, in_param->m_CublasHandle, plan->d_temp_SM);
      cufftSafeCall( cufftPlanMany( plan->fftPlanManyFwd, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, plan->MaxNumImageSets*plan->ImageDim[2]) );    // For forward FFT, X and Y can share the same spectrum of data
      cufftSafeCall( cufftPlanMany( plan->fftPlanManyInv, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, plan->MaxNumImageSets*plan->ImageDim[2]*2) );  // For inverse FFT, X and Y do NOT share the spectrum of data

      // preprocessing on fixed image specific for each similarity metric
      if(plan->SimilarityMeasureType == SIMILARITY_MEASURE_SSIM){
        cufftSafeCall( cufftPlanMany( plan->fftPlanManyInv, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, plan->MaxNumImageSets*plan->ImageDim[2]) );  // for one gaussian filter
        computeGaussianKernelSpectrum(plan->Sigma, plan->ImageDim[0], plan->ImageDim[1], plan->ImageDim[2]
                                              , plan->d_X_kernel_spectrum, *(plan->fftPlanFwd));
        normalizeImages( plan->d_fixed_images, static_cast<int>(fixedImageDim), 0, 0);
        computeGaussianGPUMulti(plan->d_fixed_images, plan->ImageDim, 1, plan->d_fixed_Xgrad, plan->d_fixed_Ygrad, plan->Sigma
                                    , plan->d_X_kernel_spectrum, plan->d_temp_padded, plan->d_temp_spectrum
                                    , *(plan->fftPlanManyFwd), *(plan->fftPlanManyInv));
        plan->NormalizationFactor = plan->ImageDim[0]*plan->ImageDim[1]; // number of pixels
        if(plan->h_mask_weight) plan->NormalizationFactor -= countZeroPixels(plan->d_mask_weight, fixedImageDim);   // if mask is used, subtract number of zero pixels
        FillData(plan->d_WeightVector, fixedImageDim*plan->MaxNumImageSets, 1.0f/plan->NormalizationFactor);
      } else if(plan->SimilarityMeasureType == SIMILARITY_MEASURE_GC){
        computeGaussianGradientKernelSpectrum(plan->Sigma, plan->ImageDim[0], plan->ImageDim[1], plan->ImageDim[2]
                                              , plan->d_X_kernel_spectrum, plan->d_Y_kernel_spectrum, *(plan->fftPlanFwd));
        computeGaussianGradientGPUMulti(plan->d_fixed_images, plan->ImageDim, 1, plan->d_fixed_Xgrad, plan->d_fixed_Ygrad, plan->Sigma
                                    , plan->d_X_kernel_spectrum, plan->d_Y_kernel_spectrum, plan->d_temp_padded, plan->d_temp_spectrum
                                    , *(plan->fftPlanManyFwd), *(plan->fftPlanManyInv), plan->h_GI_threshold);
        if(plan->d_mask_weight)  MultData(plan->d_fixed_images, plan->d_mask_weight, plan->d_fixed_images, fixedImageDim);

		plan->NormalizationFactor = plan->ImageDim[0] * plan->ImageDim[1] *plan->ImageDim[2];
        if(plan->h_mask_weight) plan->NormalizationFactor -= countZeroPixels(plan->d_mask_weight, fixedImageDim);   // if mask is used, subtract number of zero pixels
        //FillData(plan->d_WeightVector, fixedImageDim, 1.0f/plan->NormalizationFactor);
        zeromean_stddivide_Images(plan->d_fixed_Xgrad, plan->d_intermediate_images, fixedImageDim, 1, in_param->m_CublasHandle, plan->d_temp_SM, plan->d_WeightVector, 1.0f/plan->NormalizationFactor, plan->d_mask_weight, plan->d_mask_weight);
        zeromean_stddivide_Images(plan->d_fixed_Ygrad, plan->d_intermediate_images, fixedImageDim, 1, in_param->m_CublasHandle, plan->d_temp_SM, plan->d_WeightVector, 1.0f/plan->NormalizationFactor, plan->d_mask_weight, plan->d_mask_weight);
      } else if(plan->SimilarityMeasureType == SIMILARITY_MEASURE_GI_STD_NORM){
        computeGaussianGradientKernelSpectrum(plan->Sigma, plan->ImageDim[0], plan->ImageDim[1], plan->ImageDim[2]
                                              , plan->d_X_kernel_spectrum, plan->d_Y_kernel_spectrum, *(plan->fftPlanFwd));
        computeGaussianGradientGPUMulti(plan->d_fixed_images, plan->ImageDim, 1, plan->d_fixed_Xgrad, plan->d_fixed_Ygrad, plan->Sigma
                                    , plan->d_X_kernel_spectrum, plan->d_Y_kernel_spectrum, plan->d_temp_padded, plan->d_temp_spectrum
                                    , *(plan->fftPlanManyFwd), *(plan->fftPlanManyInv), plan->h_GI_threshold);
        if(plan->d_mask_weight)  MultData(plan->d_fixed_images, plan->d_mask_weight, plan->d_fixed_images, fixedImageDim);

        plan->NormalizationFactor = plan->ImageDim[0]*plan->ImageDim[1]*plan->ImageDim[2];
        if(plan->h_mask_weight) plan->NormalizationFactor -= countZeroPixels(plan->d_mask_weight, fixedImageDim);   // if mask is used, subtract number of zero pixels
        //FillData(plan->d_WeightVector, fixedImageDim, 1.0f/plan->NormalizationFactor);
        zeromean_stddivide_Images(plan->d_fixed_Xgrad, plan->d_intermediate_images, fixedImageDim, 1, in_param->m_CublasHandle, plan->d_temp_SM, plan->d_WeightVector, 1.0f/plan->NormalizationFactor, plan->d_mask_weight);
        zeromean_stddivide_Images(plan->d_fixed_Ygrad, plan->d_intermediate_images, fixedImageDim, 1, in_param->m_CublasHandle, plan->d_temp_SM, plan->d_WeightVector, 1.0f/plan->NormalizationFactor, plan->d_mask_weight);
      } else if(plan->SimilarityMeasureType == SIMILARITY_MEASURE_MSE){
        // do nothing
      } else {  // GI
        computeGaussianGradientKernelSpectrum(plan->Sigma, plan->ImageDim[0], plan->ImageDim[1], plan->ImageDim[2]
                                              , plan->d_X_kernel_spectrum, plan->d_Y_kernel_spectrum, *(plan->fftPlanFwd));
        computeGaussianGradientGPUMulti(plan->d_fixed_images, plan->ImageDim, 1, plan->d_fixed_Xgrad, plan->d_fixed_Ygrad, plan->Sigma
                                    , plan->d_X_kernel_spectrum, plan->d_Y_kernel_spectrum, plan->d_temp_padded, plan->d_temp_spectrum
                                    , *(plan->fftPlanManyFwd), *(plan->fftPlanManyInv), plan->h_GI_threshold);
        bool isSingleModality = (plan->SimilarityMeasureType == SIMILARITY_MEASURE_GI_SINGLE);
        int exclusive_norm = (plan->SimilarityMeasureType == SIMILARITY_MEASURE_GI_SINGLE_ALWAYS_FLOATING_NORM) ? 0 : -1;
        computeGradientInformation(plan->d_fixed_Xgrad, plan->d_fixed_Ygrad, plan->d_fixed_Xgrad, plan->d_fixed_Ygrad, plan->d_intermediate_images
                , static_cast<int>(fixedImageDim), plan->d_mask_weight, isSingleModality, &(plan->NormalizationFactor), 1
                , in_param->m_CublasHandle, plan->d_temp_SM, plan->h_temp_SM, plan->d_WeightVector, 1.0, exclusive_norm);
      }
      //print_and_log("NormalizationFactor = %f\n", plan->NormalizationFactor);
    }
  }
  if(plan->SimilarityMeasureType == SIMILARITY_MEASURE_NCC){
    FillData(plan->d_WeightVector, fixedImageDim, 1.0f); // this initialization is important for some similarity metrics
    plan->NormalizationFactor = plan->ImageDim[0]*plan->ImageDim[1]*plan->ImageDim[2];
    //FillData(plan->d_WeightVector, fixedImageDim, 1.0f/plan->NormalizationFactor);
    zeromean_stddivide_Images(plan->d_fixed_images, plan->d_intermediate_images, fixedImageDim, 1, in_param->m_CublasHandle, plan->d_temp_SM, plan->d_WeightVector, 1.0f/plan->NormalizationFactor, plan->d_mask_weight);
    print_and_log("NormalizationFactor = %f\n", plan->NormalizationFactor);
  }
}

void RegTools::copyDataToCudaArray(struct cudaArray *d_array, float *src, int dst_index, int src_index, cudaExtent dimension, int numVolumes, cudaMemcpyKind kind)
{
  // copy host/device data to 3D cudaArray
  // Note: this is a very slow operation because of internal reordering (space-filling curving) for cudaArray 
  // (see, for example, http://www.cudahandbook.com/uploads/Chapter_10._Texturing.pdf for detail)

  int image_size = static_cast<int>(dimension.width*dimension.height);
  dimension.depth *= numVolumes;
//  print_and_log("RegTools::copyDataToCudaArray(), dimension: (%d, %d, %d), dst_index: %d, src_index: %d, numVolumes: %d\n", 
//    dimension.width, dimension.height, dimension.depth, dst_index, src_index, numVolumes);
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr((void*)(src+src_index*image_size), dimension.width*sizeof(float), dimension.width, dimension.height);
  copyParams.dstArray = d_array;
  copyParams.dstPos.x = 0;
  copyParams.dstPos.y = 0;
  copyParams.dstPos.z = dst_index;
  copyParams.extent   = dimension;
  copyParams.kind     = kind;
  cutilSafeCall( cudaMemcpy3D(&copyParams) );
  cutilSafeCall( cudaThreadSynchronize() );
}

void RegTools::RegToolsThread_SetPBO(unsigned int pbo, cudaGraphicsResource **resource)
{
  if(*resource)    cutilSafeCall(cudaGraphicsUnregisterResource(*resource));
  else             cutilSafeCall(cudaGraphicsGLRegisterBuffer(resource, pbo, cudaGraphicsMapFlagsNone));
}

void RegTools::RegToolsThread_MultVolume(RegToolsThreadParam *in_param)
{
  int array_size = in_param->m_VolumeDim.width*in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth;
  MultScalar( in_param->m_VolumePlan_cudaArray->d_raw_volume[in_param->m_CudaDeviceID_Sequential], in_param->m_ScalarVal, array_size );
}

void RegTools::RegToolsThread_CMAESPopulation(RegToolsThreadParam *in_param)
{
  // generate normally distributed random numbers
  float *d_arz = in_param->m_VolumePlan_cudaArray->d_raw_volume[in_param->m_CudaDeviceID_Sequential];
  randn_CURAND(in_param->m_CurandGenerator, d_arz, in_param->m_VolumeDim.width*in_param->m_VolumeDim.height);

  // compute mean + arz * std
  float *d_OneVector;
  cutilSafeCall( cudaMalloc(&d_OneVector, in_param->m_VolumeDim.height) );
  FillData(d_OneVector, in_param->m_VolumeDim.height, 1.0f);
  generateCMAESPopulation(d_arz, in_param->d_arx, in_param->d_arxvalid, in_param->d_CMAES_xmean, in_param->d_CMAES_diagD
    , in_param->d_CMAES_lbounds, in_param->d_CMAES_ubounds, d_OneVector, in_param->m_CublasHandle, in_param->m_VolumeDim.width, in_param->m_VolumeDim.height);
  cutilSafeCall( cudaFree(d_OneVector) );
}

void RegTools::RegToolsThread_CopyDeviceMemoryToCudaArray(RegToolsThreadParam *in_param)
{
  int dst_index = in_param->m_VolumePlan_cudaArray_out->volumeIndex * in_param->m_VolumeDim.depth;
  int src_index = in_param->m_VolumePlan_cudaArray->volumeIndex * in_param->m_VolumeDim.depth;
  copyDataToCudaArray(in_param->m_VolumePlan_cudaArray_out->d_volume[in_param->m_CudaDeviceID_Sequential], in_param->m_VolumePlan_cudaArray->d_raw_volume[in_param->m_CudaDeviceID_Sequential], 
                        dst_index, src_index, in_param->m_VolumeDim, in_param->m_NumVolumes, cudaMemcpyDeviceToDevice );
}

void RegTools::RegToolsThread_CopyDeviceMemoryToCudaArray_Multi(RegToolsThreadParam *in_param)
{
  // copy one volume in device memory to one cudaArray (consider all volumes as a whole)
  int dev_id = in_param->m_CudaDeviceID_Sequential;
  if(in_param->m_NumCudaArrays==0)  return;
  //cudaExtent extent = make_cudaExtent(in_param->m_VolumePlans[0]->VolumeDim[0], in_param->m_VolumePlans[0]->VolumeDim[1], in_param->m_VolumePlans[0]->VolumeDim[2]*in_param->m_VolumePlans[0]->numVolumes);
  cudaExtent extent = make_cudaExtent(in_param->m_VolumePlans[0]->VolumeDim[0], in_param->m_VolumePlans[0]->VolumeDim[1], in_param->m_VolumePlans[0]->VolumeDim[2]);
  for(int i=0;i<in_param->m_NumCudaArrays;i++){
//    print_and_log("RegToolsThread_CopyDeviceMemoryToCudaArray_Multi(), src_index = %d, extent: (%d, %d, %d)\n", i*in_param->m_VolumePlan_cudaArray->VolumeDim[2], extent.width, extent.height, extent.depth);
    //print_and_log("RegToolsThread_CopyDeviceMemoryToCudaArray_Multi(), src extent: (%d, %d, %d, %d)\n", in_param->m_VolumePlans[i]->VolumeDim[0], in_param->m_VolumePlans[i]->VolumeDim[1], in_param->m_VolumePlans[i]->VolumeDim[2], in_param->m_VolumePlans[i]->numVolumes);
    copyDataToCudaArray(in_param->m_VolumePlans[i]->d_volume[dev_id], in_param->m_VolumePlan_cudaArray->d_raw_volume[dev_id], 
                        0, i*in_param->m_VolumePlan_cudaArray->VolumeDim[2], extent, 1, cudaMemcpyDeviceToDevice );
  }
}

void RegTools::RegToolsThread_GetGPUProjection(RegToolsThreadParam *in_param)
{
  int num_projections_in_one_set = in_param->m_NumEnabledProjections/in_param->m_NumProjectionSets;
  int array_size = in_param->m_ProjectionWidth*in_param->m_ProjectionHeight*num_projections_in_one_set;
  int set_get_offset = array_size * in_param->m_ProjectionSetIndex;
//  print_and_log("RegTools::RegToolsThread_GetGPUProjection(), m_NumProjectionSets: %d, m_NumEnabledProjections: %d, m_Projections: %d, array_size: %d, set_get_offset: %d\n"
//    , in_param->m_NumProjectionSets, in_param->m_NumEnabledProjections, in_param->m_Projections, array_size, set_get_offset);
  // get data from default projection to host memory
  if(!in_param->d_Projections){
    print_and_log("RegToolsThread_GetGPUProjection(), d_Projections is NULL\n");
    return;
  }
  cutilSafeCall( cudaMemcpy(in_param->m_Projections, in_param->d_Projections+set_get_offset, array_size*sizeof(float), cudaMemcpyDeviceToHost) );
}

void RegTools::RegToolsThread_VolumePlan(RegToolsThreadParam *in_param)
{
  int dev_id = in_param->m_CudaDeviceID_Sequential;
  int array_size = in_param->m_VolumeDim.width*in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth*in_param->m_VolumePlan_cudaArray->numVolumes;
  int set_get_offset = in_param->m_VolumeDim.width*in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth*in_param->m_VolumePlan_cudaArray->volumeIndex;
//  print_and_log("RegTools::RegToolsThread_VolumePlan(), volumeIndex = %d, set_get_offset = %d\n", in_param->m_VolumePlan_cudaArray->volumeIndex, set_get_offset);
  if(in_param->m_VolumePlan_cudaArray->h_volume_set){
    // set data from host memory to cudaArray 
    cutilSafeCall( cudaMemcpy(in_param->m_VolumePlan_cudaArray->d_raw_volume[dev_id]+set_get_offset, in_param->m_VolumePlan_cudaArray->h_volume_set, array_size*sizeof(float), cudaMemcpyHostToDevice) );
  } else if(in_param->m_VolumePlan_cudaArray->h_volume){
    if(in_param->m_VolumePlan_cudaArray->d_raw_volume[dev_id]){
      // get data from cudaArray to host memory
      cutilSafeCall( cudaMemcpy(in_param->m_VolumePlan_cudaArray->h_volume, in_param->m_VolumePlan_cudaArray->d_raw_volume[dev_id]+set_get_offset, array_size*sizeof(float), cudaMemcpyDeviceToHost) );
    } else if(in_param->d_Volume) {
//      print_and_log("RegToolsThread_VolumePlan(), get default volume from GPU, (%d, %d, %d), volumendex: %d, numVolumes: %d\n", 
//        in_param->m_VolumeDim.width, in_param->m_VolumeDim.height, in_param->m_VolumeDim.depth, in_param->m_VolumePlan_cudaArray->volumeIndex, in_param->m_VolumePlan_cudaArray->numVolumes);
      // get data from default volume to host memory
      cutilSafeCall( cudaMemcpy(in_param->m_VolumePlan_cudaArray->h_volume, in_param->d_Volume+set_get_offset, array_size*sizeof(float), cudaMemcpyDeviceToHost) );
    }
  } else {
    // create volume plan
    cutilSafeCall( cudaMalloc((void**)&(in_param->m_VolumePlan_cudaArray->d_raw_volume[dev_id]), array_size*sizeof(float)) );
    cutilSafeCall( cudaMemcpy(in_param->m_VolumePlan_cudaArray->d_raw_volume[dev_id], in_param->m_Volume, array_size*sizeof(float), cudaMemcpyHostToDevice) );
  }
}

void RegTools::RegToolsThread_VolumePlan_cudaArray(RegToolsThreadParam *in_param)
{
  int dev_id = in_param->m_CudaDeviceID_Sequential;
  int array_size = in_param->m_VolumeDim.width*in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth*in_param->m_NumVolumes;
  int set_get_offset_slice = in_param->m_VolumeDim.depth*in_param->m_VolumePlan_cudaArray->volumeIndex;
  if(in_param->m_VolumePlan_cudaArray->h_volume_set){
    // set data from host memory to cudaArray 
    copyDataToCudaArray(in_param->m_VolumePlan_cudaArray->d_volume[dev_id], in_param->m_VolumePlan_cudaArray->h_volume_set, set_get_offset_slice, 0, in_param->m_VolumeDim, 1, cudaMemcpyHostToDevice);
  } else if(in_param->m_VolumePlan_cudaArray->h_volume){
//    print_and_log("RegTools::RegToolsThread_VolumePlan_cudaArray(), get data from cudaArray, volumeIndex = %d, set_get_offset_slice = %d, NumVolumes: %d\n", 
//      in_param->m_VolumePlan_cudaArray->volumeIndex, set_get_offset_slice, in_param->m_NumVolumes);
    // get data from cudaArray to host memory
    copyDataFromCudaArray(in_param->m_VolumePlan_cudaArray->h_volume, in_param->m_VolumePlan_cudaArray->d_volume[dev_id], 0, set_get_offset_slice, in_param->m_VolumeDim, 1);
  } else if(in_param->m_VolumePlan_cudaArray->d_volume[dev_id]){
    // delete volume plan with cudaArray
    cutilSafeCall( cudaFreeArray( in_param->m_VolumePlan_cudaArray->d_volume[dev_id] ) );
    in_param->m_VolumePlan_cudaArray->d_volume[dev_id] = NULL;
  } else if(in_param->m_VolumePlan_cudaArray->d_raw_volume[dev_id]){
    // delete volume plan with raw volume
    cutilSafeCall( cudaFree( in_param->m_VolumePlan_cudaArray->d_raw_volume[dev_id] ) );
    in_param->m_VolumePlan_cudaArray->d_raw_volume[dev_id] = NULL;
  } else {
    // create interpolator plan
//    print_and_log("create interpolator plan (GPU:%d), volume dim: (%d, %d, %d), voxel size: (%f, %f, %f)\n"
//          , dev_id, in_param->m_VolumeDim.width, in_param->m_VolumeDim.height, in_param->m_VolumeDim.depth, in_param->m_VoxelSize_mm.x, in_param->m_VoxelSize_mm.y, in_param->m_VoxelSize_mm.z);
    cudaMallocForTexture(&(in_param->m_VolumePlan_cudaArray->d_volume[dev_id]), make_cudaExtent(in_param->m_VolumeDim.width, in_param->m_VolumeDim.height, in_param->m_VolumeDim.depth*in_param->m_NumVolumes));
    copyDataToCudaArray( in_param->m_VolumePlan_cudaArray->d_volume[dev_id], in_param->m_Volume, 0, 0, in_param->m_VolumeDim, in_param->m_NumVolumes, cudaMemcpyHostToDevice);
  }
}

void RegTools::RegToolsThread_LCNComputationPlan(RegToolsThreadParam *in_param)
{
	int dev_id = in_param->m_CudaDeviceID_Sequential;
	int *p = in_param->m_LCNComputationPlan->ProjectionDim;
	unsigned int halfsize = getKernelSize(in_param->m_LCNComputationPlan->LCN_sigma);
	unsigned int kernel_size = 2 * halfsize + 1;
	const int fftH = snapTransformSize(p[1] + kernel_size - 1), fftW = snapTransformSize(p[0] + kernel_size - 1);
	int n[2] = { fftH, fftW };
	RegToolsThread_MemGetInfo(in_param->m_FreeMem, in_param->m_TotalMem);
	print_and_log("RegToolsThread_LCNComputationPlan(), before, available memory: %f MB, projection_dim: (%d,%d,%d), NumProjectionSets: %d, LCN_sigma: %f\n", (float)in_param->m_FreeMem / 1024.0f / 1024.0f, p[0], p[1], p[2], in_param->m_LCNComputationPlan->NumProjectionSets, in_param->m_LCNComputationPlan->LCN_sigma);
	print_and_log("RegToolsThread_LCNComputationPlan(), fftH: %d, fftW: %d, kernel_size: %d\n", fftH, fftW, kernel_size);
	if (in_param->m_LCNComputationPlan->create_flag) {
		print_and_log("create LCN computation plan\n");
//		in_param->m_LCNComputationPlan->fftPlanLCNFwd[dev_id] = new cufftHandle;
//		in_param->m_LCNComputationPlan->fftPlanLCNManyFwd[dev_id] = new cufftHandle;
//		in_param->m_LCNComputationPlan->fftPlanLCNManyInv[dev_id] = new cufftHandle;
		cufftSafeCall(cufftPlan2d(&in_param->m_LCNComputationPlan->fftPlanLCNFwd[dev_id], fftH, fftW, CUFFT_R2C));
		cufftSafeCall(cufftPlanMany(&in_param->m_LCNComputationPlan->fftPlanLCNManyFwd[dev_id], 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, p[2] * in_param->m_LCNComputationPlan->NumProjectionSets));  // For forward FFT
		cufftSafeCall(cufftPlanMany(&in_param->m_LCNComputationPlan->fftPlanLCNManyInv[dev_id], 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, p[2] * in_param->m_LCNComputationPlan->NumProjectionSets));  // For forward FFT
	}
	else {
		print_and_log("delete LCN computation plan\n");
		cufftSafeCall(cufftDestroy(in_param->m_LCNComputationPlan->fftPlanLCNFwd[dev_id]));	   //delete in_param->m_LCNComputationPlan->fftPlanLCNFwd[dev_id]; in_param->m_LCNComputationPlan->fftPlanLCNFwd[dev_id] = NULL;
		cufftSafeCall(cufftDestroy(in_param->m_LCNComputationPlan->fftPlanLCNManyFwd[dev_id])); //delete in_param->m_LCNComputationPlan->fftPlanLCNManyFwd[dev_id]; in_param->m_LCNComputationPlan->fftPlanLCNManyFwd[dev_id] = NULL;
		cufftSafeCall(cufftDestroy(in_param->m_LCNComputationPlan->fftPlanLCNManyInv[dev_id])); //delete in_param->m_LCNComputationPlan->fftPlanLCNManyInv[dev_id]; in_param->m_LCNComputationPlan->fftPlanLCNManyInv[dev_id] = NULL;
	}
	RegToolsThread_MemGetInfo(in_param->m_FreeMem, in_param->m_TotalMem);
	print_and_log("RegToolsThread_LCNComputationPlan(), after, available memory: %f MB, projection_dim: (%d,%d,%d), LCN_sigma: %f\n", (float)in_param->m_FreeMem / 1024.0f / 1024.0f, p[0], p[1], p[2], in_param->m_LCNComputationPlan->LCN_sigma);
}

void RegTools::RegToolsThread_CopyVarToCudaMemory(RegToolsThreadParam *in_param)
{
  copyProjectorMode( in_param->m_ProjectorMode );
  copyVoxelSize( in_param->m_VoxelSize_mm );
  copyVolumeDimensions( (int)in_param->m_VolumeDim.width, (int)in_param->m_VolumeDim.height, (int)in_param->m_VolumeDim.depth );
  float3 vol_dim = make_float3(static_cast<float>(in_param->m_VolumeDim.width), static_cast<float>(in_param->m_VolumeDim.height), static_cast<float>(in_param->m_VolumeDim.depth));
  
  if(in_param->m_DifferentVolumePerProjectionSet){
    // TODO: clean-up this interface  
    copyVolumeDimensions( (int)in_param->m_VolumeDim.width, (int)in_param->m_VolumeDim.height, (int)in_param->m_VolumeDim.depth/in_param->m_NumProjectionSets );
    vol_dim.z /= in_param->m_NumProjectionSets;
  }
  
  copyVolumeCorner( (vol_dim*in_param->m_VoxelSize_mm)/2 );
  copyProjectionDim( in_param->m_ProjectionWidth, in_param->m_ProjectionHeight );
  copyNumberOfProjectionSets( in_param->m_NumProjectionSets );
  copyStepSize( in_param->m_StepSize );
  copyRayCastingLOD( in_param->m_RayCastingLOD );
  copyRayCastingThreshold( in_param->m_RayCastingThreshold );
  copyRayCastingDistanceFalloffCoefficient( in_param->m_RayCastingDistanceFalloffCoefficient );
  copyCountNonIntersectedPixel( in_param->m_CountNonIntersectedPixel );
  copyDifferentVolumePerProjectionSet( in_param->m_DifferentVolumePerProjectionSet );
  copyDepthMapBack(in_param->m_DepthMapBack);
}

void RegTools::RegToolsThread_RunInterpolator(RegToolsThreadParam *in_param, float *d_Result)
{
	// run interpolator
	cudaEventRecord(in_param->h_timer_start, 0);
	if (!d_Result) { print_and_log("error at RegTools::RegToolsThread_RunInterpolator()\n"); return; }
	initCudaTexture(false, false);  // non-normalized, point (nearest-neighbor) interpolation mode

	launch_Interpolator_BBoxCheck(d_Result, in_param->m_VolumePlan_cudaArray->VolumeDim, in_param->m_VolumePlan_cudaArray->VoxelSize
		, in_param->m_Interpolator_transform, in_param->m_Interpolator_num_transform_element, in_param->m_Interpolator_type, in_param->m_Interpolator_order, in_param->m_Interpolator_bicubic_a
		, in_param->m_Interpolator_back_ground, in_param->m_Interpolator_volume_center, in_param->m_Interpolator_scattered_pnts
		, in_param->m_Interpolator_num_scattered_pnts, in_param->m_Interpolator_IsWarp, in_param->m_Interpolator_num_transforms);

	float kernel_running_time;
	cudaEventRecord(in_param->h_timer_stop, 0);
	cudaEventSynchronize(in_param->h_timer_stop);
	cudaEventElapsedTime(&kernel_running_time, in_param->h_timer_start, in_param->h_timer_stop);
	if (in_param->m_ElapsedTime) *(in_param->m_ElapsedTime) += kernel_running_time;

	//  print_and_log("m_VolumeDim: %d, %d, %d, m_NumProjectionSets: %d, d_Result: %d, m_Projections: %d\n", in_param->m_VolumeDim.width, in_param->m_VolumeDim.height, in_param->m_VolumeDim.depth, 
	//    in_param->m_NumProjectionSets, d_Result, in_param->m_Projections);
	if (in_param->m_Projections){
		int num_voxels = in_param->m_VolumeDim.width*in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth;
	    cutilSafeCall(cudaMemcpy(in_param->m_Projections, d_Result, num_voxels*in_param->m_NumProjectionSets * sizeof(float), cudaMemcpyDeviceToHost));
		//float sum = 0;
		//for (int i = 0; i < num_voxels; i++) sum += in_param->m_Projections[i];
		//print_and_log("RegTools::RegToolsThread_RunInterpolator(), in_param->m_Projections, sum: %f\n", sum);
	}
	/*
	int num_voxels = in_param->m_VolumeDim.width*in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth*in_param->m_NumProjectionSets;
	float *temp = new float[num_voxels];
	cutilSafeCall(cudaMemcpy(temp, d_Result, num_voxels * sizeof(float), cudaMemcpyDeviceToHost));
	float sum = 0;
	for (int i = 0; i < num_voxels; i++) sum += temp[i];
	print_and_log("RegTools::RegToolsThread_RunInterpolator(), sum: %f\n", sum);
	free(temp);
	*/
}

void RegTools::RegToolsThread_ComputeLinearCombination(RegToolsThreadParam *in_param)
{
  //
  // compute linear combination of deformation modes and copy the resulted warp to texture memory
  // d_ModeArray: (m x k) array (m_NumModeDims arrays are stacked), m: number of voxels, k: number of volumes (subjects)
  // d_ModeWeights: (k x n) array, n: number of mode weights
  // d_WarpArray: (m x n) array (m_NumModeDims arrays are stacked)
  //

  // error check (number of modes in the array is not necessarily equal to the number of mode weights)
  if(in_param->m_ModeArray_NumberOfModes<in_param->m_NumModes){
    print_and_log("Error in RegTools::RegToolsThread_ComputeLinearCombination(), number of modes are larger than array\n");
    return;
  }
//  print_and_log("RegTools::RegToolsThread_ComputeLinearCombination()\n");
//  print_and_log("m_NumModeDims: %d\n", in_param->m_NumModeDims);
//  cudaEventRecord(in_param->h_timer_start, 0);
  int volume_size = in_param->m_VolumeDim.width*in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth;
  float alpha = 1.0f, beta = 0.0f;
  // compute linear combination for X, Y, and Z component separately (this makes the later memcpy faster)
  for(int i=0;i<in_param->m_NumModeDims;i++)
    cublasStatus_t status = cublasSgemm(in_param->m_CublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, volume_size, in_param->m_NumVolumes, in_param->m_NumModes, 
                        &alpha, in_param->d_ModeArray+i*volume_size*in_param->m_ModeArray_NumberOfModes, volume_size, in_param->d_ModeWeights, in_param->m_NumModes, 
                        &beta, in_param->d_WarpArray+i*volume_size*in_param->m_NumVolumes, volume_size);
  
//  float kernel_running_time;
//  cudaEventRecord(in_param->h_timer_stop, 0);
//  cudaEventSynchronize(in_param->h_timer_stop);
//  cudaEventElapsedTime(&kernel_running_time, in_param->h_timer_start, in_param->h_timer_stop);
//  print_and_log("cublasSgemm() took %f msec\n", kernel_running_time);
//  print_and_log("volume_size: %d, in_param->m_VolumeDim: %d, %d, %d, cublasHandle: %d, numModes: %d (def_mode_array: %d), numModeDims: %d, numVolumes: %d\n", 
//    volume_size, in_param->m_VolumeDim.width, in_param->m_VolumeDim.height, in_param->m_VolumeDim.depth, 
//    in_param->m_CublasHandle, in_param->m_NumModes, in_param->m_ModeArray_NumberOfModes, in_param->m_NumModeDims, in_param->m_NumVolumes);

//  RegToolsThread_MemGetInfo( in_param->m_FreeMem, in_param->m_TotalMem );
//  print_and_log("RegToolsThread_ComputeLinearCombination(), available memory: %f MB\n", (float)in_param->m_FreeMem/1024.0f/1024.0f);
}

CUT_THREADPROC RegTools::RegToolsThread(RegToolsThreadParam *in_param)
{
  RegToolsThread_startup(in_param);
  while(1){   // this thread should be terminated by setting in_param->m_ThreadCompleted to 'true'
    // set m_ProjectorCompletedEvent event to tell host process that this process is waiting for a singnal
#if _WIN32
    ::SetEvent(in_param->m_ProjectorCompletedEvent);
    ::WaitForSingleObject(in_param->m_ProjectorDataReadyEvent, INFINITE);
    ::ResetEvent(in_param->m_ProjectorDataReadyEvent);
    if(in_param->m_ThreadCompleted) break;
    RegToolsThread_main(in_param);
#endif
  }
  RegToolsThread_cleanup(in_param);
//  ::SetEvent(in_param->m_ProjectorCompletedEvent);
  CUT_THREADEND;
}

void RegTools::RegToolsThread_startup(RegToolsThreadParam *in_param)
{
  // initialize necessary field of RegToolsThreadParam
  in_param->d_ProjectionTextureArray = in_param->d_VolumeTextureArray = NULL;
  in_param->d_Projections = in_param->d_Volume = in_param->d_Volume2 = NULL;  // global memory space to keep projection images and volume (projection results are stored here)
  in_param->previousResultWidth = in_param->previousResultHeight = -1;
  in_param->previousResultWidth2 = in_param->previousResultHeight2 = -1;

  in_param->m_CudaDeviceID = InitializeCudaDevice(in_param->m_CudaDeviceID, in_param->m_CudaDeviceLoad, in_param->m_WithGL, in_param->m_Messages, in_param->m_MessagesPtr);

  // cublas is used for efficient computation of similarity metrics
  cublasStatus_t stat = cublasCreate(&(in_param->m_CublasHandle));
  if (stat != CUBLAS_STATUS_SUCCESS) {
    print_and_log("CUBLAS initialization failed\n");
  } else {
    print_and_log("CUBLAS initialization successful\n");
  }

  // curand is used for random number generation in CMA-ES
  CURAND_CALL( curandCreateGenerator(&(in_param->m_CurandGenerator), CURAND_RNG_PSEUDO_DEFAULT) );
  CURAND_CALL( curandSetPseudoRandomGeneratorSeed(in_param->m_CurandGenerator, 1234ULL) );

  // prepare cuda timer
  cudaEventCreate(&in_param->h_timer_start);
  cudaEventCreate(&in_param->h_timer_stop);
}

void RegTools::RegToolsThread_Initialize(RegToolsThreadParam *in_param)
{
  // deprecated (do nothing)
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("GPU #%d is initialized\n", in_param->m_CudaDeviceID);
  ::Sleep(200); // just to make sure everything is printed
#endif
}

void RegTools::RegToolsThread_InitializeMemory(RegToolsThreadParam *in_param)
{
  unBindCUDATexture();
  unBindWarpTexture();
  
  if (in_param->d_ProjectionTextureArray)
  {
    unBindCUDATexture();
    cutilSafeCall( cudaFreeArray(in_param->d_ProjectionTextureArray) );
    in_param->d_ProjectionTextureArray = NULL;
  }
  
  if (in_param->d_VolumeTextureArray)
  {
    unBindCUDATexture();
    cutilSafeCall( cudaFreeArray(in_param->d_VolumeTextureArray) );
    in_param->d_VolumeTextureArray = NULL;
  }

  if (in_param->d_Projections)
  {
    cutilSafeCall( cudaFree( in_param->d_Projections ) );
    in_param->d_Projections = NULL;
  }
  
  if (in_param->d_Volume)
  {
    cutilSafeCall( cudaFree( in_param->d_Volume ) );
    in_param->d_Volume = NULL;
  }
  
  if (in_param->d_Volume2)
  {
    cutilSafeCall( cudaFree( in_param->d_Volume2 ) );
    in_param->d_Volume2 = NULL;
  }

  // in_param->m_ProjectionsInit is managed by the library user (e.g. MATLAB)

  if (in_param->d_ProjectionsInit)
  {
    cutilSafeCall( cudaFree( in_param->d_ProjectionsInit ) );
    in_param->d_ProjectionsInit = NULL;
  }

  in_param->d_ProjectionsInit_capacity_bytes = 0;

  in_param->previousResultWidth  = in_param->previousResultHeight  = -1;
  in_param->previousResultWidth2 = in_param->previousResultHeight2 = -1;
}

void RegTools::RegToolsThread_cleanup(RegToolsThreadParam *in_param)
{
  // cleanup process
//  print_and_log("cleanup process for device %d\n", in_param->m_CudaDeviceID)
  cudaEventDestroy(in_param->h_timer_start);
  cudaEventDestroy(in_param->h_timer_stop);

  RegToolsThread_InitializeMemory(in_param);
  cublasDestroy(in_param->m_CublasHandle);
  CURAND_CALL(curandDestroyGenerator(in_param->m_CurandGenerator));
  // cleanup CUDA context
#if (CUDA_VERSION_MAJOR>=4)
  cudaDeviceReset();
#else
  CUcontext contextID;
  cuCtxPopCurrent( &contextID );
  cuCtxDestroy( contextID );
#endif
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("Cleaning up projector thread\n");
  ::Sleep(100); // just to make sure everything is printed
#endif
}

void RegTools::ConstructProjectionParameterArray(int index, struct ProjectionParameters *projectionParam
                                                  , double *w_v_col /* T_World_Volume */, double *volume_size, int u_pix, int v_pix
                                                  , float *h_PreComputedMatrix_array)
{
  // transform source point using T_Volume_World
  double source_position[3], *pm_row = projectionParam->ProjectionMatrix_3x4;
  double in_col[9], inv_in_col[9]; // left-hand side matrix (3x3) of projection matrix in column-major

  if(w_v_col){
    double new_pm_row[12];
    // concatenate T_World_Volume (column-major) to 3x4 projection matrix (row-major)
    for(int i=0;i<3;i++) for(int j=0;j<4;j++) new_pm_row[i*4+j] = pm_row[i*4]*w_v_col[j*4] + pm_row[i*4+1]*w_v_col[j*4+1] + pm_row[i*4+2]*w_v_col[j*4+2] + pm_row[i*4+3]*w_v_col[j*4+3];
    in_col[0] = new_pm_row[0]; in_col[3] = new_pm_row[1]; in_col[6] = new_pm_row[2];
    in_col[1] = new_pm_row[4]; in_col[4] = new_pm_row[5]; in_col[7] = new_pm_row[6];
    in_col[2] = new_pm_row[8]; in_col[5] = new_pm_row[9]; in_col[8] = new_pm_row[10];
    RegTools::ComputeSourcePosition(new_pm_row, source_position);
  } else {
//    print_and_log("T_World_Volume was not specified\n");
    in_col[0] = pm_row[0]; in_col[3] = pm_row[1]; in_col[6] = pm_row[2];
    in_col[1] = pm_row[4]; in_col[4] = pm_row[5]; in_col[7] = pm_row[6];
    in_col[2] = pm_row[8]; in_col[5] = pm_row[9]; in_col[8] = pm_row[10];
    RegTools::ComputeSourcePosition(pm_row, source_position);
  }

  float *p = h_PreComputedMatrix_array+12*index;
  for(int i=0;i<3;i++) p[9+i] = (float)source_position[i];   // float -> double conversion
//  print_and_log("source_pos = (%f, %f, %f)\n", source_pos.x, source_pos.y, source_pos.z);

  RegTools::Inverse3x3d(in_col, inv_in_col);
  p[0] = inv_in_col[6]; p[1] = inv_in_col[7]; p[2] = inv_in_col[8];                                        // left-bottom corner (at unit distance plane)
  p[3] = inv_in_col[0]*(float)u_pix; p[4] = inv_in_col[1]*(float)u_pix; p[5] = inv_in_col[2]*(float)u_pix; // left-bottom -> right-bottom vector (at unit distance plane)
  p[6] = inv_in_col[3]*(float)v_pix; p[7] = inv_in_col[4]*(float)v_pix; p[8] = inv_in_col[5]*(float)v_pix; // left-bottom -> left-top vector
#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("source_pos = (%f, %f, %f)\n", source_position[0], source_position[1], source_position[2]);
  for(int i=0;i<3;i++)  print_and_log("p[%d] = (%f, %f, %f)\n", i, p[i*3+0], p[i*3+1], p[i*3+2]);
#endif
}

void RegTools::ComputeBoxProjection(double *pm, double *box_center, double *box_size, double *bsquare_min, double *bsquare_max)
{
  bsquare_min[0] = DBL_MAX; bsquare_min[1] = DBL_MAX; bsquare_max[0] = -DBL_MAX; bsquare_max[1] = -DBL_MAX;
  double vert_w[8][3] = { {-1,-1,-1}, {-1,-1, 1}, {-1, 1,-1}, {-1, 1, 1}, { 1,-1,-1}, { 1,-1, 1}, { 1, 1,-1}, { 1, 1, 1} }, vert[3], projected_vert[3];
  for(int j=0;j<8;j++){
    for(int k=0;k<3;k++)  vert[k] = box_center[k]+box_size[k]/2*vert_w[j][k];
    // project one vertex
    for(int k=0;k<3;k++)  projected_vert[k] = pm[k*4+0]*vert[0] + pm[k*4+1]*vert[1] + pm[k*4+2]*vert[2] + pm[k*4+3];
    projected_vert[0] /= projected_vert[2]; projected_vert[1] /= projected_vert[2];
    // find the bounding square
    if(projected_vert[0] < bsquare_min[0])  bsquare_min[0] = projected_vert[0];
    if(projected_vert[1] < bsquare_min[1])  bsquare_min[1] = projected_vert[1];
    if(projected_vert[0] > bsquare_max[0])  bsquare_max[0] = projected_vert[0];
    if(projected_vert[1] > bsquare_max[1])  bsquare_max[1] = projected_vert[1];
  }
}

void RegTools::RegToolsThread_main(RegToolsThreadParam *in_param)
{
	//print_and_log("RegTools::RegToolsThread_main(), in_param->m_ProcessingMode: %d\n", in_param->m_ProcessingMode);
  if(in_param->m_ProcessingMode == ProcessingMode_Initialize){                        RegToolsThread_Initialize( in_param );  return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_MemInfoQuery){                 RegToolsThread_MemGetInfo( in_param->m_FreeMem, in_param->m_TotalMem );  return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_VolumePlan){                   RegToolsThread_VolumePlan(in_param);              return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_MultVolume){                   RegToolsThread_MultVolume(in_param);            return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_CMAESPopulation){              RegToolsThread_CMAESPopulation(in_param);            return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_VolumePlan_cudaArray){         RegToolsThread_VolumePlan_cudaArray(in_param);              return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_GetGPUProjection){             RegToolsThread_GetGPUProjection(in_param);      return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_SimilarityMeasureComputation){ RegToolsThread_SimilarityMeasureComputation(in_param, in_param->d_Projections, in_param->d_Volume); return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_CopyDeviceMemoryToCudaArray){  RegToolsThread_CopyDeviceMemoryToCudaArray(in_param); return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_CopyDeviceMemoryToCudaArray_Multi){  RegToolsThread_CopyDeviceMemoryToCudaArray_Multi(in_param); return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_SetPBO){                       RegToolsThread_SetPBO(in_param->m_PBO, &(in_param->m_ProjectionImagePBOResource[in_param->m_PBO_index])); return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_ComputeLinearCombination){     RegToolsThread_ComputeLinearCombination(in_param); return; }
  else if (in_param->m_ProcessingMode == ProcessingMode_CopyHostInitProjectionToDevice) { RegToolsThread_CopyHostInitProjectionToDevice(in_param); return; }
  else if (in_param->m_ProcessingMode == ProcessingMode_ClearDeviceInitProjection) { RegToolsThread_ClearDeviceInitProjection(in_param); return; }
  else if (in_param->m_ProcessingMode == ProcessingMode_LCNCompuatationPlan){		  RegToolsThread_LCNComputationPlan(in_param);              return; }
  else if(in_param->m_ProcessingMode == ProcessingMode_DoNothing){                    return; }

  // start forward/back projector process
  float kernel_running_time;
  if(in_param->m_ElapsedTime) *(in_param->m_ElapsedTime) = 0;
  int block_num = (in_param->m_NumEnabledProjections % in_param->m_TransferBlockSize) > 0 ? (in_param->m_NumEnabledProjections/in_param->m_TransferBlockSize+1)  : in_param->m_NumEnabledProjections/in_param->m_TransferBlockSize;

#if defined RegTools_VERBOSE_MESSAGE
  print_and_log("start recon tools thread with %d projections ( ", in_param->m_NumEnabledProjections);
  for(int i=0;i<(int)in_param->m_NumEnabledProjections;i++)  print_and_log("%d ", in_param->m_EnabledProjection[i]);
  print_and_log(")\n");
  print_and_log("RegTools - number of transfer block: %d\n", block_num);
#endif
  
  // copy variables to CUDA const memory
  RegToolsThread_CopyVarToCudaMemory( in_param );

  // check if PBO rendering or not
  bool isPBO_enabled = (in_param->m_ProjectionImagePBOResource!=NULL), isPBO_rendering = false;
  if(isPBO_enabled)	isPBO_rendering = (in_param->m_ProjectionImagePBOResource[in_param->m_PBO_index]!=NULL);

  // set up texture array (in case of forward projection, we do this only once for all projections)
  if(in_param->m_ProcessingMode == ProcessingMode_ForwardProjection || 
     in_param->m_ProcessingMode == ProcessingMode_Interpolator){
//      GPUmemCheck("start forward projection");
	// prepare cuda texture for the input volume
    if(in_param->d_ProjectionTextureArray){ unBindCUDATexture(); cutilSafeCall( cudaFreeArray(in_param->d_ProjectionTextureArray) ); in_param->d_ProjectionTextureArray = NULL; }
    cudaArray *plannedVolume = in_param->m_VolumePlan_cudaArray ? in_param->m_VolumePlan_cudaArray->d_volume[in_param->m_CudaDeviceID_Sequential] : NULL;
    if(!in_param->m_Volume && !in_param->d_Volume && !in_param->d_VolumeTextureArray && !plannedVolume && !in_param->m_VolumePlan_cudaArray_warpX){ print_and_log("there is no data to forward project\n"); return; }
	//print_and_log("RegTools::RegToolsThread_main(), start forward projection or interpolation, plannedVolume: %d, in_param->m_Volume: %d, in_param->d_Volume: %d\n", plannedVolume, in_param->m_Volume, in_param->d_Volume);
	//print_and_log("in_param->m_VolumePlan_cudaArray: %d, in_param->m_CudaDeviceID_Sequential: %d, in_param->m_VolumePlan_cudaArray->d_volume[in_param->m_CudaDeviceID_Sequential]: %d\n",
	//	in_param->m_VolumePlan_cudaArray, in_param->m_CudaDeviceID_Sequential, in_param->m_VolumePlan_cudaArray->d_volume[in_param->m_CudaDeviceID_Sequential]);
    if(plannedVolume){
      unBindCUDATexture();
      bindCUDATexture( plannedVolume );
    } else if(in_param->m_Volume){
      if(in_param->d_VolumeTextureArray){ unBindCUDATexture(); cutilSafeCall( cudaFreeArray(in_param->d_VolumeTextureArray) ); in_param->d_VolumeTextureArray = NULL; }
      if(in_param->d_Volume){ cutilSafeCall( cudaFree(in_param->d_Volume) ); in_param->d_Volume = NULL; } // clear the volume in the global memory
      //print_and_log("in_param->m_VolumeDim: (%d, %d, %d)\n", in_param->m_VolumeDim.width, in_param->m_VolumeDim.height, in_param->m_VolumeDim.depth);
      cudaMallocForTexture(&(in_param->d_VolumeTextureArray), in_param->m_VolumeDim);
      copyDataToCudaArray(in_param->d_VolumeTextureArray, in_param->m_Volume, 0, 0, in_param->m_VolumeDim, in_param->m_NumVolumes, cudaMemcpyHostToDevice); // use the volume in host memory
      bindCUDATexture(in_param->d_VolumeTextureArray);
    } else if(in_param->d_Volume){
      if(in_param->d_VolumeTextureArray){
        //print_and_log("RegToolsThread_main(), initialize in_param->d_VolumeTextureArray\n");
        unBindCUDATexture(); cutilSafeCall( cudaFreeArray(in_param->d_VolumeTextureArray) ); in_param->d_VolumeTextureArray = NULL; 
      }
      //print_and_log("RegToolsThread_main(), in_param->m_VolumeDim: (%d, %d, %d)\n", in_param->m_VolumeDim.width, in_param->m_VolumeDim.height, in_param->m_VolumeDim.depth);
      cudaMallocForTexture(&(in_param->d_VolumeTextureArray), in_param->m_VolumeDim);
      // move volume from global memory to cudaArray
      copyDataToCudaArray(in_param->d_VolumeTextureArray, in_param->d_Volume, 0, 0, in_param->m_VolumeDim, in_param->m_NumVolumes, cudaMemcpyDeviceToDevice);
      cutilSafeCall( cudaFree(in_param->d_Volume) ); in_param->d_Volume = NULL;
      bindCUDATexture(in_param->d_VolumeTextureArray);
    }
    if((in_param->m_ProcessingMode == ProcessingMode_Interpolator && in_param->m_Interpolator_IsWarp) ||
      (in_param->m_ProcessingMode == ProcessingMode_ForwardProjection && in_param->m_ProjectorMode == ProjectorMode_LinearInterpolationDeformable)){
//      print_and_log("RegTools::RegToolsThread_main(), apply deformation field\n");
      unBindWarpTexture();
      cudaArray *warpX = in_param->m_VolumePlan_cudaArray_warpX->d_volume[in_param->m_CudaDeviceID_Sequential];
      cudaArray *warpY = in_param->m_VolumePlan_cudaArray_warpY->d_volume[in_param->m_CudaDeviceID_Sequential];
      cudaArray *warpZ = in_param->m_VolumePlan_cudaArray_warpZ->d_volume[in_param->m_CudaDeviceID_Sequential];
      bindWarpTexture(warpX, warpY, warpZ);
    }
    // if the volume exists in texture array (d_VolumeTextureArray), we do nothing, we use it as it is
  }

  // If this is Interpolation process, run Interpolator and done
  if(in_param->m_ProcessingMode == ProcessingMode_Interpolator){
    // TODO: add PBO rendering
    // if(isPBO_rendering)

    if(in_param->d_Projections){ cutilSafeCall( cudaFree( in_param->d_Projections ) ); in_param->d_Projections = NULL; }
    if(in_param->m_VolumePlan_cudaArray_out){
      if(in_param->d_Volume)  cutilSafeCall( cudaFree( in_param->d_Volume ) );
      in_param->d_Volume = in_param->m_VolumePlan_cudaArray_out->d_raw_volume[in_param->m_CudaDeviceID_Sequential];
      int *dim = in_param->m_VolumePlan_cudaArray_out->VolumeDim;
      //print_and_log("store interpolation result to the volume plan. in_param->d_Volume: %d (%d, %d, %d, %d), device: %d\n"
      //  , in_param->d_Volume, dim[0], dim[1], dim[2], in_param->m_VolumePlan_cudaArray_out->numVolumes, in_param->m_CudaDeviceID_Sequential);
    } else if((in_param->previousResultWidth == in_param->m_VolumeDim.width) && (in_param->previousResultHeight == in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth && in_param->d_Volume)){
//      print_and_log("skip memory allocation, resultWidth: %d, resultHeight: %d, previousResultWidth: %d, previousResultHeight: %d\n"
//        , in_param->m_VolumeDim.width, in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth, in_param->previousResultWidth, in_param->previousResultHeight);
    } else {
      if(in_param->d_Volume){  cutilSafeCall( cudaFree( in_param->d_Volume ) ); /* print_and_log("cudaFree(in_param->d_Volume);\n"); */ }
//      cutilSafeCall( cudaMallocPitch((void**)&in_param->d_Volume, &in_param->pitch, in_param->m_VolumeDim.width*sizeof(float), in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth) );
      cutilSafeCall( cudaMalloc((void**)&in_param->d_Volume, in_param->m_VolumeDim.width*in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth*sizeof(float)) );
      //print_and_log("cudaMalloc: %d x %d x %d\n", in_param->m_VolumeDim.width, in_param->m_VolumeDim.height, in_param->m_VolumeDim.depth);
    }
    RegToolsThread_RunInterpolator(in_param, in_param->d_Volume);
	//print_and_log("RegTools::RegToolsThread_main(): %d x %d x %d\n", in_param->m_VolumeDim.width, in_param->m_VolumeDim.height, in_param->m_VolumeDim.depth);
	/*
	int num_voxels = in_param->m_VolumeDim.width*in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth*in_param->m_NumProjectionSets;
	float *temp = new float[num_voxels];
	cutilSafeCall(cudaMemcpy(temp, in_param->d_Volume, num_voxels * sizeof(float), cudaMemcpyDeviceToHost));
	float sum = 0;
	for (int i = 0; i < num_voxels; i++) sum += temp[i];
	print_and_log("RegTools::RegToolsThread_main(), sum: %f\n", sum);
	free(temp);
	*/
    in_param->previousResultWidth = in_param->m_VolumeDim.width; in_param->previousResultHeight = in_param->m_VolumeDim.height*in_param->m_VolumeDim.depth;
    if(in_param->m_VolumePlan_cudaArray_out) in_param->d_Volume = NULL;
    return; 
  }

  // set up memory space for projection result
  float *d_ProjectionResult = NULL;  // this can be either Projections or Volume
  size_t resultWidth = 0, resultHeight = 0;
  resultWidth = in_param->m_ProjectionWidth;
  size_t num_projections_in_memory = MIN(in_param->m_TransferBlockSize, in_param->m_NumEnabledProjections);
  resultHeight = in_param->m_ProjectionHeight * num_projections_in_memory;

//  print_and_log("(resultWidth, resultHeight) = (%d, %d), previous = (%d, %d), d_ProjectionResult = %d\n", resultWidth, resultHeight, in_param->previousResultWidth, in_param->previousResultHeight, d_ProjectionResult);
  if(isPBO_rendering){
    // render forward projection result to PBO
    // currently, we support only forward projection for PBO rendering
    size_t size;
    cutilSafeCall( cudaGraphicsMapResources(1, &(in_param->m_ProjectionImagePBOResource[in_param->m_PBO_index])) );
    cutilSafeCall( cudaGraphicsResourceGetMappedPointer((void **)(&d_ProjectionResult), &size, in_param->m_ProjectionImagePBOResource[in_param->m_PBO_index]) );
//    print_and_log("ReconTools::ReconToolsThread_main(), mapped d_ProjectionResult to PBO resource, size: %d\n", size);
  } else {
    if(resultWidth>0 && resultHeight>0){
      if(in_param->m_VolumePlan_cudaArray_out){
        if(d_ProjectionResult) cutilSafeCall( cudaFree( d_ProjectionResult ) );
        d_ProjectionResult = in_param->m_VolumePlan_cudaArray_out->d_raw_volume[in_param->m_CudaDeviceID_Sequential];
  //      print_and_log("store reconstruction result to the volume plan\n");
      } else if(resultWidth == in_param->previousResultWidth && resultHeight == in_param->previousResultHeight && in_param->d_Projections){
        // if resultWidth and resultHeight is equal to the previous width & height, we don't need to allocate it again
        //print_and_log("skip reallocation for projections\n");
        d_ProjectionResult = in_param->d_Projections;
      } else if(resultWidth == in_param->previousResultWidth && resultHeight == in_param->previousResultHeight && in_param->d_Volume){
        //print_and_log("skip reallocation for volume\n");
        d_ProjectionResult = in_param->d_Volume;
      } else {
        //print_and_log("reallocate\n");
        if(in_param->d_Volume){      cutilSafeCall( cudaFree( in_param->d_Volume ) );      in_param->d_Volume = NULL; }
        if(in_param->d_Projections){ cutilSafeCall( cudaFree( in_param->d_Projections ) ); in_param->d_Projections = NULL; }
        if(d_ProjectionResult){ cutilSafeCall( cudaFree( d_ProjectionResult ) ); d_ProjectionResult = NULL; }
        cutilSafeCall( cudaMalloc((void**)&d_ProjectionResult, resultWidth*resultHeight*sizeof(float)) );
        if(d_ProjectionResult == NULL) print_and_log("Error malloc\n");
      }
    }
  }

  //  print_and_log("memory store mode = %d\n", in_param->m_MemoryStoreMode);
  if(in_param->m_MemoryStoreMode == MemoryStoreMode_Replace) {
    if (in_param->m_ProjectionsInit || in_param->d_ProjectionsInit)
    {
	    size_t arraySize = in_param->m_NumViews * in_param->m_ProjectionWidth * in_param->m_ProjectionHeight;

#if defined RegTools_VERBOSE_MESSAGE
	    print_and_log("Load initData of size: %d (%d * %d * %d)\n", arraySize, in_param->m_NumViews, in_param->m_ProjectionWidth, in_param->m_ProjectionHeight);
	    print_and_log("Result size (number of proj in memory): %d (%lu)\n", resultWidth*resultHeight, num_projections_in_memory);
#endif

      // if a device pointer is provided, prefer that over a provided host pointer.
      if (in_param->d_ProjectionsInit)
      {
#if defined RegTools_VERBOSE_MESSAGE
        print_and_log("Loading initial projection from DEVICE memory: %p\n", (void*) in_param->d_ProjectionsInit);
#endif
        cutilSafeCall( cudaMemcpy(d_ProjectionResult, in_param->d_ProjectionsInit, arraySize * sizeof(float), cudaMemcpyDeviceToDevice));
      }
      else
      {
#if defined RegTools_VERBOSE_MESSAGE
        print_and_log("Loading initial projection from HOST memory: %p\n", (void*) in_param->m_ProjectionsInit);
#endif
	      cutilSafeCall( cudaMemcpy(d_ProjectionResult, in_param->m_ProjectionsInit, arraySize * sizeof(float), cudaMemcpyHostToDevice));
      }

      // finish the replication across the number of projections to generate
	    for (size_t i = 1; i < (num_projections_in_memory/in_param->m_NumViews); i++)
      {
		    cutilSafeCall( cudaMemcpy(d_ProjectionResult + i * arraySize, d_ProjectionResult, arraySize*sizeof(float), cudaMemcpyDeviceToDevice) );
	    }
	  }
    else
    {

#if defined RegTools_VERBOSE_MESSAGE
	    print_and_log("No initData, memory set to 0\n");
#endif

      cutilSafeCall( cudaMemset(d_ProjectionResult, 0, resultWidth*resultHeight*sizeof(float) ) );
	  }
  }

  in_param->previousResultWidth = resultWidth; in_param->previousResultHeight = resultHeight;
  //RegToolsThread_MemGetInfo( in_param->m_FreeMem, in_param->m_TotalMem ); 
  //print_and_log("end of memory allocation on device %d: %f MB available (total: %f MB)\n", in_param->m_CudaDeviceID, (float)in_param->m_FreeMem/1024.0f/1024.0f, (float)in_param->m_TotalMem/1024.0f/1024.0f)
  //print_and_log("RegTools::RegToolsThread_main, resultWidth: %d, resultHeight: %d, in_param->m_ProjectionWidth: %d, in_param->m_ProjectionHeight: %d\n", resultWidth, resultHeight, in_param->m_ProjectionWidth, in_param->m_ProjectionHeight);

  // variables setup for forward/back projector
  initCudaTexture(false, true);   // only for LinearInterpolation projector

  int proj_size = in_param->m_ProjectionWidth*in_param->m_ProjectionHeight;
  float *h_PreComputedMatrix_array = new float[in_param->m_TransferBlockSize*12];
  float *d_PrecomputedMatrix;
  cutilSafeCall( cudaMalloc(&d_PrecomputedMatrix, in_param->m_TransferBlockSize*sizeof(float)*12) );
  for(int i=0;i<block_num;i++){
    int block_size = (i+1)*in_param->m_TransferBlockSize - 1 < in_param->m_NumEnabledProjections ? in_param->m_TransferBlockSize : 
                      in_param->m_NumEnabledProjections-in_param->m_TransferBlockSize*(block_num-1);
    copyProjectionBlockSize(block_size);

#if !defined(RegTools_ENABLE_CUDA20_CAPABILITY_FEATURES)
    // if 3D grid is not available, we tile the projection images (ray-tracing threads) in 2D
    // calculate tile size
    int tileX, tileY, tiledImageX, tiledImageY;
    findTileSize(block_size, in_param->m_ProjectionWidth, in_param->m_ProjectionHeight, tileX, tileY, tiledImageX, tiledImageY);
    copyTileSize(tileX, tileY);

    dim3 block(16, 16, 1);
    dim3 grid(iDivUp(tiledImageX, block.x), iDivUp(tiledImageY, block.y), 1);
    //print_and_log("tiledImageSize: (%d, %d), grid: (%d, %d), block: (%d, %d)\n", tiledImageX, tiledImageY, grid.x, grid.y, block.x, block.y);
#else
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 grid(iDivUp(in_param->m_ProjectionWidth, block.x), iDivUp(in_param->m_ProjectionHeight, block.y), iDivUp(block_size, block.z));
#endif

#if defined RegTools_VERBOSE_MESSAGE
    print_and_log("RegTools - start projection block #%d with %d projections\n", i, block_size);
#endif

    double volume_size[3] = {in_param->m_VolumeDim.width*in_param->m_VoxelSize_mm.x, in_param->m_VolumeDim.height*in_param->m_VoxelSize_mm.y, in_param->m_VolumeDim.depth*in_param->m_VoxelSize_mm.z};
    for(int j=0;j<block_size;j++){
      ConstructProjectionParameterArray(j, &(in_param->m_ProjectionParameters[in_param->m_EnabledProjection[i*in_param->m_TransferBlockSize+j]])
                                  , in_param->m_World_Volume_col, volume_size, in_param->m_ProjectionWidth, in_param->m_ProjectionHeight
                                  , h_PreComputedMatrix_array);
#if defined RegTools_VERBOSE_MESSAGE
      print_and_log("copyProjectionParameter: %d, %d, %d\n", j, i*in_param->m_TransferBlockSize+j, in_param->m_EnabledProjection[i*in_param->m_TransferBlockSize+j]);
      print_and_log("projection matrix:\n");
      double *pm = in_param->m_ProjectionParameters[in_param->m_EnabledProjection[i*in_param->m_TransferBlockSize+j]].ProjectionMatrix_3x4;
      for(int k=0;k<3;k++) print_and_log("%f %f %f %f\n", pm[k*4], pm[k*4+1], pm[k*4+2], pm[k*4+3]);
      print_and_log("\n");
      print_and_log("T_World_Volume:\n");
      for(int k=0;k<4;k++) print_and_log("%f %f %f %f\n", in_param->m_World_Volume_col[k], in_param->m_World_Volume_col[4+k], in_param->m_World_Volume_col[8+k], in_param->m_World_Volume_col[12+k]);
#endif
    }
    cutilSafeCall( cudaMemcpy( d_PrecomputedMatrix, h_PreComputedMatrix_array, block_size*sizeof(float)*12, cudaMemcpyHostToDevice ) );

#if defined RegTools_VERBOSE_MESSAGE
/*
    print_and_log("start projector: grid(%d, %d), block(%d, %d), tile(%d, %d), tiledImage(%d, %d), enabled projections: %d\n"
      , grid.x, grid.y, block.x, block.y, tileX, tileY, tiledImageX, tiledImageY, in_param->m_NumEnabledProjections);
*/
#endif

    int *d_ZeroPixelCount = NULL;
    if(in_param->m_CountNonIntersectedPixel){
      cutilSafeCall( cudaMalloc((void**)&d_ZeroPixelCount, block_size*sizeof(int)) );
      //print_and_log("cudaMalloc(d_ZeroPixelCount, %d)\n", block_size);
      FillData_i(d_ZeroPixelCount, block_size, 0);
    }
    // start kernel running time counter
    cudaEventRecord(in_param->h_timer_start, 0);

    bindPrecomputedMatrixTexture(d_PrecomputedMatrix, block_size);
    if(in_param->m_ProjectorMode == ProjectorMode_LinearInterpolation)
      launch_LinearInterpolationProjector(d_ProjectionResult, d_ZeroPixelCount, in_param->m_NumEnabledProjections/in_param->m_NumProjectionSets, grid, block);
    else if(in_param->m_ProjectorMode == ProjectorMode_Siddon)
      launch_SiddonProjector(d_ProjectionResult, in_param->pitch, grid, block, NULL); //, d_RandomSequence);
    else if(in_param->m_ProjectorMode == ProjectorMode_RayCasting)
      launch_RayCastingProjector(d_ProjectionResult, grid, block);
    else if (in_param->m_ProjectorMode == ProjectorMode_DepthMap)
      launch_DepthMapProjector(d_ProjectionResult, grid, block);
    else if(in_param->m_ProjectorMode == ProjectorMode_LinearInterpolationDeformable)
      launch_LinearInterpolationDeformableProjector(d_ProjectionResult, d_ZeroPixelCount, in_param->m_NumEnabledProjections/in_param->m_NumProjectionSets, grid, block);

	if (in_param->m_LCN_sigma > 0) {
		RegToolsThread_LocalContrastNormalization(in_param, d_ProjectionResult, in_param->m_LCN_sigma);
	}

//    print_and_log("number of projections in one set: %d, m_DifferentVolumePerProjectionSet: %d, block_size: %d\n", in_param->m_NumEnabledProjections/in_param->m_NumProjectionSets, in_param->m_DifferentVolumePerProjectionSet, block_size);
    unBindPrecomputedMatrixTexture();
    unBindWarpTexture();

    // normalize projection result if needed
    int one_image_size = in_param->m_ProjectionWidth*in_param->m_ProjectionHeight*block_size;
    if(in_param->m_NormalizeMax > in_param->m_NormalizeMin)
      normalizeImages(d_ProjectionResult, one_image_size, in_param->m_NormalizeMax, in_param->m_NormalizeMin);

    // stop timer
    cudaEventRecord(in_param->h_timer_stop, 0);
    cudaEventSynchronize(in_param->h_timer_stop);
    cudaEventElapsedTime(&kernel_running_time, in_param->h_timer_start, in_param->h_timer_stop);
    if(in_param->m_ElapsedTime) *(in_param->m_ElapsedTime) += kernel_running_time;

    if(in_param->m_Projections){
      cutilSafeCall( cudaMemcpy(in_param->m_Projections + i*in_param->m_TransferBlockSize*proj_size, 
        d_ProjectionResult, proj_size*block_size*sizeof(float), cudaMemcpyDeviceToHost) );
    }

    if(in_param->m_CountNonIntersectedPixel){
      cutilSafeCall( cudaMemcpy(in_param->m_ZeroPixelCount + i*in_param->m_TransferBlockSize, d_ZeroPixelCount, block_size*sizeof(int), cudaMemcpyDeviceToHost) );
      //for(int i=0;i<block_size;i++){
      //  print_and_log("image #%d, number of zero pixels = %d/%d, weight = %f\n", i, in_param->m_ZeroPixelCount[i], proj_size, 
      //    (double)(proj_size) / (double)(proj_size - in_param->m_ZeroPixelCount[i]));
      //}
      cutilSafeCall( cudaFree( d_ZeroPixelCount ) );
    }

#if defined RegTools_VERBOSE_MESSAGE
      print_and_log("forward projection done, image size: (%d, %d, %d, %d)\n"
        , in_param->m_ProjectionWidth, in_param->m_ProjectionHeight, in_param->m_NumEnabledProjections, in_param->m_NumProjectionSets);
#endif
  }
  delete []h_PreComputedMatrix_array;
  cutilSafeCall( cudaFree( d_PrecomputedMatrix ) );

  // if needed, compute maximum and minimum value of projection result
  if(in_param->m_MaxValue || in_param->m_MinValue){
    ComputeMaxMin(d_ProjectionResult, resultWidth*resultHeight, in_param->m_MaxValue, in_param->m_MinValue); // this 'size' works for both forward and back projection
//    print_and_log("in_param->m_MaxValue: %d, in_param->m_MinValue: %d\n", in_param->m_MaxValue, in_param->m_MinValue);
//    print_and_log("ComputeMaxMin on device %d end\n", in_param->m_CudaDeviceID);
  }
    
  if(isPBO_rendering){
    // post process for PBO rendering
    cutilSafeCall( cudaGraphicsUnmapResources(1, &(in_param->m_ProjectionImagePBOResource[in_param->m_PBO_index])) );
//    print_and_log("ReconTools::ReconToolsThread_main(), unmapped PBO resource\n");
  } else {
    // post processing
    in_param->d_Projections = d_ProjectionResult;   in_param->d_Volume = NULL;
  }
}

void RegTools::RegToolsThread_LocalContrastNormalization(RegToolsThreadParam *in_param, float *d_ProjectionResult, int LCN_sigma)
{
	unsigned int halfsize = getKernelSize(LCN_sigma);
	unsigned int kernel_size = 2 * halfsize + 1;
	fComplex *d_LCN_gaussian_kernel;
	cufftHandle *fftPlanLCNFwd;
	cufftHandle *fftPlanLCNManyFwd;
	cufftHandle *fftPlanLCNManyInv;
	int dim[3] = { in_param->m_ProjectionWidth, in_param->m_ProjectionHeight, ceil((float)(in_param->m_NumEnabledProjections)/(float)(in_param->m_NumProjectionSets)) };
	const int fftH = snapTransformSize(dim[1] + kernel_size - 1), fftW = snapTransformSize(dim[0] + kernel_size - 1);
	float *d_centered_images, *d_std_images;
	float *d_temp_padded;
	fComplex *d_temp_spectrum;
	RegToolsThread_MemGetInfo(in_param->m_FreeMem, in_param->m_TotalMem);
	print_and_log("RegToolsThread_LocalContrastNormalization(GPU_ID:%d), before, available memory: %f MB\n", in_param->m_CudaDeviceID_Sequential, (float)in_param->m_FreeMem/1024.0f/1024.0f);
	print_and_log("m_NumEnabledProjections: %d, m_NumProjectionSets:%d, dim: (%d,%d,%d)\n", in_param->m_NumEnabledProjections, in_param->m_NumProjectionSets, dim[0], dim[1], dim[2]);
	print_and_log("kernel_size: %d, fftH: %d, fftW:%d\n", kernel_size, fftH, fftW);
	cutilSafeCall(cudaMalloc(&(d_centered_images), dim[0] * dim[1] * dim[2]* in_param->m_NumProjectionSets * sizeof(float)));
//	cutilSafeCall(cudaMalloc(&(d_std_images), dim[0] * dim[1] * dim[2] * in_param->m_NumProjectionSets * sizeof(float)));
	cutilSafeCall(cudaMalloc(&(d_temp_padded), fftH*fftW*dim[2] * in_param->m_NumProjectionSets * sizeof(float)));
	cutilSafeCall(cudaMalloc(&(d_temp_spectrum), fftH *(fftW / 2 + 1)*dim[2] * in_param->m_NumProjectionSets * sizeof(fComplex)));
	cutilSafeCall(cudaMalloc(&(d_LCN_gaussian_kernel), fftH * (fftW / 2 + 1) * sizeof(fComplex)));
	int n[2] = { fftH, fftW };
	// cufft functions seem to consume 28MB at its initial function call (maybe for initialization... don't know the detail, but cannot release the 28MB even after cufftDestroy)
	if (!in_param->m_fftPlanLCNFwd) {
		print_and_log("in_param->m_fftPlanLCNFwd is NULL\n");
		fftPlanLCNFwd = new cufftHandle;
		cufftSafeCall(cufftPlan2d(fftPlanLCNFwd, fftH, fftW, CUFFT_R2C));
	}
	else {
		print_and_log("in_param->m_fftPlanLCNFwd is not NULL\n");
		fftPlanLCNFwd = in_param->m_fftPlanLCNFwd;
	}
	if (!in_param->m_fftPlanLCNManyFwd) {
		print_and_log("in_param->m_fftPlanLCNManyFwd is NULL\n");
		fftPlanLCNManyFwd = new cufftHandle;
		cufftSafeCall(cufftPlanMany(fftPlanLCNManyFwd, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, dim[2] * in_param->m_NumProjectionSets));  // For forward FFT
	}
	else {
		print_and_log("in_param->m_fftPlanLCNManyFwd is not NULL\n");
		fftPlanLCNManyFwd = in_param->m_fftPlanLCNManyFwd;
	}
	if (!in_param->m_fftPlanLCNManyInv) {
		print_and_log("in_param->m_fftPlanLCNManyInv is NULL\n");
		fftPlanLCNManyInv = new cufftHandle;
		cufftSafeCall(cufftPlanMany(fftPlanLCNManyInv, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, dim[2] * in_param->m_NumProjectionSets));  // For forward FFT
	}
	else {
		print_and_log("in_param->m_fftPlanLCNManyInv is not NULL\n");
		fftPlanLCNManyInv = in_param->m_fftPlanLCNManyInv;
	}
	RegToolsThread_MemGetInfo(in_param->m_FreeMem, in_param->m_TotalMem);
	print_and_log("RegToolsThread_LocalContrastNormalization(GPU_ID:%d), after 1, available memory: %f MB\n", in_param->m_CudaDeviceID_Sequential, (float)in_param->m_FreeMem / 1024.0f / 1024.0f);

	computeGaussianKernelSpectrum(LCN_sigma, dim[0], dim[1], 0, d_LCN_gaussian_kernel, *fftPlanLCNFwd);
	computeLocalContrastNormalizationGPUMulti(d_ProjectionResult, dim, in_param->m_NumProjectionSets, d_centered_images, NULL, d_ProjectionResult, LCN_sigma
		, d_LCN_gaussian_kernel, d_temp_padded, d_temp_spectrum, *fftPlanLCNManyFwd, *fftPlanLCNManyInv);
//	computeLocalContrastNormalizationGPUMulti(d_ProjectionResult, dim, in_param->m_NumProjectionSets, d_centered_images, d_std_images, d_ProjectionResult, LCN_sigma
//		, d_LCN_gaussian_kernel, d_temp_padded, d_temp_spectrum, *(fftPlanLCNManyFwd), *(fftPlanLCNManyInv));
	//cutilSafeCall(cudaMemcpy(d_ProjectionResult, d_centered_images, dim[0] * dim[1] * dim[2] * in_param->m_NumProjectionSets * sizeof(float), cudaMemcpyDeviceToDevice));

	RegToolsThread_MemGetInfo(in_param->m_FreeMem, in_param->m_TotalMem);
	print_and_log("RegToolsThread_LocalContrastNormalization(), after 2, available memory: %f MB\n", (float)in_param->m_FreeMem / 1024.0f / 1024.0f);

	cutilSafeCall(cudaFree(d_LCN_gaussian_kernel));
	if (!in_param->m_fftPlanLCNFwd) {
		cufftSafeCall(cufftDestroy(*(fftPlanLCNFwd)));	   delete fftPlanLCNFwd; fftPlanLCNFwd = NULL;
	}
	if (!in_param->m_fftPlanLCNManyFwd) {
		cufftSafeCall(cufftDestroy(*(fftPlanLCNManyFwd))); delete fftPlanLCNManyFwd; fftPlanLCNManyFwd = NULL;
	}
	if (!in_param->m_fftPlanLCNManyInv) {
		cufftSafeCall(cufftDestroy(*(fftPlanLCNManyInv))); delete fftPlanLCNManyInv; fftPlanLCNManyInv = NULL;
	}
	cutilSafeCall(cudaFree(d_centered_images));
//	cutilSafeCall(cudaFree(d_std_images));
	cutilSafeCall(cudaFree(d_temp_padded));
	cutilSafeCall(cudaFree(d_temp_spectrum));
	RegToolsThread_MemGetInfo(in_param->m_FreeMem, in_param->m_TotalMem);
	print_and_log("RegToolsThread_LocalContrastNormalization(), after 3, available memory: %f MB\n", (float)in_param->m_FreeMem / 1024.0f / 1024.0f);
}

void RegTools::RegToolsThread_CopyHostInitProjectionToDevice(RegToolsThreadParam *in_param)
{
  const size_t host_buf_len_bytes = in_param->m_ProjectionsInit_len * 4;

  if (in_param->d_ProjectionsInit_capacity_bytes < host_buf_len_bytes)
  {
    if (in_param->d_ProjectionsInit)
    {
      cutilSafeCall(cudaFree(in_param->d_ProjectionsInit));
    }

    cutilSafeCall(cudaMalloc((void**) &in_param->d_ProjectionsInit, host_buf_len_bytes));
    in_param->d_ProjectionsInit_capacity_bytes = host_buf_len_bytes;
  }

  cutilSafeCall(cudaMemcpy(in_param->d_ProjectionsInit, in_param->m_ProjectionsInit, host_buf_len_bytes, cudaMemcpyHostToDevice));
}

void RegTools::RegToolsThread_ClearDeviceInitProjection(RegToolsThreadParam *in_param)
{
  if (in_param->d_ProjectionsInit)
  {
    cutilSafeCall(cudaFree(in_param->d_ProjectionsInit));
  }
  in_param->d_ProjectionsInit = 0;
  in_param->d_ProjectionsInit_capacity_bytes = 0;
}

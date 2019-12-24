#include <cstdlib>
#include <iostream>

// CUDA
#include <cuda_runtime.h>


int main(int argc, char *argv[]) {
  int device_count;
  struct cudaDeviceProp properties;
  int max_major = 0;
  int max_minor = 0;

  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    device_count = 0;
  }

  for (int device = 0; device < device_count; ++device) {
    cudaGetDeviceProperties(&properties, device);
    if (properties.major == 9999) {
      continue;
    } else if (properties.major > max_major) {
      max_major = properties.major;
      max_minor = properties.minor;
    } else if (properties.major == max_major &&
               properties.minor > max_minor) {
      max_minor = properties.minor;
    } else {
      continue;
    }
  }

  if (max_major == 0 && max_minor == 0) {
    return EXIT_FAILURE;
  }
  std::cout << max_major << "." << max_minor;
  std::cout.flush();
  return EXIT_SUCCESS;
}

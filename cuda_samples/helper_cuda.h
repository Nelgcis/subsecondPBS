#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

template <typename T>
void check_cuda(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n",
                file, line, static_cast<unsigned int>(result), func);
        exit(1);
    }
}

#endif // HELPER_CUDA_H

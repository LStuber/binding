// Simple tool to determine compute capability of GPU0
#include <stdio.h>
#include <cuda_runtime_api.h>

int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    cudaError_t status;
    int device_count;

    status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        fprintf(stderr, "Could not get GPU count! %s\\n", cudaGetErrorString(status));
        return -1;
    }

    if (device_count == 0) {
        fprintf(stderr, "Could not find any GPUs on the system!\\n");
        return -1;
    }

    // get device properties for device 0
    status = cudaGetDeviceProperties(&prop, 0);
    if (status != cudaSuccess) {
        fprintf(stderr, "Could not get GPU0 properties! %s\\n", cudaGetErrorString(status));
        return -1;
    }

    // display version
    int version = prop.major * 10 + prop.minor;
    printf("%d\n", version);
}

#include "memory-color.cuh"
#include <stdlib.h>

// Handle errors raised by the GPU
// From
// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans)                                                                             \
    {                                                                                              \
        gpuAssert((ans), __FILE__, __LINE__);                                                      \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort)
            exit(code);
    }
}

// Memory management
extern void init_cuda()
{
    double* dummy;
    gpuErrchk(cudaMalloc((void**)&dummy, sizeof(double)));
    gpuErrchk(cudaFree(dummy));
}

extern void
init_memory(struct memory_cuda* mem, struct res_cuda* res, int sx, int sy, int sc, double* u)
{
    // GPU buffers
    double *dev_xie, *dev_xio,
        *dev_u; // gpu,  xie -> even coordinates of xi, xio -> odd coordinates
    double* dev_xiobar; // gpu, xiobar -> over-relaxation of xio
    double *dev_gle, *dev_glo; // gap arrays

    res->it = 0;
    res->msec = 0;
    res->gap = 0;
    res->rmse = 0;

    int sxyz = sx * sy * sc;
    int Ke = sx / 2, Le = sy / 2;
    int Ko = (sx - 2) / 2, Lo = (sy - 2) / 2;

    int factor = 4;

    // Memory management
    gpuErrchk(cudaMalloc((void**)&dev_xie, Ke * Le * sc * factor * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&dev_xio, Ko * Lo * sc * factor * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&dev_xiobar, Ko * Lo * sc * factor * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&dev_u, sxyz * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&dev_gle, Ke * Le * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&dev_glo, Ko * Lo * sizeof(double)));

    gpuErrchk(cudaMemcpy(dev_u, u, sxyz * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(dev_xie, 0, Ke * Le * sc * factor * sizeof(double)));
    gpuErrchk(cudaMemset(dev_xio, 0, Ko * Lo * sc * factor * sizeof(double)));
    // Additional sanitization, not in original code
    gpuErrchk(cudaMemset(dev_xiobar, 0, Ko * Lo * sc * factor * sizeof(double)));
    gpuErrchk(cudaMemset(dev_gle, 0, Ke * Le * sizeof(double)));
    gpuErrchk(cudaMemset(dev_glo, 0, Ko * Lo * sizeof(double)));

    mem->dev_xie = dev_xie;
    mem->dev_xio = dev_xio;
    mem->dev_u = dev_u;
    mem->dev_xiobar = dev_xiobar;
    mem->dev_gle = dev_gle;
    mem->dev_glo = dev_glo;
    return;
}

extern void free_memory(struct memory_cuda* mem)
{
    gpuErrchk(cudaFree(mem->dev_xie));
    gpuErrchk(cudaFree(mem->dev_xio));
    gpuErrchk(cudaFree(mem->dev_xiobar));
    gpuErrchk(cudaFree(mem->dev_u));
    gpuErrchk(cudaFree(mem->dev_gle));
    gpuErrchk(cudaFree(mem->dev_glo));
    return;
}


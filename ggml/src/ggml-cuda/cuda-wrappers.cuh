#pragma once

#include "ggml-cuda/common.cuh"

#ifdef GGML_USE_HIP
#define TRACE_FN_PREFIX "hip"
#else
#define TRACE_FN_PREFIX "cuda"
#endif

// CUDA memory management tracing wrappers
static cudaError_t ggml_cuda_malloc(void **ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    fprintf(stdout, TRACE_FN_PREFIX "Malloc: allocated ptr=%p, size=%zu, result=%s\n", *ptr, size, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_free(void *ptr) {
    cudaError_t err = cudaFree(ptr);
    fprintf(stdout, TRACE_FN_PREFIX "Free: ptr=%p, result=%s\n", ptr, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_malloc_managed(void **ptr, size_t size) {
    cudaError_t err = cudaMallocManaged(ptr, size);
    fprintf(stdout, TRACE_FN_PREFIX "MallocManaged: ptr=%p, size=%zu, result=%s\n", ptr, size, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_host_alloc(void **ptr, size_t size) {
    cudaError_t err = cudaMallocHost(ptr, size);
    fprintf(stdout, TRACE_FN_PREFIX "MallocHost: ptr=%p, size=%zu, result=%s\n", ptr, size, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_host_free(void *ptr) {
    cudaError_t err = cudaFreeHost(ptr);
    fprintf(stdout, TRACE_FN_PREFIX "FreeHost: ptr=%p, result=%s\n", ptr, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_memcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    fprintf(stdout, TRACE_FN_PREFIX "Memcpy: dst=%p, src=%p, count=%zu, kind=%d, result=%s\n",
            dst, src, count, kind, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_memcpy_async(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, count, kind, stream);
    fprintf(stdout, TRACE_FN_PREFIX "MemcpyAsync: dst=%p, src=%p, count=%zu, kind=%d, stream=%p, result=%s\n",
            dst, src, count, kind, (void*)stream, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_memcpy2d_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
    fprintf(stdout, TRACE_FN_PREFIX "Memcpy2DAsync: dst=%p, dpitch=%zu, src=%p, spitch=%zu, width=%zu, height=%zu, kind=%d, stream=%p, result=%s\n",
            dst, dpitch, src, spitch, width, height, kind, (void*)stream, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_memcpy_peer_async(void *dst, int dstDevice, void *src, int srcDevice, size_t count, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
    fprintf(stdout, TRACE_FN_PREFIX "MemcpyPeerAsync: dst=%p, dstDevice=%d, src=%p, srcDevice=%d, count=%zu, stream=%p, result=%s\n",
            dst, dstDevice, src, srcDevice, count, (void*)stream, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_memset(void *devPtr, int value, size_t count) {
    cudaError_t err = cudaMemset(devPtr, value, count);
    fprintf(stdout, TRACE_FN_PREFIX "Memset: devPtr=%p, value=%d, count=%zu, result=%s\n",
            devPtr, value, count, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_memset_async(void *devPtr, int value, size_t count, cudaStream_t stream) {
    cudaError_t err = cudaMemsetAsync(devPtr, value, count, stream);
    fprintf(stdout, TRACE_FN_PREFIX "MemsetAsync: devPtr=%p, value=%d, count=%zu, stream=%p, result=%s\n",
            devPtr, value, count, (void*)stream, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_host_register(void *ptr, size_t size, unsigned int flags) {
    cudaError_t err = cudaHostRegister(ptr, size, flags);
    fprintf(stdout, TRACE_FN_PREFIX "HostRegister: ptr=%p, size=%zu, flags=%u, result=%s\n", ptr, size, flags, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_host_unregister(void *ptr) {
    cudaError_t err = cudaHostUnregister(ptr);
    fprintf(stdout, TRACE_FN_PREFIX "HostUnregister: ptr=%p, result=%s\n", ptr, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_stream_synchronize(cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    fprintf(stdout, TRACE_FN_PREFIX "StreamSynchronize: stream=%p, result=%s\n", stream, cudaGetErrorString(err));
    return err;
}

static cudaError_t ggml_cuda_stream_create_with_flags(cudaStream_t* pStream, unsigned int flags)
{
    cudaError_t err = cudaStreamCreateWithFlags(pStream, flags);
    fprintf(stdout, TRACE_FN_PREFIX "StreamCreateWithFlags: created stream=%p, flags=%u, result=%s\n", *pStream, flags, cudaGetErrorString(err));
    return err;
}

#pragma once
#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define BLOCK 512
#define RANDOM(x) (rand() % x)

namespace gpu {
	void im2col_gpu(float *im, float *data_col, int channels, int height, int width, int ksize, int stride, int pad, int out_w, int out_h);

	void matmul(float * a, float * b, int inp_h, int io_wh, int out_w, float * out, bool transpose_a = false, bool transpose_b = false);

	void conv2d(float *input, float *filters, float *output, int channels, int height, int width, int ksize, int stride, int pad, int out_w, int out_h);
}
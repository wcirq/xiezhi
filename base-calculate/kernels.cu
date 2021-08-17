#include "kernels.h"
#include "class_timer.hpp"

namespace gpu {
	__global__ void im2col_gpu_kernel(const int n, const float* data_im,
		const int height, const int width, const int ksize,
		const int pad,
		const int stride,
		const int height_col, const int width_col,
		float *data_col) {
		int index = blockIdx.x*blockDim.x + threadIdx.x;

		// printf("blockIdx.x %d threadIdx.x %d\n", blockIdx.x, threadIdx.x);
		// printf("%d %d\n", blockDim.x, gridDim.x);

		for (; index < n; index += blockDim.x*gridDim.x) {
			int w_out = index % width_col;
			int h_index = index / width_col;
			int h_out = h_index % height_col;
			int channel_in = h_index / height_col;
			int channel_out = channel_in * ksize * ksize;
			int h_in = h_out * stride - pad;
			int w_in = w_out * stride - pad;
			float* data_col_ptr = data_col;
			data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
			const float* data_im_ptr = data_im;
			data_im_ptr += (channel_in * height + h_in) * width + w_in;
			for (int i = 0; i < ksize; ++i) {
				for (int j = 0; j < ksize; ++j) {
					int h = h_in + i;
					int w = w_in + j;

					*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
						data_im_ptr[i * width + j] : 0;

					data_col_ptr += height_col * width_col;
				}
			}
		}
	}

	__global__ void test(float * a) {
		int index = threadIdx.x*blockDim.x + threadIdx.y;
		printf("%d %f\n", index, a[index]);
	}

	__global__ void matmul_gpu(float * a, float * b, int inp_h, int io_wh, int out_w, float * out, bool transpose_a = false, bool transpose_b = false) {
		test << <1, 2>> > (a);

		int index = blockIdx.x*blockDim.x + threadIdx.x;
		float total=0;
		int a_index, b_index;
		for (size_t i = 0; i < io_wh; i++)
		{
			if (transpose_a) {
				a_index = (int)(index / inp_h) * (inp_h * io_wh) + (index % inp_h) + i * inp_h;
			}
			else {
				a_index = (int)(index / out_w) * (out_w * io_wh) + (index % inp_h) * io_wh + i;
			}
		
			if (transpose_b) {
				b_index = index % io_wh + i;
			}
			else {
				b_index = index % out_w + i;
			}

			
			total += (a[a_index] * b[b_index]);
			// if (index==10) printf("index %d (%d %2.0f) (%d %f)\n", index, a_index, a[a_index], b_index, b[b_index]);
		}
		out[index] = total;

	}

	void im2col_gpu(float *im, float *data_col,
		int channels, int height, int width,
		int ksize, int stride, int pad, int out_w, int out_h) {
		Timer timer;
		cudaSetDevice(0);
		int inputs = height * width * channels;
		int workspace_size = out_h * out_w * ksize * ksize * channels;

		timer.reset();
		cudaError_t error = cudaSuccess;
		float *dev_inp, *dev_out;
		error = cudaMalloc((void **)&dev_inp, sizeof(float) * inputs);
		error = cudaMalloc((void **)&dev_out, sizeof(float) * workspace_size);
		timer.out("cudaMalloc");

		timer.reset();
		if (error != cudaSuccess) {
			printf("Fail to cudaMalloc on GPU");
		}
		cudaMemcpy(dev_inp, im, sizeof(float) * inputs, cudaMemcpyHostToDevice);
		timer.out("cudaMemcpy");

		timer.reset();
		int num_kernels = channels * out_h * out_w;
		unsigned int s1 = (num_kernels + BLOCK - 1) / BLOCK;
		unsigned int s2 = BLOCK;

		im2col_gpu_kernel << <s1, s2 >> > (
				num_kernels, dev_inp, height, width, ksize, pad,
				stride, out_h,
				out_w, dev_out);
		timer.out("im2col_gpu_kernel");

		timer.reset();
		cudaMemcpy(data_col, dev_out, sizeof(float) * workspace_size, cudaMemcpyDeviceToHost);
		timer.out("cudaMemcpy");

		cudaFree(dev_inp);
		cudaFree(dev_out);
		
	}

	void matmul(float * a, float * b, int inp_h, int io_wh, int out_w, float * out, bool transpose_a, bool transpose_b)
	{
		Timer timer;
		int aSize = inp_h * io_wh * out_w;
		int bSize = out_w * io_wh;
		int outSize = inp_h * out_w;

		cudaError_t error = cudaSuccess;
		float *dev_a, *dev_b, *dev_out;
		error = cudaMalloc((void **)&dev_a, sizeof(float) * aSize);
		error = cudaMalloc((void **)&dev_b, sizeof(float) * bSize);
		error = cudaMalloc((void **)&dev_out, sizeof(float) * outSize);

		cudaMemcpy(dev_a, a, sizeof(float) * aSize, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, sizeof(float) * bSize, cudaMemcpyHostToDevice);

		unsigned int s1 = (outSize + BLOCK - 1) / BLOCK;
		unsigned int s2 = BLOCK;

		matmul_gpu<< <s1, s2>> >(dev_a, dev_b, inp_h, io_wh, out_w, dev_out, transpose_a, transpose_b);

		cudaMemcpy(out, dev_out, sizeof(float) * outSize, cudaMemcpyDeviceToHost);

		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_out);
	}
	void conv2d(float * input, float * filters, float * output, int channels, int height, int width, int ksize, int stride, int pad, int out_w, int out_h)
	{
	}
}


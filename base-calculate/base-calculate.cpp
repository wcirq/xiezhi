#include <stdio.h>
#include <stdlib.h>
#include "class_timer.hpp"
#include "kernels.h"

int conv_out_height(int h, int pad, int size, int stride) {
	return (h + 2 * pad - size) / stride + 1;
}

int conv_out_width(int w, int pad, int size, int stride) {
	return (w + 2 * pad - size) / stride + 1;
}

int im2col_get_pixel(float *im, int height, int width, int channels,
	int row, int col, int channel, int pad)
{
	row -= pad;
	col -= pad;

	if (row < 0 || col < 0 ||
		row >= height || col >= width) return 0;
	return im[col + width * (row + height * channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
	int channels, int height, int width,
	int ksize, int stride, int pad, float* data_col)
{
	int c, h, w;
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;

	int channels_col = channels * ksize * ksize;
	for (c = 0; c < channels_col; ++c) { //卷积核参数个数
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int c_im = c / ksize / ksize;
		for (h = 0; h < height_col; ++h) {
			for (w = 0; w < width_col; ++w) {
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * height_col + h) * width_col + w;
				data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
					im_row, im_col, c_im, pad);
			}
		}
	}
}

int main(int argc, char* argv[]) {
	Timer timer;

	float *data_im = NULL;
	float *data_k = NULL;
	float *data_col = NULL;
	float *data_out = NULL;
	int channels = 3, height = 4, width = 4;
	int ksize = 2, stride = 1, pad = 0;
	int out_w, out_h;
	int inp_size, col_size, filter_size;

	filter_size = ksize * ksize * channels;
	inp_size = height * width * channels;
	data_im = (float*)malloc(inp_size * sizeof(float));
	data_k = (float*)malloc(filter_size * sizeof(float));
	if (!data_im) {
		printf("malloc error\n");
		exit(EXIT_FAILURE);
	}

	out_w = conv_out_width(width, pad, ksize, stride);
	out_h = conv_out_width(height, pad, ksize, stride);
	col_size = out_h * out_w * ksize * ksize * channels;

	data_col = (float*)malloc(col_size * sizeof(float));
	data_out = (float*)malloc(out_w*out_h*channels * sizeof(float));
	if (!data_col) {
		printf("malloc error\n");
		exit(EXIT_FAILURE);
	}

	//init image
	for (int i = 0; i < inp_size; i++) data_im[i] = RANDOM(5);
	for (int i = 0; i < filter_size; i++) data_k[i] = 1;

	timer.reset();
	//im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);
	gpu::im2col_gpu(data_im, data_col, channels, height, width, ksize, stride, pad, out_w, out_h);
	timer.out("im2col_gpu time");
	timer.reset();
	int inp_h = out_w * out_h;
	int io_wh = ksize * ksize;
	int matmul_out_w = channels;
	gpu::matmul(data_col, data_k, inp_h, io_wh, matmul_out_w, data_out, true);
	timer.out("matmul time");

	printf("data_im:\n");
	for (int i = 0; i < inp_size; i++) {
		if (i > 100) break;
		printf("%2.0f ", data_im[i]);
		if( (i+1) % width == 0) printf("\n");
		if( (i+1) % (width*height) == 0) printf("\n");
	}
	printf("\n");

	printf("\ndata_col:\n");
	for (int i = 0; i < col_size; i++) {
		if (i > 100) break;
		printf("%2.0f ", data_col[i]);
		if( (i+1) % (out_h * out_w) == 0) printf("\n");
		if( (i+1) % (ksize * ksize * out_h * out_w) == 0) printf("\n");
	}
	printf("\n");

	printf("\ndata_k:\n");
	for (int i = 0; i < filter_size; i++) {
		if (i > 100) break;
		printf("%2.0f ", data_k[i]);
		if ((i + 1) % ksize == 0) printf("\n");
		if ((i + 1) % (ksize*ksize) == 0) printf("\n");
	}
	printf("\n");

	printf("\ndata_out:\n");
	for (int i = 0; i < out_w*out_h*channels; i++) {
		if (i > 100) break;
		printf("%2.0f ", data_out[i]);
		if ((i + 1) % out_w == 0) printf("\n");
		if ((i + 1) % (out_w*out_h) == 0) printf("\n");
	}
	printf("\n");

	free(data_im);
	free(data_col);

	exit(EXIT_SUCCESS);
}
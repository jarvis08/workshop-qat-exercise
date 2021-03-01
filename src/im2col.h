#ifndef IM2COL_H
#define IM2COL_H

#include "darknet.h"
#include <stdint.h>

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

void im2col_int4_cpu(int4_t* data_im,
                     int channels,  int height,  int width,
                     int ksize,  int stride, int pad,
                     int4_t QZ, int4_t* data_col);

void im2col_quant_cpu(int8_t* data_im,
                      int channels,  int height,  int width,
                      int ksize,  int stride, int pad,
                      int8_t QZ, int8_t* data_col);
#ifdef GPU

void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

void im2col_quant_gpu(int8_t *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,
         const int8_t QZ, int8_t *data_col);

void im2col_fake_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,
         const int8_t QZ, float *data_col);
#endif
#endif

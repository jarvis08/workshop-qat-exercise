#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
#include "quant_utils.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabsf(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabsf(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
#endif

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
}

void quantization_aware_training_convolutional_layer_int4_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    float _min, _max;
    get_min_max_cpu(l.weights, &_min, &_max, l.nweights);
    fake_quantize_int4(l.weights_gpu, l.nweights, _min, _max);

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    if (l.ema_init[0]) {
        ema_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs, net.ema_smooth, l.activation == RELU);
        fake_quantize_int4(l.output_gpu, l.batch * l.outputs, l.act_range[0], l.act_range[1]);
    } else {
        get_min_max_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs);
        l.ema_init[0] = 1;
    }
}

void quantization_aware_training_convolutional_layer_int8_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    float _min, _max;
    get_min_max_cpu(l.weights, &_min, &_max, l.nweights);
    fake_quantize_int8(l.weights_gpu, l.nweights, _min, _max);

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    if(l.ema_init[0]) {
        ema_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs, net.ema_smooth, l.activation == RELU);
        fake_quantize_int8(l.output_gpu, l.batch * l.outputs, l.act_range[0], l.act_range[1]);
    } else {
        get_min_max_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs);
        l.ema_init[0] = 1;
    }
}

void qat_convolutional_layer_with_batch_of_mixed_clusters_int4_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    float _min, _max;
    get_min_max_cpu(l.weights, &_min, &_max, l.nweights);
    fake_quantize_int4(l.weights_gpu, l.nweights, _min, _max);

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    int ctr, n_data;
    int idx = 0;
    float* r;
    for(i = 0; i < net.ctr_k; i++) {
        ctr = net.ctr_info[i * 2];
        if(ctr == -1) break;

        n_data = l.outputs * net.ctr_info[i * 2 + 1];
        r = &l.act_cluster[ctr * 2];
        if(l.ema_init[ctr]) {
            ema_cpu(&l.output[idx], r, r + 1, n_data, net.ema_smooth, l.activation == RELU);
            fake_quantize_int4(&l.output_gpu[idx], n_data, *r, *(r + 1));
        } else {
            get_min_max_cpu(&l.output[idx], r, r + 1, n_data);
            l.ema_init[ctr] = 1;
        }
        idx += n_data;
    }
}

void qat_per_cluster_convolutional_layer_int4_gpu(layer l, network net, const int ctr)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    float _min, _max;
    get_min_max_cpu(l.weights, &_min, &_max, l.nweights);
    fake_quantize_int4(l.weights_gpu, l.nweights, _min, _max);

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    float *r = &l.act_cluster[ctr * 2];
    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    if(l.ema_init[ctr]) {
        ema_cpu(l.output, r, r + 1, l.batch * l.outputs, net.ema_smooth, l.activation == RELU);
        fake_quantize_int4(l.output_gpu, l.batch * l.outputs, *r, *(r+1));
    } else {
        get_min_max_cpu(l.output, r, r + 1, l.batch * l.outputs);
        l.ema_init[ctr] = 1;
    }
}

void qat_per_cluster_convolutional_layer_int8_gpu(layer l, network net, const int ctr)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    float _min, _max, QS;
    int8_t QZ;
    get_min_max_cpu(l.weights, &_min, &_max, l.nweights);
    cal_qsz(_min, _max, &QS, &QZ);
    fake_quantize_int8_gpu(l.weights_gpu, l.nweights, QS, QZ);

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    float *r = &l.act_cluster[ctr * 2];
    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    if(l.ema_init[ctr]) {
        ema_cpu(l.output, r, r + 1, l.batch * l.outputs, net.ema_smooth, l.activation == RELU);
        float QS;
        int8_t QZ;
        cal_qsz(*r, *(r + 1), &QS, &QZ);
        fake_quantize_int8_gpu(l.output_gpu, l.batch * l.outputs, QS, QZ);
    } else {
        get_min_max_cpu(l.output, r, r + 1, l.batch * l.outputs);
        l.ema_init[ctr] = 1;
    }
}


void forward_convolutional_layer_ema_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    if(l.ema_init[0]) {
        ema_gpu(l.output_gpu, l.batch * l.outputs, l.act_range, net.ema_smooth);
    } else {
        set_min_max_gpu(l.output_gpu, l.batch * l.outputs, l.act_range);
        l.ema_init[0] = 1;
    }
}

void forward_convolutional_layer_qparams_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    set_range_gpu(l.output_gpu, l.act_range, l.batch * l.outputs, l.activation == RELU);
}

void forward_convolutional_layer_quant_gpu(convolutional_layer l, network net, int dequant)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            int8_t *a = l.weights_int8_gpu + j*l.nweights/l.groups;
            int8_t *b = net.workspace_int8_gpu;
            int32_t *c32 = l.output_int32_gpu + (i*l.groups + j)*n*m;
            int8_t *im = net.input_int8_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_quant_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, l.qz[0], b);
            }
            //gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);

            int8_t *c = l.output_int8 + (i*l.groups + j)*n*m;
            cuda_pull_array_int32(c32, l.output_int32, n);
            get_nn_totalsum_gpu(m, n, k, a, k, b, n, c, n, l.output_int32, l.biases_int32, l.qs, l.qz);
        }
    }

    if(dequant) {
        dequantize_matrix(l.output_int8, l.output, l.batch, l.outputs, l.qs[2], l.qz[2]);
        cuda_push_array(l.output_gpu, l.output, l.outputs*l.batch);
    } else {
        cuda_push_array_int8(l.output_int8_gpu, l.output_int8, l.outputs*l.batch);
    }
}

void forward_convolutional_layer_fake_int4_gpu(convolutional_layer l, network net, int dequant)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

	int4_to_float(net.input_int4, net.input, l.inputs * l.batch);
	cuda_push_array(net.input_gpu, net.input, l.inputs * l.batch);

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
			// for gemm
            float *a_fake = l.fake_int_weights_gpu + j*l.nweights/l.groups; // fixed, (float)weights_int8: to use gemm_gpu
            float *b_fake = net.workspace; // fixed
            float *im_fake = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w; // made above 
            float *c32_fake = l.output_gpu + (i*l.groups + j)*n*m; // fixed

			// for totalsum
            int4_t *a = l.weights_int4 + j*l.nweights/l.groups;
            int4_t *b = net.workspace_int4;
            int4_t *c = l.output_int4 + (i*l.groups + j)*n*m;
            int4_t *im = net.input_int4 + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            int32_t *c32 = l.output_int32 + (i*l.groups + j)*n*m;

            if (l.size == 1){
                b_fake = im_fake;
                b = im;
            } else {
                im2col_fake_gpu(im_fake, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, l.qz_int4[0].el, b_fake);
				im2col_int4_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, l.qz_int4[0], b);
            }
            gemm_gpu(0,0,m,n,k,1,a_fake,k,b_fake,n,1,c32_fake,n);

            cuda_pull_array(c32_fake, l.output, m * n);
			float_to_int32(l.output, c32, m * n);
            totalsum_int4_cpu(m, n, k, a, b, c, c32, l.biases_int32, l.qs, l.qz_int4, 0);
        }
    }
    if(dequant) {
        dequantize_int4_cpu(l.output_int4, l.output, l.batch * l.outputs, l.qs[2], l.qz_int4[2].el);
        cuda_push_array(l.output_gpu, l.output, l.outputs*l.batch);
    }
}

void forward_convolutional_layer_fake_int8_gpu(convolutional_layer l, network net, int dequant)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    // to use input_gpu as fake int
    cuda_pull_array_int8(net.input_int8_gpu, net.input_int8, l.inputs*l.batch);
	int8_to_float(net.input_int8, net.input, l.inputs*l.batch);
	cuda_push_array(net.input_gpu, net.input, l.inputs*l.batch);

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
			// for gemm
            float *a_fake = l.fake_int_weights_gpu + j*l.nweights/l.groups; // fixed, (float)weights_int8: to use gemm_gpu
            float *b_fake = net.workspace; // fixed
            float *im_fake = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w; // made above 
            float *c32_fake = l.output_gpu + (i*l.groups + j)*n*m; // fixed

			// for totalsum
            int8_t *a = l.weights_int8 + j*l.nweights/l.groups;
            int8_t *b = net.workspace_int8;
            int8_t *c = l.output_int8 + (i*l.groups + j)*n*m;
            int8_t *im = net.input_int8 + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            int32_t *c32 = l.output_int32 + (i*l.groups + j)*n*m;

            if (l.size == 1){
                b_fake = im_fake;
                b = im;
            } else {
                im2col_fake_gpu(im_fake, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, l.qz[0], b_fake);
				im2col_quant_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, l.qz[0], b);
            }
            gemm_gpu(0,0,m,n,k,1,a_fake,k,b_fake,n,1,c32_fake,n);

            cuda_pull_array(c32_fake, l.output, m * n);
			float_to_int32(l.output, c32, m * n);
            //get_nn_totalsum_gpu(m, n, k, a, k, b, n, c, n, c32, l.biases_int32, l.qs, l.qz);
            totalsum_int8_cpu(m, n, k, a, b, c, c32, l.biases_int32, l.qs, l.qz, 0);
        }
    }
    if(dequant) {
        dequantize_matrix(l.output_int8, l.output, l.batch, l.outputs, l.qs[2], l.qz[2]);
        cuda_push_array(l.output_gpu, l.output, l.outputs*l.batch);
    } else {
        cuda_push_array_int8(l.output_int8_gpu, l.output_int8, l.outputs*l.batch);
    }
}

void forward_convolutional_layer_cudnn(convolutional_layer l, network net, int dequant)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_int8_gpu,
                l.weightDesc,
                l.weights_int8_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace_int8_gpu,
                l.workspace_cudnn_size,
                //l.workspace_int8_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);
#endif
    cuda_pull_array_int8(net.input_int8_gpu, net.input_int8, l.batch * l.inputs);
    cuda_pull_array_int8(l.weights_int8_gpu, l.weights_int8, l.nweights);
    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    float_to_int32(l.output, l.output_int32, l.batch * l.outputs);

	// for totalsum
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    int i, j;
    //for(i=0;i<l.outputs/100;i++){
    //    printf("%f\n", l.output[i]);
    //}
    //exit(0);
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            int8_t *b = net.workspace_int8;
            int8_t *im = net.input_int8 + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            if (l.size == 1){
                b = im;
            } else {
				im2col_quant_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, l.qz[0], b);
            }

            int8_t *a = l.weights_int8 + j*l.nweights/l.groups;
            int8_t *c = l.output_int8 + (i*l.groups + j)*n*m;
            int32_t *c32 = l.output_int32 + (i*l.groups + j)*n*m;
            get_nn_totalsum_gpu(m, n, k, a, k, b, n, c, n, c32, l.biases_int32, l.qs, l.qz);
        }
    }

    if(dequant) {
        dequantize_matrix(l.output_int8, l.output, l.batch, l.outputs, l.qs[2], l.qz[2]);
        cuda_push_array(l.output_gpu, l.output, l.outputs*l.batch);
    } else {
        cuda_push_array_int8(l.output_int8_gpu, l.output_int8, l.outputs*l.batch);
    }
}

__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.f);
    int h_offset = -(size/2.f);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    float *original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(net.delta_gpu){
        if(l.binary || l.xnor) swap_binary(&l);
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                net.delta_gpu);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
    }

#else
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;

            float *im  = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta_gpu) {
                if (l.binary || l.xnor) swap_binary(&l);
                a = l.weights_gpu + j*l.nweights/l.groups;
                b = l.delta_gpu + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
                if(l.binary || l.xnor) {
                    swap_binary(&l);
                }
            }
            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
        }
    }
#endif
}

void pull_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_quantized_convolutional_layer(layer l)
{
    cuda_push_array_int8(l.weights_int8_gpu, l.weights_int8, l.nweights);
}

//void update_convolutional_layer_gpu(layer l, update_args a)
//{
//    float learning_rate = a.learning_rate*l.learning_rate_scale;
//    float momentum = a.momentum;
//    float decay = a.decay;
//    int batch = a.batch;
//
//    if(a.adam){
//        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
//        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
//        if(l.scales_gpu){
//            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
//        }
//    }else{
//        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
//        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
//        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);
//
//        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
//        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);
//
//        if(l.scales_gpu){
//            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
//            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
//        }
//    }
//    if(l.clip){
//        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
//    }
//}

void update_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        momentum_gpu(l.n, momentum, learning_rate/batch, l.bias_updates_gpu, l.bias_velocity_gpu);
        axpy_gpu(l.n, 1, l.bias_velocity_gpu, 1, l.biases_gpu, 1);

        momentum_gpu(l.nweights, momentum, learning_rate/batch, l.weight_updates_gpu, l.weight_velocity_gpu);
        axpy_gpu(l.nweights, 1, l.weight_velocity_gpu, 1, l.weights_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    if(l.clip){
        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
    }
}

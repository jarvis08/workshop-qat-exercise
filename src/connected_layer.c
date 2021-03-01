#include "connected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"
#include "quant_utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

    l.output = calloc(batch*outputs, sizeof(float));
    l.delta = calloc(batch*outputs, sizeof(float));

    l.weight_updates = calloc(inputs*outputs, sizeof(float));
    l.bias_updates = calloc(outputs, sizeof(float));

    l.weights = calloc(outputs*inputs, sizeof(float));
    l.biases = calloc(outputs, sizeof(float));

    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }

    for(i = 0; i < outputs; ++i){
        l.biases[i] = 0;
    }

    if(adam){
        l.m = calloc(l.inputs*l.outputs, sizeof(float));
        l.v = calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }
    if(batch_normalize){
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch*outputs, sizeof(float));
        l.x_norm = calloc(batch*outputs, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_connected_layer_gpu;
    l.backward_gpu = backward_connected_layer_gpu;
    l.update_gpu = update_connected_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
    if (adam) {
        l.m_gpu =       cuda_make_array(0, inputs*outputs);
        l.v_gpu =       cuda_make_array(0, inputs*outputs);
        l.bias_m_gpu =  cuda_make_array(0, outputs);
        l.bias_v_gpu =  cuda_make_array(0, outputs);
        l.scale_m_gpu = cuda_make_array(0, outputs);
        l.scale_v_gpu = cuda_make_array(0, outputs);
    }

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
#endif
    }
#endif
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

layer make_connected_layer_quant(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam, char *argv)
{
    int i;
    layer l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch = batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    l.delta = calloc(batch*outputs, sizeof(float));
    l.weights = calloc(outputs*inputs, sizeof(float));
    l.weight_updates = calloc(inputs*outputs, sizeof(float));
    l.output = calloc(batch*outputs, sizeof(float));
    l.biases = calloc(outputs, sizeof(float));
    l.bias_updates = calloc(outputs, sizeof(float));
    
    // TF default initializer : glorot_uniform (Xavier)
    // But, xavier makes too much zeros if used with ReLU
    // float scale = sqrt(6./(inputs + outputs));
    // He initializer
     float scale = sqrt(6./inputs);
    //float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }
    for(i = 0; i < outputs; ++i){
        l.biases[i] = 0;
    }

    l.weight_velocity = calloc(inputs*outputs, sizeof(float));
    l.bias_velocity = calloc(outputs, sizeof(float));

    // Fine-tuning
    l.weights_mask = calloc(outputs*inputs, sizeof(float));

    // Quantization
    l.weights_int8 = calloc(outputs*inputs, sizeof(int8_t));
    l.biases_int32 = calloc(outputs, sizeof(int32_t));
    l.output_int8 = calloc(batch*outputs, sizeof(int8_t));
    l.qs = calloc(3, sizeof(float));
    l.qz = calloc(3, sizeof(int8_t));
    l.ema_init = calloc(10, sizeof(int));
    l.act_range = calloc(2, sizeof(float));
    l.act_cluster = calloc(20, sizeof(float));
    l.weights_range = calloc(2, sizeof(float));

    int4_t tmp;
    l.qz_int4 = malloc(3 * sizeof(int4_t));
    l.weights_int4 = malloc(outputs * inputs * sizeof(int4_t));
    l.output_int4 = malloc(outputs * batch * sizeof(int4_t));
    for(i = 0; i < 3; i++) l.qz_int4[i] = tmp;
    for(i = 0; i < outputs * inputs; i++) l.weights_int4[i] = tmp;
    for(i = 0; i < outputs * batch; i++) l.output_int4[i] = tmp;

    l.fake_int_weights = calloc(outputs * inputs, sizeof(float));

    l.forward_int8 = forward_connected_layer_int8;
    l.forward_cmp = forward_connected_layer_comparison;
    l.forward_csv = forward_connected_layer_save_into_csv;
    l.forward_qparams = forward_connected_layer_qparams;
    l.forward_qat_int8 = quantization_aware_training_connected_layer_int8;

    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    if(adam){
        l.m = calloc(l.inputs*l.outputs, sizeof(float));
        l.v = calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }
    if(batch_normalize){
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch*outputs, sizeof(float));
        l.x_norm = calloc(batch*outputs, sizeof(float));
    }

#ifdef GPU
    l.forward_int4_gpu = forward_connected_layer_fake_int4_gpu;
    l.forward_int8_gpu = forward_connected_layer_int8_gpu;

    l.forward_qat_int4_gpu = quantization_aware_training_connected_layer_int4_gpu;
    l.forward_qat_int8_gpu = quantization_aware_training_connected_layer_int8_gpu;
    l.forward_bmc_int4_gpu = qat_connected_layer_with_batch_of_mixed_clusters_int4_gpu;
    l.forward_qat_cluster_int4_gpu = qat_per_cluster_connected_layer_int4_gpu;
    l.forward_qat_cluster_int8_gpu = qat_per_cluster_connected_layer_int8_gpu;

    l.forward_qparams_gpu = forward_connected_layer_qparams_gpu;
    l.forward_ema_gpu = forward_connected_layer_ema_gpu;

    l.forward_gpu = forward_connected_layer_gpu;
    l.backward_gpu = backward_connected_layer_gpu;
    l.update_gpu = update_connected_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.weight_velocity_gpu = cuda_make_array(l.weight_velocity, outputs*inputs);
    l.bias_velocity_gpu = cuda_make_array(l.bias_velocity, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);

    // quantization
    l.fake_int_weights_gpu = cuda_make_array(l.fake_int_weights, outputs*inputs);
    l.weights_int8_gpu = cuda_make_array_int8(l.weights_int8, outputs*inputs);
    l.output_int8_gpu = cuda_make_array_int8(l.output_int8, batch*outputs);

    l.output_int32 = calloc(batch*outputs, sizeof(int32_t));
    l.output_int32_gpu = cuda_make_array_int32(l.output_int32, outputs*batch);

    if (adam) {
        l.m_gpu =       cuda_make_array(0, inputs*outputs);
        l.v_gpu =       cuda_make_array(0, inputs*outputs);
        l.bias_m_gpu =  cuda_make_array(0, outputs);
        l.bias_v_gpu =  cuda_make_array(0, outputs);
        l.scale_m_gpu = cuda_make_array(0, outputs);
        l.scale_v_gpu = cuda_make_array(0, outputs);
    }

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
#endif
    }
#endif
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

void forward_connected_layer(layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;
    
    /*
    // Save weights into csv...
    char weight_buff[256];
    sprintf(weight_buff, "cifar-pruned.csv");
    printf("CSV save mode >> %s\n", weight_buff);
    FILE* weight_csv = fopen(weight_buff, "w");

    int i;
    int len_weights = k * n;
    for(i = 0; i < len_weights; i++) fprintf(weight_csv, "%d%s", i, (i < len_weights - 1 ? "," : "\n"));
    for(i = 0; i < len_weights; i++) fprintf(weight_csv, "%f%s", l.weights[i], (i < len_weights - 1 ? "," : "\n"));

    fclose(weight_csv);
    exit(0);
    // Saving Done.
    // */
    
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void forward_connected_layer_qparams(layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    /*
    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;

    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
    push_connected_layer(l);
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
    pull_connected_layer(l);
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);

    set_min_max(l.output, l.outputs*l.batch, l.qs3_min_max);

    add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    set_min_max(l.output, l.outputs*l.batch, l.qa_min_max);

    activate_array(l.output, l.outputs*l.batch, l.activation);
    */
}

void forward_connected_layer_int8(layer l, network net, int dequant)
{
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

    fill_cpu_int8(l.outputs*l.batch, 0, l.output_int8, 1);
    int8_t *a = net.input_int8;
    int8_t *b = l.weights_int8;
    int8_t *c = l.output_int8;
    int32_t *bias = l.biases_int32;
    gemm_nt_quant(m,n,k,a,k,b,k,c,n,bias,l.qs,l.qz);
    if(dequant) dequantize_matrix(l.output_int8, l.output, m, n, l.qs[2], l.qz[2]);
}

void forward_connected_layer_comparison(layer l, network net, int dequant)
{
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

    // original float gemm
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    float *oa = net.input;
    float *ob = l.weights;
    float *oc = l.output;
    gemm(0,1,m,n,k,1,oa,k,ob,k,1,oc,n);

    add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
    //if(dequant) {
    //    int i;
    //    for(i=0; i<10;i++) printf("%f\n", i, l.output[i]);
    //    printf("\n");
    //}

    fill_cpu_int8(l.outputs*l.batch, 0, l.output_int8, 1);
    int8_t *qa = net.input_int8;
    int8_t *qb = l.weights_int8;
    int8_t *qc = l.output_int8;
    int32_t *bias = l.biases_int32;
    gemm_nt_quant(m,n,k,qa,k,qb,k,qc,n,bias,l.qs,l.qz);
    //if(dequant) {
    //    char buff_batch[256];
    //    sprintf(buff_batch, "test.csv");
    //    FILE *fp_batch = fopen(buff_batch, "a");
    //    int j;
    //    for(j = 0; j < l.outputs; j++) {
    //            fprintf(fp_batch, "%f%s", l.output[j], (j < l.outputs - 1 ? "," : "\n"));
    //    }
    //    fflush(fp_batch);
    //    dequantize_matrix(l.output_int8, l.output, l.batch, l.outputs, l.qs[2], l.qz[2]);
    //    for(j = 0; j < l.outputs; j++) {
    //            fprintf(fp_batch, "%f%s", l.output[j], (j < l.outputs - 1 ? "," : "\n"));
    //    }
    //    fflush(fp_batch);
    //    for(j = 0; j < l.outputs; j++) {
    //            fprintf(fp_batch, "%d%s", l.output_int8[j], (j < l.outputs - 1 ? "," : "\n"));
    //    }
    //    fflush(fp_batch);
    //}
    //if(dequant) {
    //    dequantize_matrix(l.output_int8, l.output, l.batch, l.outputs, l.qs[2], l.qz[2]);
    //    int i;
    //    for(i=0; i<10;i++) printf("%f\n", i, l.output[i]);
    //    printf("\n");
    //}

    compare_output(l.output_int8, l.output, m, n, l.qs[2], l.qz[2]);
}

void forward_connected_layer_save_into_csv(layer l, network net, int dequant, FILE *output_csv, FILE *weight_csv)
{
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

    // original float gemm
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    float *oa = net.input;
    float *ob = l.weights;
    float *oc = l.output;
    gemm(0,1,m,n,k,1,oa,k,ob,k,1,oc,n);

    // write output before bias addition
    /*
    if(output_csv) {
        int i;
        int len_output = m * n;
        
        for(i = 0; i < len_output; i++) fprintf(output_csv, "%f%s", l.output[i], (i < len_output - 1 ? "," : "\n"));
    }
    */

    add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);

    // quantized gemm
    fill_cpu_int8(l.outputs*l.batch, 0, l.output_int8, 1);
    int8_t *qa = net.input_int8;
    int8_t *qb = l.weights_int8;
    int8_t *qc = l.output_int8;
    int32_t *bias = l.biases_int32;
    gemm_nt_quant(m,n,k,qa,k,qb,k,qc,n,bias,l.qs,l.qz);
    if(!dequant) activate_array_relu_quant(l.output_int8, l.outputs*l.batch, l.qz[3]);

    /*
    if(output_csv) {
        int i;
        int len_output = m * n;
        float *output_dequant = calloc(len_output, sizeof(float));
        dequantize_matrix(l.output_int8, output_dequant, m, n, l.qs[3], l.qz[3]);
        
        //for(i = 0; i < len_output; i++) fprintf(output_csv, "%d%s", l.output_int8[i], (i < len_output - 1 ? "," : "\n"));
        for(i = 0; i < len_output; i++) fprintf(output_csv, "%f%s", output_dequant[i], (i < len_output - 1 ? "," : "\n"));
        for(i = 0; i < len_output; i++) fprintf(output_csv, "%f%s", l.output[i], (i < len_output - 1 ? "," : "\n"));
        free(output_dequant);
    }
    */

    if(weight_csv) {
        int i;
        int len_weights = k * n;
        //float *weights_dequant = calloc(len_weights, sizeof(float));
        //dequantize_matrix(l.weights_int8, weights_dequant, m, n, l.qs[1], l.qz[1]);
        
        if(n == 1024) {
            for(i = 0; i < len_weights; i++) fprintf(weight_csv, "%d%s", i, (i < len_weights - 1 ? "," : "\n"));
        }
        //for(i = 0; i < len_weights; i++) fprintf(weight_csv, "%d%s", l.weights_int8[i], (i < len_weights - 1 ? "," : "\n"));
        //for(i = 0; i < len_weights; i++) fprintf(weight_csv, "%f%s", weights_dequant[i], (i < len_weights - 1 ? "," : "\n"));
        //for(i = 0; i < len_weights; i++) fprintf(weight_csv, "%f%s", l.weights[i], (i < len_weights - 1 ? "," : "\n"));
        float _min = l.weights[0];
        float _max = l.weights[0];
        for(i = 0; i < len_weights; i++) {
            if(l.weights[i] > _max) _max = l.weights[i];
            if(l.weights[i] < _min) _min = l.weights[i];
            fprintf(weight_csv, "%f%s", l.weights[i], (i < len_weights - 1 ? "," : "\n"));
        }
        printf("min = %f\t|\tmax = %f\n", _min, _max);
        //free(weights_dequant);
    }

    // cal and print diff between float arith & integer arith
    compare_output(l.output_int8, l.output, m, n, l.qs[3], l.qz[3]);
}

void quantization_aware_training_connected_layer_int8(layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;

    int step = get_current_batch(&net);
    if(step > net.qat_init_step) {
        float _min, _max;
        get_min_max_cpu(l.weights, &_min, &_max, k * n);
        fake_quantize_int4(l.weights, k * n, _min, _max);
    }

    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);

    if(step > net.qat_init_step) {
        ema_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs, net.ema_smooth, l.activation == RELU);
        fake_quantize_int8(l.output, l.batch * l.outputs, l.act_range[0], l.act_range[1]);
    } else {
        get_min_max_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs);
    }
}


void backward_connected_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = net.input;
    float *c = l.weight_updates;
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;

    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}


void update_connected_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize){
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}


void denormalize_connected_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.outputs; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .000001);
        for(j = 0; j < l.inputs; ++j){
            l.weights[i*l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}


void statistics_connected_layer(layer l)
{
    if(l.batch_normalize){
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l.rolling_mean, l.outputs);
           printf("Rolling Variance ");
           print_statistics(l.rolling_variance, l.outputs);
         */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}

#ifdef GPU
void pull_connected_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);

    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void push_connected_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);

    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void push_connected_layer_quant(layer l)
{
    cuda_push_array_int8(l.weights_int8_gpu, l.weights_int8, l.inputs*l.outputs);
}

void forward_connected_layer_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;

//    char weight_buff[256];
//    sprintf(weight_buff, "cifar-pruned.csv");
//    printf("CSV save mode >> %s\n", weight_buff);
//    FILE* weight_csv = fopen(weight_buff, "a");
//
//    double times;
//    times = what_time_is_it_now();
//
//    fprintf(weight_csv, "%lf%s", what_time_is_it_now()-times, (dequant ? "\n" : ","));
//    fclose(weight_csv);

    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void quantization_aware_training_connected_layer_int4_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;

    cuda_pull_array(l.weights_gpu, l.weights, k * n);
    float _min, _max;
    get_min_max_cpu(l.weights, &_min, &_max, k * n);
    fake_quantize_int4(l.weights_gpu, k * n, _min, _max);

    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
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

void quantization_aware_training_connected_layer_int8_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;

    cuda_pull_array(l.weights_gpu, l.weights, k * n);
    float _min, _max, QS;
    int8_t QZ;
    get_min_max_cpu(l.weights, &_min, &_max, k * n);
    cal_qsz(_min, _max, &QS, &QZ);
    fake_quantize_int8_gpu(l.weights_gpu, k * n, QS, QZ);

    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    if(l.ema_init[0]) {
        ema_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs, net.ema_smooth, l.activation == RELU);
        float QS;
        int8_t QZ;
        cal_qsz(l.act_range[0], l.act_range[1], &QS, &QZ);
        fake_quantize_int8_gpu(l.output_gpu, l.batch * l.outputs, QS, QZ);
    } else {
        get_min_max_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs);
        l.ema_init[0] = 1;
    }
}

void qat_connected_layer_with_batch_of_mixed_clusters_int4_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;

    cuda_pull_array(l.weights_gpu, l.weights, k * n);
    float _min, _max;
    get_min_max_cpu(l.weights, &_min, &_max, k * n);
    fake_quantize_int4(l.weights_gpu, k * n, _min, _max);

    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
    add_bias_gpu(l.output_gpu, l.biases_gpu, m, n, 1);
    activate_array_gpu(l.output_gpu, m * n, l.activation);

    cuda_pull_array(l.output_gpu, l.output, m * n);
    int i, ctr, n_data;
    int idx = 0;
    float* r;
    for(i = 0; i < net.ctr_k; i++) {
        ctr = net.ctr_info[i * 2];
        if(ctr == -1) break;

        n_data = n * net.ctr_info[i * 2 + 1];
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

void qat_per_cluster_connected_layer_int4_gpu(layer l, network net, const int ctr)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;

    cuda_pull_array(l.weights_gpu, l.weights, k * n);
    float _min, _max;
    get_min_max_cpu(l.weights, &_min, &_max, k * n);
    fake_quantize_int4(l.weights_gpu, k * n, _min, _max);

    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
    add_bias_gpu(l.output_gpu, l.biases_gpu, m, n, 1);
    activate_array_gpu(l.output_gpu, m * n, l.activation);

    float *r = &l.act_cluster[ctr * 2];
    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    if(l.ema_init[ctr]) {
        ema_cpu(l.output, r, r + 1, m * n, net.ema_smooth, l.activation == RELU);
        fake_quantize_int4(l.output_gpu, m * n, *r, *(r + 1));
    } else {
        get_min_max_cpu(l.output, r, r + 1, m * n);
        l.ema_init[ctr] = 1;
    }
}

void qat_per_cluster_connected_layer_int8_gpu(layer l, network net, const int ctr)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;

    cuda_pull_array(l.weights_gpu, l.weights, k * n);
    float _min, _max;
    get_min_max_cpu(l.weights, &_min, &_max, k * n);
    fake_quantize_int8(l.weights_gpu, k * n, _min, _max);

    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    activate_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation);

    float *r = &l.act_cluster[ctr * 2];
    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    if(l.ema_init[ctr]) {
        ema_cpu(l.output, r, r + 1, m * n, net.ema_smooth, l.activation == RELU);
        fake_quantize_int8(l.output_gpu, m * n, *r, *(r + 1));
    } else {
        get_min_max_cpu(l.output, r, r + 1, m * n);
        l.ema_init[ctr] = 1;
    }
}

void forward_connected_layer_fake_int4_gpu(layer l, network net, int dequant)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

	int4_to_float(net.input_int4, net.input, m * k);
	cuda_push_array(net.input_gpu, net.input,  m * k);

    float *a_fake = net.input_gpu;
    float *b_fake = l.fake_int_weights_gpu;
    float *c32_fake = l.output_gpu;
    gemm_gpu(0,1,m,n,k,1,a_fake,k,b_fake,k,1,c32_fake,n);

    int4_t *a = net.input_int4;
    int4_t *b = l.weights_int4;
    int4_t *c = l.output_int4;
    int32_t *c32 = l.output_int32;

    cuda_pull_array(c32_fake, l.output, m * n);
    float_to_int32(l.output, c32, m * n);

    totalsum_int4_cpu(m, n, k, a, b, c, c32, l.biases_int32, l.qs, l.qz_int4, 1);

    if(dequant) {
        dequantize_int4_cpu(c, l.output, m * n, l.qs[2], l.qz_int4[2].el);
        cuda_push_array(l.output_gpu, l.output, m * n);
    }
}

void forward_connected_layer_int8_gpu(layer l, network net, int dequant)
{
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;

    fill_gpu_int32(l.outputs*l.batch, 0, l.output_int32_gpu, 1);

    int8_t * a = net.input_int8_gpu;
    int8_t * b = l.weights_int8_gpu;
    int8_t * c = l.output_int8;        // int8 real output, allocated calloc
    int32_t *c32 = l.output_int32_gpu;    // int32 output [q1(i, j) * q2(j, k)], allocated cuda
    int32_t *bias = l.biases_int32;

//    char weight_buff[256];
//    sprintf(weight_buff, "cifar-pruned.csv");
//    printf("CSV save mode >> %s\n", weight_buff);
//    FILE* weight_csv = fopen(weight_buff, "a");
//
//    double times;
//    times = what_time_is_it_now();
//
//    fprintf(weight_csv, "%lf%s", what_time_is_it_now()-times, (dequant ? "\n" : ","));
//    fclose(weight_csv);

    // matrix multiplication, [q1(i, j) * q2(j, k)]
    gemm_gpu_cublasGemmEx_int32(0, 1, m, n, k, 1, a, k, b, k, 1, c32, n);

    int32_t *c32_cpu = calloc(l.outputs*l.batch,sizeof(int32_t));
    cuda_pull_array_int32(c32, c32_cpu, n * m);

    // extract real output, int8
    cuda_pull_array_int8(net.input_int8_gpu, net.input_int8, m * k);
    //get_nt_totalsum_gpu(m, n, k, net.input_int8, k, l.weights_int8, k, c, n, c32_cpu, bias, l.qs, l.qz);
    totalsum_int8_cpu(m, n, k, net.input_int8, l.weights_int8, c, c32_cpu, bias, l.qs, l.qz, 1);

    free(c32_cpu);

    if(dequant) {
        dequantize_matrix(l.output_int8, l.output, m, n, l.qs[2], l.qz[2]);
        cuda_push_array(l.output_gpu, l.output, l.outputs*l.batch);
    } else {
        cuda_push_array_int8(l.output_int8_gpu, l.output_int8, l.outputs*l.batch);
    }

}

void forward_connected_layer_ema_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input_gpu;
    float *b = l.weights_gpu;
    float *c = l.output_gpu;

    gemm_gpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    if(l.ema_init[0]) {
        ema_gpu(l.output_gpu, l.batch * l.outputs, l.act_range, net.ema_smooth);
    } else {
        set_min_max_gpu(l.output_gpu, l.batch * l.outputs, l.act_range);
        l.ema_init[0] = 1;
    }
}

void forward_connected_layer_qparams_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input_gpu;
    float *b = l.weights_gpu;
    float *c = l.output_gpu;

    gemm_gpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    set_range_gpu(l.output_gpu, l.act_range, l.batch * l.outputs, l.activation == RELU);
}

void backward_connected_layer_gpu(layer l, network net)
{
    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float * a = l.delta_gpu;
    float * b = net.input_gpu;
    float * c = l.weight_updates_gpu;
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = net.delta_gpu;

    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
}

//void update_connected_layer_gpu(layer l, update_args a)
//{
//    float learning_rate = a.learning_rate*l.learning_rate_scale;
//    float momentum = a.momentum;
//    float decay = a.decay;
//    int batch = a.batch;
//    if(a.adam){
//        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
//        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
//        if(l.scales_gpu){
//            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
//        }
//    }else{
//        axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
//        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);
//
//        axpy_gpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
//        axpy_gpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
//        scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
//    }
//}

void update_connected_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        }
    }else{
        momentum_gpu(l.outputs, momentum, learning_rate/batch, l.bias_updates_gpu, l.bias_velocity_gpu);
        axpy_gpu(l.outputs, 1, l.bias_velocity_gpu, 1, l.biases_gpu, 1);

        momentum_gpu(l.inputs*l.outputs, momentum, learning_rate/batch, l.weight_updates_gpu, l.weight_velocity_gpu);
        axpy_gpu(l.inputs*l.outputs, 1, l.weight_velocity_gpu, 1, l.weights_gpu, 1);
    }
}
#endif

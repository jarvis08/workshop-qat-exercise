#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer convolutional_layer;

#ifdef GPU
void push_quantized_convolutional_layer(convolutional_layer layer);
void quantization_aware_training_convolutional_layer_int4_gpu(layer l, network net);
void quantization_aware_training_convolutional_layer_int8_gpu(layer l, network net);
void qat_convolutional_layer_with_batch_of_mixed_clusters_int4_gpu(layer l, network net);
void qat_per_cluster_convolutional_layer_int4_gpu(layer l, network net, const int ctr);
void qat_per_cluster_convolutional_layer_int8_gpu(layer l, network net, const int ctr);
void forward_convolutional_layer_ema_gpu(convolutional_layer l, network net);
void forward_convolutional_layer_qparams_gpu(convolutional_layer l, network net);
void forward_convolutional_layer_int8_gpu(convolutional_layer l, network net, int dequant);
void forward_convolutional_layer_fake_int4_gpu(convolutional_layer l, network net, int dequant);
void forward_convolutional_layer_fake_int8_gpu(convolutional_layer l, network net, int dequant);

void forward_convolutional_layer_gpu(convolutional_layer layer, network net);
void backward_convolutional_layer_gpu(convolutional_layer layer, network net);
void update_convolutional_layer_gpu(convolutional_layer layer, update_args a);

void push_convolutional_layer(convolutional_layer layer);
void pull_convolutional_layer(convolutional_layer layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
#ifdef CUDNN
void cudnn_quantized_convolutional_setup(layer *l);
void cudnn_convolutional_setup(layer *l);
void forward_convolutional_layer_cudnn(convolutional_layer l, network net, int dequant);
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);

convolutional_layer make_convolutional_layer_quant(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, char* argv);
void forward_convolutional_layer_int8(const convolutional_layer layer, network net, int dequant);
void forward_convolutional_layer_comparison(layer l, network net, int dequant);
void forward_convolutional_layer_save_into_csv(layer l, network net, int dequant, FILE *output_csv, FILE *weight_csv);
void quantization_aware_training_convolutional_layer_int8(layer l, network net);

void resize_convolutional_layer(convolutional_layer *layer, int w, int h);
void forward_convolutional_layer(const convolutional_layer layer, network net);
void update_convolutional_layer(convolutional_layer layer, update_args a);
image *visualize_convolutional_layer(convolutional_layer layer, char *window, image *prev_weights);
void binarize_weights(float *weights, int n, int size, float *binary);
void swap_binary(convolutional_layer *l);
void binarize_weights2(float *weights, int n, int size, char *binary, float *scales);

void backward_convolutional_layer(convolutional_layer layer, network net);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

image get_convolutional_image(convolutional_layer layer);
image get_convolutional_delta(convolutional_layer layer);
image get_convolutional_weight(convolutional_layer layer, int i);

int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);

#endif


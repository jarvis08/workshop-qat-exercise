#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

// quantization 
layer make_connected_layer_quant(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam, char *argv);
void forward_connected_layer_int8(layer l, network net, int dequant);
void forward_connected_layer_qparams(layer l, network net);
void forward_connected_layer_comparison(layer l, network net, int dequant);
void forward_connected_layer_save_into_csv(layer l, network net, int dequant, FILE *output_csv, FILE *weight_csv);
void quantization_aware_training_connected_layer_int8(layer l, network net);

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);
void forward_connected_layer(layer l, network net);
void backward_connected_layer(layer l, network net);
void update_connected_layer(layer l, update_args a);

#ifdef GPU
// quantization gpu
void quantization_aware_training_connected_layer_int4_gpu(layer l, network net);
void quantization_aware_training_connected_layer_int8_gpu(layer l, network net);
void qat_connected_layer_with_batch_of_mixed_clusters_int4_gpu(layer l, network net);
void qat_per_cluster_connected_layer_int4_gpu(layer l, network net, const int ctr);
void qat_per_cluster_connected_layer_int8_gpu(layer l, network net, const int ctr);

void forward_connected_layer_qparams_gpu(layer l, network net);
void forward_connected_layer_ema_gpu(layer l, network net);

void forward_connected_layer_int8_gpu(layer l, network net, int dequant);
void forward_connected_layer_fake_int4_gpu(layer l, network net, int dequant);

void forward_connected_layer_gpu(layer l, network net);
void backward_connected_layer_gpu(layer l, network net);
void update_connected_layer_gpu(layer l, update_args a);
void push_connected_layer(layer l);
void pull_connected_layer(layer l);

void push_connected_layer_quant(layer l);
#endif

#endif


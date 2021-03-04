#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"
#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "parser.h"
#include "data.h"
#include "gemm.h"
#include "quant_utils.h"

load_args get_base_args(network *net)
{
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}

network *load_network(char *cfg, char *weights, int clear)
{
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}

network *load_network_from_txt(char *cfg, char *weights, int clear)
{
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights_from_txt(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}

network *load_pruned_network(char *cfg, char *weights, char *argv)
{
    network *net = parse_quantized_network_cfg(cfg, argv);
    if(weights && weights[0] != 0){
        load_pruned_weights(net, weights, 0, net->n);
    }
    return net;
}

network *load_transfer_learning_network(char *cfg, char *weights, char *argv)
{
    network *net = parse_quantized_network_cfg(cfg, argv);
    if(weights && weights[0] != 0){
        load_weights_for_transfer_learning(net, weights);
    }
    return net;
}

network *load_network_qparams(char *cfg, char *weights, char *argv)
{
    network *net = parse_quantized_network_cfg(cfg, argv);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    return net;
}

network *load_network_pruned_qparams(char *cfg, char *weights, char *pruning_params, char *argv)
{
    network *net = parse_quantized_network_cfg(cfg, argv);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(pruning_params && pruning_params[0] != 0){
        load_pruning_params(net, pruning_params, 0, net->n);
    }
    return net;
}

network *load_quantized_network_int4(char *cfg, char *weights, char *qweights, char *qsz, int clear, char *argv)
{
    network *net = parse_quantized_network_cfg(cfg, argv);
    if(weights && weights[0] != 0)
        load_weights(net, weights);
    if(qweights && qweights[0] != 0)
        load_quantized_weights_only_int4(net, qweights, 0, net->n);
    if(qsz && qsz[0] != 0)
		load_qparams_only_int4(net, qsz, 0, net->n); 
    if(clear) (*net->seen) = 0;
    return net;
}

network *load_quantized_network(char *cfg, char *weights, char *qweights, char *qsz, int clear, char *argv)
{
    network *net = parse_quantized_network_cfg(cfg, argv);
    if(weights && weights[0] != 0)
        load_weights(net, weights);
    if(qweights && qweights[0] != 0)
        load_quantized_weights_only(net, qweights, 0, net->n);
    if(qsz && qsz[0] != 0)
		load_qparams_only(net, qsz, 0, net->n); 
    if(clear) (*net->seen) = 0;
    return net;
}

network *load_quantized_network_pruned(char *cfg, char *qweights, char *qsz, char *pruning_params, int clear, char *argv)
{
    network *net = parse_quantized_network_cfg(cfg, argv);
    if(qweights && qweights[0] != 0){
        load_quantized_weights_only(net, qweights, 0, net->n);
    if(qsz && qsz[0] != 0)
		load_qparams_only(net, qsz, 0, net->n); 
    }
    if(pruning_params && pruning_params[0] != 0){
        load_pruning_params(net, pruning_params, 0, net->n);
    }
    if(clear) (*net->seen) = 0;
    return net;
}

size_t get_current_batch(network *net)
{
    size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
    return batch_num;
}

void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i) {
        #ifdef GPU
        layer l = net->layers[i];
        if(l.state_gpu){
            fill_gpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
        }
        if(l.h_gpu){
            fill_gpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
        }
        #endif
    }
}

void reset_rnn(network *net)
{
    reset_network_state(net, 0);
}

float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy) {
        case CONSTANT:
            return net->learning_rate;
        case STEP:
            return net->learning_rate * pow(net->scale, batch_num/net->step);
        case STEPS:
            rate = net->learning_rate;
            for(i = 0; i < net->num_steps; ++i){
                if(net->steps[i] > batch_num) return rate;
                rate *= net->scales[i];
            }
            return rate;
        case EXP:
            return net->learning_rate * pow(net->gamma, batch_num);
        case POLY:
            return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
        case RANDOM:
            return net->learning_rate * pow(rand_uniform(0,1), net->power);
        case SIG:
            return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case LSTM:
	    return "lstm";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case YOLO:
            return "yolo";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    return net;
}

void forward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        forward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }

        l.forward(l, net);
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }

        //if(l.type == CONNECTED) {
        //    // Save weights into csv...
        //    char weight_buff[256];
        //    //sprintf(weight_buff, "tuning_graphs/cifar-clamped_weight-20th-l2reg_0.5-layer-%d.csv", i+1);
        //    sprintf(weight_buff, "tuning_graphs/cifar-tuned-default-layer-%d.csv", i+1);
        //    printf("CSV save mode >> %s\n", weight_buff);
        //    FILE* weight_csv = fopen(weight_buff, "w");

        //    int j;
        //    int len_weights = l.inputs * l.outputs;
        //    for(j = 0; j < len_weights; j++) fprintf(weight_csv, "%d%s", j, (j < len_weights - 1 ? "," : "\n"));
        //    for(j = 0; j < len_weights; j++) fprintf(weight_csv, "%f%s", l.weights[j], (j < len_weights - 1 ? "," : "\n"));

        //    fclose(weight_csv);
        //    if(i == 3)
        //        exit(0);
        //}
    }
    calc_network_cost(netp);
}

void forward_quantization_simulated_network_int8(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        forward_quantization_simulated_network_int8_gpu(netp);
        return;
    }
#endif
    network net = *netp;

    int i;
    int step = get_current_batch(&net);

    // ema
    if(step >= net.qat_init_step) {
        if(step > net.qat_init_step) {
            ema_cpu(net.input, &net.input_range[0], &net.input_range[1], net.batch * net.inputs, 0.05, 0);
        } else {
            set_min_max(net.input, &net.input_range[0], &net.input_range[1], net.inputs * net.batch);
        }
    }

    // propagation
    for(i = 0; i < net.n; ++i) {
        net.index = i;
        layer l = net.layers[i];

        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }

        if(l.type == CONVOLUTIONAL && step >= net.qat_init_step)
        {
            l.forward_qat_int8(l, net);
        }
        else if (l.type == CONNECTED && step >= net.qat_init_step)
        {
            l.forward_qat_int8(l, net);
        }
        else
        {
            l.forward(l, net);
        }

        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}

void forward_network_qparams_staged(network *netp, int stage)
{
    int i;
    network net = *netp;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.type == CONNECTED) {
            if(i < stage) {
                int dequant = (net.layers[i + 1].type == SOFTMAX);
                l.forward_int8(l, net, dequant);
                if(dequant) net.input = l.output;
                else net.input_int8 = l.output_int8;
            } else if(i == stage) {
                dequantize_matrix(net.input_int8, net.input, l.batch, l.inputs, l.qs[0], l.qz[0]);
                l.forward_qparams(l, net);
                net.input = l.output;
            }
        }
    }
}

void forward_network_int8(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        forward_network_int8_gpu(netp);
        return;
    }
#endif
    network net = *netp;

    int i;
    for(i = 0; i < net.n; ++i) {
        net.index = i;
        layer l = net.layers[i];
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        if(l.type == CONNECTED) 
        {
            int dequant = (net.layers[i + 1].type == SOFTMAX);
            l.forward_int8(l, net, dequant);
            if (dequant) net.input = l.output;
            else net.input_int8 = l.output_int8;
        }
        else if(l.type == CONVOLUTIONAL || l.type == MAXPOOL)
        {
            int dequant = (net.layers[i + 1].type == AVGPOOL);
            l.forward_int8(l, net, dequant);
            if (dequant) net.input = l.output;
            else net.input_int8 = l.output_int8;
        }
        else 
        {
            l.forward(l, net);
            net.input = l.output;
        }
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}

void forward_network_comparison(network *netp)
{
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.type == CONNECTED)
        {
            int dequant = (net.layers[i + 1].type == SOFTMAX);
            //int dequant = (net.layers[i - 1].type != CONVOLUTIONAL && net.layers[i+1].type != SOFTMAX);
            l.forward_cmp(l, net, dequant);
            net.input_int8 = l.output_int8;
            net.input = l.output;
            //if(dequant) exit(0);
        }
        else if(l.type == CONVOLUTIONAL)
        {
			l.forward_cmp(l, net, 0);
            net.input_int8 = l.output_int8;
            net.input = l.output;
        }
        else if(l.type == MAXPOOL)
        {
			l.forward(l, net);
            net.input = l.output;
			l.forward_int8(l, net, 0);
            net.input_int8 = l.output_int8;
        }
        else 
        {
            //printf("SOFTMAX\n");
            //int j;
            //for(j=0;j<l.outputs;j++) printf("%f\n", net.input[j]);
            //printf("\n");
            l.forward(l, net);
            //for(j=0;j<l.outputs;j++) printf("%f\n", l.output[j]);
            net.input = l.output;
        }
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}

void forward_network_quant_csv(network *netp, FILE *output_csv, FILE *weight_csv)
{
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.type == CONNECTED)
        {
            int dequant = (net.layers[i + 1].type == SOFTMAX);
            l.forward_csv(l, net, dequant, output_csv, weight_csv);
            if(dequant) net.input = l.output;
            else net.input_int8 = l.output_int8;
        }
        else if(l.type == CONVOLUTIONAL)
        {
            int dequant = (net.layers[i + 1].type == AVGPOOL);
            int first_layer = (i == 0);
			l.forward_csv(l, net, dequant, first_layer, 0);
            if (dequant) net.input = l.output;
            else net.input_int8 = l.output_int8;
        }
        else if(l.type == MAXPOOL)
        {
            l.forward_int8(l, net, 0);
            net.input_int8 = l.output_int8;
        }
        else 
        {
            l.forward(l, net);
            net.input = l.output;
        }
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}

void update_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        update_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = *net.t;

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, a);
        }
    }
}

void calc_network_cost(network *netp)
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum/count;
}

int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}

void backward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        backward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
        }
        net.index = i;
        l.backward(l, net);
    }
}

float train_network_datum(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    forward_network(net);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float train_imagenet_pretrained_network_datum(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    forward_network(net);
    //backward_imagenet_pretrained_network_gpu(net);
    float error = *net->cost;
    //if(((*net->seen)/net->batch)%net->subdivisions == 0) update_imagenet_pretrained_network_gpu(net);
    return error;
}

float train_quantization_simulated_network_of_tiny_imagenet_datum_int4(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    net->ema_smooth += net->ema_decay_factor;
    //forward_quantization_simulated_network_of_tiny_imagenet_int4_gpu(net);
    //backward_imagenet_pretrained_network_gpu(net);
    float error = *net->cost;
    //if(((*net->seen)/net->batch)%net->subdivisions == 0) update_imagenet_pretrained_network_gpu(net);
    return error;
}

float train_quantization_simulated_network_of_tiny_imagenet_datum_int8(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    net->ema_smooth += net->ema_decay_factor;
    //forward_quantization_simulated_network_of_tiny_imagenet_int8_gpu(net);
    //backward_imagenet_pretrained_network_gpu(net);
    float error = *net->cost;
    //if(((*net->seen)/net->batch)%net->subdivisions == 0) update_imagenet_pretrained_network_gpu(net);
    return error;
}

float train_quantization_simulated_network_datum_int4(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    net->ema_smooth += net->ema_decay_factor;
    //forward_quantization_simulated_network_int4_gpu(net);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float train_quantization_simulated_network_datum_int8(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    net->ema_smooth += net->ema_decay_factor;
    forward_quantization_simulated_network_int8(net);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float qat_network_with_batch_of_mixed_clusters_datum_int4(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    net->ema_smooth += net->ema_decay_factor;
    //forward_qat_with_batch_of_mixed_clusters_int4_gpu(net);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float qat_per_cluster_network_datum_int4(network *net, const int cluster)
{
    *net->seen += net->batch;
    net->train = 1;
    net->ema_smooth += net->ema_decay_factor;
    //forward_qat_per_cluster_int4_gpu(net, cluster);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float qat_per_cluster_network_datum_int8(network *net, const int cluster)
{
    *net->seen += net->batch;
    net->train = 1;
    net->ema_smooth += net->ema_decay_factor;
    //forward_qat_per_cluster_int8_gpu(net, cluster);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float train_tuned_network_datum(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    forward_network(net);
#ifdef GPU
    backward_tuned_network_gpu(net);
#endif
    //backward_network_gpu(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float train_network_datum_quant(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    forward_network_int8(net);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

void train_network_qparams_staged(network *net, data d, int stage)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;
    net->train = 0;

    int i;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        *net->seen += net->batch;

        layer first = net->layers[0];
        int len_input = first.batch * first.inputs;
        int8_t quantized_input[len_input];
        for(int i = 0; i < len_input; i++)
            quantized_input[i] = 0;
        make_quantized_matrix(net->input, quantized_input, first.batch, first.inputs, &first.qs[0], &first.qz[0]);
        net->input_int8 = quantized_input;
#ifdef GPU
        forward_network_qparams_staged_gpu(net, stage);
#endif
    }
}

void ema_through_dataset(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;
    //int batch_input = batch * net->inputs;

    int i, j;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        *net->seen += net->batch;
#ifdef GPU
        //forward_network_qparams_gpu(net);
        forward_network_ema_gpu(net);
#endif
    }
}

void min_max_through_dataset(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;
    //int batch_input = batch * net->inputs;

    int i, j;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        *net->seen += net->batch;

        //char buff_batch[256];
        //sprintf(buff_batch, "batch_%d-top_3.csv", batch);
        //FILE *fp_batch = fopen(buff_batch, "a");
        //if(get_current_batch(net) == 1) {
        //    for(j = 0; j < batch_input; j++) {
        //        fprintf(fp_batch, "%d%s", j, (j < batch_input - 1 ? "," : "\n"));
        //    }
        //}
        //for(j = 0; j < batch_input; j++) fprintf(fp_batch, "%f%s", net->input[j], (j < batch_input - 1 ? "," : "\n"));
        //fclose(fp_batch);

#ifdef GPU
        forward_network_qparams_gpu(net);
#endif
    }
}

void batch_min_max_through_dataset(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;
    int batch_input = batch * net->inputs;

    int i, j;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        *net->seen += net->batch;

        char buff_batch[256];
        sprintf(buff_batch, "batch_%d-top_3.csv", batch);
        FILE *fp_batch = fopen(buff_batch, "a");
        if(get_current_batch(net) == 1) {
            for(j = 0; j < batch_input; j++) {
                fprintf(fp_batch, "%d%s", j, (j < batch_input - 1 ? "," : "\n"));
            }
        }
        for(j = 0; j < batch_input; j++) fprintf(fp_batch, "%f%s", net->input[j], (j < batch_input - 1 ? "," : "\n"));
        fclose(fp_batch);

#ifdef GPU
        forward_network_batch_qparams_gpu(net);
#endif
    }
}

void get_qparams_of_inputs(network *net, data d, char *cfgname)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;
    int batch_input = batch * net->inputs;

    int i;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        *net->seen += net->batch;
#ifdef GPU
        forward_network_input_qparams_gpu(net);
#endif
    }
    char buff_qsz[256];
    char buff_range[256];
    sprintf(buff_qsz, "backup/%s.input_qparams", cfgname);
    sprintf(buff_range, "backup/%s.input_ranges", cfgname);
    save_input_qparams(net, buff_qsz, buff_range, net->n);
}

float train_network_sgd(network *net, data d, int n)
{
    int batch = net->batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_network(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_imagenet_pretrained_network(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_imagenet_pretrained_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_quantization_simulated_network_of_tiny_imagenet_int4(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_quantization_simulated_network_of_tiny_imagenet_datum_int4(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_quantization_simulated_network_of_tiny_imagenet_int8(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_quantization_simulated_network_of_tiny_imagenet_datum_int8(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_quantization_simulated_network_int4(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_quantization_simulated_network_datum_int4(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_quantization_simulated_network_int8(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_quantization_simulated_network_datum_int8(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float qat_network_with_batch_of_mixed_clusters_int4(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = qat_network_with_batch_of_mixed_clusters_datum_int4(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float qat_per_cluster_network_int4(network *net, data d, const int cluster)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = qat_per_cluster_network_datum_int4(net, cluster);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float qat_per_cluster_network_int8(network *net, data d, const int cluster)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = qat_per_cluster_network_datum_int8(net, cluster);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_tuned_network(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_tuned_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_network_quant(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_network_datum_quant(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

void set_temp_network(network *net, float t)
{
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].temperature = t;
    }
}


void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_quantized_convolutional_setup(net->layers + i);
            //cudnn_convolutional_setup(net->layers + i);
        }
        if(net->layers[i].type == DECONVOLUTIONAL){
            layer *l = net->layers + i;
            cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, l->out_h, l->out_w);
            cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
        }
#endif
    }
}

int resize_network(network *net, int w, int h)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    cuda_free(net->workspace);
#endif
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
        }else if(l.type == YOLO){
            resize_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if(l.type == SHORTCUT){
            resize_shortcut_layer(&l, w, h);
        }else if(l.type == UPSAMPLE){
            resize_upsample_layer(&l, w, h);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else{
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if(l.workspace_size > 2000000000) assert(0);
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
    layer out = get_network_output_layer(net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    if(gpu_index >= 0){
        cuda_free(net->input_gpu);
        cuda_free(net->truth_gpu);
        net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
        net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
        if(workspace_size){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }
    }else {
        free(net->workspace);
        net->workspace = calloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = calloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

layer get_network_detection_layer(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers[i].type == DETECTION){
            return net->layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    layer l = {0};
    return l;
}

image get_network_image_layer(network *net, int i)
{
    layer l = net->layers[i];
#ifdef GPU
    //cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network *net)
{
    int i;
    for(i = net->n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network *net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net->n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}


float *network_predict(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}

float *network_predict_int4(network *net, float *input)
{
    // 작성 X
    float* i;
    return i;
}

float *network_predict_int8(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    
    layer first = net->layers[0];
    // line 1248: 아래 정보를 활용, net->input을 quantize 하여 net->input_int8에 저장하도록 quantize_int8_cpu(...) 함수를 사용하세요.
    // net->input      :: FP input mat
    // net->input_int8 :: INT input mat
    // net->batch      :: batch size
    // net->inputs     :: # of elements per data
    // first.qs        :: 1'st layer's array of S
    // first.qz        :: 1'st layer's array of Z

    net->truth = 0;
    net->train = 0;
    net->delta = 0;

    forward_network_int8(net);
    float *out = net->output;
    *net = orig;
    return out;
}

float *network_predict_with_closest_input(network *net, float *input)
{
    // 작성 X
    float* i;
    return i;
}

float *network_predict_comparison(network *net, float *input)
{
    // 작성 X
    float* i;
    return i;
}

float *network_predict_csv(network *net, float *input, FILE *output_csv, FILE *weight_csv)
{
    // 작성 X
    float* i;
    return i;
}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
            s += yolo_num_detections(l, thresh);
        }
        if(l.type == DETECTION || l.type == REGION){
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    if(num) *num = nboxes;
    detection *dets = calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = calloc(l.classes, sizeof(float));
        if(l.coords > 4){
            dets[i].mask = calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == YOLO){
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
        if(l.type == REGION){
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION){
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}

float *network_predict_image(network *net, image im)
{
    image imr = letterbox_image(im, net->w, net->h);
    set_batch_network(net, 1);
    float *p = network_predict(net, imr.data);
    free_image(imr);
    return p;
}

int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}

matrix network_predict_data_multi(network *net, data test, int n)
{
    int i,j,b,m;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net->batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}

matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network *n1, network *n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den); 
}

float network_accuracy(network *net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network *net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

float network_accuracy_multi(network *net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);
#ifdef GPU
    if(net->input_int8_gpu) cuda_free_int8(net->input_int8_gpu);

    if(net->input_gpu) cuda_free(net->input_gpu);
    if(net->truth_gpu) cuda_free(net->truth_gpu);
#endif
    free(net);
}

// Some day...
// ^ What the hell is this comment for?


layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}

#ifdef GPU

void forward_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }

    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }

        l.forward_gpu(l, net);

        //if(l.type==SOFTMAX || net.layers[i+1].type == SOFTMAX) {
        //    cuda_pull_array(l.output_gpu, l.output, l.outputs);
        //    int j;
        //    for(j=0;j<l.outputs;j++) fprintf(stdout, "%f\n", l.output[j]);
        //}
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void forward_quantization_simulated_network_of_tiny_imagenet_int4_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);

    int step = get_current_batch(&net);
    float *r = &net.input_range[0];
    if(net.ema_init[0]) {
        ema_cpu(net.input, r, r + 1, net.batch * net.inputs, net.ema_smooth, 1);
        fake_quantize_int4(net.input_gpu, net.batch * net.inputs, *r, *(r + 1));
    } else {
        get_min_max_cpu(net.input,  r, r + 1, net.inputs * net.batch);
        net.ema_init[0] = 1;
    }

    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        // Only fake quantize output values before 5th CONV
        if(i >= 6) {
            if(l.type == CONVOLUTIONAL || l.type == CONNECTED) {
                l.forward_qat_int4_gpu(l, net);
            } else {
                l.forward_gpu(l, net);
            }
        } else {
            l.forward_gpu(l, net);
            if(l.type == CONVOLUTIONAL || l.type == CONNECTED) {
                cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
                if (l.ema_init[0]) {
                    ema_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs, net.ema_smooth, l.activation == RELU);
                    fake_quantize_int4(l.output_gpu, l.batch * l.outputs, l.act_range[0], l.act_range[1]);
                } else {
                    get_min_max_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs);
                    l.ema_init[0] = 1;
                }
            }
        }
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void forward_quantization_simulated_network_of_tiny_imagenet_int8_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);

    int step = get_current_batch(&net);
    float *r = &net.input_range[0];
    if(net.ema_init[0]) {
        ema_cpu(net.input, r, r + 1, net.batch * net.inputs, net.ema_smooth, 1);
        fake_quantize_int8(net.input_gpu, net.batch * net.inputs, *r, *(r + 1));
    } else {
        get_min_max_cpu(net.input,  r, r + 1, net.inputs * net.batch);
        net.ema_init[0] = 1;
    }

    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        // Only fake quantize output values before 5th CONV
        if(i >= 6) {
            if(l.type == CONVOLUTIONAL || l.type == CONNECTED) {
                l.forward_qat_int8_gpu(l, net);
            } else {
                l.forward_gpu(l, net);
            }
        } else {
            l.forward_gpu(l, net);
            if(l.type == CONVOLUTIONAL || l.type == CONNECTED) {
                cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
                if (net.ema_init[0]) {
                    ema_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs, net.ema_smooth, l.activation == RELU);
                    fake_quantize_int8(l.output_gpu, l.batch * l.outputs, l.act_range[0], l.act_range[1]);
                } else {
                    get_min_max_cpu(l.output, &l.act_range[0], &l.act_range[1], l.batch * l.outputs);
                    net.ema_init[0] = 1;
                }
            }
        }
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void forward_quantization_simulated_network_int4_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);

    int step = get_current_batch(&net);
    float *r = &net.input_range[0];
    if(net.ema_init[0]) {
        ema_cpu(net.input, r, r + 1, net.batch * net.inputs, net.ema_smooth, 1);
        fake_quantize_int4(net.input_gpu, net.batch * net.inputs, *r, *(r + 1));
    } else {
        get_min_max_cpu(net.input,  r, r + 1, net.inputs * net.batch);
        net.ema_init[0] = 1;
    }

    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        if(l.type == CONVOLUTIONAL || l.type == CONNECTED) {
            l.forward_qat_int4_gpu(l, net);
        } else {
            l.forward_gpu(l, net);
        }
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void forward_quantization_simulated_network_int8_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);

    int step = get_current_batch(&net);
    float *r = &net.input_range[0];
    if(net.ema_init[0]) {
        ema_cpu(net.input, r, r + 1, net.batch * net.inputs, net.ema_smooth, 1);
        fake_quantize_int8(net.input_gpu, net.batch * net.inputs, *r, *(r + 1));
    } else if(step == net.qat_init_step) {
        get_min_max_cpu(net.input,  r, r + 1, net.inputs * net.batch);
        net.ema_init[0] = 1;
    }

    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        } if(l.type == CONVOLUTIONAL || l.type == CONNECTED) {
            l.forward_qat_int8_gpu(l, net);
        } else {
            l.forward_gpu(l, net);
        }
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void forward_qat_with_batch_of_mixed_clusters_int4_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs * net.batch);

    int i, ctr, n_data;
    int idx = 0;
    float* r;                          // cluster's ema array
    for(i = 0; i < net.ctr_k; i++) {
        ctr = net.ctr_info[i * 2];     // cluster id
        if(ctr == -1) break;

        n_data = net.inputs * net.ctr_info[i * 2 + 1];  // # of data in the c
        r = &net.input_cluster[ctr * 2];                // cluster's ema array
        if(net.ema_init[ctr]) {
            ema_cpu(&net.input[idx], r, r + 1, n_data, net.ema_smooth, 1);
            fake_quantize_int4(&net.input_gpu[idx], n_data, *r, *(r + 1));
        } else {
            get_min_max_cpu(&net.input[idx], r, r + 1, n_data);
            net.ema_init[ctr] = 1;
        }
        idx += n_data;
    }

    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }

    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        } if(l.type == CONVOLUTIONAL || l.type == CONNECTED) {
            l.forward_bmc_int4_gpu(l, net);
        } else {
            l.forward_gpu(l, net);
        }
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void forward_qat_per_cluster_int4_gpu(network *netp, const int ctr_idx)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs * net.batch);

    float *r = &net.input_cluster[ctr_idx * 2];
    if(net.ema_init[ctr_idx]) {
        ema_cpu(net.input, r, r + 1, net.inputs * net.batch, net.ema_smooth, 1);
        fake_quantize_int4(net.input_gpu, net.inputs * net.batch, *r, *(r + 1));
    } else {
        get_min_max_cpu(net.input, r, r + 1, net.inputs * net.batch);
        net.ema_init[ctr_idx] = 1;
    }

    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }

    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        } if(l.type == CONVOLUTIONAL || l.type == CONNECTED) {
            l.forward_qat_cluster_int4_gpu(l, net, ctr_idx);
        } else {
            l.forward_gpu(l, net);
        }
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void forward_qat_per_cluster_int8_gpu(network *netp, const int ctr_idx)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs * net.batch);

    float *r = &net.input_cluster[ctr_idx * 2];
    if(net.ema_init[ctr_idx]) {
        ema_cpu(net.input, r, r + 1, net.inputs * net.batch, net.ema_smooth, 1);
        fake_quantize_int8(net.input, net.inputs * net.batch, *r, *(r + 1));
    } else {
        get_min_max_cpu(net.input, r, r + 1, net.inputs * net.batch);
        net.ema_init[ctr_idx] = 1;
    }

    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }

    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        } if(l.type == CONVOLUTIONAL || l.type == CONNECTED) {
            l.forward_qat_cluster_int8_gpu(l, net, ctr_idx);
        } else {
            l.forward_gpu(l, net);
        }
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void forward_network_int4_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }

    int i;
    for ( i = 0; i < net.n; ++i ) {
        net.index = i;
        layer l = net.layers[i];
        if ( l.type == CONVOLUTIONAL ) {
            l.forward_int4_gpu(l, net, 0);
            net.input_int4 = l.output_int4;
        } else if ( l.type == MAXPOOL ) {
            l.forward_int4(l, net, 0);
            net.input_int4 = l.output_int4;
        } else if ( l.type == CONNECTED ) {
            int dequant = (net.layers[i + 1].type == SOFTMAX);
            l.forward_int4_gpu(l, net, dequant);
            net.input_int4 = l.output_int4;
        } else {
            l.forward_gpu(l, net);
        }
        net.input_gpu = l.output_gpu;
        net.input = l.output;

        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void forward_network_int8_gpu(network *netp)
{
    network net = *netp;
    if(net.train) exit(0);
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs * net.batch);
    cuda_push_array_int8(net.input_int8_gpu, net.input_int8, net.inputs * net.batch);
    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }

    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL || l.type == MAXPOOL)
        {
            int dequant = (net.layers[i + 1].type == AVGPOOL);
            //if(i == 0) {
            //    printf("000000000\n");
            //    l.forward_int8_gpu(l, net, dequant);
            //} else {
            //    printf("111111111\n");
            //    l.forward_cudnn_gpu(l, net, dequant);
            //}
            l.forward_int8_gpu(l, net, dequant);
            //l.forward_cudnn_gpu(l, net, dequant);
            net.input_int8 = l.output_int8;
            net.input_int8_gpu = l.output_int8_gpu;
            net.input = l.output;
            net.input_gpu = l.output_gpu;
        } else if(l.type == CONNECTED) {
            int dequant = (net.layers[i + 1].type == SOFTMAX);
            l.forward_int8_gpu(l, net, dequant);
            if(dequant) {
                net.input_int8 = l.output_int8;
                net.input_int8_gpu = l.output_int8_gpu;
                net.input_gpu = l.output_gpu;
            } else {
                net.input_int8 = l.output_int8;
                net.input_int8_gpu = l.output_int8_gpu;
                net.input = l.output;
                net.input_gpu = l.output_gpu;
            }
        } else {
            l.forward_gpu(l, net);
            net.input_gpu = l.output_gpu;
            net.input = l.output;
        }
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void forward_network_ema_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);

    int step = get_current_batch(&net);
    if(net.ema_init[0]) {
        ema_cpu(net.input, &net.input_range[0], &net.input_range[1], net.batch * net.inputs, net.ema_smooth, 1);
        fake_quantize_int8(net.input, net.inputs, net.input_range[0], net.input_range[1]);
    } else if(step == net.qat_init_step) {
        get_min_max_cpu(net.input, &net.input_range[0], &net.input_range[1], net.inputs * net.batch);
        net.ema_init[0] = 1;
    }

    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.type == CONNECTED || l.type == CONVOLUTIONAL) {
            l.forward_ema_gpu(l, net);
        }
        else l.forward_gpu(l, net);
        net.input_gpu = l.output_gpu;
        net.input = l.output;
    }
    pull_network_output(netp);
}

void forward_network_qparams_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    set_min_max(net.input, &net.input_range[0], &net.input_range[1], net.inputs * net.batch);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.type == CONNECTED || l.type == CONVOLUTIONAL) {
            l.forward_qparams_gpu(l, net);
        }
        else l.forward_gpu(l, net);
        net.input_gpu = l.output_gpu;
        net.input = l.output;
    }
    pull_network_output(netp);
}

void forward_network_input_qparams_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.type == CONNECTED || l.type == CONVOLUTIONAL) {
            l.act_range[0] = (l.activation != RELU);
            l.act_range[1] = -999;
            l.forward_qparams_gpu(l, net);
        }
        else l.forward_gpu(l, net);
        net.input_gpu = l.output_gpu;
        net.input = l.output;
    }
    pull_network_output(netp);
}

void forward_network_batch_qparams_gpu(network *netp)
{
    network net = *netp;
    /*
    int step = get_current_batch(&net);
    // 1. (min, max) per batch
    // 2. (S1, S3), (Z1, Z3) per layer
    int idx = (step - 1) * 2;
    //set_min_max(net.input, &net.input_range[0], &net.input_range[1], net.inputs * net.batch);
    set_batch_range(net.input, &net.batch_range[idx], &net.batch_range[idx + 1], net.inputs * net.batch);
    */
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.type == CONNECTED || l.type == CONVOLUTIONAL) {
            /*
            l.act_range[0] = 0;
            l.act_range[1] = 0;
            */
            l.forward_qparams_gpu(l, net);
        }
        else l.forward_gpu(l, net);
        net.input_gpu = l.output_gpu;
        net.input = l.output;
    }
    /*
    float next_qs1;
    int8_t next_qz1;
    net.input_range[0] = net.batch_range[idx];
    net.input_range[1] = net.batch_range[idx + 1];
    cal_qsz(net.input_range, &next_qs1, &next_qz1);
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.type == CONNECTED || l.type == CONVOLUTIONAL) {
            set_batch_qsz(l, &next_qs1, &next_qz1, idx);
        }
    }
    */
    pull_network_output(netp);
}

void forward_network_qparams_staged_gpu(network *netp, int stage)
{
    int i;
    network net = *netp;
    cuda_set_device(net.gpu_index);
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.type == CONNECTED) {
            if(i < stage) {
                cuda_push_array_int8(net.input_int8_gpu, net.input_int8, l.inputs*l.batch);
                int dequant = (net.layers[i + 1].type == SOFTMAX);
                l.forward_int8_gpu(l, net, dequant);
                net.input_int8 = l.output_int8;
            } else if(i == stage) {
                dequantize_matrix(net.input_int8, net.input, l.batch, l.inputs, l.qs[0], l.qz[0]);
                cuda_push_array(net.input_gpu, net.input, l.inputs*l.batch);
                l.forward_qparams_gpu(l, net);
                net.input_gpu = l.output_gpu;
                break;
            }
        }
    }
}

void backward_network_gpu(network *netp)
{
    int i;
    network net = *netp;
    network orig = net;
    cuda_set_device(net.gpu_index);
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        net.index = i;
        l.backward_gpu(l, net);
    }
}

void backward_imagenet_pretrained_network_gpu(network *netp)
{
    int i;
    network net = *netp;
    network orig = net;
    cuda_set_device(net.gpu_index);
    // backward until 5'th CONV layer
    for(i = net.n-1; i >= 6; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        net.index = i;
        l.backward_gpu(l, net);
    }
}

void backward_tuned_network_gpu(network *netp)
{
    int i;
    network net = *netp;
    network orig = net;
    cuda_set_device(net.gpu_index);
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        net.index = i;
        l.backward_gpu(l, net);
        if(i == 3) continue;
        if(l.type == CONNECTED) {
            mask_matrix(l.weights_gpu, l.weights_mask, l.outputs * l.inputs);
            //if(i != 3) clamp_mat_gpu(l.weights_gpu, l.outputs * l.inputs, l.weights_range);
        }
    }
}

void update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = (*net.t);

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update_gpu){
            l.update_gpu(l, a);
        }
    }
}

void update_imagenet_pretrained_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = (*net.t);

    for(i = 6; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update_gpu){
            l.update_gpu(l, a);
        }
    }
}

void harmless_update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.weight_updates_gpu) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
        if(l.bias_updates_gpu) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
        if(l.scale_updates_gpu) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
    }
}

void mask_network_gpu(network *netp)
{
    int i;
    network net = *netp;
    network orig = net;
    cuda_set_device(net.gpu_index);
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.type == CONNECTED) {
            mask_matrix(l.weights_gpu, l.weights_mask, l.outputs * l.inputs);
            clamp_mat_gpu(l.weights_gpu, l.outputs * l.inputs, l.weights_range);
        }
    }
}

typedef struct {
    network *net;
    data d;
    float *err;
} train_args;

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net->gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network *net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.nweights, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.nweights);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
        if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if (l.type == CONNECTED) {
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


/*
   void pull_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }
   void push_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }
   void update_layer(layer l, network net)
   {
   int update_batch = net.batch*net.subdivisions;
   float rate = get_current_rate(net);
   l.t = get_current_batch(net);
   if(l.update_gpu){
   l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
   }
   }
   void merge_updates(layer l, layer base)
   {
   if (l.type == CONVOLUTIONAL) {
   axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
   if (l.scale_updates) {
   axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
   }
   } else if(l.type == CONNECTED) {
   axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
   }
   }
   void distribute_updates(layer l, layer base)
   {
   if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
   if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
   }
   }
 */

/*
   void sync_layer(network *nets, int n, int j)
   {
   int i;
   network net = nets[0];
   layer base = net.layers[j];
   scale_weights(base, 0);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   pull_weights(l);
   merge_weights(l, base);
   }
   scale_weights(base, 1./n);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   distribute_weights(l, base);
   }
   }
 */

void sync_layer(network **nets, int n, int j)
{
    int i;
    network *net = nets[0];
    layer base = net->layers[j];
    scale_weights(base, 0);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        distribute_weights(l, base);
    }
}

typedef struct{
    network **nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network **nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network **nets, int n, int interval)
{
    int j;
    int layers = nets[0]->n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *(nets[0]->seen) += interval * (n-1) * nets[0]->batch * nets[0]->subdivisions;
    for (j = 0; j < n; ++j){
        *(nets[j]->seen) = *(nets[0]->seen);
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network **nets, int n, data d, int interval)
{
    int i;
    int batch = nets[0]->batch;
    int subdivisions = nets[0]->subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

void pull_network_output(network *net)
{
    layer l = get_network_output_layer(net);
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
}

#endif

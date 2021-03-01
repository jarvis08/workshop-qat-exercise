#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h> 
#include <stdint.h>

#include "activation_layer.h"
#include "logistic_layer.h"
#include "l2norm_layer.h"
#include "activations.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "iseg_layer.h"
#include "reorg_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "lstm_layer.h"
#include "utils.h"
#include "gemm.h"
#include "quant_utils.h"

typedef struct{
    char *type;
    list *options;
}section;

list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[region]")==0) return REGION;
    if (strcmp(type, "[yolo]")==0) return YOLO;
    if (strcmp(type, "[iseg]")==0) return ISEG;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[deconv]")==0
            || strcmp(type, "[deconvolutional]")==0) return DECONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[logistic]")==0) return LOGXENT;
    if (strcmp(type, "[l2norm]")==0) return L2NORM;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[lstm]") == 0) return LSTM;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[reorg]")==0) return REORG;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[route]")==0) return ROUTE;
    if (strcmp(type, "[upsample]")==0) return UPSAMPLE;
    return BLANK;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i){
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;

local_layer parse_local(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before local layer must output image.");

    local_layer layer = make_local_layer(batch,h,w,c,n,size,stride,pad,activation);

    return layer;
}

layer parse_deconvolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before deconvolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;

    layer l = make_deconvolutional_layer(batch,h,w,c,n,size,stride,padding, activation, batch_normalize, params.net->adam);

    return l;
}


convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}

convolutional_layer parse_convolutional_quant(list *options, size_params params, char* argv)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional_layer layer = make_convolutional_layer_quant(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam, argv);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}

layer parse_crnn(list *options, size_params params)
{
    int output_filters = option_find_int(options, "output_filters",1);
    int hidden_filters = option_find_int(options, "hidden_filters",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_crnn_layer(params.batch, params.w, params.h, params.c, hidden_filters, output_filters, params.time_steps, activation, batch_normalize);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_rnn(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_rnn_layer(params.batch, params.inputs, output, params.time_steps, activation, batch_normalize, params.net->adam);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_gru(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);
    l.tanh = option_find_int_quiet(options, "tanh", 0);

    return l;
}

layer parse_lstm(list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);

    return l;
}

layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);
    return l;
}

layer parse_connected_quant(list *options, size_params params, char *argv)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    layer l = make_connected_layer_quant(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam, argv);
    return l;
}

layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups",1);
    layer l = make_softmax_layer(params.batch, params.inputs, groups);
    l.temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    l.w = params.w;
    l.h = params.h;
    l.c = params.c;
    l.spatial = option_find_float_quiet(options, "spatial", 0);
    l.noloss =  option_find_int_quiet(options, "noloss", 0);
    return l;
}

int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',')+1;
        }
        *num = n;
    }
    return mask;
}

layer parse_yolo(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes);
    assert(l.outputs == params.inputs);

    l.max_boxes = option_find_int_quiet(options, "max",90);
    l.jitter = option_find_float(options, "jitter", .2);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

layer parse_iseg(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int ids = option_find_int(options, "ids", 32);
    layer l = make_iseg_layer(params.batch, params.w, params.h, classes, ids);
    assert(l.outputs == params.inputs);
    return l;
}

layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.background = option_find_int_quiet(options, "background", 0);
    l.max_boxes = option_find_int_quiet(options, "max",30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.rescore = option_find_int_quiet(options, "rescore",0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_int_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.mask_scale = option_find_float(options, "mask_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match",0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    char *a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

detection_layer parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 0);
    int num = option_find_int(options, "num", 1);
    int side = option_find_int(options, "side", 7);
    detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    layer.softmax = option_find_int(options, "softmax", 0);
    layer.sqrt = option_find_int(options, "sqrt", 0);

    layer.max_boxes = option_find_int_quiet(options, "max",90);
    layer.coord_scale = option_find_float(options, "coord_scale", 1);
    layer.forced = option_find_int(options, "forced", 0);
    layer.object_scale = option_find_float(options, "object_scale", 1);
    layer.noobject_scale = option_find_float(options, "noobject_scale", 1);
    layer.class_scale = option_find_float(options, "class_scale", 1);
    layer.jitter = option_find_float(options, "jitter", .2);
    layer.random = option_find_int_quiet(options, "random", 0);
    layer.reorg = option_find_int_quiet(options, "reorg", 0);
    return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    layer.ratio =  option_find_float_quiet(options, "ratio",0);
    layer.noobject_scale =  option_find_float_quiet(options, "noobj", 1);
    layer.thresh =  option_find_float_quiet(options, "thresh",0);
    return layer;
}

crop_layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);
    float saturation = option_find_float(options, "saturation",1);
    float exposure = option_find_float(options, "exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before crop layer must output image.");

    int noadjust = option_find_int_quiet(options, "noadjust",0);

    crop_layer l = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int reverse = option_find_int_quiet(options, "reverse",0);
    int flatten = option_find_int_quiet(options, "flatten",0);
    int extra = option_find_int_quiet(options, "extra",0);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before reorg layer must output image.");

    layer layer = make_reorg_layer(batch,w,h,c,stride,reverse, flatten, extra);
    return layer;
}

maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch,h,w,c,size,stride,padding);
    return layer;
}

maxpool_layer parse_maxpool_quant(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer_quant(batch,h,w,c,size,stride,padding);
    return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.");

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}

dropout_layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

layer parse_normalization(list *options, size_params params)
{
    float alpha = option_find_float(options, "alpha", .0001);
    float beta =  option_find_float(options, "beta" , .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l;
}

layer parse_batchnorm(list *options, size_params params)
{
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c);
    return l;
}

layer parse_shortcut(list *options, size_params params, network *net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if(index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net->layers[index];

    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    s.alpha = option_find_float_quiet(options, "alpha", 1);
    s.beta = option_find_float_quiet(options, "beta", 1);
    return s;
}


layer parse_l2norm(list *options, size_params params)
{
    layer l = make_l2norm_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}


layer parse_logistic(list *options, size_params params)
{
    layer l = make_logistic_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}

layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;

    return l;
}

layer parse_upsample(list *options, size_params params, network *net)
{

    int stride = option_find_int(options, "stride",2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    return l;
}

route_layer parse_route(list *options, size_params params, network *net)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if(!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    route_layer layer = make_route_layer(batch, n, layers, sizes);

    convolutional_layer first = net->layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        convolutional_layer next = net->layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random")==0) return RANDOM;
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->notruth = option_find_int_quiet(options, "notruth",0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, "random", 0);

    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .0000001);
    }

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);
    net->max_ratio = option_find_float_quiet(options, "max_ratio", (float) net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, "min_ratio", (float) net->min_crop / net->w);
    net->center = option_find_int_quiet(options, "center",0);
    net->clip = option_find_float_quiet(options, "clip", 0);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    net->power = option_find_float_quiet(options, "power", 4);
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS){
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

void parse_net_options_quant(list *options, network *net, char* argv)
{
    net->batch = option_find_int(options, "batch", 1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->notruth = option_find_int_quiet(options, "notruth",0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, "random", 0);

    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .0000001);
    }

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);
    net->max_ratio = option_find_float_quiet(options, "max_ratio", (float) net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, "min_ratio", (float) net->min_crop / net->w);
    net->center = option_find_int_quiet(options, "center",0);
    net->clip = option_find_float_quiet(options, "clip", 0);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    net->power = option_find_float_quiet(options, "power", 4);
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS){
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
    }
    net->max_batches = option_find_int(options, "max_batches", 0);

    // QAT
    net->qat_init_step = option_find_int(options, "qat_init_step", 1);
    net->qat_end_epoch = option_find_int(options, "qat_end_epoch", 1);
    net->ema_smooth = option_find_float(options, "ema_smooth", 0.95);
    net->ema_decay = option_find_int(options, "ema_decay", 0);
    net->ema_convergence = option_find_float(options, "ema_convergence", 0.99); 
    if(net->ema_decay) {
        net->ema_decay_factor = (net->ema_convergence - net->ema_smooth) / (net->max_batches - net->qat_init_step);
    } else {
        net->ema_decay_factor = 0;
    }

    // Cluster QAT
    net->ctr_k = option_find_int(options, "ctr_k", 10);
    char *m = option_find_str(options, "ctr_method", "minmax");
    net->ctr_method = calloc(strlen(m), sizeof(char));
    strcpy(net->ctr_method, m);
}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}

network *parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections");
    network *net = make_network(sections->size - 1);
    net->gpu_index = gpu_index;
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while(n){
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = {0};
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params);
        }else if(lt == DECONVOLUTIONAL){
            l = parse_deconvolutional(options, params);
        }else if(lt == LOCAL){
            l = parse_local(options, params);
        }else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }else if(lt == LOGXENT){
            l = parse_logistic(options, params);
        }else if(lt == L2NORM){
            l = parse_l2norm(options, params);
        }else if(lt == RNN){
            l = parse_rnn(options, params);
        }else if(lt == GRU){
            l = parse_gru(options, params);
        }else if (lt == LSTM) {
            l = parse_lstm(options, params);
        }else if(lt == CRNN){
            l = parse_crnn(options, params);
        }else if(lt == CONNECTED){
            l = parse_connected(options, params);
        }else if(lt == CROP){
            l = parse_crop(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
        }else if(lt == REGION){
            l = parse_region(options, params);
        }else if(lt == YOLO){
            l = parse_yolo(options, params);
        }else if(lt == ISEG){
            l = parse_iseg(options, params);
        }else if(lt == DETECTION){
            l = parse_detection(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }else if(lt == NORMALIZATION){
            l = parse_normalization(options, params);
        }else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }else if(lt == REORG){
            l = parse_reorg(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == ROUTE){
            l = parse_route(options, params, net);
        }else if(lt == UPSAMPLE){
            l = parse_upsample(options, params, net);
        }else if(lt == SHORTCUT){
            l = parse_shortcut(options, params, net);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params);
            l.output = net->layers[count-1].output;
            l.delta = net->layers[count-1].delta;
#ifdef GPU
            l.output_gpu = net->layers[count-1].output_gpu;
            l.delta_gpu = net->layers[count-1].delta_gpu;
#endif
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        l.clip = net->clip;
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontsave = option_find_int_quiet(options, "dontsave", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.numload = option_find_int_quiet(options, "numload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        net->layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    net->output_gpu = out.output_gpu;
    net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
    net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
#endif
    if(workspace_size){
        //printf("%ld\n", workspace_size);
#ifdef GPU
        if(gpu_index >= 0){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }else {
            net->workspace = calloc(1, workspace_size);
        }
#else
        net->workspace = calloc(1, workspace_size);
#endif
    }
    return net;
}

network *parse_quantized_network_cfg(char *filename, char *argv)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections");
    network *net = make_network(sections->size - 1);
    net->gpu_index = gpu_index;
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options_quant(options, net, argv);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;

    size_t workspace_size = 0;
    size_t workspace_int8_size = 0;
    size_t workspace_int4_size = 0;
#ifdef CUDNN
    size_t workspace_cudnn_size = 0;
#endif
    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while(n){
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = {0};
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional_quant(options, params, argv);
        }else if(lt == CONNECTED){
            l = parse_connected_quant(options, params, argv);
        }else if(lt == COST){
            l = parse_cost(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }else if(lt == MAXPOOL){
            l = parse_maxpool_quant(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }

        l.clip = net->clip;
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontsave = option_find_int_quiet(options, "dontsave", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.numload = option_find_int_quiet(options, "numload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);

        option_unused(options);
        net->layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if (l.workspace_int8_size > workspace_int8_size) workspace_int8_size = l.workspace_int8_size;
        if (l.workspace_int4_size > workspace_int4_size) workspace_int4_size = l.workspace_int4_size;
#ifdef CUDNN
        if (l.workspace_cudnn_size > workspace_cudnn_size) workspace_cudnn_size = l.workspace_cudnn_size;
#endif
        free_section(s);

        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }

    free_list(sections);
    layer out = get_network_output_layer(net);

    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;

    // quantization
    // qsz selection
    int i;
    net->n_qsz_layer = 0;
    for(i = 0; i < net->n; i++) {
        layer l = net->layers[i];
        if(l.type == CONNECTED || l.type == CONVOLUTIONAL)
            net->n_qsz_layer += 1;
    }
    printf("net->n_qsz_layer = %d\n", net->n_qsz_layer);
    printf("batch size = %d\n", net->batch);
    net->n_qsz_set = 100; // 100 samples of S, Z sets
    net->n_qsz = 3; // 3 S, Z per set
    net->ctr_info = calloc(net->ctr_k * 2 + 1, sizeof(int));
    net->ema_init = calloc(net->ctr_k, sizeof(int));
    net->input_range = calloc(2, sizeof(float));
    net->input_cluster = calloc(2 * net->ctr_k, sizeof(float));
    //net->input_ranges = calloc(2 * net->n_qsz_set, sizeof(float));
    //net->input_values = calloc(net->inputs * net->n_qsz_set, sizeof(float));
    //net->batch_range = calloc((50000/net->batch + 1) * 2, sizeof(float)); // min-max set for 50000/batch_size

    //net->inputs_qs = calloc(net->n_qsz_layer * 3 * net->n_qsz_set, sizeof(float));
    //net->inputs_qz = calloc(net->n_qsz_layer * 3 * net->n_qsz_set, sizeof(int8_t));

    int4_t tmp;
    net->input_int4 = malloc(net->inputs * net->batch * sizeof(int4_t));
    net->input_int8 = calloc(net->inputs*net->batch, sizeof(int8_t));
    net->output_int4 = out.output_int4;
    net->output_int8 = out.output_int8;

    net->output = out.output;
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    net->output_gpu = out.output_gpu;
    net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
    net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);

    // quantization
    net->output_int8_gpu = out.output_int8_gpu;
    net->input_int8_gpu = cuda_make_array_int8(net->input_int8, net->inputs*net->batch);
#endif
    if(workspace_size){
#ifdef GPU
        if(gpu_index >= 0){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
            net->workspace_int8 = calloc(1, workspace_int8_size);

            net->workspace_int4 = malloc(workspace_int4_size);
            for(i = 0; i < workspace_int4_size / sizeof(int4_t); i++) net->workspace_int4[i] = tmp;

#ifdef CUDNN
            net->workspace_int8_gpu = cuda_make_array_int8(0, (workspace_int8_size-1)/sizeof(int8_t)+1);
            if(workspace_cudnn_size) {
                printf("Workspace_size = %d\n", workspace_cudnn_size);
                cuda_free_int8(net->workspace_int8_gpu);
                net->workspace_int8_gpu = cuda_make_array_int8(0, (workspace_cudnn_size-1)/sizeof(int8_t)+1);
            }
#else
            //net->workspace_int8_gpu = cuda_make_array_int8(0, (workspace_int8_size-1)/sizeof(int8_t)+1);
#endif
        }else {
            net->workspace = calloc(1, workspace_size);
            net->workspace_int8 = calloc(1, workspace_int8_size);
        }
#else
        net->workspace = calloc(1, workspace_size);
        net->workspace_int8 = calloc(1, workspace_int8_size);
#endif
    }
    return net;
}

list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = malloc(sizeof(section));
                list_insert(options, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

void save_convolutional_weights_binary(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
    int size = l.c*l.size*l.size;
    int i, j, k;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    for(i = 0; i < l.n; ++i){
        float mean = l.binary_weights[i*size];
        if(mean < 0) mean = -mean;
        fwrite(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                if (l.binary_weights[index + k] > 0) c = (c | 1<<k);
            }
            fwrite(&c, sizeof(char), 1, fp);
        }
    }
}

void save_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_converted_convolutional_weights(layer l, FILE *fp)
{
    int num = l.nweights;
    int i;
    fwrite(l.biases, sizeof(float), l.n, fp);
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_convolutional_weights_int4(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    quantize_int32_cpu(l.biases, l.biases_int32, l.n, l.qs[0] * l.qs[1], 0);
    quantize_int4_cpu(l.weights, l.weights_int4, l.nweights, l.qs[1], l.qz_int4[1].el);
    
    fwrite(l.biases_int32, sizeof(int32_t), l.n, fp);
    fwrite(l.weights_int4, sizeof(int4_t), l.nweights, fp);
}

void save_convolutional_weights_int8(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    quantize_biases(l.biases, l.biases_int32, 1, l.n, l.qs[0] * l.qs[1], 0);
    quantize_matrix(l.weights, l.weights_int8, 1, l.nweights, l.qs[1], l.qz[1]);
    
    fwrite(l.biases_int32, sizeof(int32_t), l.n, fp);
    fwrite(l.weights_int8, sizeof(int8_t), l.nweights, fp);
}

void save_batchnorm_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_batchnorm_layer(l);
    }
#endif
    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_converted_connected_weights(layer l, FILE *fp)
{
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
}

void save_pruning_param(layer l, FILE *fp)
{
    printf("Pruned gap = %f\n", l.pruned_gap);
    fwrite(&l.pruned_gap, sizeof(float), 1, fp);
}

void save_connected_weights_int8(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    quantize_biases(l.biases, l.biases_int32, 1, l.outputs, l.qs[0] * l.qs[1], 0);
    quantize_matrix(l.weights, l.weights_int8, l.inputs, l.outputs, l.qs[1], l.qz[1]);
    
    fwrite(l.biases_int32, sizeof(int32_t), l.outputs, fp);
    fwrite(l.weights_int8, sizeof(int8_t), l.outputs*l.inputs, fp);
}

void save_connected_weights_int4(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    quantize_int32_cpu(l.biases, l.biases_int32, l.outputs, l.qs[0] * l.qs[1], 0);
    quantize_int4_cpu(l.weights, l.weights_int4, l.inputs * l.outputs, l.qs[1], l.qz_int4[1].el);
    
    fwrite(l.biases_int32, sizeof(int32_t), l.outputs, fp);
    fwrite(l.weights_int4, sizeof(int4_t), l.inputs * l.outputs, fp);
}

void save_pruned_connected_weights_int8(layer l, FILE *fp)
{
    int i;
    float closest_gap[2];
    closest_gap[0] = -9999;
    closest_gap[1] = 9999;
    for(i = 0; i < l.inputs * l.outputs; i++) {
        if(l.weights[i] < - l.pruned_gap && l.weights[i] > closest_gap[0]) closest_gap[0] = l.weights[i];
        if(l.weights[i] > - l.pruned_gap && l.weights[i] < closest_gap[1]) closest_gap[1] = l.weights[i];
    }

    for(i = 0; i < l.inputs * l.outputs; i++) {
        //if(l.weights[i] >= - l.pruned_gap && l.weights[i] <= l.pruned_gap) continue;
        if(l.weights[i] >= - l.pruned_gap && l.weights[i] <= l.pruned_gap) {
            if(l.weights[i] < 0) l.weights[i] = closest_gap[0] + l.pruned_gap;
            else if(l.weights[i] > 0) l.weights[i] = closest_gap[1] - l.pruned_gap;
        }
        else if(l.weights[i] > 0) l.weights[i] -= l.pruned_gap;
        else if(l.weights[i] < 0) l.weights[i] += l.pruned_gap;
    }
    
    quantize_matrix(l.weights, l.weights_int8, l.inputs, l.outputs, l.qs[1], l.qz[1]);
    quantize_biases(l.biases, l.biases_int32, 1, l.outputs, l.qs[0] * l.qs[1], 0);

    fwrite(l.weights_int8, sizeof(int8_t), l.outputs*l.inputs, fp);
    fwrite(l.biases_int32, sizeof(int32_t), l.outputs, fp);
}

void save_qsz_int4(layer l, FILE *fp)
{   
    int i;
    //for(i = 0; i<3;i++) printf("S%d: %f, Z%d: %d\n", i + 1, l.qs[i], i + 1, l.qz_int4[i].el);
    fwrite(l.qs, sizeof(float), 3, fp);
    fwrite(l.qz_int4, sizeof(int4_t), 3, fp);
}

void save_qsz(layer l, FILE *fp)
{   
    int i;
    //for(i = 0; i<3;i++) printf("S%d: %f, Z%d: %d\n", i + 1, l.qs[i], i + 1, l.qz[i]);
    fwrite(l.qs, sizeof(float), 3, fp);
    fwrite(l.qz, sizeof(int8_t), 3, fp);
}

/*
void save_batch_qsz(layer l, FILE *fp)
{   
    int i;
    for(i = 0; i<4;i++) printf("S%d: %f, Z%d: %d\n", i + 1, l.batch_qs[i], i + 1, l.batch_qz[i]);
    fwrite(l.batch_qs, sizeof(float), 1000, fp);
    fwrite(l.batch_qz, sizeof(int8_t), 1000, fp);
}
*/

void save_input_range(float* input_range, FILE *fp)
{   
    //printf("Save input's range\n");
    //printf("min=%f | max=%f\n", input_range[0], input_range[1]);
    fwrite(input_range, sizeof(float), 2, fp);
}

void save_input_value(float* input, int n, FILE *fp)
{   
    printf("Save input's values\n");
    fwrite(input, sizeof(float), n, fp);
}

void set_quantization_stage(network *net, int stage)
{
    layer l = net->layers[stage];
    // set S2, Z2
    float w_minmax[2];
    w_minmax[0] = l.weights[0];
    w_minmax[1] = l.weights[0];

#ifdef GPU
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
#endif
    int i;
    if(l.type == CONNECTED) {
        for(i = 1; i < l.inputs * l.outputs; i++) {
           if(l.weights[i] > w_minmax[1]) w_minmax[1] = l.weights[i]; 
           else if(l.weights[i] < w_minmax[0]) w_minmax[0] = l.weights[i]; 
        }
    }
    cal_qsz(w_minmax[0], w_minmax[1], &l.qs[1], &l.qz[1]);
    quantize_biases(l.biases, l.biases_int32, 1, l.outputs, l.qs[0] * l.qs[1], 0);
    quantize_matrix(l.weights, l.weights_int8, l.inputs, l.outputs, l.qs[1], l.qz[1]);
#ifdef GPU
    if(stage != 3) {
        push_connected_layer_quant(l);
    }
#endif
    cal_qsz(l.qs3_min_max[0], l.qs3_min_max[1], &l.qs[2], &l.qz[2]);
    cal_qsz(l.qa_min_max[0], l.qa_min_max[1], &l.qs[3], &l.qz[3]);
    if(net->layers[stage + 1].type == CONNECTED) {
        net->layers[stage + 1].qs[0] = l.qs[3];
        net->layers[stage + 1].qz[0] = l.qz[3];
    }

    int j;
    for(i = 0; i <= stage; ++i){
        layer l = net->layers[i];
        if(l.type == CONNECTED) {
            for(j=0;j<4;j++) {
                printf("qs[%d] = %f\t|\tqz[%d] = %d\n", j, l.qs[j], j, l.qz[j]);
            }
        }
    }
}

void set_qsz_int4(layer l, float *next_qs1, int4_t *next_qz1)
{
#ifdef GPU
    if(gpu_index >= 0){
        if(l.type == CONVOLUTIONAL) pull_convolutional_layer(l);
        else if(l.type == CONNECTED) pull_connected_layer(l);
    }
#endif
    // set S1, Z1 
    l.qs[0] = *next_qs1;
    l.qz_int4[0] = *next_qz1;
    
    // set S2, Z2
    float w_minmax[2];
    w_minmax[0] = l.weights[0];
    w_minmax[1] = l.weights[0];

    int i, n;
    if(l.type == CONNECTED) n = l.inputs * l.outputs;
    else n = l.nweights;

    for(i = 1; i < n; i++) {
       if(l.weights[i] > w_minmax[1]) w_minmax[1] = l.weights[i]; 
       else if(l.weights[i] < w_minmax[0]) w_minmax[0] = l.weights[i]; 
    }
    cal_qsz_int4(w_minmax[0], w_minmax[1], &l.qs[1], &l.qz_int4[1]);

    // throw S3, Z3 for  S1, Z1 of next layer
    cal_qsz_int4(l.act_range[0], l.act_range[1], &l.qs[2], &l.qz_int4[2]);
    *next_qs1 = l.qs[2];
    *next_qz1 = l.qz_int4[2];
}

void set_qsz(layer l, float *next_qs1, int8_t *next_qz1)
{
#ifdef GPU
    if(gpu_index >= 0){
        if(l.type == CONVOLUTIONAL) pull_convolutional_layer(l);
        else if(l.type == CONNECTED) pull_connected_layer(l);
    }
#endif
    // set S1, Z1 
    l.qs[0] = *next_qs1;
    l.qz[0] = *next_qz1;
    
    // set S2, Z2
    float w_minmax[2];
    w_minmax[0] = l.weights[0];
    w_minmax[1] = l.weights[0];

    int i, n;
    if(l.type == CONNECTED) n = l.inputs * l.outputs;
    else n = l.nweights;

    for(i = 1; i < n; i++) {
       if(l.weights[i] > w_minmax[1]) w_minmax[1] = l.weights[i]; 
       else if(l.weights[i] < w_minmax[0]) w_minmax[0] = l.weights[i]; 
    }
    cal_qsz(w_minmax[0], w_minmax[1], &l.qs[1], &l.qz[1]);

    // throw S3, Z3 for  S1, Z1 of next layer
    cal_qsz(l.act_range[0], l.act_range[1], &l.qs[2], &l.qz[2]);
    *next_qs1 = l.qs[2];
    *next_qz1 = l.qz[2];
}

void set_cluster_qsz_int4(layer l, float *next_qs1, int4_t *next_qz1, const int ctr)
{
#ifdef GPU
    if(gpu_index >= 0){
        if(l.type == CONVOLUTIONAL) pull_convolutional_layer(l);
        else if(l.type == CONNECTED) pull_connected_layer(l);
    }
#endif
    // Set S1, Z1 
    l.qs[0] = *next_qs1;
    l.qz_int4[0] = *next_qz1;
    
    // Set S2, Z2
    int i, n;
    float w_min, w_max;
    w_min = l.weights[0];
    w_max = l.weights[0];
    if(l.type == CONNECTED) n = l.inputs * l.outputs;
    else n = l.nweights;
    for(i = 1; i < n; i++) {
       if(l.weights[i] > w_max) w_max = l.weights[i]; 
       else if(l.weights[i] < w_min) w_min = l.weights[i]; 
    }
    cal_qsz_int4(w_min, w_max, &l.qs[1], &l.qz_int4[1]);

    // Set S3, Z3 and throw them to  S1, Z1 of next layer
    cal_qsz_int4(l.act_cluster[ctr], l.act_cluster[ctr + 1], &l.qs[2], &l.qz_int4[2]);
    *next_qs1 = l.qs[2];
    *next_qz1 = l.qz_int4[2];
}

void set_cluster_qsz(layer l, float *next_qs1, int8_t *next_qz1, const int ctr)
{
#ifdef GPU
    if(gpu_index >= 0){
        if(l.type == CONVOLUTIONAL) pull_convolutional_layer(l);
        else if(l.type == CONNECTED) pull_connected_layer(l);
    }
#endif
    // set S1, Z1 
    l.qs[0] = *next_qs1;
    l.qz[0] = *next_qz1;
    
    // set S2, Z2
    float w_minmax[2];
    w_minmax[0] = l.weights[0];
    w_minmax[1] = l.weights[0];

    int i, n;
    if(l.type == CONNECTED) n = l.inputs * l.outputs;
    else n = l.nweights;

    for(i = 1; i < n; i++) {
       if(l.weights[i] > w_minmax[1]) w_minmax[1] = l.weights[i]; 
       else if(l.weights[i] < w_minmax[0]) w_minmax[0] = l.weights[i]; 
    }
    cal_qsz(w_minmax[0], w_minmax[1], &l.qs[1], &l.qz[1]);

    // throw S3, Z3 for  S1, Z1 of next layer
    cal_qsz(l.act_cluster[ctr], l.act_cluster[ctr + 1], &l.qs[2], &l.qz[2]);
    *next_qs1 = l.qs[2];
    *next_qz1 = l.qz[2];
}

void set_pruned_qsz(layer l, int l_idx, float *prev_qs1, int8_t *prev_qz1)
{
    // set S1, Z1 
    if(l_idx == 0) {
        // image의 경우 최대 255 pixel이므로, int8로 표현 가능
        l.qs[0] = 1;
        l.qz[0] = -128;
    } else {
        l.qs[0] = *prev_qs1;
        l.qz[0] = *prev_qz1;
    }
    
    // set S2, Z2
    float w_minmax[2];
    w_minmax[0] = l.weights[0];
    w_minmax[1] = l.weights[0];

    // No Clamping
    int i, n;
    if(l.type == CONNECTED) n = l.inputs * l.outputs;
    else n = l.nweights;

    for(i = 1; i < n; i++) {
       if(l.weights[i] > w_minmax[1]) w_minmax[1] = l.weights[i]; 
       else if(l.weights[i] < w_minmax[0]) w_minmax[0] = l.weights[i]; 
    }

    printf("Pruning gap = %f\n", l.pruned_gap);
    printf("Before pruning = %f\t| After pruning = %f\n", w_minmax[1], w_minmax[1] - l.pruned_gap);
    w_minmax[0] += l.pruned_gap; 
    w_minmax[1] -= l.pruned_gap; 

    cal_qsz(w_minmax[0], w_minmax[1], &l.qs[1], &l.qz[1]);

    // set S3, Z3
    cal_qsz(l.qs3_min_max[0], l.qs3_min_max[1], &l.qs[2], &l.qz[2]);

    // Cal activation S & Z, save as S4, Z4, index 3 of l.qz&l.qs
    cal_qsz(l.qa_min_max[0], l.qa_min_max[1], &l.qs[3], &l.qz[3]);
    *prev_qs1 = l.qs[3];
    *prev_qz1 = l.qz[3];
}

void save_weights_upto(network *net, char *filename, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            save_convolutional_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
        } if(l.type == RNN){
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        } if (l.type == LSTM) {
            save_connected_weights(*(l.wi), fp);
            save_connected_weights(*(l.wf), fp);
            save_connected_weights(*(l.wo), fp);
            save_connected_weights(*(l.wg), fp);
            save_connected_weights(*(l.ui), fp);
            save_connected_weights(*(l.uf), fp);
            save_connected_weights(*(l.uo), fp);
            save_connected_weights(*(l.ug), fp);
        } if (l.type == GRU) {
            if(1){
                save_connected_weights(*(l.wz), fp);
                save_connected_weights(*(l.wr), fp);
                save_connected_weights(*(l.wh), fp);
                save_connected_weights(*(l.uz), fp);
                save_connected_weights(*(l.ur), fp);
                save_connected_weights(*(l.uh), fp);
            }else{
                save_connected_weights(*(l.reset_layer), fp);
                save_connected_weights(*(l.update_layer), fp);
                save_connected_weights(*(l.state_layer), fp);
            }
        }  if(l.type == CRNN){
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
        } if(l.type == LOCAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_local_layer(l);
            }
#endif
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.weights, sizeof(float), size, fp);
        }
    }
    fclose(fp);
}

void save_converted_weights(network *net, char *filename)
{
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            save_converted_convolutional_weights(l, fp);
        } if(l.type == CONNECTED){
            save_converted_connected_weights(l, fp);
        }
    }
    fclose(fp);
}

void save_weights_upto_int4(network *net, char *filename, char *filename_qw, char *filename_qsz, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s, %s, %s\n", filename, filename_qw, filename_qsz);
    FILE *fp = fopen(filename, "wb");
    FILE *fp_qw = fopen(filename_qw, "wb");
    FILE *fp_qsz = fopen(filename_qsz, "wb");
    
    if(!fp) file_error(filename);
    if(!fp_qw) file_error(filename_qw);
    if(!fp_qsz) file_error(filename_qsz);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    float next_qs1;
    int4_t next_qz1;
    cal_qsz_int4(net->input_range[0], net->input_range[1], &next_qs1, &next_qz1);
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            save_convolutional_weights(l, fp);
            set_qsz_int4(l, &next_qs1, &next_qz1);
            save_convolutional_weights_int4(l, fp_qw);
            save_qsz_int4(l, fp_qsz);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
            set_qsz_int4(l, &next_qs1, &next_qz1);
            save_connected_weights_int4(l, fp_qw);
            save_qsz_int4(l, fp_qsz);
        }
    }
    fclose(fp);
    fclose(fp_qw);
    fclose(fp_qsz);
}

void save_weights_upto_int8(network *net, char *filename, char *filename_qw, char *filename_qsz, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s, %s, %s\n", filename, filename_qw, filename_qsz);
    FILE *fp = fopen(filename, "wb");
    FILE *fp_qw = fopen(filename_qw, "wb");
    FILE *fp_qsz = fopen(filename_qsz, "wb");
    
    if(!fp) file_error(filename);
    if(!fp_qw) file_error(filename_qw);
    if(!fp_qsz) file_error(filename_qsz);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    float next_qs1;
    int8_t next_qz1;
    cal_qsz(net->input_range[0], net->input_range[1], &next_qs1, &next_qz1);
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            save_convolutional_weights(l, fp);
            set_qsz(l, &next_qs1, &next_qz1);
            save_convolutional_weights_int8(l, fp_qw);
            save_qsz(l, fp_qsz);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
            set_qsz(l, &next_qs1, &next_qz1);
            save_connected_weights_int8(l, fp_qw);
            save_qsz(l, fp_qsz);
        }
    }
    fclose(fp);
    fclose(fp_qw);
    fclose(fp_qsz);
}

void save_cluster_qparams_int4(network *net, char *backup_dir, char *base, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    int i, c;
    char filename[256];
    sprintf(filename, "%s/%s.%s.k%d.int4.weights", backup_dir, base, net->ctr_method, net->ctr_k);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            save_convolutional_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        }
    }
    fclose(fp);
    for(c = 0; c < net->ctr_k; c++) {
        char filename_qw[256];
        char filename_qsz[256];
        sprintf(filename_qw, "%s/%s.%s.k%d.c%d.int4.qweights", backup_dir, base, net->ctr_method, net->ctr_k, c);
        sprintf(filename_qsz, "%s/%s.%s.k%d.c%d.int4.qparams", backup_dir, base, net->ctr_method, net->ctr_k, c);

        fprintf(stderr, "Saving weights to %s, %s\n", filename_qw, filename_qsz);
        FILE *fp_qw = fopen(filename_qw, "wb");
        FILE *fp_qsz = fopen(filename_qsz, "wb");
        
        if(!fp_qw) file_error(filename_qw);
        if(!fp_qsz) file_error(filename_qsz);

        int ctr = c * 2;
        float next_qs1;
        int4_t next_qz1;
        cal_qsz_int4(net->input_cluster[ctr], net->input_cluster[ctr + 1], &next_qs1, &next_qz1);
        for(i = 0; i < net->n && i < cutoff; ++i){
            layer l = net->layers[i];
            if (l.dontsave) continue;
            if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
                set_cluster_qsz_int4(l, &next_qs1, &next_qz1, ctr);
                save_convolutional_weights_int4(l, fp_qw);
                save_qsz_int4(l, fp_qsz);
            } if(l.type == CONNECTED){
                set_cluster_qsz_int4(l, &next_qs1, &next_qz1, ctr);
                save_connected_weights_int4(l, fp_qw);
                save_qsz_int4(l, fp_qsz);
            }
        }
        fclose(fp_qw);
        fclose(fp_qsz);
    }
}

void save_cluster_qparams(network *net, char *backup_dir, char *base, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    int c;
    for(c = 0; c < 10; c++) {
        char filename[256];
        char filename_qw[256];
        char filename_qsz[256];
        sprintf(filename, "%s/%s.qat.c%d.int8.weights", backup_dir, base, c);
        sprintf(filename_qw, "%s/%s.qat.c%d.int8.qweights", backup_dir, base, c);
        sprintf(filename_qsz, "%s/%s.qat.c%d.int8.qparams", backup_dir, base, c);

        fprintf(stderr, "Saving weights to %s, %s, %s\n", filename, filename_qw, filename_qsz);
        FILE *fp = fopen(filename, "wb");
        FILE *fp_qw = fopen(filename_qw, "wb");
        FILE *fp_qsz = fopen(filename_qsz, "wb");
        
        if(!fp) file_error(filename);
        if(!fp_qw) file_error(filename_qw);
        if(!fp_qsz) file_error(filename_qsz);

        int major = 0;
        int minor = 2;
        int revision = 0;
        fwrite(&major, sizeof(int), 1, fp);
        fwrite(&minor, sizeof(int), 1, fp);
        fwrite(&revision, sizeof(int), 1, fp);
        fwrite(net->seen, sizeof(size_t), 1, fp);

        int i;
        int ctr = c * 2;
        float next_qs1;
        int8_t next_qz1;
        cal_qsz(net->input_cluster[ctr], net->input_cluster[ctr + 1], &next_qs1, &next_qz1);
        for(i = 0; i < net->n && i < cutoff; ++i){
            layer l = net->layers[i];
            if (l.dontsave) continue;
            if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
                save_convolutional_weights(l, fp);
                set_cluster_qsz(l, &next_qs1, &next_qz1, ctr);
                save_convolutional_weights_int8(l, fp_qw);
                save_qsz(l, fp_qsz);
            } if(l.type == CONNECTED){
                save_connected_weights(l, fp);
                set_cluster_qsz(l, &next_qs1, &next_qz1, ctr);
                save_connected_weights_int8(l, fp_qw);
                save_qsz(l, fp_qsz);
            }
        }
        fclose(fp);
        fclose(fp_qw);
        fclose(fp_qsz);
    }
}

void save_pruned_weights(network *net, char *filename, char *pruningfile, int cutoff, char* argv)
{
    fprintf(stderr, "Saving pruned weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    FILE *fp_pruning = fopen(pruningfile, "wb");
    if(!fp) file_error(filename);
    if(!fp_pruning) file_error(pruningfile);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            save_convolutional_weights(l, fp);
        } if(l.type == CONNECTED){
#ifdef GPU
            if(gpu_index >= 0){
                pull_connected_layer(l);
            }
            prune_matrix(l.weights, l.outputs * l.inputs, &l.pruned_gap, 0.7);
#endif
            save_pruning_param(l, fp_pruning);
#ifdef GPU
            cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
#endif
            save_connected_weights(l, fp);
        }
    }
    fclose(fp);
}

void save_input_qparams(network *net, char *filename_qsz, char *filename_range, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    //fprintf(stderr, "Saving qparams & inputs' ranges to %s, %s\n", filename_qsz, filename_range);
    FILE *fp_qsz = fopen(filename_qsz, "ab");
    FILE *fp_range = fopen(filename_range, "ab");
    if(!fp_qsz) file_error(filename_qsz);
    if(!fp_range) file_error(filename_range);

    net->input_range[0] = 1;
    net->input_range[1] = 0;
    set_min_max(net->input, &net->input_range[0], &net->input_range[1], net->inputs);
    save_input_range(net->input_range, fp_range);

    //char buff_value[256];
    //sprintf(buff_value, "backup/test.input_value");
    //FILE *fp_value = fopen(buff_value, "ab");
    //save_input_value(net->input, net->inputs, fp_value);
    //fclose(fp_value);

    int i;
    float next_qs1;
    int8_t next_qz1;
    cal_qsz(net->input_range[0], net->input_range[1], &next_qs1, &next_qz1);
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == CONNECTED){
            set_qsz(l, &next_qs1, &next_qz1);
            save_qsz(l, fp_qsz);
        }
    }
    fclose(fp_qsz);
    fclose(fp_range);
}

void save_qparams_upto(network *net, char *filename_qw, char *filename_qsz, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s, %s\n", filename_qw, filename_qsz);
    FILE *fp_qw = fopen(filename_qw, "wb");
    FILE *fp_qsz = fopen(filename_qsz, "wb");
    
    if(!fp_qw) file_error(filename_qw);
    if(!fp_qsz) file_error(filename_qsz);

    int i;
    float next_qs1;
    int8_t next_qz1;
    cal_qsz(net->input_range[0], net->input_range[1], &next_qs1, &next_qz1);
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONNECTED){
            set_qsz(l, &next_qs1, &next_qz1);
            save_qsz(l, fp_qsz);
            save_connected_weights_int8(l, fp_qw);
        }
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            set_qsz(l, &next_qs1, &next_qz1);
            save_qsz(l, fp_qsz);
            save_convolutional_weights_int8(l, fp_qw);
        }
    }
    fclose(fp_qw);
    fclose(fp_qsz);
}

void save_tuned_qparams(network *net, char *filename_qw, char *filename_qsz, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s, %s\n", filename_qw, filename_qsz);
    FILE *fp_qw = fopen(filename_qw, "wb");
    FILE *fp_qsz = fopen(filename_qsz, "wb");
    
    if(!fp_qw) file_error(filename_qw);
    if(!fp_qsz) file_error(filename_qsz);

    int i;
    //float next_qs1 = 1;
    //int8_t next_qz1 = -128;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONNECTED){
#ifdef GPU
            if(gpu_index >= 0){
                pull_connected_layer(l);
            }
#endif
            /*
            //set_pruned_qsz(l, i, &next_s1, &next_z1);
            if(i == 3) {
                set_qsz(l, &next_s1, &next_z1);
                save_connected_weights_int8(l, fp_qw);
            } else {
                set_pruned_qsz(l, i, &next_s1, &next_z1);
                save_pruned_connected_weights_int8(l, fp_qw);
            }
            //save_pruned_connected_weights_int8(l, fp_qw);
            */
            //set_qsz(l, &next_s1, &next_z1);
            save_connected_weights_int8(l, fp_qw);
            save_qsz(l, fp_qsz);
        }
    }
    fclose(fp_qw);
    fclose(fp_qsz);
}

void save_weights(network *net, char *filename)
{
    save_weights_upto(net, filename, net->n);
}

void save_weights_int4(network *net, char *filename, char *filename_qw, char *filename_qsz)
{
    save_weights_upto_int4(net, filename, filename_qw, filename_qsz, net->n);
}

void save_weights_int8(network *net, char *filename, char *filename_qw, char *filename_qsz)
{
    save_weights_upto_int8(net, filename, filename_qw, filename_qsz, net->n);
}

void save_qparams(network *net, char *filename_qw, char *filename_qsz)
{
    save_qparams_upto(net, filename_qw, filename_qsz, net->n);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void transpose_matrix_quant(int8_t *a, int rows, int cols)
{
    float *transpose = calloc(rows*cols, sizeof(int8_t));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(int8_t));
    free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);

    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer(l);
    }
#endif
}

void load_connected_weights_without_xpose_info(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer(l);
    }
#endif
}

void load_connected_weights_from_txt(layer l, FILE *fp)
{
    int i;
    for (i = 0; i < l.outputs * l.inputs; i++) fscanf(fp, "%f\n", &l.weights[i]);
    for (i = 0; i < l.outputs; i++) fscanf(fp, "%f\n", &l.biases[i]);
}

void load_connected_weights_int8(layer l, FILE *fp, int transpose)
{
    fread(l.biases_int32, sizeof(int32_t), l.outputs, fp);
    fread(l.weights_int8, sizeof(int8_t), l.outputs*l.inputs, fp);

    if(transpose){
        transpose_matrix_quant(l.weights_int8, l.inputs, l.outputs);
    }
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer_quant(l);
    }
#endif
}

void load_connected_weights_int4(layer l, FILE *fp)
{
    fread(l.biases_int32, sizeof(int32_t), l.outputs, fp);
    fread(l.weights_int4, sizeof(int4_t),  l.inputs * l.outputs, fp);
}

void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_batchnorm_layer(l);
    }
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    int size = l.c*l.size*l.size;
    int i, j, k;
    for(i = 0; i < l.n; ++i){
        float mean = 0;
        fread(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            fread(&c, sizeof(char), 1, fp);
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                l.weights[index + k] = (c & 1<<k) ? mean : -mean;
            }
        }
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_convolutional_weights(layer l, FILE *fp)
{
    int num = l.c/l.groups*l.n*l.size*l.size;
    fread(l.biases, sizeof(float), l.n, fp);
    fread(l.weights, sizeof(float), num, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_convolutional_weights_from_txt(layer l, FILE *fp)
{
    int num = l.c/l.groups*l.n*l.size*l.size;
    int i;
    for (i = 0; i < num; i++) fscanf(fp, "%f\n", &l.weights[i]);
    for (i = 0; i < l.n; i++) fscanf(fp, "%f\n", &l.biases[i]);
}

void load_convolutional_weights_int8(layer l, FILE *fp)
{
    fread(l.biases_int32, sizeof(int32_t), l.n, fp);
    fread(l.weights_int8, sizeof(int8_t), l.nweights, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_quantized_convolutional_layer(l);
    }
#endif
}

void load_convolutional_weights_int4(layer l, FILE *fp)
{
    fread(l.biases_int32, sizeof(int32_t), l.n, fp);
    fread(l.weights_int4, sizeof(int4_t), l.nweights, fp);
}

void assign_qsz(network *net, layer l, int idx)
{
    int i;
    for(i = 0; i < 3; i++) {
        l.qs[i] = net->inputs_qs[idx + i];
        l.qz[i] = net->inputs_qz[idx + i];
    }
}

void load_qsz_int4(layer l, FILE *fp)
{
    fread(l.qs, sizeof(float), 3, fp);
    fread(l.qz_int4, sizeof(int4_t), 3, fp);
    int i;
    for(i = 0; i < 3 ;i++) printf("S%d: %f | Z%d: %d\n", i + 1, l.qs[i], i + 1, l.qz_int4[i].el);
}

void load_qsz(layer l, FILE *fp)
{
    fread(l.qs, sizeof(float), 3, fp);
    fread(l.qz, sizeof(int8_t), 3, fp);
    int i;
    for(i = 0; i < 3; i++) printf("S%d: %f | Z%d: %d\n", i + 1, l.qs[i], i + 1, l.qz[i]);
}

void load_grad_qsz(layer l, int layer_idx)
{
    if(layer_idx==0) {
        l.qs[1] = 0.009279;
        l.qz[1] = 11;
        l.qs[2] = 0.028158;
        l.qz[2] = 20;
    } else if(layer_idx==2) {
        l.qs[0] = 0.028158;
        l.qz[0] = 20;
        l.qs[1] = 0.012329;
        l.qz[1] = -19;
        l.qs[2] = 0.058192;
        l.qz[2] = -20;
    } else if(layer_idx==3) {
        l.qs[0] = 0.058192;
        l.qz[0] = -20;
        l.qs[1] = 0.006457;
        l.qz[1] = 1;
        l.qs[2] = 0.110812;
        l.qz[2] = 2;
    } else if(layer_idx==5) {
        l.qs[0] = 0.110812;
        l.qz[0] = 2;
        l.qs[1] = 0.006807;
        l.qz[1] = -16;
        l.qs[2] = 0.173217;
        l.qz[2] = -13;
    } else if(layer_idx==6) {
        l.qs[0] = 0.173217;
        l.qz[0] = -13;
        l.qs[1] = 0.003931;
        l.qz[1] = -6;
        l.qs[2] = 0.274032;
        l.qz[2] = 31;
    } else if(layer_idx==7) {
        l.qs[0] = 0.274032;
        l.qz[0] = 31;
        l.qs[1] = 0.00385;
        l.qz[1] = -15;
        l.qs[2] = 0.170158;
        l.qz[2] = 3;
    } else if(layer_idx==8) {
        l.qs[0] = 0.170158;
        l.qz[0] = 3;
        l.qs[1] = 0.008612;
        l.qz[1] = -31;
        l.qs[2] = 0.319470;
        l.qz[2] = -80;
    }
    int j;
    for(j =0;j<3;j++)
        printf("S = %f\tZ = %d\n", l.qs[j], l.qz[j]);
    quantize_biases(l.biases, l.biases_int32, 1, l.n, l.qs[0] * l.qs[1], 0);
    quantize_matrix(l.weights, l.weights_int8, 1, l.nweights, l.qs[1], l.qz[1]);
}

void load_mask(layer l) {
    int i;
    for(i = 0; i < l.inputs * l.outputs; i++) {
        if(l.weights[i] == 0) l.weights_mask[i] = 0;
        else l.weights_mask[i] = 1;
    }
}

void initialize_pruned_weights(layer l) {
    int i;

    printf("initialize mat\n");
    float scale = sqrt(2./l.inputs);
    for(i = 0; i < l.outputs*l.inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }
#ifdef GPU
    mask_matrix(l.weights_gpu, l.weights_mask, l.outputs * l.inputs);
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_weights_for_transfer_learning(network *net, char *filename)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading imagenete pretrained weights from %s...\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int transpose = 0;
    // stop parsing before last FC layer(right before Softmax)
    int i;
    for(i = 0; i < net->n && i < net->n - 2; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
        if(l.type == BATCHNORM){
            load_batchnorm_weights(l, fp);
        }
        if(l.type == CRNN){
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == RNN){
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if (l.type == LSTM) {
            load_connected_weights(*(l.wi), fp, transpose);
            load_connected_weights(*(l.wf), fp, transpose);
            load_connected_weights(*(l.wo), fp, transpose);
            load_connected_weights(*(l.wg), fp, transpose);
            load_connected_weights(*(l.ui), fp, transpose);
            load_connected_weights(*(l.uf), fp, transpose);
            load_connected_weights(*(l.uo), fp, transpose);
            load_connected_weights(*(l.ug), fp, transpose);
        }
        if (l.type == GRU) {
            if(1){
                load_connected_weights(*(l.wz), fp, transpose);
                load_connected_weights(*(l.wr), fp, transpose);
                load_connected_weights(*(l.wh), fp, transpose);
                load_connected_weights(*(l.uz), fp, transpose);
                load_connected_weights(*(l.ur), fp, transpose);
                load_connected_weights(*(l.uh), fp, transpose);
            }else{
                load_connected_weights(*(l.reset_layer), fp, transpose);
                load_connected_weights(*(l.update_layer), fp, transpose);
                load_connected_weights(*(l.state_layer), fp, transpose);
            }
        }
        if(l.type == LOCAL){
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_local_layer(l);
            }
#endif
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_weights_without_xpose_info(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights_without_xpose_info(l, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_weights_from_txt_upto(network *net, char *filename, int start, int cutoff)
{
    fprintf(stderr, "Loading weights from %s...\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "r");
    if(!fp) file_error(filename);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights_from_txt(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights_from_txt(l, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_pruned_weights(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading pruned weights from %s...\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
            load_mask(l);
            //initialize_pruned_weights(l);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_quantized_weights_only(network *net, char *filename_qw, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading quantized weights from %s...\n", filename_qw);
    fflush(stdout);
    FILE *fp_qw = fopen(filename_qw, "rb");
    if(!fp_qw) file_error(filename_qw);

    int i;
    int transpose = 0;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights_int8(l, fp_qw);
        }
        if(l.type == CONNECTED){
            load_connected_weights_int8(l, fp_qw, transpose);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp_qw);
}

void make_fake_integer_weights(network *net)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Make fake integer weights...\n");
    fflush(stdout);

    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            int8_to_float(l.weights_int8, l.fake_int_weights, l.nweights);
#ifdef GPU
            cuda_push_array(l.fake_int_weights_gpu, l.fake_int_weights, l.nweights);
#endif
        }
    }
}

void load_quantized_weights_only_int4(network *net, char *filename_qw, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading int4 qweights from %s...\n", filename_qw);
    fflush(stdout);
    FILE *fp_qw = fopen(filename_qw, "rb");
    if(!fp_qw) file_error(filename_qw);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if(l.dontload) continue;
        if(l.type == CONVOLUTIONAL){
            load_convolutional_weights_int4(l, fp_qw);
            int4_to_float(l.weights_int4, l.fake_int_weights, l.nweights);
#ifdef GPU
            cuda_push_array(l.fake_int_weights_gpu, l.fake_int_weights, l.nweights);
#endif
        }
        if(l.type == CONNECTED){
            load_connected_weights_int4(l, fp_qw);
            int4_to_float(l.weights_int4, l.fake_int_weights, l.inputs * l.outputs);
#ifdef GPU
            cuda_push_array(l.fake_int_weights_gpu, l.fake_int_weights, l.inputs * l.outputs);
#endif
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp_qw);
}

void load_qparams_only_int4(network *net, char *filename_qsz, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading int4 qparams from %s...\n", filename_qsz);
    fflush(stdout);
    FILE *fp_qsz = fopen(filename_qsz, "rb");
    if(!fp_qsz) file_error(filename_qsz);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if ( l.dontload ) continue;
        if ( l.type == CONVOLUTIONAL || l.type == CONNECTED )
        {
            load_qsz_int4(l, fp_qsz);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp_qsz);
}

void load_weights_and_quantize(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading  weights from %s...\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights(l, fp);
            quantize_biases(l.biases, l.biases_int32, 1, l.n, l.qs[0] * l.qs[1], 0);
            quantize_matrix(l.weights, l.weights_int8, 1, l.nweights, l.qs[1], l.qz[1]);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_input_qparams(network *net, char* filename_qsz, char* filename_ranges)
{
    FILE *fp_qsz = fopen(filename_qsz, "rb");
    FILE *fp_ranges = fopen(filename_ranges, "rb");
    if(!fp_qsz) file_error(filename_qsz);
    if(!fp_ranges) file_error(filename_ranges);

    printf("Load inputs' qsz sets..\n");
    int i, j;
    int a = net->n_qsz;       // 3: number of S, Z per layer
    int b = net->n_qsz_layer; // 8: number of layers which have S, Z
    int c = net->n_qsz_set;   // 100: number of sampled qsz sets
    for(i = 0; i < c; i++) {
        for(j = 0; j < b; j++) {
            fread(&net->inputs_qs[i*a*b + j*a], sizeof(float), a, fp_qsz);
            fread(&net->inputs_qz[i*a*b + j*a], sizeof(int8_t), a, fp_qsz);
        }
    }

    printf("Load inputs' ranges..\n");
    fread(net->input_ranges, sizeof(float), net->n_qsz_set * 2, fp_ranges);

    //FILE *fp_value = fopen("backup/cifar_conv_qat_rand.input_value", "rb");
    //if(!fp_value) file_error("backup/cifar_conv_qat_rand.input_value");
    //fread(net->input_values, sizeof(float), net->n_qsz_set * net->inputs, fp_value);
    //fclose(fp_value);

    fclose(fp_qsz);
    fclose(fp_ranges);
}

void assign_selected_qparams(network *net, int selected, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    int i;
    int loaded = 0;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if(l.type == CONNECTED){
            assign_qsz(net, l, selected * net->n_qsz_layer * net->n_qsz + loaded * net->n_qsz);
            quantize_biases(l.biases, l.biases_int32, 1, l.outputs, l.qs[0] * l.qs[1], 0);
            quantize_matrix(l.weights, l.weights_int8, l.inputs, l.outputs, l.qs[1], l.qz[1]);
            loaded += 1;
#ifdef GPU
            if(gpu_index >= 0){
                push_connected_layer_quant(l);
            }
#endif
        }
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            assign_qsz(net, l, selected * net->n_qsz_layer * net->n_qsz + loaded * net->n_qsz);
            quantize_biases(l.biases, l.biases_int32, 1, l.n, l.qs[0] * l.qs[1], 0);
            quantize_matrix(l.weights, l.weights_int8, 1, l.nweights, l.qs[1], l.qz[1]);
            loaded += 1;
        }
    }
}

void select_qparams(network *net)
{
    float cur_range[2] = {1, 0};
    set_min_max(net->input, &cur_range[0], &cur_range[1], net->inputs);
    //printf("current\tmin = %f | max = %f\n", cur_range[0], cur_range[1]);
    
    int i;
    int closest_idx = 0;
    float difference, closest_diff;
    closest_diff = 100000;
    for(i = 0; i < net->n_qsz_set; i++) {
        difference = fabs(net->input_ranges[i*2] - cur_range[0]) + fabs(net->input_ranges[i*2+1] - cur_range[1]);
        if(closest_diff > difference) {
            closest_idx = i;
            closest_diff = difference;
        }
    }
    //printf("closest\tmin = %f | max = %f\n", net->input_ranges[closest_idx*2], net->input_ranges[closest_idx*2+1]);
    //printf("closest\tindex = %d\n", closest_idx);
    //printf("closest\tdiff = %f\n", closest_diff);

    //float distance, closest_dist;
    //distance = 0;
    //closest_dist = 100000;
    //for(i = 0; i < net->n_qsz_set; i++) {
    //    for(j = 0; j < net->inputs; j++) {
    //        distance += fabs(net->input_values[i*net->inputs + j] - net->input[j]);
    //    }
    //    if(closest_dist > distance) {
    //        closest_idx = i;
    //        closest_dist = distance;
    //    }
    //    distance = 0;
    //}
    //printf("closest\tindex = %d\n", closest_idx);
    //printf("closest\tdistance = %f\n", closest_dist);
    assign_selected_qparams(net, closest_idx, net->n);
}

void load_qparams_only(network *net, char *filename_qsz, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading qparams from %s...\n", filename_qsz);
    fflush(stdout);
    FILE *fp_qsz = fopen(filename_qsz, "rb");
    if(!fp_qsz) file_error(filename_qsz);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_qsz(l, fp_qsz);
        }
        if(l.type == CONNECTED){
            load_qsz(l, fp_qsz);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp_qsz);
}

void load_weights_upto_quant(network *net, char *filename, char *filename_qw, char *filename_qsz, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    FILE *fp_qw = fopen(filename_qw, "rb");
    FILE *fp_qsz = fopen(filename_qsz, "rb");
    if(!fp) file_error(filename);
    if(!fp_qw) file_error(filename_qw);
    if(!fp_qsz) file_error(filename_qsz);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights(l, fp);
            load_convolutional_weights_int8(l, fp_qw);
            load_qsz(l, fp_qsz);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
            load_connected_weights_int8(l, fp_qw, transpose);
            load_qsz(l, fp_qsz);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
    fclose(fp_qw);
    fclose(fp_qsz);
}

void load_pruning_params(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading pruning params from %s...\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            fread(&net->layers[i].pruned_gap, sizeof(float), 1, fp);
        }
        if(l.type == CONNECTED){
            fread(&net->layers[i].pruned_gap, sizeof(float), 1, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n);
}

void load_weights_from_txt(network *net, char *filename)
{
    load_weights_from_txt_upto(net, filename, 0, net->n);
}

void load_weights_int8(network *net, char *filename, char *filename_qw, char *filename_qsz)
{
    load_weights_upto_quant(net, filename, filename_qw, filename_qsz, 0, net->n);
}

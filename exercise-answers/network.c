float *network_predict_int8(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    
    layer first = net->layers[0];
    int i, len_input;
    len_input = net->batch * net->inputs;
    int8_t quantized_input[len_input];
    for(i = 0; i < len_input; i++) {
        quantized_input[i] = 0;
    }

    quantize_int8_cpu(input, quantized_input, net->batch * net->inputs, first.qs[0], first.qz[0]);

    net->input_int8 = quantized_input;

    net->truth = 0;
    net->train = 0;
    net->delta = 0;

    forward_network_int8(net);
    float *out = net->output;
    *net = orig;
    return out;
}

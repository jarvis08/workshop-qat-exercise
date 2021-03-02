float *network_predict_int8(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    
    layer first = net->layers[0];
    quantize_int8_cpu(net->input, net->input_int8, net->batch * net->inputs, first.qs[0], first.qz[0]);

    net->truth = 0;
    net->train = 0;
    net->delta = 0;

    forward_network_int8(net);
    float *out = net->output;
    *net = orig;
    return out;
}

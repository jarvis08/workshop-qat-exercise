# Quantization Aware Training in Darknet

- Darknet's default files
    - `*.data`: dataset's configuration file
    - `*.cfg`: network & train/test configuration file
    - `*.weights`: model's parameters(weights & biases)
- Quantization files
    - `*.qat.int8.qweights`: quantized model's parameters
    - `*.qat.int8.qparams`: quantization params

## Exercises

### [FP DNN -> INT8 DNN] by Fake Quantization

```bash
# [src/gemm.cpp] `void cal_qsz(...)`
min, max 값을 사용하여 S와 Z를 계산하고, 저장하는 함수 작성

# [src/quant_utils.c] `void fake_quantize_int8_cpu(...)`
Quantize/dequantize를 수행하는 fake quantization 함수 작성

# [src/quant_utils.c] `void ema_cpu(...)`
Min/max 값을 지수 평균으로 누적하는 ema 함수 작성
```

### [INT8] Integer-arithmetic Only Inference

```bash
# [src/network.cpp] `void network_predict_int8(...)`
FP input을 INT8로 변환하도록 quantize_int8_cpu(...) 함수를 사용

# [src/quant_utils.c] `void totalsum_int8_cpu(...)`
실수 M값을 계산하는 코드를 올바르게 작성

# [src/quant_utils.c] `void totalsum_int8_cpu(...)`
Bias를 더해주는 코드를 삽입
```

## Commands

### Pre-training FP model

```bash
# Train
./darknet classifier train_randseq cfg/mnist.data cfg/mnist_fc4.cfg

# Validate
./darknet classifier valid cfg/mnist.data cfg/mnist_fc4.cfg backup/mnist_fc4.weights
```

### Quantization Aware Training

```bash
# Train
./darknet classifier qat_int8 cfg/mnist.data cfg/mnist_fc4.cfg backup/mnist_fc4.weights

# Validate
./darknet classifier qat_int8 cfg/mnist.data cfg/mnist_fc4.cfg backup/mnist_fc4.qat.int8.qweights  backup/mnist_fc4.qat.int8.qparams
```

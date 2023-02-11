<!--
 * @Author: kavinbj
 * @Date: 2022-11-24 13:25:32
 * @LastEditTime: 2023-02-11 12:47:58
 * @FilePath: README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by kavinbj, All Rights Reserved. 
-->
# torch-quantizer
pytorch quantization tools


pytorch 模型 量化工具


pytorch版本 v1.13

## 量化简介

在深度学习中，量化指的是使用更少的bit来存储原本以浮点数存储的tensor，以及使用更少的bit来完成原本以浮点数完成的计算。这么做的好处主要有如下几点：

- 更少的模型体积，接近4倍的减少；
- 可以更快的计算，由于更少的内存访问和更快的int8计算，可以快2~4倍。

一个量化后的模型，其部分或者全部的tensor操作会使用int类型来计算，而不是使用量化之前的float类型。当然，量化还需要底层硬件支持，x86 CPU（支持AVX2）、ARM CPU、Google TPU、Nvidia Volta/Turing/Ampere、Qualcomm DSP这些主流硬件都对量化提供了支持。

PyTorch 支持多种量化深度学习模型的方法。在大多数情况下，模型是在 FP32 中训练的，然后模型被转换为 INT8。此外，PyTorch 还支持量化感知训练，它使用伪量化模块对前向和后向传递中的量化误差进行建模。请注意，整个计算是在浮点数中进行的。在量化感知训练结束时，PyTorch 提供了转换函数，将训练好的模型转换为较低的精度。

在较低级别，PyTorch 提供了一种表示量化张量并对其执行运算的方法。它们可用于直接构建以较低精度执行全部或部分计算的模型。提供了更高级别的 API，其中包含将 FP32 模型转换为较低精度且精度损失最小的典型工作流程。


example

eager mode quantization:
1、Post Training Dynamic Quantization 
2、Post Training Static Quantization (PTQ)
   steps:
   1、modify fload model, add QuantStub(), DeQuantStub(), and use nn.quantized.FloatFunctional() replace some float Function.
   2、fuse_model() and model.eval()
   3、qconfig
   4、prepare(myModel, inplace=True)
   5、calibration
   6、convert(myModel, inplace=True)
   7、evaluate accuracy
   note: per_tensor vs per_channel qconfig,   per_channel config can have better accuracy in cnn example

3、Quantization Aware Training for Static Quantization  (QAT)
   steps:
   1、modify fload model, add QuantStub(), DeQuantStub(), and use nn.quantized.FloatFunctional() replace some float Function.
   2、fuse_model() and model.training()
   3、qconfig
   4、prepare(myModel, inplace=True)
   5、training_loop
   6、convert(myModel, inplace=True)
   7、evaluate accuracy

fx mode quantization:
1、Post Training Dynamic Quantization
2、Post Training Static Quantization (PTQ_fx)
   steps:
   1、config and model.eval()  
   2、prepare_fx
   3、calibration
   4、convert_fx
   5、evaluate
   note: per_tensor vs per_channel qconfig,   per_channel config can have better accuracy in cnn example
   
3、Quantization Aware Training for Static Quantization (QAT_fx)
    steps:
   1、config and model.eval()  
   2、prepare_qat_fx
   3、training_loop
   4、convert_fx
   5、evaluate



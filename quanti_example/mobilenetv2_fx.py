'''
Author: error: git config user.name & please set dead value or install git
Date: 2023-02-04 21:45:25
LastEditTime: 2023-02-05 18:21:01
FilePath: mobilenetv2_fx.py
Description: 

Copyright (c) 2023 by ${git_name}, All Rights Reserved. 
'''
# from torchvision.models.quantization import mobilenet_v2
from torchvision.models import MobileNetV2, MobileNet_V2_Weights, mobilenet_v2
from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig_mapping
from torch.ao.quantization.qconfig import default_qconfig
from torch.ao.quantization.quantize import prepare, convert, prepare_qat
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx

from torch import jit
import torch
from torch import nn
from .qutils import load_data_cifar10, train_fine_tuning, print_size_of_model, evaluate, load_model


saved_model_dir = 'models/'
float_model_file_fx = 'mobilenet_pretrained_float_fx.pth'
scripted_float_model_file_fx = 'mobilenet_float_scripted_fx.pth'
scripted_ptq_model_file_fx = 'mobilenet_ptq_scripted_fx.pth'
scripted_quantized_model_file_fx = 'mobilenet_quantization_scripted_quantized_fx.pth'
scripted_qat_model_file_fx = 'mobilenet_qat_scripted_quantized_fx.pth'

learning_rate = 5e-5
num_epochs = 30
batch_size = 32
num_classes = 10

# 设置评估策略
criterion = nn.CrossEntropyLoss()

train_iter, test_iter = load_data_cifar10(batch_size=batch_size, resize=224, num_workers=4)

num_train, num_eval = (50000, 10000)

def load_data():
    train_iter, test_iter = load_data_cifar10(batch_size=batch_size, resize=224, num_workers=4)
    num_train = sum(len(ys) for _, ys in train_iter)
    num_eval = sum(len(ys) for _, ys in test_iter)
    print(num_train, num_eval)

# 定义模型  注意，这里的模型为原始的fload模型，没有QuantStub DeQuantStub 组件
def create_model(num_classes=10,
                 pretrained=True):
    float_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

    # 匹配 ``num_classes``
    float_model.classifier[1] = nn.Linear(float_model.last_channel,
                                          num_classes)
    return float_model


def fine_tuning_model():
    # print('训练、测试批次分别为：', len(train_iter), len(test_iter))
    float_model = create_model(pretrained=True, num_classes=num_classes)
    # print(float_model)
    train_fine_tuning(float_model, train_iter, test_iter, 
                      learning_rate=learning_rate,
                      num_epochs=num_epochs,
                      device='cuda:0',
                      param_group=True)
    
    torch.save(float_model.state_dict(), saved_model_dir + float_model_file_fx)

def load_float_model():
    float_model = create_model(pretrained=True, num_classes=num_classes)
    float_model = load_model(float_model, saved_model_dir + float_model_file_fx)
    # print(float_model)

    print_size_of_model(float_model)
    # print(float_model.features[1].conv)

    top1, top5 = evaluate(float_model, criterion, test_iter)

    print(f'\n浮点模型：\n\t在 {num_eval} 张图片上评估 accuracy 为: {top1.avg:2.2f}')
    # 保存
    jit.save(jit.script(float_model), saved_model_dir + scripted_float_model_file_fx)

def PTQ_fx_per_tensor():
    # 加载模型
    float_model = create_model(pretrained=True, num_classes=num_classes)
    float_model = load_model(float_model, saved_model_dir + float_model_file_fx)

    float_model.eval()
    qconfig = get_default_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    example_inputs = (next(iter(train_iter))[0])

    prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)
    print('\n 查看观测者插入后的 inverted residual \n\n', prepared_model)

    num_calibration_batches = 200 # 取部分训练集做校准
    evaluate(prepared_model, criterion, train_iter, neval_batches=num_calibration_batches)
    print('\nPTQ：校准完成！')

    quantized_model = convert_fx(prepared_model)
    print(quantized_model)
    print('PTQ：转换完成！')

    print_size_of_model(quantized_model)

    top1, top5 = evaluate(quantized_model, criterion, test_iter)
    print(f'\nPTQ_fx 模型：\n\t在 {num_eval} 张图片上评估 accuracy 为: {top1.avg:2.2f}')
    jit.save(jit.script(quantized_model), saved_model_dir + scripted_ptq_model_file_fx)
    
    ## PTQ_fx 模型：
        # 在 10000 张图片上评估 accuracy 为: 50.98

def PTQ_fx_per_channel():
     # 加载模型
    float_model = create_model(pretrained=True, num_classes=num_classes)
    float_model = load_model(float_model, saved_model_dir + float_model_file_fx)

    float_model.eval()
    ## 默认参数
    # qconfig = get_default_qconfig("fbgemm")
    # qconfig_mapping = QConfigMapping().set_global(qconfig)

    # print('qconfig_mapping', qconfig_mapping)
    # per_channel 配置
    qconfig = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(
        qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
        ),
        weight=torch.ao.quantization.observer.default_per_channel_weight_observer
    )
    qconfig_mapping = {"": qconfig}

    example_inputs = (next(iter(train_iter))[0])

    prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)
    print('\n 查看观测者插入后的 inverted residual \n\n', prepared_model)

    num_calibration_batches = 200 # 取部分训练集做校准
    evaluate(prepared_model, criterion, train_iter, neval_batches=num_calibration_batches)
    print('\nPTQ：校准完成！')

    model_type = 'PTQ 模型（直方图观测器）'
    quantized_model = convert_fx(prepared_model)
    print_size_of_model(quantized_model)

    top1, top5 = evaluate(quantized_model, criterion, test_iter)
    print(f'\n{model_type}：\n\t在 {num_eval} 张图片上评估 accuracy 为: {top1.avg:2.2f}')
    jit.save(jit.script(quantized_model), saved_model_dir + scripted_quantized_model_file_fx)

    ## 模型大小：8.909929 MB
    # PTQ_fx_per_channel 模型（直方图观测器）：
        # 在 10000 张图片上评估 accuracy 为: 93.90


def QAT_fx():
     # 加载模型
    float_model = create_model(pretrained=True, num_classes=num_classes)
    float_model = load_model(float_model, saved_model_dir + float_model_file_fx)
    
    float_model.train()
    
    qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
    # prepare
    example_inputs = (next(iter(train_iter))[0])
    model_prepared = prepare_qat_fx(float_model, qconfig_mapping, example_inputs)

    # training loop
    train_fine_tuning(model_prepared,
                     train_iter,
                     test_iter,
                     learning_rate=learning_rate,
                     num_epochs=1,
                     device='cuda:0',
                     param_group=True,
                     is_freeze=False,
                     is_quantized_acc=False,
                     need_qconfig=False,
                     ylim=[0.8, 1])

    # quantize
    quantized_model = convert_fx(model_prepared)

    # print_size_of_model(quantized_model)
    # top1, top5 = evaluate(quantized_model, criterion, test_iter)
    # print(f'\nQAT_FX模型：\n\t在 {num_eval} 张图片上评估 accuracy 为: {top1.avg:2.2f}')
    # jit.save(jit.script(quantized_model), saved_model_dir + scripted_qat_model_file_fx)


def test():
    print('mobilenetv2_fx', torch.__version__)

    # 1、load data

    # 2、fine_tuning_model   
    # fine_tuning_model()
    # loss 0.057, train acc 0.981, test acc 0.939

    # 3、load_float_model
    # load_float_model()
    # 浮点模型：
        # 在 10000 张图片上评估 accuracy 为: 93.90


    # 4、PTQ_per_tensor
    # PTQ_fx_per_tensor()
    
    # 5、PTQ_fx_per_channel
    # PTQ_fx_per_channel()

    # 6、QAT_fx
    QAT_fx()


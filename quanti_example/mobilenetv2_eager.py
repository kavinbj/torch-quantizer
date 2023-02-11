'''
Author: kavinbj
Date: 2023-02-04 00:30:06
LastEditTime: 2023-02-04 21:10:41
FilePath: mobilenetv2_eager.py
Description: 

Copyright (c) 2023 by ${git_name}, All Rights Reserved. 
'''
from torchvision.models.quantization import mobilenet_v2
from torch.ao.quantization.qconfig import default_qconfig
from torch.ao.quantization.quantize import prepare, convert, prepare_qat

from torch import jit
import torch
from torch import nn
from .qutils import load_data_cifar10, train_fine_tuning, print_size_of_model, evaluate, load_model

saved_model_dir = 'models/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_float_scripted.pth'
scripted_ptq_model_file = 'mobilenet_ptq_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'
scripted_qat_model_file = 'mobilenet_qat_scripted_quantized.pth'

learning_rate = 5e-5
num_epochs = 30
batch_size = 32
num_classes = 10

# 设置评估策略
criterion = nn.CrossEntropyLoss()

train_iter, test_iter = load_data_cifar10(batch_size=batch_size, resize=224, num_workers=4)

# num_train = sum(len(ys) for _, ys in train_iter)
# num_eval = sum(len(ys) for _, ys in test_iter)
num_train, num_eval = (50000, 10000)

def load_data():
    train_iter, test_iter = load_data_cifar10(batch_size=batch_size, resize=224, num_workers=4)
    num_train = sum(len(ys) for _, ys in train_iter)
    num_eval = sum(len(ys) for _, ys in test_iter)
    print(num_train, num_eval)

# 定义模型
def create_model(quantize=False,
                 num_classes=10,
                 pretrained=False):
    float_model = mobilenet_v2(pretrained=pretrained,
                               quantize=quantize)
    # 匹配 ``num_classes``
    float_model.classifier[1] = nn.Linear(float_model.last_channel,
                                          num_classes)
    return float_model


def fine_tuning_model():
    # print('训练、测试批次分别为：', len(train_iter), len(test_iter))
    float_model = create_model(pretrained=True,
                           quantize=False,
                           num_classes=num_classes)
    # print(float_model)
    train_fine_tuning(float_model, train_iter, test_iter, 
                      learning_rate=learning_rate,
                      num_epochs=num_epochs,
                      device='cuda:0',
                      param_group=True)
    
    torch.save(float_model.state_dict(), saved_model_dir + float_model_file)

def print_info(model,
               model_type='浮点模型',
               test_iter=test_iter,
               criterion=criterion, num_eval=num_eval):
    '''打印信息'''
    print_size_of_model(model)
    top1, top5 = evaluate(model, criterion, test_iter)
    print(f'\n{model_type}：\n\t'
          f'在 {num_eval} 张图片上评估 accuracy 为: {top1.avg:2.5f}')


def load_float_model():
    float_model = create_model(quantize=False, num_classes=num_classes)
    float_model = load_model(float_model, saved_model_dir + float_model_file)
    # print_info(float_model)
    print(float_model.features[1].conv)

    float_model.fuse_model(is_qat=None)

    print(float_model.features[1].conv)
    model_type = '融合后的浮点模型'
    print("baseline 模型大小")
    print_size_of_model(float_model)
    top1, top5 = evaluate(float_model, criterion, test_iter)

    ## baseline 模型大小
    ## 模型大小：9.177753 MB

    print(f'\n{model_type}：\n\t在 {num_eval} 张图片上评估 accuracy 为: {top1.avg:2.2f}')
    # 保存
    jit.save(jit.script(float_model), saved_model_dir + scripted_float_model_file)

# 
def PTQ_per_tensor():
    # 加载模型
    myModel = create_model(pretrained=False, quantize=False, num_classes=num_classes)
    myModel = load_model(myModel, saved_model_dir + float_model_file)
    myModel.eval()

    # 融合
    myModel.fuse_model()

    # 指定量化配置
    myModel.qconfig = default_qconfig
    print(myModel.qconfig)

    print('PTQ 准备：插入观测者')
    prepare(myModel, inplace=True)
    print('\n 查看观测者插入后的 inverted residual \n\n', myModel.features[1].conv)

    num_calibration_batches = 200 # 取部分训练集做校准
    evaluate(myModel, criterion, train_iter, neval_batches=num_calibration_batches)
    print('\nPTQ：校准完成！')

    convert(myModel, inplace=True)
    print('PTQ：转换完成！')
    print('\n 查看转换完成后的 inverted residual \n\n', myModel.features[1].conv)

    print_size_of_model(myModel)

    model_type = 'PTQ 模型'
    top1, top5 = evaluate(myModel, criterion, test_iter)
    print(f'\n{model_type}：\n\t在 {num_eval} 张图片上评估 accuracy 为: {top1.avg:2.2f}')
    jit.save(jit.script(myModel), saved_model_dir + scripted_ptq_model_file)

    # PTQ 模型： 模型大小：2.349283 MB
        # 在 10000 张图片上评估 accuracy 为: 49.04
 
def PTQ_per_channel():
    per_channel_quantized_model = create_model(quantize=False,
                                           num_classes=num_classes)
    per_channel_quantized_model = load_model(per_channel_quantized_model,
                                         saved_model_dir + float_model_file)
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    per_channel_quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')

    print('per_channel_quantized_model.qconfig', per_channel_quantized_model.qconfig)
    
    prepare(per_channel_quantized_model, inplace=True)
    num_calibration_batches = 200 # 仅仅取 200 个批次 # 取部分训练集做校准
    evaluate(per_channel_quantized_model, criterion, train_iter, num_calibration_batches)

    model_type = 'PTQ 模型（直方图观测器）'
    convert(per_channel_quantized_model, inplace=True)
    print_size_of_model(per_channel_quantized_model)

    top1, top5 = evaluate(per_channel_quantized_model, criterion, test_iter)
    print(f'\n{model_type}：\n\t在 {num_eval} 张图片上评估 accuracy 为: {top1.avg:2.2f}')
    jit.save(jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)

    # PTQ 模型（直方图观测器）： 模型大小：2.642769 MB
        # 在 10000 张图片上评估 accuracy 为: 67.16

def create_qat_model(num_classes,
                     model_path,
                     quantize=False,
                     backend='fbgemm'):
    qat_model = create_model(quantize=quantize,
                             num_classes=num_classes)
    qat_model = load_model(qat_model, model_path)
    qat_model.fuse_model()
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend=backend)
    return qat_model

def QAT_test():

    model_path = saved_model_dir + float_model_file
    qat_model = create_qat_model(num_classes, model_path)
    qat_model = prepare_qat(qat_model)

    print('\n 查看qat转换完成后的 inverted residual \n\n', qat_model.features[1].conv)

    train_fine_tuning(qat_model,
                     train_iter,
                     test_iter,
                     learning_rate=learning_rate,
                     num_epochs=10,
                     device='cpu',
                     param_group=True,
                     is_freeze=False,
                     is_quantized_acc=False,
                     need_qconfig=False,
                     ylim=[0.8, 1])

    
    convert(qat_model.cpu().eval(), inplace=True)
    qat_model.eval()
    print_info(qat_model,'QAT 模型')

    top1, top5 = evaluate(qat_model, criterion, test_iter)
    print(f'\nQAT模型：\n\t在 {num_eval} 张图片上评估 accuracy 为: {top1.avg:2.2f}')

    jit.save(jit.script(qat_model), saved_model_dir + scripted_qat_model_file)


def test():
    print('quanti_PTQ', torch.__version__)
    # 1、
    # load_data()
    
    # 2、
    # fine_tuning_model()
    # result: loss 0.011, train acc 0.997, test acc 0.950


    # 3、load_float_model
    # load_float_model()
    # 浮点模型：
        # 在 10000 张图片上评估 accuracy 为: 94.98000


    # 融合后的浮点模型：
        # 在 10000 张图片上评估 accuracy 为: 94.98

    # 4、PTQ_per_tensor
    # PTQ_per_tensor()

    # 5、PTQ_per_channel
    # PTQ_per_channel()


    # 6、QAT_test
    QAT_test()
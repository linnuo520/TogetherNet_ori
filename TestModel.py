from nets.darknet import CSPDarknet,CSPDarknet_SADC
from nets.yolo import YoloBody
import torch
import torch.nn as nn
# from utils.utils import print_network
# 创建 CSPDarknet 实例
from utils.utils import get_classes

dep_mul = 0.33  # 深度乘子
wid_mul = 0.5 # 宽度乘子
classes_path    = 'model_data/rtts_classes.txt'
class_names, num_classes = get_classes(classes_path)# No pretrained weights
phi='s'
# backbone = CSPDarknet(dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"))
backbone = CSPDarknet_SADC(dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"))
# model = YoloBody(num_classes,phi)
# print_network(backbone)
input_tensor = torch.randn(16, 3, 640, 640)
# 执行前向传播
outputs = backbone.forward(input_tensor)
# outputs = model.forward(input_tensor)
# print(outputs.__sizeof__())
# 打印输出特征图的尺寸
for key, value in outputs.items():
    print(f"{key}: {value.shape}")
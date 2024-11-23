import os
from datetime import datetime
class Config:
    def __init__(self,
                 exp_name,
                 unfreeze_epoch=100,
                 unfreeze_batch_size=4,
                 model_type = "ori", # ori 表示原始网络结构
                 init_lr=1e-2,
                 min_lr=None,
                 model_path='model_data/yolox_s.pth',
                 train_datasets='voc',
                 phi='s',
                 num_workers=4,
                 Parallel = True, #是否使用多卡训练，默认不使用
                 mosaic=False,
                 init_epoch=0,
                 freeze_epoch=0,
                 freeze_batch_size=16,
                 freeze_train=False,
                 optimizer_type="sgd",
                 momentum=0.937,
                 weight_decay=5e-4,
                 lr_decay_type="cos",
                 save_period=1,
                 classes_path = 'model_data/rtts_classes.txt'

                 ):

        # 训练数据集  实验名称
        self.train_datasets = train_datasets
        self.model_type = model_type
        self.exp_name = exp_name
        self.class_names, self.num_classes = self.get_classes(classes_path)

        self.num_workers = num_workers
        self.UnFreeze_Epoch = unfreeze_epoch
        self.Unfreeze_batch_size = unfreeze_batch_size
        self.Init_Epoch = init_epoch
        # 保存模型的周期
        self.save_period = save_period

        # 学习率和优化器
        self.Init_lr = init_lr
        self.Min_lr = min_lr if min_lr is not None else self.Init_lr * 0.01
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.weight_decay = weight_decay
        # 学习率衰减
        self.lr_decay_type = lr_decay_type

        # 可选的，预训练模型 以及 权重路径
        self.model_path = model_path
        self.phi = phi #默认为yolox s
        self.Parallel = Parallel

        # 其他训练参数
        self.mosaic = mosaic
        self.Freeze_Train = freeze_train
        self.Freeze_Epoch = freeze_epoch
        self.Freeze_batch_size = freeze_batch_size


    def get_classes(self, classes_path):
        # 读取类别文件并返回类别名称和数量（示例实现）
        with open(classes_path, 'r') as f:
            class_names = f.read().strip().split('\n')
        return class_names, len(class_names)

    def print_config(self):
        # 打印超参数配置
        print("超参数配置:")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

    def print_config_part(self):
        print("主要的超参数配置:")
        keys = ['exp_name','UnFreeze_Epoch','Unfreeze_batch_size','input_shape','train_datasets']
        for key in keys:
            if key in self.__dict__:
                print(f"{key}: {self.__dict__[key]}")


    def save_partial_config(self):
        # 定义日志目录
        log_dir = 'logs'
        # 定义实验文件夹路径
        experiment_dir = os.path.join(log_dir, self.exp_name)

        # 确保文件夹存在
        os.makedirs(experiment_dir, exist_ok=True)

        # 定义配置文件的完整路径
        config_file_path = os.path.join(experiment_dir, 'config.txt')
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 保存所有指定参数到文件
        with open(config_file_path, 'w', encoding='utf-8') as f:
            f.write(f"实验时间: {current_time}\n")
            f.write("实验参数配置:\n")
            # f.write(f"exp_name: {self.exp_name}\n")
            # f.write(f"unfreeze_epoch: {self.UnFreeze_Epoch}\n")
            # f.write(f"unfreeze_batch_size: {self.Unfreeze_batch_size}\n")
            # f.write(f"model_path: {self.model_path}\n")
            # f.write(f"train_datasets: {self.train_datasets}\n")
            # f.write(f"数据集类别: {self.class_names  }\n")
            # f.write(f"init_lr: {self.Init_lr}\n")
            # f.write(f"min_lr: {self.Min_lr}\n")
            # f.write(f"phi: {self.phi}\n")
            # f.write(f"num_workers: {self.num_workers}\n")
            # f.write(f"mosaic: {self.mosaic}\n")
            # f.write(f"init_epoch: {self.Init_Epoch}\n")
            # f.write(f"freeze_epoch: {self.Freeze_Epoch}\n")
            # f.write(f"freeze_batch_size: {self.Freeze_batch_size}\n")
            # f.write(f"freeze_train: {self.Freeze_Train}\n")
            # f.write(f"optimizer_type: {self.optimizer_type}\n")
            # f.write(f"momentum: {self.momentum}\n")
            # f.write(f"weight_decay: {self.weight_decay}\n")
            # f.write(f"lr_decay_type: {self.lr_decay_type}\n")
            # f.write(f"save_period: {self.save_period}\n")
            for key, value in vars(self).items():
                f.write(f"{key}: {value}\n")

        print(f"参数已保存到 {config_file_path}")


if __name__ == "__main__":
        config = Config(exp_name='orignal')
        config.print_config_part()

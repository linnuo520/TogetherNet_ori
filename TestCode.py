from torch.utils.data import DataLoader

from utils.dataloader import YoloDataset
from utils.utils import get_classes

if __name__ == "__main__":
    train_annotation_path = '2007_train_fog.txt'
    clear_annotation_path = '2007_train.txt'
    input_shape = [640, 640]
    mosaic = False
    num_workers = 4
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 16
    classes_path = 'model_data/rtts_classes.txt'
    # model_path      = 'model_data/yolox_s.pth'                 # Pretrained weights for better performance (COCO or VOC）
    model_path = ''
    class_names, num_classes = get_classes(classes_path)  # No pretrained weights
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(clear_annotation_path, encoding='utf-8') as f:
        clear_lines = f.readlines()

    train_dataset = YoloDataset(train_lines, clear_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch,
                                    mosaic=mosaic, train=True)
    data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=num_workers)
    for iteration, (images, boxes, clear_images) in enumerate(data_loader):
        print(f'Iteration {iteration + 1}')
        print('Images shape:', images.shape)  # 打印模糊图像的形状
        print('Clear Images shape:', clear_images.shape)  # 打印清晰图像的形状
        print('Boxes:', boxes)  # 打印边界框信息

        # 这里可以添加可视化代码，查看模糊图像和清晰图像
        # 例如使用 matplotlib 显示图像
        if iteration >= 5:  # 只测试前5个批次
            break
    # for images, boxes, clear_images in data_loader:
    #     print("Batch of images shape:", images.shape)  # 应该是 (batch_size, 3, 416, 416)
    #     print("Batch of boxes:", boxes)  # 每个样本的边界框信息
    #     print("Batch of clear images shape:", clear_images.shape)  # 应该是 (batch_size, 3, 416, 416)
    #     break  # 只测试一个批次
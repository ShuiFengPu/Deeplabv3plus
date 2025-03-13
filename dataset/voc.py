import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
# 在 VOC 类外部定义颜色映射（全局变量）
VOC_COLORMAP = [
    (0, 0, 0),        # 背景（黑色）
    (128, 0, 0),      # aeroplane（红色）
    (0, 128, 0),      # bicycle（绿色）
    (128, 128, 0),    # bird（黄绿色）
    (0, 0, 128),      # boat（蓝色）
    (128, 0, 128),    # bottle（紫色）
    (0, 128, 128),    # bus（青蓝色）
    (128, 128, 128),  # car（灰色）
    (64, 0, 0),       # cat（暗红色）
    (192, 0, 0),      # chair（亮红色）
    (64, 128, 0),     # cow（橄榄绿）
    (192, 128, 0),    # dining table（橙黄色）
    (64, 0, 128),     # dog（蓝紫色）
    (192, 0, 128),    # horse（粉紫色）
    (64, 128, 128),   # motorbike（蓝绿色）
    (192, 128, 128),  # person（淡蓝色）
    (0, 64, 0),       # potted plant（深绿色）
    (128, 64, 0),     # sheep（棕色）
    (0, 192, 0),      # sofa（亮绿色）
    (128, 192, 0),    # train（黄绿色）
    (0, 64, 128),     # tv/monitor（深蓝色）
]
class VOC(Dataset):
    def __init__(self,
                 root='data',
                 mode='train',
                 augment=True,#是否数据增强
                 transform=None):
        if mode == 'test':
            self.data_dir = os.path.join(root, 'VOC2012_test')#拼接文件路径
        else:
            self.data_dir = os.path.join(root, 'VOC2012_train_val')

        # 关键路径定义（适配截图结构）
        self.img_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.mask_dir = os.path.join(self.data_dir, 'SegmentationClass')
        self.split_file = os.path.join(self.data_dir, 'ImageSets', 'Segmentation', f'{mode}.txt')

        # 验证路径是否存在
        self._validate_paths()

        # 加载数据列表
        self.img_ids = self._load_split_file()

        # 预处理流程
        self.transform = self._build_pipeline(augment)

    def _validate_paths(self):
        """验证本地路径正确性"""
        required_dirs = [
            self.img_dir,
            self.mask_dir,
            os.path.dirname(self.split_file)
        ]
        for d in required_dirs:
            if not os.path.exists(d):
                raise FileNotFoundError(f"关键路径缺失: {d}")

    def _load_split_file(self):
        """加载本地划分文件"""
        with open(self.split_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _build_pipeline(self, augment):
        # 基础预处理流程（云服务器显存充足，提升输入分辨率至512x512）
        base_transforms = [
            T.Resize((512, 512)),
            # 将PIL图像转换为Tensor格式（通道维度变为第一维 CxHxW）
            T.ToTensor(),
            # 使用ImageNet标准归一化参数（与预训练模型兼容）
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        if augment:
            # 训练集增强流程（利用云服务器强大的GPU资源进行复杂增强）
            return T.Compose([
                # 随机水平翻转（50%概率触发，增加水平对称性数据的泛化性）
                T.RandomHorizontalFlip(p=0.5),
                # 随机垂直翻转（30%概率，适用于具有垂直对称性的场景如医学影像）
                T.RandomVerticalFlip(p=0.3),
                # 随机旋转（-15度到+15度之间，增强旋转鲁棒性）
                T.RandomRotation(degrees=15),
                # 随机仿射变换（平移10%图像尺寸，模拟视角变化）
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                # 颜色空间扰动（亮度、对比度、饱和度各30%幅度变化）
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                # 高斯模糊（30%概率应用3x3模糊核，提升抗噪能力）
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
                *base_transforms  # 合并基础预处理流程
            ])

        return T.Compose(base_transforms)

    def __getitem__(self, index):
        img_id = self.img_ids[index]

        # 加载数据（适配截图中的JPEGImages路径）
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png")

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # 应用预处理
        img = self.transform(img)
        mask = T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST)(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask

    def __len__(self):
        return len(self.img_ids)


def decode_target(target):
    """
    将单通道的类别索引标签（H, W）转换为 RGB 彩色图（H, W, 3）
    """
    # 如果是 PyTorch 张量，转换为 NumPy数组
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # 初始化 RGB 图像
    h, w = target.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # 遍历所有类别，将对应颜色赋给标签位置
    for class_id, color in enumerate(VOC_COLORMAP):
        rgb[target == class_id] = color

    return rgb


# 本地运行示例
if __name__ == "__main__":
    # 训练集示例
    train_set = VOC(mode='train')
    print(f"训练集样本数: {len(train_set)}")
    img, mask = train_set[0]
    print(f"图像尺寸: {img.shape}, 掩码尺寸: {mask.shape}")
    decoded_mask = train_set.decode_target(mask)
    # 验证路径是否正确（根据截图结构）
    print("测试第一个样本路径:")
    print("图片路径:", os.path.join(train_set.img_dir, f"{train_set.img_ids[0]}.jpg"))
    print("掩码路径:", os.path.join(train_set.mask_dir, f"{train_set.img_ids[0]}.png"))
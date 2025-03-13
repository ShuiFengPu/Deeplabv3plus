# 基础环境
Python 3.8+
PyTorch 1.10+
CUDA 11.3

# 安装依赖
pip install -r requirements.txt

# 数据集准备
下载PASCAL VOC 2012数据集至 data/VOC2012

# 运行预处理脚本：
python tools/preprocess_voc.py --data-path data/VOC2012

# 训练与推理
python train.py 

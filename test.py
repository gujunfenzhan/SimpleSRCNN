import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from model import SRCNN
from pylab import mpl

# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
transform_to_tensor = transforms.ToTensor()
transform_to_pil = transforms.ToPILImage(mode="YCbCr")
image_path = "image/illust_83516948_20230611_230308.jpg"
test_image = Image.open(image_path).convert("YCbCr")
model = torch.load('model/model0.0005458736550891897.pth')
y, cb, cr = transform_to_tensor(test_image).to(device)
y = y.unsqueeze(0)
with torch.no_grad():
    y = model(y).clamp(0.0, 1.0)

sr_image = transform_to_pil(torch.stack((y.squeeze(), cb, cr)))

# 创建包含两个子图的图形
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# 显示第一张图像
axes[0].imshow(test_image)
axes[0].axis('off')  # 不显示坐标轴
axes[0].set_title('原图')

# 显示第二张图像
axes[1].imshow(sr_image)
axes[1].axis('off')  # 不显示坐标轴
axes[1].set_title('超分辨')

# 调整子图布局
plt.tight_layout()

# 显示图形
plt.show()

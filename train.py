import argparse
import math
import random
import time
from pathlib import Path

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils import ImageUtils
import torch.cuda
from torch import nn
from Dataset import SRCNNDataset, SRCNNDataLoader
import model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="super_resolution", type=str, help="Name of the dataset to be used for "
                                                                            "training")
parser.add_argument('--lr', default=0.001, type=float, help="Learning rate for the optimizer")
parser.add_argument('--model', default='SimpleSRCNN', type=str, help="Model architecture to be used")
parser.add_argument('--patch_size', default=64, type=int, help="Width and height of image patches")
parser.add_argument('--patch_stride', default=28, type=int, help="Stride for extracting patches")
parser.add_argument('--num_epochs', default=50, type=int, help="Number of training epochs")
parser.add_argument('--num_res_blk', default=16, type=int, help="Number of residual blocks in the model")
parser.add_argument('--kernel_size', default=3, type=int, help="Size of convolution kernels")
parser.add_argument('--minimum_scale', default=0.2, type=float, help="Minimum scale factor for image definition")
parser.add_argument('--maximum_scale', default=0.4, type=float, help="Maximum scale factor for image definition")
parser.add_argument('--num_image', default=-1, type=int, help="Number of images to use for training, -1 for all")
parser.add_argument('--num_sample', default=64, type=int, help="Number of images to sample from the dataset in each "
                                                               "iteration")
parser.add_argument('--batch_size', default=256, type=int, help="Batch size for training patches extracted from "
                                                                "sampled images")
parser.add_argument('--no_bp_train', action='store_true', default=False, help='Disable breakpoint training')
parser.add_argument('--padding', default=1, type=int, help="Padding size for convolution layers")
parser.add_argument('--backup', action='store_true', default=False)

args = parser.parse_args()
# 遍历并输出所有参数及其值
for arg, value in args.__dict__.items():
    print(f"{arg}: {value}")

transform_to_tensor = transforms.ToTensor()

if torch.cuda.is_available():
    print("使用CUDA")
    device = torch.device("cuda")
else:
    print("使用CPU")
    device = torch.device("cpu")

train_path = Path("data/%s/train" % args.dataset)
val_path = Path("data/%s/val" % args.dataset)

best_loss = 999
if not args.no_bp_train and Path('model.pth').exists():
    model = torch.load('model.pth')
    print("继续训练")
else:
    class_ = getattr(model, args.model)
    model = class_(num_residual_blocks=args.num_res_blk, kernel_size=args.kernel_size, padding=args.padding).to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_image_paths = list(train_path.glob("*.png"))
if args.num_image > 0:
    train_image_paths = train_image_paths[0:args.num_image]
train_dataset = SRCNNDataset(train_image_paths, args.minimum_scale, args.maximum_scale, args.patch_size,
                             args.patch_stride)
val_dataset = SRCNNDataset(list(val_path.glob("*.png")), args.minimum_scale, args.maximum_scale, args.patch_size,
                           args.patch_stride)
train_dataloader = SRCNNDataLoader(train_dataset, batch_size=args.num_sample, shuffle=True, num_workers=0)
val_dataloader = SRCNNDataLoader(val_dataset, batch_size=args.num_sample, shuffle=True, num_workers=0)
try:
    for epoch in range(args.num_epochs):
        print("第%s轮训练" % (epoch + 1))
        outer_tq = tqdm(train_dataloader, desc="训练集")
        outer_total_loss = 0
        for sample_group in outer_tq:
            inner_tq = tqdm(range(0, len(sample_group), args.batch_size), leave=False, desc="训练集")
            total_loss = 0
            for batch_idx in inner_tq:
                x, y = zip(*sample_group[batch_idx:batch_idx + args.batch_size])
                x = torch.stack(x).to(device)
                y = torch.stack(y).to(device)
                y_predict = model(x)
                loss = criterion(y_predict, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                inner_tq.set_postfix(loss=loss.item())
                total_loss += loss.item()
            average_loss = total_loss / len(inner_tq)
            outer_total_loss += average_loss
            outer_tq.set_postfix(loss=average_loss)
            inner_tq.close()
            if args.backup:
                torch.save(model, "backup/model-backup-%s-%s.pth" % ((epoch+1),time.strftime("%Y-%m-%d-%H-%M-%S")))
        outer_tq.close()
        train_outer_average_loss = outer_total_loss / len(outer_tq)

        outer_tq = tqdm(val_dataloader, desc="测试集")
        outer_total_loss = 0
        for sample_group in outer_tq:
            inner_tq = tqdm(range(0, len(sample_group), args.batch_size), leave=False, desc="测试集")
            total_loss = 0
            for batch_idx in inner_tq:
                with torch.no_grad():
                    x, y = zip(*sample_group[batch_idx:batch_idx + args.batch_size])
                    x = torch.stack(x).to(device)
                    y = torch.stack(y).to(device)
                    y_predict = model(x)
                    loss = criterion(y_predict, y)
                    inner_tq.set_postfix(loss=loss.item())
                    total_loss += loss.item()
            average_loss = total_loss / len(inner_tq)
            outer_total_loss += average_loss
            outer_tq.set_postfix(loss=average_loss)
            inner_tq.close()
        outer_tq.close()
        val_outer_average_loss = outer_total_loss / len(outer_tq)
        print("训练集平均loss:%s\t测试集平均loss:%s" % (train_outer_average_loss, val_outer_average_loss))
        if val_outer_average_loss <= best_loss:
            best_loss = val_outer_average_loss
            torch.save(model, "model/model%s.pth" % val_outer_average_loss)
except KeyboardInterrupt as e:
    name = input("模型名字:")
    torch.save(model, "model/%s" % name)
    print("保存成功")
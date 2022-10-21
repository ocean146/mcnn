from config import get_args
from model import MCNN
from dataset import myDatasets
import torchvision
from torch.utils.data import DataLoader
import torch
from torch import nn
import time
from utils import get_mse_mae
import os

args = get_args()

if args.dataset == 'ShanghaiTechA':
    if os.name == 'nt':
        # for windows
        train_imgs_path = args.dataset_path + r'\train_data\images\\'
        train_labels_path = args.dataset_path+r'\train_data\ground-truth\\'
        test_imgs_path = args.dataset_path+r'\test_data\images\\'
        test_labels_path = args.dataset_path+r'\test_data\ground-truth\\'
    else:
        # for linux
        train_imgs_path = os.path.join(args.dataset_path,r'\train_data\images\\')
        train_labels_path = os.path.join(args.dataset_path,r'\train_data\ground-truth\\')
        test_imgs_path = os.path.join(args.dataset_path,r'\test_data\images\\')
        test_labels_path = os.path.join(args.dataset_path,r'\test_data\ground-truth\\')
    # print(F"{train_imgs_path=}\n{train_labels_path=}\n{test_imgs_path=}\n{test_labels_path=}")
else:
    raise Exception(F'Dataset {args.dataset} Not Implement')

# 数据集
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    # torchvision.transforms.Resize((768,1024))
])
train_datasets = myDatasets(train_imgs_path, train_labels_path,down_sample=True,transform=transform)
train_data_loader = DataLoader(train_datasets, batch_size=args.batch_size)
test_datasets = myDatasets(test_imgs_path, test_labels_path,down_sample=True,transform=transform)
test_data_loader = DataLoader(test_datasets, batch_size=args.batch_size)

model = MCNN().to(args.device)

if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.95)
else:
    raise Exception(F"optimizer {args.optimizer} not implemented")

# 损失函数
loss_fn = nn.MSELoss(reduction='sum').to(args.device)

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

best_mse = float('inf')
for epoch in range(args.epochs):
    if (epoch+1) % args.print_freq == 0:
        print(F"Epoch[{epoch+1}/{args.epochs}]: ",end='')
    model.train()
    start_time = time.time()
    train_loss = .0
    loss = None
    for imgs,targets in train_data_loader:
        imgs = imgs.to(args.device)
        targets = targets.to(args.device)
        outputs = model(imgs)
        outputs = outputs.squeeze(0)
        # print(F"{targets.shape=},{outputs.shape=}") # targets.shape=torch.Size([1, 192, 256]),outputs.shape=torch.Size([1, 1, 192, 256])
        # if loss is None:
        #     loss = loss_fn(outputs,targets)
        # else:
        #     loss += loss_fn(outputs,targets)
    # train_loss += loss.item()
        loss = loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % args.print_freq == 0:
        print(F"train_loss: {loss.item()} ",end='')

    # 测试
    test_mse = 0.0
    test_mae = 0.0
    targets_list = []
    outputs_list = []
    # 正确率
    total_accuracy = 0
    model.eval()
    with torch.no_grad():
        for imgs,targets in test_data_loader:
            imgs = imgs.to(args.device)
            targets = targets.to(args.device)[0,::,::].sum()
            outputs = model(imgs)[0,0,::,::].sum()
            targets_list.append(targets)
            outputs_list.append(outputs)
        test_mse,test_mae = get_mse_mae(outputs_list,targets_list)
        # test_mse += loss
    end_time = time.time()

    torch.save(model.state_dict(),args.save_path + "last.pth")
    if test_mse < best_mse:
        best_mse = test_mse
        torch.save(model.state_dict(),args.save_path + "best.pth")
    
    if (epoch+1) % args.print_freq == 0:
        print(F"test_mse: {test_mse},test_mae: {test_mae}, time: {end_time-start_time}, best_mse: {best_mse}")
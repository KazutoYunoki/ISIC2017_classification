
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import models

from ImageTransfom import ImageTransform
from Dataset import IsicDataset, make_datapath_list, create_dataloader
from Train_model import train_model

#訓練データと検証データのパス
train_list = make_datapath_list(phase = 'train')
val_list = make_datapath_list(phase= 'val')

#画像表示と確認
'''
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
img = train_list[100]
img = Image.open(img)
plt.imshow(img)
plt.show()

transform = ImageTransform(size, mean, std)
img_transformed = transform(img, phase = 'train')

img_transformed = img_transformed.numpy().transpose((1, 2, 0))
img_transformed = np.clip(img_transformed, 0, 1)
plt.imshow(img_transformed)
plt.show()
'''
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

#データセットの作成
train_dataset = IsicDataset(
    file_list = train_list,
    transform = ImageTransform(size, mean, std),
    phase = 'train')

val_dataset = IsicDataset(
    file_list = val_list,
    transform = ImageTransform(size, mean, std),
    phase = 'val')


#DataLoaderを作成
batch_size = 32

#辞書型'train'と'val'のバッチサイズandシャッフル
dataloader_dict = create_dataloader(
    batch_size = batch_size,
    train_dataset = train_dataset,
    val_dataset = val_dataset)

'''
#バッチごとの動作確認
batch_iterator = iter(dataloader_dict['val'])
inputs, labels = next(batch_iterator)
print(inputs.size())
print(labels)
'''
#ネットワークモデルのロード
use_pretrained = True
net = models.resnet50(pretrained = use_pretrained)

net.fc = nn.Linear(in_features = 2048, out_features = 2, bias = True)
net.train()

#損失関数の設定
criterion = nn.CrossEntropyLoss()

#調整するパラメータの設定
params_to_update = []
update_params_names = ["fc.weight", "fc.bias"]
for name, param in net.named_parameters():
    if name in update_params_names:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

#最適化手法の設定
optimizer = optim.SGD(params = params_to_update, lr = 0.001, momentum = 0.9)

#モデルの学習
num_epochs = 2
history = train_model(net, dataloader_dict, criterion, optimizer, num_epochs)

#学習履歴のグラフ化
plt.figure()
plt.plot(range(1, num_epochs + 1, 1), history['train_loss'], label = 'train_loss')
plt.plot(range(1, num_epochs + 1, 1), history['val_loss'], label = 'val_loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('loss.png')
    
plt.figure()
plt.plot(range(1, num_epochs + 1, 1), history['train_acc'], label = 'train_acc')
plt.plot(range(1, num_epochs + 1, 1), history['val_acc'], label = 'val_acc')
plt.xlabel('epoch')
plt.legend()
plt.savefig('acc.png')
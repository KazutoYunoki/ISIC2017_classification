#ISICのデータセットを作成
import torch.utils.data as data
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
import pandas as pd
import os.path as osp
#データへのpathリストを作成
def make_datapath_list(phase ='train'):

    #ISICのimage_id と　教師データのcsvファイル読み込み
    csv_data = pd.read_csv('./ISIC-2017_Training_Part3_GroundTruth (1).csv')
    
    #csvファイルからimage_idの列を取得
    image_id = []
    image_id = csv_data['image_id']

    #画像のroot_path
    root_path = './ISIC-2017_Training_Data/'

    path_list = []
    if(phase == 'train'):
        for i in range(0, 1200):
            #root_pathとimage_idをくっつくて画像へのパスを作成
            target_path = osp.join(root_path + image_id[i] +'.jpg')
            path_list.append(target_path)
    
    if(phase == 'val'):
        for i in range(1200, 2000):
            target_path = osp.join(root_path + image_id[i] + '.jpg')
            path_list.append(target_path)

    return path_list

train_path_list = make_datapath_list(phase='train')

if __name__ == "__main__":

    #画像へのパスがきちんと通っているかの確認
    print(train_path_list[0])
    print("画像の枚数: " + str(len(train_path_list)))
    pass


def create_dataloader(batch_size, train_dataset, val_dataset):

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = True)

    dataloader_dict = {'train' : train_dataloader, 'val' : val_dataloader}
    
    return dataloader_dict

class IsicDataset(data.Dataset):
    def __init__(self, file_list, transform = None, phase = 'train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)

        #画像のラベルを読み込み
        csv_data = pd.read_csv('./ISIC-2017_Training_Part3_GroundTruth (1).csv')
        label = []
        label = csv_data['melanoma']

        label = label.astype('int64')

        if self.phase == 'train':
            #index番目のラベルを取得
            label = label[index]
        elif self.phase == 'val':
            #検証データは1200～2000番目
            label = label[index + 1200]
        
        return img_transformed, label

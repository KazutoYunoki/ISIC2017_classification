#ISICのデータセットを作成
import torch.utils.data as data
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
import pandas as pd
import os.path as osp
from ImageTransfom import ImageTransform

#データへのpathリストを作成
def make_datapath_list(phase ='train'):

    #ISICのimage_id と教師データのcsvファイル読み込み
    training_data_CSV_file = pd.read_csv('./ISIC-2017_Training_Part3_GroundTruth (1).csv')
    val_data_CSV_file = pd.read_csv('./ISIC-2017_Validation_Part3_GroundTruth.csv')
    
    #csvファイルからimage_idの列を取得
    train_image_id = []
    val_image_id = [] 
    
    train_image_id = training_data_CSV_file['image_id']
    val_image_id = val_data_CSV_file['image_id']

    #画像のroot_path
    train_path = './ISIC-2017_Training_Data/'
    val_path = './ISIC-2017_Validation_Data/'

    path_list = []

    if(phase == 'train'):
        for i in range(len(train_image_id)):
            #root_pathとimage_idをくっつくて画像へのパスを作成
            target_path = osp.join(train_path + train_image_id[i] +'.jpg')
            path_list.append(target_path)
    
    if(phase == 'val'):
        for i in range(len(val_image_id)):
            target_path = osp.join(val_path + val_image_id[i] + '.jpg')
            path_list.append(target_path)

    return path_list



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

        label = []

        #画像のラベルを読み込み
        if self.phase == 'train':
            tain_data_csv_file = pd.read_csv('./ISIC-2017_Training_Part3_GroundTruth (1).csv')
            train_label = []
            #melanomaのラベルを取得
            train_label = train_data_csv_file['melanoma']
            #labelのデータ型をfloat→int64
            train_label = train_label.astype('int64')
            #index番目のラベルを取得
            label = train_label[index]

        elif self.phase == 'val':
            val_data_csv_file = pd.read_csv('./ISIC-2017_Validation_Part3_GroundTruth.csv')
            val_label = []
            val_label = val_data_csv_file['melanoma']
            val_label = val_label.astype('int64')
            label = val_label[index]
        
        return img_transformed, label

#動作確認
if __name__ == "__main__":

    train_list = make_datapath_list(phase='train')
    val_list = make_datapath_list(phase ='val')
    #画像へのパスがきちんと通っているかの確認
    print(train_list[0])
    print("訓練画像の枚数: " + str(len(train_list)))
    print(val_list[0])
    print("検証画像の枚数: " + str(len(val_list)))

    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    #データセットのサイズとラベルの確認
    train_dataset = IsicDataset(
    file_list = train_list,
    transform = ImageTransform(size, mean, std),
    phase = 'train')

    val_dataset = IsicDataset(
    file_list = val_list,
    transform = ImageTransform(size, mean, std),
    phase = 'val')

    index = 0
    print(val_dataset.__getitem__(index)[0].size())
    print(val_dataset.__getitem__(index)[1])


import pathlib
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data

import os.path as osp
from image_transform import ImageTransform


def make_datapath_list(csv_file, data_id, data_dir):
    """
    画像へのパスを作成するメソッド

    Parameters
    ----------
    csv : string
        読み込みたいcsvファイル
    dir : string
        読み込みたいデータファイル

    Returns
    -------
    path_list: list
        画像1枚1枚のパスが格納されたリスト
    """
    # 現在のディレクトリを取得
    current_dir = pathlib.Path(__file__).resolve().parent

    # ISICのimage_id と教師データのcsvファイル読み込み
    csv_file = pd.read_csv(current_dir / "data" / csv_file)

    # csvファイルからimage_idの列を取得
    image_id = csv_file[data_id]

    path_list = []

    for i in range(len(image_id)):
        # root_pathとimage_idをくっつくて画像へのパスを作成
        target_path = current_dir / "data" / data_dir / image_id[i]
        target_path = osp.join(str(target_path) + ".jpg")
        path_list.append(target_path)

    return path_list


def create_dataloader(batch_size, train_dataset, val_dataset):
    """
    データローダを作成する関数
    Parameters
    ----------
    bacth_size : int
        画像のバッチサイズ
    train_dataset :
        学習用のデータセット
    val_dataset :
        検証用のデータセット
    Returns
    -------
    dataloader_dict : dict
        学習用と検証用のデータローダ
    """
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    dataloader_dict = {"train": train_dataloader, "test": val_dataloader}

    return dataloader_dict


class IsicDataset(data.Dataset):
    def __init__(
        self, file_list, transform=None, phase="train", csv_file=None, label_name=None
    ):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.csv_file = csv_file
        self.label_name = label_name

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)

        label = []

        # 現在のディレクトリを取得
        current_dir = pathlib.Path(__file__).resolve().parent

        # csvファイルを読み込み
        csv_file = pd.read_csv(current_dir / "data" / self.csv_file)

        # melanomaのラベルを取得
        mel_label = csv_file[self.label_name]

        # labelのデータ型をfloat → int64
        mel_label = mel_label.astype("int64")

        # index番目のラベルを取得
        label = mel_label[index]

        return img_transformed, label


# 動作確認
if __name__ == "__main__":

    train_list = make_datapath_list(
        csv_file="ISIC-2017_Training_Part3_GroundTruth (1).csv",
        data_dir="ISIC-2017_Training_Data",
    )
    val_list = make_datapath_list(
        csv_file="ISIC-2017_Validation_Part3_GroundTruth.csv",
        data_dir="ISIC-2017_Validation_Data",
    )

    # 画像へのパスがきちんと通っているかの確認
    print(train_list[0])
    print("訓練画像の枚数: " + str(len(train_list)))
    print(val_list[0])
    print("検証画像の枚数: " + str(len(val_list)))

    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # データセットのサイズとラベルの確認
    train_dataset = IsicDataset(
        file_list=train_list,
        transform=ImageTransform(size, mean, std),
        phase="train",
        csv_file="ISIC-2017_Training_Part3_GroundTruth (1).csv",
    )

    val_dataset = IsicDataset(
        file_list=val_list,
        transform=ImageTransform(size, mean, std),
        phase="val",
        csv_file="ISIC-2017_Validation_Part3_GroundTruth.csv",
    )

    index = 0
    print(val_dataset.__getitem__(index)[0].size())
    print(val_dataset.__getitem__(index)[1])

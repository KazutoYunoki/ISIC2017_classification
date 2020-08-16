import pathlib
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.datasets as dset

import os
import os.path as osp
import shutil
from image_transform import ImageTransform
import torchvision.transforms as transforms


def make_split_data(ori_dir, csv_file):
    """
    フォルダを作成して、画像のファイルをラベルごとに分ける

    Parameters
    ---------
    ori_dir: string
        分けたい画像が入っている元のディレクトリ(./data直下に配置)
    """
    # 現在のディレクトリを取得
    current_dir = pathlib.Path(__file__).resolve().parent
    print(current_dir)

    # 3種類の空のデータフォルダを作成
    os.makedirs(str(current_dir) + "/data/" + ori_dir + "/0", exist_ok=True)
    os.makedirs(str(current_dir) + "/data/" + ori_dir + "/1", exist_ok=True)
    os.makedirs(str(current_dir) + "/data/" + ori_dir + "/2", exist_ok=True)

    # 画像データが入っているpath_list
    data_dir = make_datapath_list(csv_file, "image_id", ori_dir)

    # csvファイル読み込み
    re_csv = pd.read_csv(current_dir / "data" / csv_file)

    # 画像のラベルをリストして取得
    skin_label = re_csv["skin"]

    for i in range(len(data_dir)):
        # ラベルの値を取得
        label = skin_label[i]
        shutil.move(
            data_dir[i], str(current_dir) + "/data/" + ori_dir + "/" + str(label)
        )

    print(os.listdir("./data/" + ori_dir + "/0"))
    print(os.listdir("./data/" + ori_dir + "/1"))


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


def create_dataloader(batch_size, train_dataset, val_dataset, test_dataset):
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

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }

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

        # 画像のパス出力
        # print(img_path)

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


def make_trainset(dataroot, resize, mean, std):
    """
    ImageFolder関数を用いた学習用データの作成
    """
    current_dir = pathlib.Path(__file__).resolve().parent
    print(current_dir)

    # データセットの作成
    dataset = dset.ImageFolder(
        root=str(current_dir) + dataroot,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.CenterCrop(resize),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    )

    return dataset


def make_testset(dataroot, resize, mean, std):
    """
    ImageFolder関数を用いたテスト用データの作成
    """
    current_dir = pathlib.Path(__file__).resolve().parent
    print(current_dir)

    # データセットの作成
    dataset = dset.ImageFolder(
        root=str(current_dir) + dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    )

    return dataset


# 動作確認
if __name__ == "__main__":
    make_split_data(
        "ISIC-2017_Validation_Data", "ISIC-2017_Validation_Part3_GroundTruth.csv"
    )
    make_split_data(
        "ISIC-2017_Training_Data", "ISIC-2017_Training_Part3_GroundTruth (1).csv"
    )
    make_split_data("ISIC-2017_Test_v2_Data", "ISIC-2017_Test_v2_Part3_GroundTruth.csv")

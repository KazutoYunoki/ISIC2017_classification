import matplotlib.pyplot as plt
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import hydra
import logging

import seaborn as sns
from image_transform import ImageTransform
from Dataset import IsicDataset, make_datapath_list, create_dataloader
from model import train_model, test_model

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    # 訓練データと検証データのパス
    train_list = make_datapath_list(
        csv_file=cfg.csv.train, data_id=cfg.csv.id, data_dir=cfg.data.train_dir
    )
    val_list = make_datapath_list(
        csv_file=cfg.csv.val, data_id=cfg.csv.id, data_dir=cfg.data.val_dir
    )
    test_list = make_datapath_list(
        csv_file=cfg.csv.test, data_id=cfg.csv.id, data_dir=cfg.data.test_dir
    )

    # 画像表示と確認
    """
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img = train_list[100]
    img = Image.open(img)
    plt.imshow(img)
    plt.show()

    (size, mean, std)
    img_transformed = transform(img, phase = 'train')

    img_transformed = img_transformed.numpy().transpose((1, 2, 0))
    img_transformed = np.clip(img_transformed, 0, 1)
    plt.imshow(img_transformed)
    plt.show()
    """
    # データセットの作成
    train_dataset = IsicDataset(
        file_list=train_list,
        transform=ImageTransform(cfg.image.size, cfg.image.mean, cfg.image.std),
        phase="train",
        csv_file=cfg.csv.train,
        label_name=cfg.csv.label,
    )
    val_dataset = IsicDataset(
        file_list=val_list,
        transform=ImageTransform(cfg.image.size, cfg.image.mean, cfg.image.std),
        phase="val",
        csv_file=cfg.csv.val,
        label_name=cfg.csv.label,
    )

    test_dataset = IsicDataset(
        file_list=test_list,
        transform=ImageTransform(cfg.image.size, cfg.image.mean, cfg.image.std),
        phase="test",
        csv_file=cfg.csv.test,
        label_name=cfg.csv.label,
    )

    # 辞書型'train'と'val'と'test'のデータローダを作成
    dataloaders_dict = create_dataloader(
        batch_size=cfg.image.batch_size,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    """
    #バッチごとの動作確認
    batch_iterator = iter(dataloader_dict['val'])
    inputs, labels = next(batch_iterator)
    print(inputs.size())
    print(labels)
    """
    # ネットワークモデルのロード
    net = models.resnet50(pretrained=True)

    net.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    net.train()

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()

    # 調整するパラメータの設定
    params_to_update = []
    update_params_names = cfg.train.update_param_names

    for name, param in net.named_parameters():
        if name in update_params_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    # 最適化手法の設定
    optimizer = optim.SGD(
        params=params_to_update, lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum
    )

    # 学習回数を設定ファイルから読み込む
    num_epochs = cfg.train.num_epochs

    # GPU初期設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス: ", device)

    # ネットワークをGPUへ
    net.to(device)
    torch.backends.cudnn.benchmark = True

    # 損失値と認識率を保持するリスト
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    """
    # 学習と検証
    for epoch in range(num_epochs):

        log.info("Epoch {} / {}".format(epoch + 1, num_epochs))
        log.info("----------")

        # 学習
        train_history = train_model(
            net, dataloaders_dict["train"], criterion, optimizer
        )

        # 学習したlossと認識率のリストを作成
        train_loss.append(train_history["train_loss"])
        train_acc.append(train_history["train_acc"])

        # 検証
        test_history = test_model(net, dataloaders_dict["val"], criterion)

        # 検証したlossと認識率のリストを作成
        test_loss.append(test_history["test_loss"])
        test_acc.append(test_history["test_acc"])

    # figインスタンスとaxインスタンスを作成
    fig_loss, ax_loss = plt.subplots(figsize=(10, 10))
    ax_loss.plot(range(1, num_epochs + 1, 1), train_loss, label="train_loss")
    ax_loss.plot(range(1, num_epochs + 1, 1), test_loss, label="test_loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    fig_loss.savefig("loss.png")

    fig_acc, ax_acc = plt.subplots(figsize=(10, 10))
    ax_acc.plot(r
    ange(1, num_epochs + 1, 1), train_acc, label="train_acc")
    ax_acc.plot(range(1, num_epochs + 1, 1), test_acc, label="test_acc")
    ax_acc.legend()
    ax_acc.set_xlabel("epoch")
    fig_acc.savefig("acc.png")
    

    # パラメータの保存
    current_dir = pathlib.Path(__file__).resolve().parent
    save_path = current_dir / "weights_fine_tuning.pth"
    torch.save(net.state_dict(), save_path)
    """

    # Pytorchのネットワークパラメータのロード
    # 現在のディレクトリを取得
    current_dir = pathlib.Path(__file__).resolve().parent
    print(current_dir)

    load_path = str(current_dir) + "/weights_fine_tuning.pth"
    load_weights = torch.load(load_path, map_location="cpu")
    net.load_state_dict(load_weights)

    evaluate_history = test_model(net, dataloaders_dict["test"], criterion)
    print(evaluate_history["confusion_matrix"])

    # 混同行列の作成と表示
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(evaluate_history["confusion_matlix"], annot=True, fmt="d", cmap="Blues")
    ax.set_title("confusion_matrix")
    fig.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()

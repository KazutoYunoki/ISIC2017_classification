import torch
import torch.nn as nn
from model import calculate_efficiency
from torchvision import models
import pathlib
import seaborn as sns

from model import evaluate_model
from Dataset import make_testset
import matplotlib.pyplot as plt


def test_model():

    # データセットを作成
    test_dataset = make_testset(
        dataroot="/data/skin_data/generate_skin_data",
        resize=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    net = models.vgg16_bn(pretrained=True)

    # Pytorchのネットワークパラメータのロード
    # 現在のディレクトリを取得
    current_dir = pathlib.Path(__file__).resolve().parent
    print(current_dir)
    # 学習済みのパラメータを使用したいとき
    load_path = str(current_dir) + "/weights_fine_tuning.pth"
    load_weights = torch.load(load_path)
    net.load_state_dict(load_weights)

    criterion = nn.BCELoss()

    evaluate_history = evaluate_model(net, test_dataloader, criterion, thershold=0.6)
    print(evaluate_history["confusion_matrix"])

    # 性能評価指標の計算（正解率、適合率、再現率、F1値)
    efficienct = calculate_efficiency(evaluate_history["confusion_matrix"])

    print("正解率: " + str(efficienct["accuracy"]))
    print("適合率: " + str(efficienct["precision"]))
    print("再現率: " + str(efficienct["recall"]))
    print("f1値 :" + str(efficienct["f1"]))

    # 混同行列の作成と表示
    fig_conf, ax_conf = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        evaluate_history["confusion_matrix"], annot=True, fmt="d", cmap="Reds",
    )
    ax_conf.set_title("confusion_matrix")
    ax_conf.set_xlabel("Predicted label")
    ax_conf.set_ylabel("True label")
    fig_conf.savefig("confusion_matrix.png")


if __name__ == "__main__":
    test_model()

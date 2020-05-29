from tqdm import tqdm
import torch
import logging

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score

log = logging.getLogger(__name__)


def train_model(net, train_dataloader, criterion, optimizer):
    """
    学習させるための関数

    Parameters
    ----------
    net :
        ネットワークモデル

    train_dataloader :
        学習用のデータローダ
    criterion :
        損失関数
    optimizer :
        最適化手法

    Returns
    -------
    epoch_loss : double
        エポックあたりのloss値
    epoch_acc : double
        エポック当たりの認識率
    """
    # GPU初期設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.train()

    epoch_loss = 0.0
    epoch_corrects = 0

    for inputs, labels in tqdm(train_dataloader, leave=False):

        # GPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        epoch_corrects += torch.sum(preds == labels.data)

    epoch_loss = epoch_loss / len(train_dataloader.dataset)
    epoch_acc = epoch_corrects.double() / len(train_dataloader.dataset)

    log.info("Train Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))

    return {"train_loss": epoch_loss, "train_acc": epoch_acc}


def test_model(net, test_dataloader, criterion):
    """
    モデルで検証させる関数

    Parameters
    ----------
    net :
        ネットワークモデル
    test_dataloader :
        テスト用のデータローダ
    criterion :
        損失関数

    Returns
    -------
    epoch_loss :
        エポックあたりの損失値
    epoch_acc :
        エポックあたりの認識率
    """
    # GPU初期設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 評価モードへ変更
    net.eval()

    epoch_loss = 0.0
    epoch_corrects = 0

    for inputs, labels in tqdm(test_dataloader, leave=False):
        # GPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        epoch_loss += loss.item() * inputs.size(0)
        epoch_corrects += torch.sum(preds == labels.data)

    epoch_loss = epoch_loss / len(test_dataloader.dataset)
    epoch_acc = epoch_corrects.double() / len(test_dataloader.dataset)

    log.info("Test Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))

    return {"test_loss": epoch_loss, "test_acc": epoch_acc}


def evaluate_model(net, test_dataloader, criterion):

    # GPU初期設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 評価モードへ変更
    net.eval()

    epoch_loss = 0.0
    epoch_corrects = 0

    # 予測ラベルと正解ラベルのリスト
    predlist = torch.zeros(0, dtype=torch.long, device=device)
    truelist = torch.zeros(0, dtype=torch.long, device=device)

    for inputs, labels in tqdm(test_dataloader, leave=False):
        # GPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        # confusion_matrixの作成
        predlist = torch.cat([predlist, preds.view(-1).cuda()])
        truelist = torch.cat([truelist, labels.view(-1).cuda()])
        cm = confusion_matrix(
            truelist.cpu().numpy(), predlist.cpu().numpy(), labels=[0, 1]
        )

        epoch_loss += loss.item() * inputs.size(0)
        epoch_corrects += torch.sum(preds == labels.data)

    epoch_loss = epoch_loss / len(test_dataloader.dataset)
    epoch_acc = epoch_corrects.double() / len(test_dataloader.dataset)

    log.info("Test Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))

    return {
        "test_loss": epoch_loss,
        "test_acc": epoch_acc,
        "confusion_matrix": cm,
    }


def calculate_efficiency(cm):

    # confusion_matrixを1次元化 tn=483 fp=0 , fn=117 , tp=0
    tn, fp, fn, tp = cm.flatten()

    # 全てのサンプルのうち正解したサンプルの割合
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 適合率の計算　1と判定されたものうち正解した割合
    # 偽陽性(実際は陰性だけど予測が陽性)を避けたい場合に用いる指標
    precision = tp / (tp + fp)

    # 再現率の計算　1とついてるラベルの中で正解した割合
    # 偽陰性(実際は陽性だけど予測が陰性)を避けたい場合に用いる
    recall = tp / (tp + fn)

    # f1値の計算　適合率と再現率の調和平均
    f1 = (2 * tp) / (2 * tp + fp + fn)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

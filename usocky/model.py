from tqdm import tqdm
import torch
import logging
from sklearn.metrics import confusion_matrix

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

        # BCELossを使用する場合labelsを(batch_size, 1)に変更(dtype = float32)
        # labels = labels.view(-1, 1)
        # labels = labels.float()

        # GPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = net(inputs)
            # sigmoidの出力結果を保存
            # log.info(torch.sigmoid(outputs).flatten())
            # outputs = torch.sigmoid(outputs)

            # log.info("訓練データの出力値\n" + str(torch.softmax(outputs, 1)))

            loss = criterion(outputs, labels)

            # outfeatures = 2以上　↓
            _, preds = torch.max(outputs, 1)

            # 予測ラベルの閾値処理　閾値以上なら1、以下なら0
            # preds = (outputs > 0.5).long()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)

        epoch_corrects += torch.sum(preds == labels.data)
        # 　↓BCE使うとき
        #epoch_corrects += torch.sum(preds.long() == labels.data.long())

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

        # BCELossを使用する場合labelsを(batch_size, 1)に変更(dtype = float32)
        # labels = labels.view(-1, 1)
        # labels = labels.float()

        # GPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            # outputs = torch.sigmoid(outputs)
            # log.info("検証データの出力値\n" + str(torch.softmax(outputs, 1)))

            loss = criterion(outputs, labels)

            # outfeatures = 2 以上　↓
            _, preds = torch.max(outputs, 1)

            # 予測ラベルの閾値処理　閾値以上なら1、以下なら0
            # preds = (outputs > 0.5).long()
        epoch_loss += loss.item() * inputs.size(0)
        epoch_corrects += torch.sum(preds == labels.data)
        # 　↓BCE使うとき
        # epoch_corrects += torch.sum(preds.long() == labels.data.long())

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

        # BCELossを使用する場合labelsを(batch_size, 1)に変更(dtype = float32)
        # labels = labels.view(-1, 1)
        # labels = labels.float()

        # GPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

            log.info("softmaxの出力\n" + str(torch.softmax(outputs, 1)))

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            print(preds.shape)
            print(labels.shape)
            print(preds)
            print(labels)
            # 予測ラベルの閾値処理　閾値以上なら1、以下なら0
            # preds = (outputs > 0.5).long()

        # confusion_matrixの作成
        predlist = torch.cat([predlist, preds.long().view(-1).cuda()])
        truelist = torch.cat([truelist, labels.long().view(-1).cuda()])
        cm = confusion_matrix(
            truelist.cpu().numpy(), predlist.cpu().numpy(), labels=[0, 1]
        )

        epoch_loss += loss.item() * inputs.size(0)
        log.info("予測ラベル\n" + str(preds.flatten()))
        log.info("実際のラベル\n" + str(labels.data.flatten()))
        # 　↓BCE使うとき
        epoch_corrects += torch.sum(preds.long() == labels.data.long())

        # epoch_corrects += torch.sum(preds == labels.data)

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

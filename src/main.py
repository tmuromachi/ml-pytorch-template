import torch
import torch.nn as nn
import torch.optim as optim

import sys
import yaml
import models
import dataset
from evaluate import evaluate
from util.plot import plot
from util.util import fix_seed
from train import train_model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from tqdm.notebook import tqdm  # jupyter で実行するとき
# sys.path.append('src')    # 自作モジュール探索用パス

if __name__ == '__main__':
    # リソースの選択（CPU/GPU）
    args = sys.argv
    cuda_number = "cuda:" + args[1]
    device = torch.device(cuda_number if torch.cuda.is_available() else "cpu")

    # YAMLから設定ファイル読み込み
    with open('config.yaml') as file:
        config_file = yaml.safe_load(file)

    print("==CONFIG==")
    print(config_file)
    print("==========")
    # MNISTデータ(run.shから実行する際は相対パスが変わるため注意)
    TRAIN_DATA_PATH = config_file['config']['train_path']
    TEST_DATA_PATH = config_file['config']['test_path']
    PATH_SAVED_MODEL = config_file['config']['model_path']    # 学習済みモデルの保存・ロード
    CLASSES = config_file['config']['classes']    # クラス数
    EPOCH = config_file['config']['epoch']    # エポック数
    BATCH_SIZE = config_file['config']['batch_size']    # バッチサイズ

    seed = 42
    fix_seed(seed)

    # Data preprocessing------
    train_loader, test_loader = dataset.create_dataset(TRAIN_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE)

    # Modeling------
    # モデル・損失関数・最適化アルゴリスムの設定
    model = models.Net(CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 訓練の実行
    train_loss = []
    test_loss = []
    for epoch in range(EPOCH):
        model, train_l, test_l = train_model(model, train_loader, test_loader, optimizer, criterion, device)
        train_loss.append(train_l)
        test_loss.append(test_l)
        print("epoch:", epoch + 1, "/", EPOCH)
        print("train loss:", train_l)
        print("test loss:", test_l)

    # 学習状況（ロス）の確認
    plot(train_loss, test_loss)

    # Evaluation ------
    # 予測結果と正解値を取得
    preds, labels = evaluate(model, test_loader, device)
    print(preds)

    # 評価
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")  # 多クラス分類なのでaverageを指定
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")

    print("----------")
    print("accuracy", accuracy)
    print("precision", precision)
    print("recall", recall)
    print("F1", f1)
    print("----------")

    # Other ----------------------------------------------------------------------

    # モデルの保存
    torch.save(model.state_dict(), PATH_SAVED_MODEL)
    # モデルのロード
    model = models.Net(CLASSES)
    model.load_state_dict(torch.load(PATH_SAVED_MODEL))

    print("[finish]")

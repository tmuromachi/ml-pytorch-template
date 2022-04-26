import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
import torchvision.transforms as transforms

# import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
# from tqdm import tqdm  # コマンドラインで実行するとき
# from tqdm.notebook import tqdm  # jupyter で実行するとき

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# リソースの選択（CPU/GPU）
# args = sys.argv
# cuda_number = "cuda:" + args[1]
# device = torch.device(cuda_number if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# YAMLから設定ファイル読み込み
# with open('config.yaml') as file:
#     config_file = yaml.safe_load(file)

# MNISTデータ(run.shから実行する際は相対パスが変わるため注意)
TRAIN_DATA_PATH = "../data/mnist/train/"
TEST_DATA_PATH = "../data/mnist/test/"
# 学習済みモデルの保存・ロード
PATH_SAVED_MODEL = "./model"
# クラス数
CLASSES = 10
# エポック数
EPOCH = 5
# バッチサイズ
BATCH_SIZE = 64


# 乱数シード固定（再現性の担保）
def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
fix_seed(seed)


# データローダーのサブプロセスの乱数のseedが固定
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Data preprocessing ----------------------------------------------------------

# データセットの作成
class Mydataset(torch.utils.data.Dataset):
    def __init__(self, X, y, dir_path):
        self.X = X
        self.y = y

        self.transform = transforms.Compose([
            # transforms.Resize(self.X, self.Y), # 画像のリサイズ
            transforms.ToTensor(),  # Tensor化
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 標準化
        ])

        # ここに入力データとラベルを入れる
        self.image_paths = [str(p) for p in pathlib.Path(dir_path).glob("**/*.jpg")]

        # self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]    # MNISTのクラス数を定義
        print("INPUT DATA NUM:", len(self.image_paths))

    def __len__(self):
        # データセットの大きさを返すメソッド
        return len(self.image_paths)

    def __getitem__(self, index):
        # 学習時にデータとラベルをタプルで返すメソッド
        # 学習前に全て読み込むのではなく、学習の都度読み込む
        p = self.image_paths[index]
        path = pathlib.Path(p)
        image = Image.open(path)
        image = image.convert("RGB")  # MNISTの場合1チャンネルなので3チャンネルに修正

        # 読み込みデータに対してコンストラクタで決めた前処理を行う
        if self.transform:
            out_data = self.transform(image)

        out_label = int(str(path.parent.name))  # 親ディレクトリのパスを取得し、ラベルとする
        return out_data, out_label


train_dataset = Mydataset(28, 28, TRAIN_DATA_PATH)
test_dataset = Mydataset(28, 28, TEST_DATA_PATH)

# データローダーの作成
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,  # バッチサイズ
                                           shuffle=True,  # データシャッフル
                                           num_workers=2,  # 高速化
                                           pin_memory=True,  # 高速化
                                           worker_init_fn=worker_init_fn
                                           )
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=2,
                                          pin_memory=True,
                                          worker_init_fn=worker_init_fn
                                          )


# Modeling --------------------------------------------------------------------

# モデルの定義
class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU())
        self.conv2 = torch.nn.Sequential(nn.Conv2d(16, 64, 3, 2, 1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU())

        self.fc1 = nn.Linear(3136, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(100, CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# モデル・損失関数・最適化アルゴリスムの設定
model = Mymodel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# モデル訓練関数
def train_model(model, train_loader, test_loader):
    # Train loop ----------------------------
    model.train()  # 学習モードをオン
    train_batch_loss = []
    for data, label in train_loader:
        # GPUへの転送
        data, label = data.to(device), label.to(device)
        # 1. 勾配リセット
        optimizer.zero_grad()
        # 2. 推論
        output = model(data)
        # 3. 誤差計算
        loss = criterion(output, label)
        # 4. 誤差逆伝播
        loss.backward()
        # 5. パラメータ更新
        optimizer.step()
        # train_lossの取得
        train_batch_loss.append(loss.item())

    # Test(val) loop ----------------------------
    model.eval()  # 学習モードをオフ
    test_batch_loss = []
    with torch.no_grad():  # 勾配を計算なし
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            test_batch_loss.append(loss.item())

    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)


# 訓練の実行
train_loss = []
test_loss = []

for epoch in range(EPOCH):
    model, train_l, test_l = train_model(model, train_loader, test_loader)
    train_loss.append(train_l)
    test_loss.append(test_l)
    print("epoch:", epoch, "/", EPOCH)
    print("train loss:", train_l)
    print("test loss:", test_l)

# 学習状況（ロス）の確認
plt.title("loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(train_loss, label='train_loss')
plt.plot(test_loss, label='test_loss')
plt.savefig('loss.png')


# Evaluation ----------------------------------------------------------------

# 学習済みモデルから予測結果と正解値を取得
def retrieve_result(model, dataloader):
    model.eval()
    preds = []
    labels = []
    # Retreive prediction and labels
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            # Collect data
            preds.append(output)
            labels.append(label)
    # Flatten
    preds = torch.cat(preds, axis=0)
    labels = torch.cat(labels, axis=0)
    _, preds = torch.max(preds, 1)  # 予測結果の中で最大確率のindexを取得する
    # Returns as numpy (CPU環境の場合は不要)
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return preds, labels


# 予測結果と正解値を取得
preds, labels = retrieve_result(model, test_loader)
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
model = Mymodel()
model.load_state_dict(torch.load(PATH_SAVED_MODEL))

print("[finish]")

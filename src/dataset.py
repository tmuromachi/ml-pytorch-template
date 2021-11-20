import torch
import torchvision.transforms as transforms
from PIL import Image
import pathlib

from util.util import worker_init_fn


# データセットの作成
class Dataset(torch.utils.data.Dataset):
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


def create_dataset(train_path, test_path, batch_size):
    train_dataset = Dataset(28, 28, train_path)
    test_dataset = Dataset(28, 28, test_path)

    # データローダーの作成
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,  # バッチサイズ
                                               shuffle=True,  # データシャッフル
                                               num_workers=2,  # 高速化
                                               pin_memory=True,  # 高速化
                                               worker_init_fn=worker_init_fn
                                               )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True,
                                              worker_init_fn=worker_init_fn
                                              )
    return train_loader, test_loader

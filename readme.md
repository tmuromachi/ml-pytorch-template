# PyTorch Template
### ゼロから始めるPyTorch DeepLearning 環境構築から実験管理まで

Qiitaに解説を掲載しています。
https://qiita.com/ToshikiMuromachi/items/ec1233e50b8069a22e19

初めてPythonを触る人のための全てがまとまっているコード：`./template/main.py`  
実際に使用することを想定しモジュール化されたコード：`./src/`  

### 実行方法
#### 1. MNISTデータの解凍
`unzip data/mnist.zip -d ./data/`

#### 2. PyTorchサンプルの実行

**・シェルスクリプトから実行する場合**  
`source run.sh`を実行することで対話形式で学習を開始できます  
※ CUDAデバイス選択時に0番を選択するとCPUで実行されます  

**・直接pythonスクリプトを実行する場合**  
`python ./src/main.py 1`  
※ pythonスクリプトの引数はGPU番号です(0はCPU)  
※ 学習後のモデルとloss画像がml-pytorch-template直下に生成されます
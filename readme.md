# PyTorch Template
### ゼロから始めるPyTorch DeepLearning 環境構築から実験管理まで

Qiita版の解説が最新です。このReadmeの内容は古くなっている可能性があります。
https://qiita.com/ToshikiMuromachi/items/ec1233e50b8069a22e19

初めてPythonを触る人のためのコード：`./template/main.py`  
実際に使用することを想定しモジュール化されたコード：`./src/`  

---
「できる限り変わったことをしない」「環境を汚さない」「なるべく手順を減らす」「Linux上でなるべく完結する」ことを目標とした初学者向け環境構築手法です。  
python仮想環境、vscodeによるWSL2上での開発、簡単な実験管理を紹介します。

個人的にはWindowsではWSLg+PyCharmが最強の開発環境になるという考えでしたが、
現状高dpiの対応等に難ありで当分解決されそうにないため、PyCharmを使うことを諦めます。  
(PyCharmの方が仮想環境やGit関係まで面倒見は良いです。
vscodeではRemote SSHだとGit追跡が遅かったり、変更箇所がハイライトされないなど、特にGit周りは弱く感じます。)  

この手順の他の開発環境は以下のものなどがあると思います。 
```
- VNC上でPyCharmを使用し、開発を行う(PyCharmの動作が不安定だとよく言われています)
- 手元のVM上でPyCharmを動かし、Gitでサーバーにコードを送る(自分はこれが一番好きです)
- 手元のmac/Linux上でPyCharmを動かし、Gitでサーバーにコードを送る(これも良いです)
- WindowsでPyCharmをうごかし、Gitでサーバーにコードを送る(WindowsでPython環境構築は辛いです)
- WindowsのPyCharmをWSLインタプリタを使用して動かす(WSL上の仮想環境が使えません。またGit周りでも難ありです)
- WSL2上でPyCharmを動かし、xrdpやVNCを使って画面を出す(日本語化等も含めると手順が多すぎて万人におすすめできません。)
```

---
## 特徴
- WindowsでWSL2+VScode+venv+RemoteWSL環境を構築します。
- シェルスクリプトでGPUの選択等を対話形式で行い、プログラムを実行します。
- 実行時の時間やコメントを反映したディレクトリを作成し、ログの保存・実験管理を行います。 
- 対象読者としてはLinuxコマンドが少しわかっている程度の人(情報系の学部1~2年生あたり)を想定しています。

---
## データ
データはMNIST(手書き文字データセット)を想定します。  
http://yann.lecun.com/exdb/mnist/   
60,000例のトレーニングセットと10,000例のテストセットで構成されています。  
各例は、10クラスのラベルに関連付けられた28x28のグレースケール画像です。  
わざわざ手動ダウンロードせずともPyTorchならば使えますが、
他のデータセットでも流用できるように画像データを./data/mnist.zipに用意してあるので、後で解凍して使用します。  
(vscodeではプロジェクト内に大規模データがあると.gitignoreに入れていても警告が出るため、本当はデータはプロジェクト外に置いたほうがよさそうです)

---
## VSCode Remote WSL環境を作る
今後詳細を記述するかもしれません。  
活発に開発されているので、Qiita等の記事だと情報が古い場合があります。なるべくMicrosftのドキュメントを参考に環境を作ってください。  

1. **WSL2をインストールする**   
WSL のインストール  
https://docs.microsoft.com/ja-jp/windows/wsl/install

2. **WSL2とvscodeを連携させる**  
Linux 用 Windows サブシステムで Visual Studio Code の使用を開始する  
https://docs.microsoft.com/ja-jp/windows/wsl/tutorials/wsl-vscode

---
## サンプルをGitからCloneする 
サンプルを手元のPC(WSL2の中)にダウンロードしたいので、Gitからcloneします。  
カレントディレクトリにダウンロードされるため、格納先のディレクトリを作成して移動しておいてください。
自分は~/vscodeというディレクトリを作っています。
```
# 新しくディレクトリを作る場合
$ cd
$ mkdir vscode 

# gitからプロジェクトをcloneする
$ git clone https://gitlab.com/tmuromachi/ml-pytorch-template.git
```


---

## Python環境構築 
Linux上のPythonはシステムが使用しているため、基本的にはそのPythonを使いません。 
仮想環境やDockerを使うべきです。仮想環境は様々なものがありますが、本稿ではvenvで仮想環境の作成を行います。  
venvはPythonに標準で入っており、仮想環境はvenvが最も基本的だと考えています。   
Pythonは3系であればバージョントラブルはあまりありません。どちらかというとバージョンを上げすぎると対応していないパッケージが出てくるので
少し古めのほうがよいです。そういった事情も踏まえてQiita等でよく紹介されているpyenv+venv環境は基本的には不要だと考えています。  
venvで仮想環境を作る際に違うバージョンのpythonを使えばpyenvなしでも複数バージョンのPythonを使用できます。

python.jpの仮想環境構築手順を参考にしています。   
https://www.python.jp/install/ubuntu/virtualenv.html

1. **pyenvによるビルド**  
pyenvのpython-buildを利用してPythonソースコードのダウンロードからインストールまで行います。
   ```
   Python3.8インストール例：
   $ git clone git://github.com/pyenv/pyenv.git  
   $ pyenv/plugins/python-build/bin/python-build 3.8.3 ~/python3.8
   ```

2. **仮想環境の作成をする**  
インストールしたpythonのバージョンを指定した仮想環境の作成をします。まず、このプロジェクトをクローンしたディレクトリに移動してください。  
次に、手順1で作成したディレクトリの下にbin/python3があるため、それを指定して仮想環境を作ります。  
以下のコマンドの場合だと、`.venv`というディレクトリに仮想環境が作られます。  
`$ ~/python3.8/bin/python3 -m venv .venv`  


3. **仮想環境の有効化**  
仮想環境の作成が完了した後に仮想環境をアクティブにします。  
`$ . .venv/bin/activate`  
ターミナルのプロンプトの前に(.venv)という表示が出たら成功です。  
`$ python3`と入力するとPythonのバージョンも確認できます。  
仮想環境を終了したい場合には`$ deactivate`と入力することで終了します。

---

## PyTorchサンプルを動かす
今回動かすサンプルは単純な3層CNNを使用したものです。モデル部分は参考資料のものを流用したため、誤りがある可能性があります。
(既に数箇所間違いを見つけて訂正しています)  
コードは./src以下に格納しています。今回は実行時に実験管理まで行いたいので、シェルスクリプト経由でmain.pyを実行します。


1. **必要なパッケージのインストール**  
サンプルを動かすために必要なパッケージ等(PyTotrch, numpy等)をインストールします。  
必要なパッケージの一覧は`requirements.txt`に記載されており、Pythonのパッケージを管理システムであるpipにrequirements.txtを渡すことでインストールできます。  
仮想環境を作成したばかりだと、pipのバージョンが低い場合があるので必要なパッケージをインストールする前に仮想環境を有効化した状態で、pipのアップデートを行います。    
`$ pip install --upgrade pip`  
アップデート後に以下のコマンドを実行することでrequirements.txtに記載されたパッケージがインストールされます。  
`$ pip install -r requirements.txt`

2. **データ準備**  
加工済みMNISTデータを解凍します。  
`cd ml-pytorch` (プロジェクトのルートディレクトリに移動します)  
`unzip data/mnist.zip -d ./data/` (解凍します)

3. **学習の設定**  
データセットのパスやエポック数やバッチサイズ等はconfig.yamlで設定してあります  
変更したい場合はconfig.yamlを書き換えると変更ができます。

4. **実行**  
PyTorchサンプル実行します。  
`source run.sh`というコマンドをプロジェクトルートで実行すると、まずCUDAデバイス番号を聞いてきます。  
`nvtop`等を使用して、空いているGPUを確認し、番号を入力しましょう。
機械学習用やゲーム用のGPUがない場合は番号入力時にエンターを押してスキップしてください。
スキップした場合は0番のデバイスか、CPUで計算します。  
数字を入力してエンターすると、実験時のコメントを入力できます。必要なければそのままエンターを押してください。  
コメント入力まで終わるとサンプルが実行されます。

5. **結果(実験管理)**  
実行が終了すると実行結果は./log/[実行時刻]ディレクトリ内に保存されます。

###  注意事項等
- `./template/main.py`がPyTorchサンプルコード全体となっています。   
モジュール化しておらず、シェルスクリプトも不要なため、こちらの方がわかりやすいです。
Pythonでは自作モジュールをインポートする際にはパスの設定が難しいため注意してください。


- 今回はTrain/Testしか作っていません。注意してください。  


- 実行用シェルスクリプトでは対話形式で聞いたことを変数に格納し、仮想環境をアクティベートしてから、
実行結果を保存するディレクトリを作成し、プログラムを実行します。最後に生成物をログディレクトリに移すという処理まで行っています。  
このシェルスクリプトにgit pull等を追加して、実行時に自動でコードを更新するようにするなどの工夫をすると更に使いやすくなると思います。  


- run.shから実行する場合はimportする際に、python上のカレントディレクトリがrun.shのある場所(=プロジェクトルート)になる点に注意してください。
少し分かりづらいので、最初は./template/main.pyを使ったほうがいいと思います。 


- 今回やった実験管理は簡単なもの(ただし普通ではない)ですが、Tensorboard等を使っている人をよく見かけます。MLflowなど他にも色々とやり方はあると思うので好きな方法を探してください。

---
## 参考文献
Python環境構築ガイド  
https://www.python.jp/install/ubuntu/index.html

MNIST-JPG  
https://github.com/teavanist/MNIST-JPG  
./data/mnist.zipに使用したものです。

Pytorch Template 個人的ベストプラクティス（解説付き）  
https://qiita.com/takubb/items/7d45ae701390912c7629

PyTorchでDatasetの読み込みを実装してみた  
https://qiita.com/kumonk/items/0f3cad018cc9aec67a63

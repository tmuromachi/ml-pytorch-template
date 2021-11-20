#!/bin/sh　-x

echo -n "Please select the CUDA device number:"
read num
# 変数numが空の場合の処理
if [ -z "$num" ]; then
  num=0
fi

echo -n "comment:"
read comment

# 仮想環境をアクティベートする
source ./.venv/bin/activate

# git pull
# git pull
# git log -1

# 実行とロギング
# python main.py $1
now=`date "+%F_%T"`
echo $now$comment
mkdir ./log/$now$comment
python ./src/main.py $num 2>&1 | tee ./log/$now$comment/log.txt

# python ./template/main.py $num 2>&1 | tee ./log/$today$comment/log.txt
# 生成された画像(lossの推移)をログ保存用ディレクトリに移す
mv loss.png ./log/$now$comment/
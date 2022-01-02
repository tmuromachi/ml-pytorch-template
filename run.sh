#!/bin/sh　-x

# CUDA番号指定
echo -n "Please select the CUDA device number:"
read num
if [ -z "$num" ]; then
  num=0    # 変数numが空の場合に0番指定にする
fi

# 実行コメント入力
echo -n "comment:"
read comment
comment=`echo $comment | tr -d ' '` # 空白は削除

# 仮想環境をアクティベートする
source ./.venv/bin/activate

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
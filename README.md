# コーディング課題機械学習・数理分野

PFNインターン課題2018の提出物となります.
課題の内容は[こちら](https://github.com/pfnet/intern-coding-tasks)を参照してください.

課題1のソースコードはcore/linalg.py及びfunctions/activation/に記述してあります.
テストコードはtests/以下にソースコードと同様の階層においてあります.

## Requirements
- Python >= 3.6 Only

## Usage
次のコマンドで実行してください.
--taskの引数の課題が実行されます.
```bash
git clone https://github.com/natsukium/PFNIntern2018.git
cd PFNIntern2018/src/
python main.py --task [2-4]
```

## Test
本ソースコードのテストはunittestを用いています.
```bash
cd PFNIntern2018/src/
python -m unittest discover
# if flake8 is installed
flake8
```

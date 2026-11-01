# 第2回をMacで実行する手順

この回の教材はColabのT4で動作確認されている。今回はM2 MacBook Air（16GB、macOS 14.5）で実行し、環境の違いを解答と結果に記録する。
基礎課題の2条件はGPU（MPS）で実行した。勾配の比較でCPUとMPSに大きな差があったため、追加実験では学習をCPU、回答生成・評価をMPSで行う。

## 環境

第1回の `.venv` と依存関係を混ぜないよう、学習用環境を別に作る。
リポジトリのルートから実行する。

```bash
uv venv --python 3.12 .venv/lecture02
uv pip install --python .venv/lecture02/bin/python -r lectures/02/requirements-local.txt
```

## 実行

標準設定と、学習率だけを半分にした設定を別プロセスで実行する。
それぞれ固定された元のモデルを読み込むため、前の学習で追加したアダプターは引き継がない。

```bash
.venv/lecture02/bin/python -u scripts/run_lora_local.py --name baseline
.venv/lecture02/bin/python -u scripts/run_lora_local.py --name lr_1e4 --lr 0.0001
```

モデルの取得には初回だけダウンロードが必要。Gemini APIは使わない。
結果とチェックポイントは `.venv/lecture02-runs/<name>/` に保存する。
既存の結果を上書きしないよう、同じ名前の出力ディレクトリがある場合は停止する。

追加実験は、標準設定と同じ学習率 `0.0002` を使う。学習時の計算スレッド数は4に固定する。
`--device` が学習、`--eval-device` が前後の回答生成・評価に使う計算場所。後者を省略すると学習と同じ場所になる。

```bash
.venv/lecture02/bin/python -u scripts/run_lora_local.py --name cpu_train_mps_eval --device cpu --eval-device mps
```

CPUとMPSの勾配の比較方法は [results/diagnostics/README.md](results/diagnostics/README.md) に記録する。

## 教材からの変更

- 教材ノートブックのFAQ、学習用14問・検証用7問・評価用21問、生成条件、採点関数をそのまま使う。
- 保存先をColabの `/content` からローカルのディレクトリへ変更する。
- 重みは教材と同じ32bitで読み込む。ライブラリが混合精度を自動で有効にするのを避けるため、3条件とも `fp16=False` と `bf16=False` を明示する。
- 最初の短い試行では勾配がNaNになった。32bitを明示した2ステップの再試行では有限の勾配と重みの更新を確認した。本番でも、非有限値が出たら停止する。
- lossと勾配ノルムを各ステップ記録する。NaNをログから隠すフィルターは無効にする。
- 知識質問の「FAQ にある」フラグのみ修正する。教材は評価専用の `gold` で判定するため、学習用に含まれる対照質問も「FAQ にない」と表示してしまう。今回は元のFAQの質問一覧で判定する。質問文やモデルに渡す入力、生成結果、分類、評価スコアは変更しない。

実行コードはノートブック内の対応するセルを読み込む。ノートブックの内容を変更した場合、結果に記録するSHA-256も変わる。

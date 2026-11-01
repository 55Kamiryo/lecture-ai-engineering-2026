# 第2回のローカル実行結果

`baseline/` は教材の学習率 `0.0002`、`lr_1e4/` は学習率だけを `0.0001` に変更した条件。
それぞれ別プロセスで元のモデルを読み込み、同じFAQと同じ学習・評価用質問を使う。

`cpu_train_mps_eval/` は改善の追加実験。標準設定と同じ学習率 `0.0002`・40ステップで、学習をCPU、前後の回答生成・評価をMPSで実行した。
基礎課題の学習率比較は最初の2条件で、追加実験では計算環境を変えた効果を調べる。
CPUとMPSで勾配を比べた記録と、小さな再現例は [diagnostics/README.md](diagnostics/README.md) にある。

| ファイル | 内容 |
|---|---|
| `result.json` | モデル・設定、学習前後の集計、退行した質問、知識質問、最終loss・学習時間 |
| `eval_before.json` | 学習前の21問の回答全文、形式と内容の判定 |
| `eval_after.json` | 学習後の同じ21問の回答全文、形式と内容の判定 |
| `training_log.json` | 各ステップのloss・勾配ノルム・学習率と、学習終了時の集計 |
| `run_info.json` | 実行日時、環境、ライブラリの版、実際の精度設定、進行状態 |

`result.json` の `training.final_logged_loss` は40ステップ目のloss、`average_training_loss` は全学習の平均loss。
`training.seconds` はTrainerが報告する学習時間、`train_and_save_seconds` は学習からアダプター保存までの時間。
CPUでの学習中に軽い診断も一部並行して行ったため、時間は今回の実行記録であり、厳密な速度ベンチマークではない。

モデルの重みとチェックポイントはGit管理対象外の `.venv/lecture02-runs/` に保存し、このディレクトリには含めない。
Mac向けの変更点と再実行手順は [local-mac.md](../local-mac.md) に記載する。

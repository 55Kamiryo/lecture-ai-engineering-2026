# 勾配の比較

MPSでの本番学習では、クリッピング前の勾配ノルムが `10^11` 前後になり、学習率を下げても評価スコアは改善しなかった。
そこで、学習用の先頭1問（263トークン）で重みを更新せず、CPUとMPSの順伝播・逆伝播を比較した。
モデルとLoRAの設定、seed 42、FP32を揃え、dropoutは無効にした。診断では通常の `nll`、本番では教材の既定の損失計算を使う。

| 計算場所 | loss | 勾配ノルム |
|---|---:|---:|
| 全体をCPU | 2.345047 | 7.962 |
| 全体をMPS | 2.345175 | 約2.574 × 10^11 |
| MPS、線形注意のchunk計算だけCPU | 2.345175 | 約1.853 × 10^19 |
| MPS、causal convolutionだけCPU | 2.345176 | 約8.241 × 10^6 |
| MPS、通常のRMSNormの正規化だけCPU | 2.345175 | 約4.330 × 10^11 |
| MPS、SiLUに渡す勾配を連続配置に変更 | 2.345175 | 約6.206 × 10^6 |

lossは近い値だったが、逆伝播の結果には大きな差があった。一部の処理をCPUに移しても解消しなかったため、追加実験では学習全体をCPUで実行し、前後の回答生成・評価はMPSで行う。
この比較だけでは、原因となる演算やPyTorch・macOSのどの部分に問題があるかは特定できない。MPS全般で同じ問題が起きるという結論でもない。

追加で、畳み込みとSiLUだけの小さな計算でも、SiLUへ渡す勾配が非連続なメモリ配置の場合にCPUとMPSの逆伝播に差が出た。6,144チャンネルの例では、順伝播の相対誤差は約 `6.3e-8`、入力勾配の相対誤差は約 `3.05` だった。
SiLUの出力にフックを付け、逆伝播で受け取る勾配を `.contiguous()` にすると、この例の入力勾配の相対誤差は約 `9.3e-8` になった。
ただし、モデル全体ではまだ大きな勾配が残ったため、本番用の修正には採用していない。この再現例だけで全体の原因を説明できたとはしない。

各JSONの `gradient_norms` に、学習対象パラメータごとの勾配ノルムを保存した。条件の詳細は [conditions.json](conditions.json) に記録した。

リポジトリのルートから再実行する。学習やモデル保存は行わず、結果をGit管理対象外の `.venv/lecture02-diagnostics/` に書き出す。

```bash
.venv/lecture02/bin/python scripts/diagnose_lora_backward.py --device cpu
.venv/lecture02/bin/python scripts/diagnose_lora_backward.py --device mps
.venv/lecture02/bin/python scripts/diagnose_lora_backward.py --device mps --cpu-op chunk
.venv/lecture02/bin/python scripts/diagnose_lora_backward.py --device mps --cpu-op conv
.venv/lecture02/bin/python scripts/diagnose_lora_backward.py --device mps --cpu-op rms
.venv/lecture02/bin/python scripts/diagnose_lora_backward.py --device mps --mps-silu-grad-contiguous
.venv/lecture02/bin/python scripts/diagnose_mps_silu.py
.venv/lecture02/bin/python scripts/diagnose_mps_silu.py --contiguous-grad
```

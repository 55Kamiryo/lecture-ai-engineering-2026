# 第5回の実行記録

2026年11月23日にローカルのMacからGemini APIを呼び出して記録した。
回答モデルは全実行で `gemini-3.5-flash-lite`、埋め込みは教材既定の `gemini-embedding-001`。

各条件1回の観測であり、平均速度や一般的な正答率を示すものではない。
教材と索引は変更せず、JSONにSHA-256を保存した。全文投入は全9資料438ページ、その他は第1〜3回本講義214ページを使う。

| ファイル名（拡張子を除く） | 内容 | RAG_TOP_K |
|---|---|---:|
| `retrieval_only` | `LoRA QLoRA 違い` の検索だけの結果。JSONのみ | 5 |
| `full_context_lora` | LoRAの質問、全文投入 | 5（検索なし） |
| `vector_k5_lora` | LoRAの質問、ベクトル検索 | 5 |
| `tree_lora` | LoRAの質問、Tree | 5（ベクトル検索なし） |
| `graph_lora` | LoRAの質問、Graph | 5 |
| `vector_k5_memory` | メモリの質問、ベクトル検索 | 5 |
| `vector_k1_lora` | LoRAの質問、変更後のベクトル検索 | 1 |
| `vector_k1_memory` | メモリの質問、変更後のベクトル検索 | 1 |

`.log` は教材CLIの回答と `--trace` 出力、`.json` は計測値とツール呼び出し・返り値の全記録。
全文投入のプロンプト全文やAPIキーは保存していない。
`usage.generation_cost_usd` は教材が返したDecimalを文字列として保存している。請求額ではなく、埋め込み・索引構築分を除く推定値。

## 再実行

リポジトリのルートで `uv sync --frozen` を実行する。
APIキーは無視対象の `.env` または環境変数 `GEMINI_API_KEY` に設定する。
環境変数で設定済みなら、以下の `--env-file .env` は省略できる。

提出コードは `app/config.py` の `RAG_TOP_K = 1`。変更前を再現する場合は5に戻し、別のプロセスで実行する。**Graphも同じ設定を参照するため、演習のGraphを再実行するときは5を使う。**

```bash
AIE_QUESTION='第2回の説明では、LoRAは何を固定し何を学習しますか？ QLoRAは何を加えますか？'
uv run --env-file .env python -m scripts.record_lec05 \
  --output lectures/05/results/rerun_vector_lora -- --trace "$AIE_QUESTION"
```

全文投入・Tree・Graphは、`-- --trace` の箇所をそれぞれ `-- --full-context --trace`、`-- --tree --trace`、`-- --graph --trace` に変える。`--output` も条件ごとに変える。

メモリの質問は次を指定する。

```text
第3回の例では、コンパクションとセッションをまたぐメモリはどう違いますか。日々のメモはすべて長期メモリに入りますか
```

検索だけを再実行する場合：

```bash
uv run --env-file .env python -m app.retrieval 'LoRA QLoRA 違い'
```

記録用スクリプトは既存ファイルの上書きを拒否する。再実行時は未使用の出力名を指定する。

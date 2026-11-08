# 第3回の実行記録

2026年11月8日、Macのローカル環境でGemini APIを呼び出して記録した。
モデルは教材の既定値 `gemini-3.5-flash-lite`。実行と結果整理にはCodexを利用した。

## 条件

| ディレクトリ | 内容 |
|---|---|
| `exercise/` | `app.ask` による回答と、`make_quiz(lecture=2)` の呼び出し確認 |
| `baseline/` | 教材の `INSTRUCTIONS`、2問を各3回 |
| `without_search_rule/` | 必須検索の指示1行だけ削除、同じ2問を各3回 |
| `without_search_prompts/` | 上の条件から、検索ツールの説明文にある「答える前に使うこと」も削除、同じ2問を各3回 |
| `attempts/without_search_rule_rate_limit/` | 変更条件の最初の試行。APIの回数制限で終了した記録 |

最初の比較（`baseline/` と `without_search_rule/`）では `make_quiz` を含む3ツールを両条件に登録した。ツールの実装とdocstring、モデル、用語集、質問は変えていない。
各実行は別プロセスで、会話履歴を引き継がない。temperatureやseedは指定せず、教材のCLIとAPIの既定値を使った。
標準条件6回の後に変更条件6回を実行したため、実行順序やAPI側の変動の影響までは取り除けていない。
変更条件の最初の試行はAPIの回数制限で終了したため、1分以上空けて同じ条件を再実行した。失敗した試行は比較表に含めず、`attempts/` に残した。

追加実験の `without_search_prompts/` は、`without_search_rule/` から `search_glossary` のdocstringの1文だけを削除した条件。機能と引数の説明は残した。他の指示、ツールの処理、モデル、依存関係、質問は同じ。記録用スクリプトには条件名と検査を追加したため、スクリプトのSHA-256は以前の記録と異なる。前の条件を同時に再実行した比較ではなく、実行時刻の差も残る。

## ファイル

- `metadata.json`：実行日時、モデル、依存関係、指示文、ツールの説明、ソースファイルのSHA-256。
- `<質問名>_<反復番号>.json`：実際のコマンド、終了コード、回答とtraceを含む標準出力、標準エラー、ツール名と引数、LLMの呼び出し回数。
- `runs.json`：その条件の実行記録をまとめた一覧。個別JSONと同じ内容。

`requests` はLLMへのリクエスト数。ツールの実行回数とは区別する。
`app.ask` はusageを表示しないため、その記録の `requests` は未計測の `null` としている。
traceのツール戻り値は教材の表示どおり先頭120文字まで。回答本文とusageは省略せず保存した。

## 再実行する場合

リポジトリ直下で、READMEのローカル手順に従って `uv sync --frozen` を実行する。
既存の `.env` に `GEMINI_API_KEY` を設定する。APIキーを結果ファイルやGitに含めない。

記録用スクリプトは、既存の結果を上書きしない。再実験は、既存の結果を退避した別の作業コピーで行う。

```bash
uv run --env-file .env python -m scripts.record_lec03 --condition exercise
uv run --env-file .env python -m scripts.record_lec03 --condition baseline
```

次に `app/agent.py` の `INSTRUCTIONS` から、次の行だけを削除する。

```text
- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
```

```bash
uv run --env-file .env python -m scripts.record_lec03 --condition without_search_rule
```

実験後はこの1行を元に戻す。今回の提出用コードにも、教材の検索指示を残している。
スクリプトは実際に `python -m app.agent --trace` を子プロセスで呼び、変更前後の指示文を各 `metadata.json` に保存する。
環境変数としてAPIキーを設定済みの場合は、上のコマンドから `--env-file .env` を省略できる。

追加実験を再実行する場合は、全体の検索指示を削除した状態で `app/tools.py` の `search_glossary` のdocstringも次の内容にする。削除するのは「AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について答える前に使うこと。」の1文だけ。

```text
講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

query には調べたい用語を1つだけ入れる（例: "LoRA"）。
```

```bash
uv run --env-file .env python -m scripts.record_lec03 --condition without_search_prompts
```

終了後は全体の指示とdocstringを両方元に戻す。保存した `metadata.json` で実験時の文面を確認できる。

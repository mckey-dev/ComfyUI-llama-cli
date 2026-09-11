# ComfyUI llama-cli

更新日: 2026-09-12

## 概要

ComfyUI 向けに公式 llama.cpp の `llama-cli` を取得し、他のカスタムノードが同じバイナリを共有するための拡張です。ノードは出ません。推論は行いません。

- Windows x64 CUDA 13: 公式 zip（`b10472`）
- Linux x64 CUDA: 同じタグを `cmake` + `nvcc` でビルド
- `storage` に必要ファイルが揃っていれば再ダウンロードも再ビルドもしない
- 初回はノード実行時。ComfyUI 起動ではビルドしない

## 配置

Notebook ルート（`notebooks/` の親）基準:

```text
./
  notebooks/comfyui/custom_node/ComfyUI-llama-cli/   # このリポジトリ
  tmp/ComfyUI-llama-cli/                             # ダウンロードと cmake。成功後に削除
  storage/ComfyUI-llama-cli/<tag>/<platform>/        # llama-cli と実行ライブラリ
```

ローカルで `tmp/` と `storage/` が無いときは、ComfyUI ルートの隣に同じ名前で作ります。

必要としているカスタムノード:

- [ComfyUI-LLM-text-processor-fork](../ComfyUI-LLM-text-processor-fork)
- [ComfyUI-JoyCaption-fork](../ComfyUI-JoyCaption-fork)

どちらかを使う場合はこの拡張を `custom_nodes`（クラウドでは `custom_node`）に置いてください。

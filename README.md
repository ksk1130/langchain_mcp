# LangChain MCP チャットアプリケーション

LangChainとModel Context Protocol (MCP)を使用したチャットアプリケーションです。GradioベースのWebUIを提供し、複数のLLMプロバイダーとMCPツールを統合します。

## 🚀 特徴

- **Gradio WebUI**: 使いやすいブラウザベースのチャットインターフェース
- **MCP統合**: Model Context Protocolを使用した外部ツールの呼び出し
- **複数LLM対応**: OpenAI、Geminiなど複数のLLMプロバイダーをサポート
- **デュアルLLM機能**: 2つのLLMを並行実行して比較できる機能
- **ツール履歴表示**: 実行されたツールの履歴を表示
- **カスタマイズ可能**: 設定ファイルで簡単にカスタマイズ

## 📋 前提条件

- Python 3.11以上
- uv (Python パッケージマネージャー)

## 🔧 インストール

### 1. uvの導入

#### Windows (PowerShell)
```powershell
# uvをインストール
iwr https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip -OutFile uv.zip
Expand-Archive uv.zip -DestinationPath .
Move-Item uv.exe $env:USERPROFILE\.local\bin\
```

#### macOS/Linux
```bash
# uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. プロジェクトのクローンと依存関係のインストール

```bash
git clone <リポジトリURL>
cd langchain_mcp
uv sync
```

## ⚙️ 設定

### 1. サーバー設定ファイル (`server_params.json`)

MCPサーバーとLLMプロバイダーの設定を行います：

```json
{
  "servers": {
    "awslabs": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "awslabs-aws-documentation-mcp-server",
        "awslabs.aws-documentation-mcp-server.exe"
      ]
    },
    "microsoft.docs.mcp": {
      "transport": "streamable_http",
      "url": "https://learn.microsoft.com/api/mcp"
    }
  },
  "llm": {
    "OpenAI": { 
      "model": "gpt-4o", 
      "base_url": "http://127.0.0.1:4000" 
    },
    "Gemini": { 
      "model": "gpt-4.1", 
      "base_url": "http://127.0.0.1:4000" 
    }
  }
}
```

#### サーバー設定オプション

**MCPサーバー設定:**
- `transport`: 通信方式 (`stdio`, `streamable_http`, `sse`, `websocket`)
- `command`: 実行コマンド (stdio使用時)
- `args`: コマンド引数 (stdio使用時)
- `url`: サーバーURL (HTTP通信時)

**LLM設定:**
- `model`: 使用するモデル名
- `base_url`: LLMプロバイダーのベースURL (ローカルプロキシ等)

### 2. LiteLLM設定ファイル (`config.yaml`)

LiteLLMプロキシサーバーの設定（オプション）：

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: "your-openai-api-key"
  
  - model_name: gpt-4.1
    litellm_params:
      model: gemini/gemini-pro
      api_key: "your-gemini-api-key"

general_settings:
  master_key: "your-master-key"
  database_url: "sqlite:///litellm.db"
```

## 🚀 実行方法

### 1. 単一LLMモード

```bash
# 直接実行
uv run main.py

# または実行スクリプト使用
./exec_single.bat    # Windows
```

### 2. デュアルLLMモード

```bash
# 直接実行
uv run main_dual.py

# または実行スクリプト使用
./exec_dual.bat      # Windows
```

### 3. LiteLLMプロキシサーバー (オプション)

```bash
# LiteLLMプロキシを起動
uv run litellm --config config.yaml

# または実行スクリプト使用
./exec_litellmproxy.bat  # Windows
```

## 📝 使用方法

### 基本的な使用方法

1. アプリケーションを起動
2. ブラウザで `http://127.0.0.1:7860` にアクセス
3. チャットタブでメッセージを入力
4. ツール呼び出しの有効/無効を選択
5. 使用するLLMを選択
6. 「利用可能なツール」タブでツール一覧を確認

### ツール機能

- **Function Calling**: 有効にするとMCPツールを自動呼び出し
- **ツール履歴表示**: 実行されたツールとその引数を表示
- **ツール一覧**: 利用可能なツールの詳細情報を表示

## 🧪 テスト

```bash
# テスト実行
uv run pytest

# または実行スクリプト使用
./exec_pytest.bat    # Windows

# 詳細出力でテスト実行
uv run pytest -v
```

## 📁 プロジェクト構造

```
langchain_mcp/
├── main.py                    # 単一LLMアプリケーション
├── main_dual.py               # デュアルLLMアプリケーション
├── langchain_mcp_utils.py     # 共通ユーティリティ関数
├── test_langchain_mcp_utils.py # テストファイル
├── server_params.json         # サーバー設定ファイル
├── config.yaml               # LiteLLM設定ファイル
├── pyproject.toml            # プロジェクト設定
├── requirements.dat          # 依存関係リスト
├── uv.lock                   # ロックファイル
└── exec_*.bat                # 実行スクリプト (Windows)
```

## 🔧 トラブルシューティング

### 一般的な問題

**1. ツール取得エラー**
```
Configuration error: Missing 'transport' key in server configuration
```
- `server_params.json`の`transport`フィールドを確認
- 有効な値: `stdio`, `streamable_http`, `sse`, `websocket`

**2. LLM接続エラー**
- `base_url`の設定を確認
- LiteLLMプロキシが起動しているか確認
- API キーが正しく設定されているか確認

**3. 依存関係エラー**
```bash
# 依存関係を再インストール
uv sync --reinstall
```

### ログとデバッグ

```bash
# デバッグモードで実行
uv run litellm --config config.yaml --debug

# Python ログレベル設定
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
uv run main.py
```

## 🤝 貢献

1. フォークを作成
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- [LangChain](https://github.com/hwchase17/langchain) - LLMアプリケーションフレームワーク
- [Gradio](https://github.com/gradio-app/gradio) - WebUIフレームワーク
- [Model Context Protocol](https://modelcontextprotocol.io/) - ツール統合プロトコル
- [LiteLLM](https://github.com/BerriAI/litellm) - LLMプロキシサーバー
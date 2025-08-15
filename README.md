# LangChain MCP チャットアプリケーション

LangChainとModel Context Protocol (MCP)を使用したチャットアプリケーションです。GradioベースのWebUIを提供し、複数のLLMプロバイダーとMCPツールを統合します。

## 🚀 特徴

- **Gradio WebUI**: 使いやすいブラウザベースのチャットインターフェース
- **MCP統合**: Model Context Protocolを使用した外部ツールの呼び出し
- **複数LLM対応**: OpenAI、Geminiなど複数のLLMプロバイダーをサポート
- **デュアルLLM機能**: 2つのLLMを並行実行して比較できる機能
- **システムプロンプト**: 内部でカスタマイズ可能なシステムプロンプト機能
- **ツール履歴表示**: 実行されたツールの履歴を表示
- **デバッグ制御**: 設定ファイルでデバッグモードの有効/無効を制御
- **包括的テスト**: pytestによる自動テスト機能
- **一括実行**: 複数アプリケーションの同時起動機能
- **カスタマイズ可能**: 設定ファイルで簡単にカスタマイズ

## 📋 前提条件

- Python 3.11以上
- uv (Python パッケージマネージャー)

## 🔧 インストール

### 1. uvの導入
uvはPythonのパッケージマネージャーで、依存関係の管理と実行を簡素化します。

**uvの利点:**
- 🔒 **分離された環境**: プロジェクトごとに独立した仮想環境を作成し、他のPythonインストールやプロジェクトに影響を与えません
- ⚡ **高速**: Rustで書かれており、pipよりも大幅に高速な依存関係解決とインストール
- 🎯 **プロジェクト管理**: `pyproject.toml`ベースの現代的なプロジェクト管理
- 🛡️ **システム保護**: システムPythonやグローバルパッケージを変更せず、クリーンな開発環境を維持

以下の手順でインストールしてください。
- [uvのインストール手順(公式)](https://docs.astral.sh/uv/#installation)

### 2. プロジェクトのクローンと依存関係のインストール

```bash
git clone <リポジトリURL>
cd langchain_mcp
uv sync
```

> 💡 **注意**: `uv sync`により、このプロジェクト専用の仮想環境が自動作成されます。システムPython環境や他のプロジェクトの依存関係には一切影響しません。

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
  },
  "debug": "true"
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

**デバッグ設定:**
- `debug`: デバッグモードの有効/無効 (`"true"` または `"false"`)
  - `"true"`: LangGraphエージェントの詳細ログを出力
  - `"false"`: 通常の実行ログのみ出力

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

### 3. 一括実行 (すべてのアプリケーション)

```bash
# LiteLLMプロキシ、単一LLM、デュアルLLMを同時起動
./exec_all.bat       # Windows
```

### 4. LiteLLMプロキシサーバー (オプション)

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
5. 使用するLLMを選択（単一LLMモードの場合）
6. 「利用可能なツール」タブでツール一覧を確認

### 高度な機能

#### システムプロンプト
- **内部設定**: システムプロンプトは内部で自動設定されます
- **カスタマイズ**: `main.py`や`main_dual.py`の`system_prompt`変数を編集することで変更可能
- **一貫性**: 全てのLLMで統一されたシステムプロンプトが適用されます

#### デバッグモード
- **設定**: `server_params.json`の`"debug"`で制御
- **有効時**: LangGraphエージェントの詳細なツール呼び出しログを出力
- **無効時**: 簡潔な実行ログのみ表示

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
├── main.py                      # 単一LLMアプリケーション
├── main_dual.py                 # デュアルLLMアプリケーション
├── langchain_mcp_utils.py       # 共通ユーティリティ関数
├── test_langchain_mcp_utils.py  # テストファイル
├── server_params.json           # サーバー設定ファイル
├── config.yaml                  # LiteLLM設定ファイル
├── pyproject.toml              # プロジェクト設定
├── requirements.dat            # 依存関係リスト
├── uv.lock                     # ロックファイル
├── exec_single.bat             # 単一LLM実行スクリプト (Windows)
├── exec_dual.bat               # デュアルLLM実行スクリプト (Windows)
├── exec_all.bat                # 一括実行スクリプト (Windows)
├── exec_litellmproxy.bat       # LiteLLMプロキシ実行スクリプト (Windows)
├── exec_pytest.bat             # テスト実行スクリプト (Windows)
└── README.md                   # このファイル
```
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

**4. デバッグログが表示されない**
- `server_params.json`の`"debug"`設定を確認
- `"true"`に設定するとLangGraphエージェントの詳細ログが表示

### カスタマイズ

**システムプロンプトの変更:**
```python
# main.py または main_dual.py の該当箇所を編集
system_prompt = "あなたのカスタムシステムプロンプトをここに設定"
```

**デバッグモードの切り替え:**
```json
{
  "debug": "true"   // 詳細ログ有効
  // または
  "debug": "false"  // 簡潔ログのみ
}
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
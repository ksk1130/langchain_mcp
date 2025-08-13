import asyncio
import json
from langchain_openai import ChatOpenAI


def extract_answer(resp) -> str:
    """
    エージェント応答から回答テキストのみを抽出する関数。
    dict, list, オブジェクト型に対応し、AIMessageのcontentや代表的な回答キーを優先して返す。
    Args:
        resp: エージェントから返された応答オブジェクト
    Returns:
        str: 回答テキスト
    """
    if isinstance(resp, dict):
        # dictの場合、AIMessageのcontentや代表的な回答キーを優先して抽出
        if "messages" in resp:
            messages = resp["messages"]
            # messagesがリストの場合、AIMessageのcontentを優先して抽出
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if (
                        hasattr(msg, "content")
                        and hasattr(msg, "__class__")
                        and "AIMessage" in str(type(msg))
                    ):
                        return msg.content
                    # messagesがdictの場合、AIMessageのcontentを優先して抽出
                    elif isinstance(msg, dict) and msg.get("type") == "ai":
                        return msg.get("content", "")

        # その他のキーをチェック
        for key in ["output", "answer", "content", "result", "text"]:
            if key in resp:
                return extract_answer(resp[key])

        # dictの場合、AIMessageのcontentや代表的な回答キーを優先して抽出
        if resp.get("type") == "ai":
            return resp.get("content", "")

        return str(resp)

    # リストの場合、各アイテムから回答を抽出
    elif isinstance(resp, list):
        for item in reversed(resp):
            if (
                hasattr(item, "content")
                and hasattr(item, "__class__")
                and "AIMessage" in str(type(item))
            ):
                return item.content

            elif isinstance(item, dict) and item.get("type") == "ai":
                return item.get("content", "")

        answers = [extract_answer(item) for item in resp]
        return "\n---\n".join(str(a) for a in answers if a)
    else:
        if hasattr(resp, "content"):
            return resp.content
        try:
            return str(resp)
        except Exception:
            return f"<{type(resp).__name__}>"


def load_server_params(path: str) -> dict:
    """
    server_params.jsonからサーバー設定を読み込む関数
    Args:
        path (str): 設定ファイルのパス
    Returns:
        dict: サーバー設定
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
        # jsonが存在しない場合や、読み込みに失敗した場合は空の辞書を返す
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


# LLMを初期化する関数
def initialize_llm(llm_name: str = "gpt-4o", base_url: str = "") -> ChatOpenAI:
    """
    LLMを初期化する関数
    Args:
        llm_name (str): LLMの名前
        base_url (str): LLMのベースURL
    Returns:
        ChatOpenAI: 初期化されたChatOpenAIインスタンス
    """
    if base_url != "":
        return ChatOpenAI(model=llm_name, base_url=base_url)
    else:
        return ChatOpenAI(model=llm_name)


def get_llm_params(params: dict) -> tuple:
    """
    LLMの名前とベースURLなどの設定を取得する関数。
    Args:
        params (dict): 設定パラメータ
    Returns:
        tuple: (model_name, base_url, llm_options, default_llm, available_llms)
    """

    # 利用可能なLLMオプションを取得
    llm_options = params.get("llm", {})
    available_llms = list(llm_options.keys()) if llm_options else ["Default"]

    # デフォルトのLLM名
    default_llm = (
        available_llms[0]
        if available_llms and available_llms[0] != "Default"
        else "Default"
    )

    # デフォルトLLMからモデル名とベースURLを取得
    llm_name = default_llm
    base_url = ""

    if llm_name in llm_options:
        llm_config = llm_options[llm_name]
        if isinstance(llm_config, dict):
            # 新しい形式: {"model": "...", "base_url": "..."}
            model_name = llm_config.get("model", "gpt-4o")
            base_url = llm_config.get("base_url")

    return model_name, base_url, llm_options, default_llm, available_llms


# 利用可能なツール一覧を取得する関数
async def get_available_tools(available_tools: list) -> str:
    """
    初期化済みのグローバルツールから利用可能なツール一覧を取得
    Args:
        available_tools (list): 利用可能なツールのリスト
    Returns:
        str: ツール一覧の文字列
    """
    try:
        result = "# 利用可能なツール一覧\n\n"

        result += f"**合計ツール数**: {len(available_tools)}\n\n"
        if available_tools:
            result += "## ツール詳細:\n"
            for i, tool in enumerate(available_tools, 1):
                tool_name = getattr(tool, "name", "Unknown")
                tool_desc = getattr(tool, "description", "説明なし")
                tool_args = getattr(tool, "args_schema", {})

                result += f"{i}. **{tool_name}**\n"
                result += f"   - 説明: {tool_desc}\n"
                if tool_args:
                    result += f"   - 引数: {tool_args}\n"
                result += "\n"
        else:
            result += "利用可能なツールが見つかりませんでした。\n"
        return result
    except Exception as e:
        return f"ツール一覧の取得中にエラーが発生しました: {type(e).__name__}: {str(e)}"


def sync_get_available_tools(available_tools: list) -> str:
    """
    利用可能なツール取得の同期ラッパー
    Returns:
        str: 利用可能なツールの一覧
    """
    try:
        # 30秒のタイムアウトを設定
        return asyncio.run(
            asyncio.wait_for(get_available_tools(available_tools), timeout=30.0)
        )
    except asyncio.TimeoutError:
        return (
            "ツール一覧の取得がタイムアウトしました。サーバーの接続を確認してください。"
        )
    except Exception as e:
        return f"ツール一覧の取得中にエラーが発生しました: {type(e).__name__}: {str(e)}"


def extract_tool_history(agent_response) -> list:
    """
    ツール履歴を抽出するヘルパー関数。
    Args:
        agent_response (dict): エージェントの応答
    Returns:
        list: ツール履歴のリスト
    """
    tool_history = []
    if isinstance(agent_response, dict):
        if "messages" in agent_response:
            messages_resp = agent_response["messages"]
            if isinstance(messages_resp, list):
                for msg in messages_resp:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if isinstance(tool_call, dict):
                                # LangChainのAIMessage.tool_calls形式
                                if "function" in tool_call:
                                    tool_name = tool_call["function"].get("name", "Unknown")
                                    tool_args = tool_call["function"].get("arguments", "{}")
                                else:
                                    tool_name = tool_call.get("name", "Unknown")
                                    tool_args = tool_call.get("args", {})
                            else:
                                # オブジェクト型のtool_call
                                tool_name = getattr(tool_call, "name", str(tool_call))
                                tool_args = getattr(tool_call, "args", {})
                            tool_history.append(
                                f"ツール名: {tool_name}, 引数: {tool_args}"
                            )
        if "tool_calls" in agent_response:
            calls = agent_response["tool_calls"]
            if calls:
                for call in calls:
                    tool_history.append(
                        f"ツール名: {call.get('tool_name', call.get('name', 'Unknown'))}, 入力: {call.get('input', call.get('args', {}))}"
                    )

    return tool_history

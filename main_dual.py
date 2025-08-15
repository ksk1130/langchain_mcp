import os
import gradio as gr
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_utils import (
    extract_answer,
    load_server_params,
    initialize_llm,
    get_llm_params,
    sync_get_available_tools,
    extract_tool_history,
)

global_client = None
global_tools = []
llm1_name = None
llm2_name = None
llm_options = {}


# LLMを初期化する関数（ローカル版）
def initialize_llm_local(llm_name: str) -> ChatOpenAI:
    """
    選択されたLLMに基づいてChatOpenAIインスタンスを初期化（main_dual専用）
    Args:
        llm_name (str): LLMの名前
    Returns:
        ChatOpenAI: 初期化されたChatOpenAIインスタンス
    """
    if llm_name in llm_options:
        llm_config = llm_options[llm_name]
        if isinstance(llm_config, dict):
            # 新しい形式: {"model": "...", "base_url": "..."}
            model = llm_config.get("model", "gpt-4o")
            base_url = llm_config.get("base_url", "")
            return initialize_llm(model, base_url)
        else:
            # 古い形式: 文字列のbase_url
            return initialize_llm("gpt-4o", llm_config)
    else:
        # デフォルトまたは不明なLLMの場合
        base_url = ""
        return initialize_llm("gpt-4o", base_url)


# 単一LLM用の非同期チャット関数
async def single_llm_chat(
    user_input, history, function_calling, llm_name, system_prompt=""
) -> str:
    """
    単一のLLMに対してチャットを実行する関数
    Args:
        user_input (str): ユーザーからの入力
        history (list): チャット履歴
        function_calling (str): ツール呼び出しの有効/無効
        llm_name (str): LLMの名前
        system_prompt (str): システムプロンプト
    Returns:
        str: LLMからの応答
    """
    try:
        # 選択されたLLMでエージェントを初期化
        current_llm = initialize_llm_local(llm_name)
        # Gradioの履歴(messages形式)をLangChainの履歴に変換
        messages = []

        # システムプロンプトがある場合、最初に追加
        if system_prompt.strip():
            messages.append({"type": "system", "content": system_prompt.strip()})

        if history:
            for msg in history:
                if msg.get("role") == "user":
                    messages.append({"type": "human", "content": msg["content"]})
                elif msg.get("role") == "assistant":
                    messages.append({"type": "ai", "content": msg["content"]})
        messages.append({"type": "human", "content": user_input})
        # グローバルツールを使用
        agent_tools = global_tools if function_calling == "有効" else []
        agent = create_react_agent(current_llm, agent_tools, debug=True)
        agent_response = await agent.ainvoke({"messages": messages})
        answer = extract_answer(agent_response)
        # ツール履歴抽出（langchain_mcp_utils.pyの関数を使用）
        tool_history = extract_tool_history(agent_response)
        if tool_history:
            answer += f"\n\n[{llm_name} - 呼び出されたツール履歴]\n" + "\n".join(
                tool_history
            )
        return answer
    except Exception as e:
        return f"エラーが発生しました ({llm_name}): {str(e)}"


# 両方のLLMに同時にプロンプトを送信する関数
async def dual_llm_chat(user_input, history1, history2, function_calling) -> tuple:
    """
    2つのLLMに同時にプロンプトを送信し、結果を返す
    Args:
        user_input (str): ユーザーからの入力
        history1 (list): LLM1のチャット履歴
        history2 (list): LLM2のチャット履歴
        function_calling (str): ツール呼び出しの有効/無効
    Returns:
        tuple: 各LLMの応答と更新された履歴
    """
    if not user_input.strip():
        return "", history1, "", history2

    # 内部でシステムプロンプトを設定
    system_prompt = """
            あなたは親切で知識豊富なAIアシスタントです。ユーザーの質問に対して、正確で分かりやすい回答を提供してください。
            なお、回答にあたり、以下のルールを守ってください。
            1. 回答は日本語で行ってください。
            2. 回答は簡潔で明確にしてください。
            3. ツール呼び出しが有効な場合は、ツールを利用してください。
            4. ユーザーの意図を理解しかねる場合は、追加の情報を求めてください。
            """

    # 両方のLLMに同時にリクエストを送信
    tasks = [
        single_llm_chat(
            user_input, history1, function_calling, llm1_name, system_prompt
        ),
        single_llm_chat(
            user_input, history2, function_calling, llm2_name, system_prompt
        ),
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    response1 = (
        responses[0]
        if not isinstance(responses[0], Exception)
        else f"エラー: {responses[0]}"
    )
    response2 = (
        responses[1]
        if not isinstance(responses[1], Exception)
        else f"エラー: {responses[1]}"
    )
    # 履歴を更新
    new_history1 = history1 + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response1},
    ]
    new_history2 = history2 + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response2},
    ]
    return "", new_history1, "", new_history2


def sync_dual_llm_chat(user_input, history1, history2, function_calling) -> tuple:
    """
    非同期dual_llm_chat関数を同期的に呼び出すラッパー
    Args:
        user_input (str): ユーザーからの入力
        history1 (list): LLM1のチャット履歴
        history2 (list): LLM2のチャット履歴
        function_calling (str): ツール呼び出しの有効/無効
    Returns:
        tuple: 各LLMの応答と更新された履歴
    """
    return asyncio.run(dual_llm_chat(user_input, history1, history2, function_calling))


# 利用可能なツール一覧を取得する関数（ローカル版）
def sync_get_available_tools_local() -> str:
    """
    利用可能なツール取得の同期ラッパー（main_dual専用）
    Returns:
        str: 利用可能なツールのリスト
    """
    return sync_get_available_tools(global_tools)


async def main() -> None:
    # server_params.jsonからサーバー設定を読み込み
    params = load_server_params("server_params.json")

    # paramsが空の辞書の場合(＝設定ファイルが存在しない場合)、エラーで終了
    if not params:
        print("設定ファイル(server_params.json)が見つからないか、無効です。")
        return

    # paramsから必要な情報を取得
    global llm_options
    _, _, llm_options, _, available_llms = get_llm_params(params)

    # 2つのLLMを取得（最初の2つ、または同じものを2回）
    global llm1_name, llm2_name
    llm1_name = available_llms[0] if len(available_llms) >= 1 else "Default"
    llm2_name = available_llms[1] if len(available_llms) >= 2 else available_llms[0]

    # アプリ起動時にclientとtoolsを一度取得して使い回す
    print("=== MCPクライアントとツールを初期化中... ===")

    # グローバルクライアントとツールを初期化
    try:
        global global_client
        global_client = MultiServerMCPClient(params.get("servers", {}))
        global global_tools
        global_tools = await global_client.get_tools()
        print(f"初期化完了: {len(global_tools)} 個のツールが利用可能です")

        # ツール一覧を表示
        for i, tool in enumerate(global_tools, 1):
            tool_name = getattr(tool, "name", "Unknown")
            print(f"  {i}. {tool_name}")
    except Exception as e:
        print(f"ツール取得エラー: {e}")
        print("プロセスを終了します...")
        os._exit(1)  # 即座にプロセスを強制終了

    # Gradio UIの構築
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            height: 100vh !important;
            display: flex !important;
            flex-direction: column !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .title {
            flex-shrink: 0 !important;
            padding: 1px !important;
            margin: 0 !important;
        }
        .dual-chat-container {
            flex: 1 !important;
            display: flex !important;
            gap: 5px !important;
            padding: 1px !important;
        }
        .chat-pane {
            flex: 1 !important;
            display: flex !important;
            flex-direction: column !important;
        }
        .controls-container {
            flex-shrink: 0 !important;
            padding: 1px !important;
            background: #f8f9fa !important;
        }
        .input-container {
            flex-shrink: 0 !important;
            padding: 1px !important;
            background: white !important;
        }
        .chatbot {
            height: calc(100vh - 350px) !important;
            min-height: 400px !important;
        }
        .llm-title {
            text-align: center !important;
            font-weight: bold !important;
            color: #2563eb !important;
            padding: 5px !important;
        }
        """,
    ) as demo:
        gr.Markdown("# LangChain MCP デュアルチャット", elem_classes=["title"])

        # タブで機能を分ける
        with gr.Tabs():
            with gr.TabItem("デュアルチャット"):
                # コントロール部分
                with gr.Row(elem_classes=["controls-container"]):
                    gr.Markdown("**ツール呼び出し（Function Calling）:**")
                    function_radio = gr.Radio(
                        ["有効", "無効"], value="有効", label=None, container=False
                    )

                # 2つのチャットボットを横並びで表示
                with gr.Row(elem_classes=["dual-chat-container"]):
                    # 左側のチャットボット
                    with gr.Column(elem_classes=["chat-pane"]):
                        gr.Markdown(f"## {llm1_name}", elem_classes=["llm-title"])
                        chatbot1 = gr.Chatbot(
                            type="messages",
                            height="calc(100vh - 350px)",
                            container=True,
                            autoscroll=True,
                            show_copy_all_button=True,
                            show_copy_button=True,
                            resizable=True,
                            elem_classes=["chatbot"],
                        )

                    # 右側のチャットボット
                    with gr.Column(elem_classes=["chat-pane"]):
                        gr.Markdown(f"## {llm2_name}", elem_classes=["llm-title"])
                        chatbot2 = gr.Chatbot(
                            type="messages",
                            height="calc(100vh - 350px)",
                            container=True,
                            autoscroll=True,
                            show_copy_all_button=True,
                            show_copy_button=True,
                            resizable=True,
                            elem_classes=["chatbot"],
                        )

                # 共通の入力フォーム
                with gr.Row(elem_classes=["input-container"]):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="両方のLLMに同時にメッセージを送信します...",
                        container=False,
                        scale=9,
                        autofocus=True,
                        submit_btn=True,
                    )

            with gr.TabItem("利用可能なツール"):
                # ツール一覧表示エリア
                tools_display = gr.Markdown(
                    "「ツール一覧を更新」ボタンをクリックして、利用可能なツールを表示してください。",
                    height=600,
                )

                # ツール一覧更新ボタン
                refresh_tools_btn = gr.Button("🔄 ツール一覧を更新", variant="primary")

                def update_tools_display() -> str:
                    """ツール一覧を更新する関数"""
                    try:
                        tools_info = sync_get_available_tools_local()
                        return tools_info
                    except Exception as e:
                        return f"ツール一覧の取得中にエラーが発生しました:\n{str(e)}"

                refresh_tools_btn.click(update_tools_display, outputs=tools_display)

        def user_submit(user_input, history1, history2, function_calling) -> tuple:
            """
            ユーザー入力を両方のLLMに送信し、履歴を更新する
            """
            result = sync_dual_llm_chat(
                user_input, history1, history2, function_calling
            )
            return result

        # イベントハンドラーを設定
        txt.submit(
            user_submit,
            [txt, chatbot1, chatbot2, function_radio],
            [txt, chatbot1, txt, chatbot2],
        )

    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)


if __name__ == "__main__":
    asyncio.run(main())

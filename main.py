import os
import gradio as gr
import asyncio
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from langchain_mcp_utils import (
    extract_answer,
    get_llm_params,
    initialize_llm,
    load_server_params,
    sync_get_available_tools,
    extract_tool_history,
)

global_client = None
global_tools = []
llm_options = {}


# Gradio用の非同期チャット関数
async def gradio_chat(
    user_input, history, function_calling, selected_llm, system_prompt=""
) -> str:
    """
    GradioのチャットUIから呼ばれる非同期チャット関数。
    Args:
        user_input (str): ユーザーの入力テキスト
        history (list): チャット履歴
        function_calling (str): ツール呼び出し有効/無効
        selected_llm (str): 選択されたLLM名
        system_prompt (str): システムプロンプト
    Returns:
        str: チャット応答
    """
    # 選択されたLLMでエージェントを初期化
    print("selected_llm:", selected_llm)
    # llm_optionsから、selected_llmに対応する設定を取得
    llm_config = llm_options.get(selected_llm, {})
    model_name = llm_config.get("model", "gpt-4o")
    base_url = llm_config.get("base_url", "")
    print("model_name:", model_name)

    current_llm = initialize_llm(model_name, base_url)
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
    # ツール履歴抽出
    tool_history = extract_tool_history(agent_response)
    if tool_history:
        answer += "\n\n[呼び出されたツール履歴]\n" + "\n".join(tool_history)
    return answer


def sync_gradio_chat(
    user_input, history, function_calling, selected_llm, system_prompt=""
) -> str:
    """
    非同期gradio_chat関数を同期的に呼び出すラッパー。
    Args:
        user_input (str): ユーザーの入力テキスト
        history (list): チャット履歴
        function_calling (str): ツール呼び出し有効/無効
        selected_llm (str): 選択されたLLM名
        system_prompt (str): システムプロンプト
    Returns:
        str: エージェントの回答
    """

    return asyncio.run(
        gradio_chat(user_input, history, function_calling, selected_llm, system_prompt)
    )


async def main() -> None:
    """
    メイン関数。Gradioアプリケーションを起動し、LLMとツールを初期化する。
    """

    # 設定ファイルからサーバーパラメータを読み込む
    params_file_name = "server_params.json"
    params = load_server_params(params_file_name)

    # paramsが空の辞書の場合(＝設定ファイルが存在しない場合)、エラーで終了
    if not params:
        print("設定ファイル({})が見つからないか、無効です。".format(params_file_name))
        return

    # paramsから必要な情報を取得
    global llm_options
    model_name, base_url, llm_options, default_llm, available_llms = get_llm_params(
        params
    )

    # 初期LLMの設定
    llm = initialize_llm(llm_name=model_name, base_url=base_url)

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
            padding: 10px !important;
            margin: 0 !important;
        }
        .chat-container {
            flex: 1 !important;
            height: calc(100vh - 350px) !important;
            min-height: calc(100vh - 350px) !important;
            max-height: calc(100vh - 350px) !important;
            overflow: auto !important;
        }
        .controls-container {
            flex-shrink: 0 !important;
            padding: 15px !important;
            background: #f8f9fa !important;
            border-top: 1px solid #ddd !important;
            border-bottom: 1px solid #ddd !important;
            min-height: 80px !important;
        }
        .input-container {
            flex-shrink: 0 !important;
            padding: 3px !important;
            margin: 0 !important;
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            background: white !important;
            border-top: 1px solid #ddd !important;
            display: flex !important;
            gap: 3px !important;
            align-items: center !important;
        }
        /* 
        .input-container .textbox {
            flex: 1 !important;
            margin: 0 !important;
        }
        .input-container .button {
            flex-shrink: 0 !important;
            width: 60px !important;
            height: 40px !important;
            margin: 0 !important;
        }
        .function-radio {
            position: fixed !important;
            bottom: 60px !important;
            left: 0 !important;
            right: 0 !important;
            border-top: 1px solid #eee !important;
            padding: 10px 20px !important;
            z-index: 10 !important;
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            height: 40px !important;
        }
        */
        /* チャットボットの高さを固定 */
        .chatbot {
            height: calc(100vh - 250px) !important;
            min-height: calc(100vh - 250px) !important;
            max-height: calc(100vh - 250px) !important;
        }
        /* プルダウンとラジオボタンの表示を確保
        .dropdown, .radio {
            min-height: 40px !important;
            z-index: 1000 !important;
        }*/
        """,
    ) as demo:
        gr.Markdown("# LangChain MCP チャット", elem_classes=["title"])

        # タブで機能を分ける
        with gr.Tabs():
            with gr.TabItem("チャット"):
                # チャットボット（画面いっぱいに表示）
                chatbot = gr.Chatbot(
                    type="messages",
                    height="calc(100vh - 250px)",
                    elem_classes=["chat-container"],
                    container=True,
                    autoscroll=True,
                    show_copy_all_button=True,
                    show_copy_button=True,
                    resizable=True,
                )

                # functionCallingラジオボタンとLLM選択プルダウン
                with gr.Row(elem_classes=["controls-container"]):
                    with gr.Column(scale=1):
                        gr.Markdown("**ツール呼び出し（Function Calling）:**")
                        function_radio = gr.Radio(
                            ["有効", "無効"],
                            value="有効",
                            label=None,
                            container=False,
                            elem_classes=["radio"],
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("**使用するLLM:**")
                        llm_dropdown = gr.Dropdown(
                            choices=available_llms,
                            value=default_llm,
                            label=None,
                            container=False,
                            elem_classes=["dropdown"],
                        )

                # 入力フォーム
                with gr.Row(elem_classes=["input-container"]):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="メッセージを入力してください...",
                        container=False,
                        scale=9,
                        autofocus=True,
                        submit_btn=True,
                    )
                    send_btn = gr.Button("📤", size="sm", variant="primary", scale=1)

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
                        tools_info = sync_get_available_tools(global_tools)
                        return tools_info
                    except Exception as e:
                        return f"ツール一覧の取得中にエラーが発生しました:\n{str(e)}"

                refresh_tools_btn.click(update_tools_display, outputs=tools_display)

        def user_submit(user_input, history, function_calling, selected_llm) -> tuple:
            """
            Gradioの送信イベントから呼ばれるコールバック関数。
            ユーザー入力・履歴・functionCalling有無・選択されたLLMを受け取り、チャット履歴を更新する。
            Args:
                user_input (str): ユーザーの入力テキスト
                history (list): チャット履歴
                function_calling (str): ツール呼び出し有効/無効
                selected_llm (str): 選択されたLLM名
            Returns:
                tuple: (空文字, 更新後履歴)
            """
            if not user_input.strip():
                return "", history
            # 内部でシステムプロンプトを設定
            system_prompt = """
            あなたは親切で知識豊富なAIアシスタントです。ユーザーの質問に対して、正確で分かりやすい回答を提供してください。
            なお、回答にあたり、以下のルールを守ってください。
            1. 回答は日本語で行ってください。
            2. 回答は簡潔で明確にしてください。
            3. ツール呼び出しが有効な場合は、ツールを利用してください。
            4. ユーザーの意図を理解しかねる場合は、追加の情報を求めてください。
            """
            response = sync_gradio_chat(
                user_input, history, function_calling, selected_llm, system_prompt
            )
            # messages形式に変換
            new_history = history + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response},
            ]
            return "", new_history

        txt.submit(
            user_submit, [txt, chatbot, function_radio, llm_dropdown], [txt, chatbot]
        )
        send_btn.click(
            user_submit, [txt, chatbot, function_radio, llm_dropdown], [txt, chatbot]
        )

    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    asyncio.run(main())

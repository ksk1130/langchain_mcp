# Create server parameters for stdio connection

import os
import gradio as gr
import asyncio
import json
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

global_client = None
global_tools = []


async def main() -> None:
    # エージェント応答から回答テキストを抽出する関数
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
            if "messages" in resp:
                messages = resp["messages"]
                if isinstance(messages, list):
                    for msg in reversed(messages):
                        if (
                            hasattr(msg, "content")
                            and hasattr(msg, "__class__")
                            and "AIMessage" in str(type(msg))
                        ):
                            return msg.content
                        elif isinstance(msg, dict) and msg.get("type") == "ai":
                            return msg.get("content", "")
            for key in ["output", "answer", "content", "result", "text"]:
                if key in resp:
                    return extract_answer(resp[key])
            if resp.get("type") == "ai":
                return resp.get("content", "")
            return str(resp)
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

    # server_params.jsonからサーバー設定を読み込み
    with open("server_params.json", encoding="utf-8") as f:
        params = json.load(f)

    # 利用可能なLLMオプションを取得
    llm_options = params.get("llm", {})
    available_llms = list(llm_options.keys()) if llm_options else ["Default"]

    # 2つのLLMを取得（最初の2つ、または同じものを2回）
    llm1_name = available_llms[0] if len(available_llms) >= 1 else "Default"
    llm2_name = available_llms[1] if len(available_llms) >= 2 else available_llms[0]

    # LLMを初期化する関数
    def initialize_llm(llm_name: str) -> ChatOpenAI:
        """選択されたLLMに基づいてChatOpenAIインスタンスを初期化"""
        if llm_name in llm_options:
            llm_config = llm_options[llm_name]
            if isinstance(llm_config, dict):
                # 新しい形式: {"model": "...", "base_url": "..."}
                model = llm_config.get("model", "gpt-4o")
                base_url = llm_config.get("base_url")
                if base_url:
                    return ChatOpenAI(model=model, base_url=base_url)
                else:
                    return ChatOpenAI(model=model)
            else:
                # 古い形式: 文字列のbase_url
                return ChatOpenAI(model="gpt-4o", base_url=llm_config)
        else:
            # デフォルトまたは不明なLLMの場合
            base_url = params.get("base_url")
            if base_url:
                return ChatOpenAI(model="gpt-4o", base_url=base_url)
            else:
                return ChatOpenAI(model="gpt-4o")

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
        print(f"初期化エラー: {e}")
        global_client = None
        global_tools = []

    # 単一LLM用の非同期チャット関数
    async def single_llm_chat(user_input, history, function_calling, llm_name) -> str:
        """
        単一のLLMに対してチャットを実行する関数
        """
        try:
            # 選択されたLLMでエージェントを初期化
            current_llm = initialize_llm(llm_name)

            # Gradioの履歴(messages形式)をLangChainの履歴に変換
            messages = []
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
            tool_history = []
            if isinstance(agent_response, dict):
                if "messages" in agent_response:
                    messages_resp = agent_response["messages"]
                    if isinstance(messages_resp, list):
                        for msg in messages_resp:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_name = (
                                        tool_call.get("name")
                                        if isinstance(tool_call, dict)
                                        else getattr(tool_call, "name", str(tool_call))
                                    )
                                    tool_args = (
                                        tool_call.get("args")
                                        if isinstance(tool_call, dict)
                                        else getattr(tool_call, "args", {})
                                    )
                                    tool_history.append(
                                        f"ツール名: {tool_name}, 引数: {tool_args}"
                                    )
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
        """
        if not user_input.strip():
            return "", history1, "", history2

        # 両方のLLMに同時にリクエストを送信
        tasks = [
            single_llm_chat(user_input, history1, function_calling, llm1_name),
            single_llm_chat(user_input, history2, function_calling, llm2_name),
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
        """
        return asyncio.run(
            dual_llm_chat(user_input, history1, history2, function_calling)
        )

    # 利用可能なツール一覧を取得する関数
    async def get_available_tools() -> str:
        """
        初期化済みのグローバルツールから利用可能なツール一覧を取得
        """
        try:
            result = "# 利用可能なツール一覧\n\n"
            result += f"**合計ツール数**: {len(global_tools)}\n\n"

            if global_tools:
                result += "## ツール詳細:\n"
                for i, tool in enumerate(global_tools, 1):
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

    def sync_get_available_tools() -> str:
        """利用可能なツール取得の同期ラッパー"""
        try:
            return asyncio.run(asyncio.wait_for(get_available_tools(), timeout=30.0))
        except asyncio.TimeoutError:
            return "ツール一覧の取得がタイムアウトしました。サーバーの接続を確認してください。"
        except Exception as e:
            return f"ツール一覧の取得中にエラーが発生しました: {type(e).__name__}: {str(e)}"

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
                        tools_info = sync_get_available_tools()
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

    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    asyncio.run(main())

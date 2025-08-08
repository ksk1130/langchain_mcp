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

async def main():
    # エージェント応答から回答テキストを抽出する関数
    def extract_answer(resp):
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
    
    # LLMの初期化
    base_url = params.get("base_url")
    if base_url:
        llm = ChatOpenAI(model="gpt-4.1", base_url=base_url)
    else:
        llm = ChatOpenAI(model="gpt-4.1")

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

    # 利用可能なツール一覧を取得する関数
    async def get_available_tools():
        """
        初期化済みのグローバルツールから利用可能なツール一覧を取得
        Returns:
            str: ツール一覧の文字列
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

    def sync_get_available_tools():
        """利用可能なツール取得の同期ラッパー"""
        try:
            # 30秒のタイムアウトを設定
            return asyncio.run(asyncio.wait_for(get_available_tools(), timeout=30.0))
        except asyncio.TimeoutError:
            return "ツール一覧の取得がタイムアウトしました。サーバーの接続を確認してください。"
        except Exception as e:
            return f"ツール一覧の取得中にエラーが発生しました: {type(e).__name__}: {str(e)}"

    # Gradio用の非同期チャット関数
    async def gradio_chat(user_input, history, function_calling):
        """
        GradioのチャットUIから呼ばれる非同期チャット関数。
        初期化済みのグローバルツールを使用する。
        """
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

        agent = create_react_agent(llm, agent_tools, debug=True)
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
            if "tool_calls" in agent_response:
                calls = agent_response["tool_calls"]
                if calls:
                    for call in calls:
                        tool_history.append(
                            f"ツール名: {call.get('tool_name', call.get('name', 'Unknown'))}, 入力: {call.get('input', call.get('args', {}))}"
                        )
        if tool_history:
            answer += "\n\n[呼び出されたツール履歴]\n" + "\n".join(tool_history)
        return answer

    def sync_gradio_chat(user_input, history, function_calling):
        """
        非同期gradio_chat関数を同期的に呼び出すラッパー。
        Args:
            user_input (str): ユーザーの入力テキスト
            history (list): チャット履歴
            function_calling (str): ツール呼び出し有効/無効
        Returns:
            str: エージェントの回答
        """
        return asyncio.run(gradio_chat(user_input, history, function_calling))

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
            height: calc(100vh - 220px) !important;
            min-height: calc(100vh - 220px) !important;
            max-height: calc(100vh - 220px) !important;
            overflow: auto !important;
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
            height: calc(100vh - 180px) !important;
            min-height: calc(100vh - 180px) !important;
            max-height: calc(100vh - 180px) !important;
        }
        """
    ) as demo:
        gr.Markdown("# LangChain MCP チャット", elem_classes=["title"])

        # タブで機能を分ける
        with gr.Tabs():
            with gr.TabItem("チャット"):
                # チャットボット（画面いっぱいに表示）
                chatbot = gr.Chatbot(
                    type="messages",
                    height="calc(100vh - 300px)",
                    elem_classes=["chat-container"],
                    container=True,
                )

                # functionCallingラジオボタン
                with gr.Row():
                    gr.Markdown("ツール呼び出し（Function Calling）を有効にするか選択してください：")
                    function_radio = gr.Radio(
                        ["有効", "無効"],
                        value="有効",
                        label=None,
                        container=False
                    )

                # 入力フォーム
                with gr.Row():
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="メッセージを入力してください...",
                        container=False,
                        scale=9
                    )
                    send_btn = gr.Button(
                        "📤", size="sm", variant="primary", scale=1
                    )

            with gr.TabItem("利用可能なツール"):
                # ツール一覧表示エリア
                tools_display = gr.Markdown(
                    "「ツール一覧を更新」ボタンをクリックして、利用可能なツールを表示してください。",
                    height=600
                )
                
                # ツール一覧更新ボタン
                refresh_tools_btn = gr.Button("🔄 ツール一覧を更新", variant="primary")
                
                def update_tools_display():
                    """ツール一覧を更新する関数"""
                    try:
                        tools_info = sync_get_available_tools()
                        return tools_info
                    except Exception as e:
                        return f"ツール一覧の取得中にエラーが発生しました:\n{str(e)}"
                
                refresh_tools_btn.click(
                    update_tools_display,
                    outputs=tools_display
                )

        def user_submit(user_input, history, function_calling):
            """
            Gradioの送信イベントから呼ばれるコールバック関数。
            ユーザー入力・履歴・functionCalling有無を受け取り、チャット履歴を更新する。
            Args:
                user_input (str): ユーザーの入力テキスト
                history (list): チャット履歴
                function_calling (str): ツール呼び出し有効/無効
            Returns:
                tuple: (空文字, 更新後履歴)
            """
            if not user_input.strip():
                return "", history
            response = sync_gradio_chat(user_input, history, function_calling)
            # messages形式に変換
            new_history = history + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response},
            ]
            return "", new_history

        txt.submit(user_submit, [txt, chatbot, function_radio], [txt, chatbot])
        send_btn.click(user_submit, [txt, chatbot, function_radio], [txt, chatbot])

    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    asyncio.run(main())

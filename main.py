# Create server parameters for stdio connection

import os
import gradio as gr
import asyncio
import json
from langchain_openai import ChatOpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent


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

    # 利用可能なツール一覧を取得する関数
    async def get_available_tools():
        """
        全サーバから利用可能なツール一覧を取得して整理する関数
        Returns:
            str: ツール一覧の文字列
        """
        all_tools_info = []
        
        for name, conf in params.get("servers", {}).items():
            try:
                tools = []
                print(f"サーバー {name} に接続中... (タイプ: {conf.get('type')})")
                
                # 各サーバ接続に個別タイムアウトを設定
                async def connect_to_server():
                    if conf.get("type") == "sse":
                        async with sse_client(conf["url"]) as (read, write):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                return await load_mcp_tools(session)
                    elif conf.get("type") == "http":
                        async with streamablehttp_client(conf["url"]) as (read, write):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                return await load_mcp_tools(session)
                    elif conf.get("type") == "stdio":
                        from mcp import StdioServerParameters
                        server_params = StdioServerParameters(
                            command=conf["command"], 
                            args=conf["args"]
                        )
                        async with stdio_client(server_params) as (read, write):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                return await load_mcp_tools(session)
                    return []
                
                # 10秒のタイムアウトで接続を試行
                tools = await asyncio.wait_for(connect_to_server(), timeout=10.0)
                print(f"サーバー {name} から {len(tools)} 個のツールを取得しました")
                
                # ツール情報を整理
                server_tools = []
                for tool in tools:
                    tool_info = {
                        "name": getattr(tool, "name", "Unknown"),
                        "description": getattr(tool, "description", "説明なし"),
                        "args": getattr(tool, "args_schema", {})
                    }
                    server_tools.append(tool_info)
                
                all_tools_info.append({
                    "server": name,
                    "type": conf.get("type", "unknown"),
                    "url": conf.get("url", conf.get("command", "N/A")),
                    "tools": server_tools
                })
                
            except asyncio.TimeoutError:
                error_msg = f"接続タイムアウト: サーバー {name} への接続がタイムアウトしました"
                print(error_msg)
                all_tools_info.append({
                    "server": name,
                    "type": conf.get("type", "unknown"),
                    "url": conf.get("url", conf.get("command", "N/A")),
                    "error": error_msg,
                    "tools": []
                })
            except ConnectionError as e:
                error_msg = f"接続エラー: {str(e)}"
                print(f"サーバー {name}: {error_msg}")
                all_tools_info.append({
                    "server": name,
                    "type": conf.get("type", "unknown"),
                    "url": conf.get("url", conf.get("command", "N/A")),
                    "error": error_msg,
                    "tools": []
                })
            except Exception as e:
                # ExceptionGroupや他の例外をまとめて処理
                error_type = type(e).__name__
                if "ExceptionGroup" in error_type or "TaskGroup" in str(e):
                    error_msg = f"TaskGroup/ExceptionGroup エラー: サーバー接続中に内部エラーが発生しました"
                else:
                    error_msg = f"予期しないエラー: {error_type}: {str(e)}"
                
                print(f"サーバー {name}: {error_msg}")
                all_tools_info.append({
                    "server": name,
                    "type": conf.get("type", "unknown"),
                    "url": conf.get("url", conf.get("command", "N/A")),
                    "error": error_msg,
                    "tools": []
                })
        
        # ツール一覧を文字列として整理
        result = "# 利用可能なツール一覧\n\n"
        total_tools = 0
        
        for server_info in all_tools_info:
            result += f"## サーバー: {server_info['server']}\n"
            result += f"- **接続タイプ**: {server_info['type']}\n"
            result += f"- **URL/コマンド**: {server_info['url']}\n"
            
            if "error" in server_info:
                result += f"- **エラー**: {server_info['error']}\n\n"
                continue
            
            result += f"- **ツール数**: {len(server_info['tools'])}\n\n"
            total_tools += len(server_info['tools'])
            
            if server_info['tools']:
                result += "### ツール詳細:\n"
                for i, tool in enumerate(server_info['tools'], 1):
                    result += f"{i}. **{tool['name']}**\n"
                    result += f"   - 説明: {tool['description']}\n"
                    if tool['args']:
                        result += f"   - 引数: {tool['args']}\n"
                    result += "\n"
            else:
                result += "ツールが見つかりませんでした。\n\n"
        
        result += f"**合計ツール数**: {total_tools}\n"
        return result

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
        ユーザー入力・履歴・functionCalling有無を受け取り、エージェント応答を返す。
        Args:
            user_input (str): ユーザーの入力テキスト
            history (list): チャット履歴
            function_calling (str): ツール呼び出し有効/無効
        Returns:
            str: エージェントの回答（ツール履歴含む場合あり）
        """
        # Gradioの履歴(messages形式)をLangChainの履歴に変換
        messages = []
        if history:
            for msg in history:
                if msg.get("role") == "user":
                    messages.append({"type": "human", "content": msg["content"]})
                elif msg.get("role") == "assistant":
                    messages.append({"type": "ai", "content": msg["content"]})
        # 今回のユーザー入力を追加
        messages.append({"type": "human", "content": user_input})

        # 各サーバからツールを取得して統合
        all_tools = []
        for name, conf in params.get("servers", {}).items():
            try:
                # 各サーバ接続に個別タイムアウトを設定
                async def connect_and_get_tools():
                    if conf.get("type") == "sse":
                        async with sse_client(conf["url"]) as (read, write):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                return await load_mcp_tools(session)
                    elif conf.get("type") == "http":
                        async with streamablehttp_client(conf["url"]) as (read, write):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                return await load_mcp_tools(session)
                    elif conf.get("type") == "stdio":
                        from mcp import StdioServerParameters
                        server_params = StdioServerParameters(
                            command=conf["command"], 
                            args=conf["args"]
                        )
                        async with stdio_client(server_params) as (read, write):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                return await load_mcp_tools(session)
                    return []
                
                # 5秒のタイムアウトで接続を試行（チャット時は短めに）
                tools = await asyncio.wait_for(connect_and_get_tools(), timeout=5.0)
                all_tools.extend(tools)
                
            except asyncio.TimeoutError:
                print(f"タイムアウト: サーバー {name} への接続がタイムアウトしました")
                continue
            except ConnectionError as e:
                print(f"接続エラー: サーバー {name} - {str(e)}")
                continue
            except Exception as e:
                error_type = type(e).__name__
                if "ExceptionGroup" in error_type or "TaskGroup" in str(e):
                    print(f"TaskGroup/ExceptionGroup エラー: サーバー {name} - 内部エラーが発生しました")
                else:
                    print(f"サーバー {name} からツール取得中にエラー: {error_type}: {str(e)}")
                continue
        
        agent_tools = all_tools if function_calling == "有効" else []
        agent = create_react_agent(llm, agent_tools)
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

    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    asyncio.run(main())

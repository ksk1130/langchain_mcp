# Create server parameters for stdio connection

import os
import gradio as gr
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import json


async def main():
    # エージェント応答から回答テキストを抽出する関数
    def extract_answer(resp):
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

    with open("server_params.json", encoding="utf-8") as f:
        params = json.load(f)
    server_params = StdioServerParameters(
        command=params["command"], args=params["args"]
    )

    # Gradio用の非同期チャット関数
    async def gradio_chat(user_input, history, function_calling):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                # functionCallingの有無でツールリストを切り替え
                agent_tools = tools if function_calling == "有効" else []
                agent = create_react_agent(model, agent_tools)
                agent_response = await agent.ainvoke({"messages": user_input})
                answer = extract_answer(agent_response)
                # ツール履歴抽出
                tool_history = []
                if isinstance(agent_response, dict):
                    if "messages" in agent_response:
                        messages = agent_response["messages"]
                        if isinstance(messages, list):
                            for msg in messages:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tool_call in msg.tool_calls:
                                        tool_name = (
                                            tool_call.get("name")
                                            if isinstance(tool_call, dict)
                                            else getattr(
                                                tool_call, "name", str(tool_call)
                                            )
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
                # 履歴表示
                if tool_history:
                    answer += "\n\n[呼び出されたツール履歴]\n" + "\n".join(tool_history)
                return answer

    def sync_gradio_chat(user_input, history, function_calling):
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

        # チャットボット（画面いっぱいに表示）
        chatbot = gr.Chatbot(
            type="messages",
            height="calc(100vh - 220px)",
            elem_classes=["chat-container"],
            container=True,
        )

        # functionCallingラジオボタン（下端の少し上に固定）
        with gr.Row(elem_classes=["function-radio"]):
            gr.Markdown("<span style='font-size:16px;'>ツール呼び出し（Function Calling）を有効にするか選択してください：</span>", show_label=False)
            function_radio = gr.Radio(
                ["有効", "無効"],
                value="有効",
                label=None,
                container=False
            )

        # 入力フォーム（下端に固定）
        with gr.Row(elem_classes=["input-container"]):
            txt = gr.Textbox(
                show_label=False,
                placeholder="メッセージを入力してください...",
                container=False,
                elem_classes=["textbox"],
            )
            send_btn = gr.Button(
                "📤", size="sm", variant="primary", elem_classes=["button"]
            )

        def user_submit(user_input, history, function_calling):
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

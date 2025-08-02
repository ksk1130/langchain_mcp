# Create server parameters for stdio connection
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio


async def main():
    server_params = StdioServerParameters(
        command="uvx",
        # Make sure to update to the full absolute path to your math_server.py file
        args=[
            "--from",
            "awslabs-aws-documentation-mcp-server",
            "awslabs.aws-documentation-mcp-server.exe",
        ],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)
            print("利用可能なツール一覧:")
            for tool in tools:
                print(f"- {getattr(tool, 'name', str(tool))}")

            # モデルを準備
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
            )

            # Create and run the agent
            agent = create_react_agent(model, tools)

            # ツール呼び出しの状況を追跡するためのコールバック関数を定義
            def tool_callback(event):
                # eventにはtool_name, input, outputなどが含まれる想定
                print("\n[ツール呼び出し]")
                print(f"ツール名: {event.get('tool_name')}")
                print(f"入力: {event.get('input')}")
                print(f"出力: {event.get('output')}")
                # ツール呼び出し履歴を保存
                tool_calls.append({
                    'tool_name': event.get('tool_name'),
                    'input': event.get('input'),
                    'output': event.get('output')
                })

            import json

            def extract_answer(resp):
                # ...existing code...
                if isinstance(resp, dict):
                    # ...existing code...
                    if "messages" in resp:
                        messages = resp["messages"]
                        if isinstance(messages, list):
                            for msg in reversed(messages):
                                if hasattr(msg, 'content') and hasattr(msg, '__class__') and 'AIMessage' in str(type(msg)):
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
                        if hasattr(item, 'content') and hasattr(item, '__class__') and 'AIMessage' in str(type(item)):
                            return item.content
                        elif isinstance(item, dict) and item.get("type") == "ai":
                            return item.get("content", "")
                    answers = [extract_answer(item) for item in resp]
                    return "\n---\n".join(str(a) for a in answers if a)
                else:
                    if hasattr(resp, 'content'):
                        return resp.content
                    try:
                        return str(resp)
                    except Exception:
                        return f"<{type(resp).__name__}>"

            # ツール呼び出し履歴を保存するリスト
            tool_calls = []

            # CLIチャットループ
            print("\n[CLIチャットモード] 終了するには Ctrl+C または Ctrl+D で終了してください。\n")
            while True:
                try:
                    user_input = input("あなた: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n終了します。")
                    break
                if not user_input:
                    continue
                # エージェントに渡す（callbacksは渡さない）
                agent_response = await agent.ainvoke({"messages": user_input})
                
                # デバッグ: エージェント応答の構造を確認
                print(f"\n[デバッグ] エージェント応答の型: {type(agent_response)}")
                if isinstance(agent_response, dict):
                    print(f"[デバッグ] キー: {list(agent_response.keys())}")
                
                answer = extract_answer(agent_response)
                print("AI: " + (answer if answer and answer.strip() and not answer.strip().startswith('<') else "[回答が見つかりませんでした]"))
                
                # agent_responseからツール呼び出し履歴を抽出して表示
                tool_calls_found = False
                if isinstance(agent_response, dict):
                    # messagesの中を確認
                    if "messages" in agent_response:
                        messages = agent_response["messages"]
                        if isinstance(messages, list):
                            for msg in messages:
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    tool_calls_found = True
                                    print("\n[呼び出されたツール履歴]")
                                    for tool_call in msg.tool_calls:
                                        tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', str(tool_call))
                                        tool_args = tool_call.get('args') if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                                        print(f"- ツール名: {tool_name}")
                                        print(f"  引数: {tool_args}")
                                elif hasattr(msg, 'content') and msg.content and hasattr(msg, 'tool_calls'):
                                    # AIMessageの場合
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                        tool_calls_found = True
                                        print("\n[呼び出されたツール履歴]")
                                        for tool_call in msg.tool_calls:
                                            print(f"- ツール名: {tool_call.get('name', 'Unknown')}")
                                            print(f"  引数: {tool_call.get('args', {})}")
                    
                    # 直接tool_callsキーがある場合
                    if "tool_calls" in agent_response:
                        calls = agent_response["tool_calls"]
                        if calls:
                            tool_calls_found = True
                            print("\n[呼び出されたツール履歴]")
                            for call in calls:
                                print(f"- ツール名: {call.get('tool_name', call.get('name', 'Unknown'))}")
                                print(f"  入力: {call.get('input', call.get('args', {}))}")
                
                if not tool_calls_found:
                    print("\n[デバッグ] ツール呼び出し履歴が見つかりませんでした")


if __name__ == "__main__":
    asyncio.run(main())

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

            # agent.ainvokeにcallbackを渡せる場合
            try:
                agent_response = await agent.ainvoke(
                    {
                        "messages": "AWSのドキュメントを検索して、プロキシ環境でCloudWatch Agentを実行する方法を教えてください"
                    },
                    callbacks=[tool_callback],
                )
            except TypeError:
                # コールバックが渡せない場合は通常通り実行
                agent_response = await agent.ainvoke(
                    {
                        "messages": "AWSのドキュメントを検索して、プロキシ環境でCloudWatch Agentを実行する方法を教えてください"
                    }
                )

            print("\n[エージェント応答]")
            import json

            def extract_answer(resp):
                # dictの場合の処理
                if isinstance(resp, dict):
                    # messagesキーがある場合（エージェントの応答形式）
                    if "messages" in resp:
                        messages = resp["messages"]
                        if isinstance(messages, list):
                            # 最後のAIMessageのcontentを取得
                            for msg in reversed(messages):
                                if hasattr(msg, 'content') and hasattr(msg, '__class__') and 'AIMessage' in str(type(msg)):
                                    return msg.content
                                elif isinstance(msg, dict) and msg.get("type") == "ai":
                                    return msg.get("content", "")
                    
                    # 通常のdict処理
                    for key in ["output", "answer", "content", "result", "text"]:
                        if key in resp:
                            return extract_answer(resp[key])
                    
                    # langchainのMessage系（HumanMessage, AIMessageなど）
                    if resp.get("type") == "ai":
                        return resp.get("content", "")
                    
                    # それ以外は全体をstr化
                    return str(resp)
                
                elif isinstance(resp, list):
                    # listの場合、AIMessageのcontentを探す
                    for item in reversed(resp):
                        if hasattr(item, 'content') and hasattr(item, '__class__') and 'AIMessage' in str(type(item)):
                            return item.content
                        elif isinstance(item, dict) and item.get("type") == "ai":
                            return item.get("content", "")
                    
                    # AIMessageが見つからない場合は従来の処理
                    answers = [extract_answer(item) for item in resp]
                    return "\n---\n".join(str(a) for a in answers if a)
                
                else:
                    # オブジェクトの場合、contentアトリビュートがあるかチェック
                    if hasattr(resp, 'content'):
                        return resp.content
                    
                    # シリアライズできないオブジェクトは型名だけ返す
                    try:
                        return str(resp)
                    except Exception:
                        return f"<{type(resp).__name__}>"

            answer = extract_answer(agent_response)
            if answer and answer.strip() and not answer.strip().startswith('<'):
                print(answer)
            else:
                print("[回答が見つかりませんでした]", agent_response)


if __name__ == "__main__":
    asyncio.run(main())

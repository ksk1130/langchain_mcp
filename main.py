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
            
            # モデルを準備
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
            )

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke(
                {
                    "messages": "AWSのドキュメントを検索して、プロキシ環境でCloudWatch Agentを実行する方法を教えてください"
                }
            )

            print(agent_response)


if __name__ == "__main__":
    asyncio.run(main())

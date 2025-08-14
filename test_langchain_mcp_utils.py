import asyncio
import langchain_mcp_utils
import pytest


def test_extract_answer_dict():
    """
    extract_answerがdict型入力で正しく値を抽出できるかをテスト。
    """
    resp = {"output": "テスト回答"}
    assert langchain_mcp_utils.extract_answer(resp) == "テスト回答"


def test_extract_answer_list():
    """
    extract_answerがlist型入力で複数要素から正しく値を抽出できるかをテスト。
    """
    resp = [
        {"type": "ai", "content": "リスト回答1"},
        {"type": "ai", "content": "リスト回答2"},
    ]
    result = langchain_mcp_utils.extract_answer(resp)
    # extract_answerはリストの最後の要素のみ返す仕様なので、リスト回答2のみを検証
    assert result == "リスト回答2"


def test_extract_answer_str():
    """
    extract_answerが単純な文字列入力でそのまま返すかをテスト。
    """
    resp = "単純な文字列"
    result = langchain_mcp_utils.extract_answer(resp)
    assert result == "単純な文字列"


def test_load_server_params():
    """
    load_server_paramsがserver_params.jsonの内容を正しく辞書で返すかをテスト。
    """
    params = langchain_mcp_utils.load_server_params("server_params.json")
    assert isinstance(params, dict)
    assert "servers" in params
    assert "llm" in params


def test_load_server_params_file_not_found(tmp_path, monkeypatch):
    """
    server_params.jsonが存在しない場合にFileNotFoundErrorが発生することをテスト。
    """
    monkeypatch.chdir(tmp_path)
    result = langchain_mcp_utils.load_server_params("server_params.json")
    assert result == {}


def test_load_server_params_invalid_json(tmp_path, monkeypatch):
    """
    server_params.jsonが壊れている場合にJSONDecodeErrorが発生することをテスト。
    """
    file = tmp_path / "server_params.json"
    file.write_text("{ invalid json }", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = langchain_mcp_utils.load_server_params(str(file))
    assert result == {}


def test_get_llm_params():
    """
    get_llm_paramsがparamsから正しい(model_name, base_url, llm_options, default_llm, available_llms)を返すかをテスト。
    """
    params = {
        "llm": {
            "OpenAI": {"model": "gpt-4.1", "base_url": "http://127.0.0.1:4000"},
            "Gemini": {"model": "gemini-pro", "base_url": "http://localhost:5000"},
        }
    }
    model_name, base_url, llm_options, default_llm, available_llms = (
        langchain_mcp_utils.get_llm_params(params)
    )
    assert model_name == "gpt-4.1"
    assert base_url == "http://127.0.0.1:4000"
    assert llm_options == params["llm"]
    assert default_llm == "OpenAI"
    assert available_llms == ["OpenAI", "Gemini"]


def test_initialize_llm():
    """
    initialize_llm関数が正しくChatOpenAIインスタンスを作成するかをテスト。
    """
    # base_urlを指定しない場合
    llm = langchain_mcp_utils.initialize_llm("gpt-4o", "")
    assert llm.model_name == "gpt-4o"
    assert llm.openai_api_base is None
    
    # base_urlを指定する場合
    llm = langchain_mcp_utils.initialize_llm("gpt-3.5-turbo", "http://localhost:8000")
    assert llm.model_name == "gpt-3.5-turbo"
    assert llm.openai_api_base == "http://localhost:8000"


def test_load_server_params_file_not_found(tmp_path, monkeypatch):
    """
    server_params.jsonが存在しない場合に空の辞書が返されることをテスト。
    """
    # 一時ディレクトリに移動してserver_params.jsonが存在しないことを確保
    monkeypatch.chdir(tmp_path)
    result = langchain_mcp_utils.load_server_params("server_params.json")
    assert result == {}

def test_get_available_tools():
    """
    get_available_toolsがツールリストから正しい文字列を返すかをテスト。
    """
    from langchain_mcp_utils import get_available_tools

    class DummyTool:
        def __init__(self, name, description, args_schema=None):
            self.name = name
            self.description = description
            self.args_schema = args_schema or {}

    tools = [
        DummyTool("ToolA", "説明A", {"arg1": "str"}),
        DummyTool("ToolB", "説明B"),
    ]
    result = asyncio.run(get_available_tools(tools))
    assert "ToolA" in result
    assert "ToolB" in result
    assert "説明A" in result
    assert "説明B" in result
    assert "合計ツール数: 2" in result or "合計ツール数**: 2" in result


def test_sync_get_available_tools(monkeypatch):
    """
    sync_get_available_toolsがglobal_toolsの内容に応じて正しい文字列を返すかをテスト。
    """
    from langchain_mcp_utils import sync_get_available_tools, get_available_tools

    # ダミーツール
    class DummyTool:
        def __init__(self, name, description, args_schema=None):
            self.name = name
            self.description = description
            self.args_schema = args_schema or {}

    # 正常系
    tools = [DummyTool("ToolA", "説明A"), DummyTool("ToolB", "説明B")]
    result = sync_get_available_tools(tools)
    assert "ToolA" in result and "ToolB" in result
    # 空リスト
    result = sync_get_available_tools([])
    assert (
        "利用可能なツールが見つかりませんでした" in result
        or "合計ツール数**: 0" in result
    )

    # 例外系
    class BadList:
        def __iter__(self):
            raise RuntimeError("dummy error")

    result = sync_get_available_tools(BadList())
    assert "ツール一覧の取得中にエラーが発生しました" in result


def test_get_available_tools_empty():
    """
    get_available_toolsが空リストで"利用可能なツールが見つかりませんでした"を返すかをテスト。
    """
    from langchain_mcp_utils import get_available_tools

    result = asyncio.run(get_available_tools([]))
    assert "利用可能なツールが見つかりませんでした" in result


def test_get_available_tools_exception():
    """
    get_available_toolsで例外が発生した場合にエラーメッセージが返るかをテスト。
    """
    from langchain_mcp_utils import get_available_tools

    class BadList:
        def __iter__(self):
            raise RuntimeError("dummy error")

    result = asyncio.run(get_available_tools(BadList()))
    assert "ツール一覧の取得中にエラーが発生しました" in result


# gradio_chat, sync_gradio_chatのテスト
@pytest.mark.asyncio
async def test_gradio_chat(monkeypatch):
    """
    gradio_chatが正常に応答を返すかをテスト。
    """
    import main as mainmod

    # ChatOpenAI, create_react_agent, extract_answer, global_tools, llm_optionsをモック
    class DummyAgent:
        async def ainvoke(self, _):
            return {"output": "dummy answer", "messages": []}

    def dummy_create_react_agent(llm, tools, debug):
        return DummyAgent()

    monkeypatch.setattr(mainmod, "create_react_agent", dummy_create_react_agent)
    monkeypatch.setattr(mainmod, "extract_answer", lambda resp: resp["output"])
    monkeypatch.setattr(mainmod, "global_tools", [object()])
    monkeypatch.setattr(mainmod, "llm_options", {"TestLLM": {"model": "gpt-4o"}})
    # user_input, history, function_calling, selected_llm
    result = await mainmod.gradio_chat("テスト入力", [], "有効", "TestLLM")
    assert "dummy answer" in result


def test_sync_gradio_chat(monkeypatch):
    """
    sync_gradio_chatがgradio_chatを同期的に呼び出し、応答を返すかをテスト。
    """
    import main as mainmod

    async def dummy_gradio_chat(user_input, history, function_calling, selected_llm):
        return f"sync:{user_input}:{selected_llm}"

    monkeypatch.setattr(mainmod, "gradio_chat", dummy_gradio_chat)
    result = mainmod.sync_gradio_chat("hello", [], "有効", "TestLLM")
    assert result == "sync:hello:TestLLM"


@pytest.mark.asyncio
async def test_gradio_chat_exception(monkeypatch):
    """
    gradio_chatでagent.ainvokeが例外を投げた場合に例外が伝播するかをテスト。
    """
    import main as mainmod

    class DummyAgent:
        async def ainvoke(self, _):
            raise RuntimeError("dummy error")

    def dummy_create_react_agent(llm, tools, debug):
        return DummyAgent()

    monkeypatch.setattr(mainmod, "create_react_agent", dummy_create_react_agent)
    monkeypatch.setattr(mainmod, "extract_answer", lambda resp: "should not reach")
    monkeypatch.setattr(mainmod, "global_tools", [object()])
    monkeypatch.setattr(mainmod, "llm_options", {"TestLLM": {"model": "gpt-4o"}})
    with pytest.raises(RuntimeError):
        await mainmod.gradio_chat("テスト入力", [], "有効", "TestLLM")


def test_extract_tool_history_messages_langchain():
    """
    messagesフィールドからツール履歴を抽出するテスト（LangChain形式）。
    """
    # LangChainのAIMessage.tool_calls形式（function形式）
    Msg = lambda calls: type('Msg', (), {"tool_calls": calls})
    agent_response = {
        "messages": [
            Msg([
                {"function": {"name": "microsoft_docs_search", "arguments": '{"query":"Azure Active Directory set up"}'}},
                {"function": {"name": "calc", "arguments": '{"x": 1, "y": 2}'}},
            ]),
            Msg([]),
        ]
    }
    result = langchain_mcp_utils.extract_tool_history(agent_response)
    assert 'ツール名: microsoft_docs_search, 引数: {"query":"Azure Active Directory set up"}' in result
    assert 'ツール名: calc, 引数: {"x": 1, "y": 2}' in result
    assert len(result) == 2


def test_extract_tool_history_tool_calls():
    """
    tool_callsフィールドからツール履歴を抽出するテスト。
    """
    agent_response = {
        "tool_calls": [
            {"tool_name": "search", "input": {"q": "test"}},
            {"name": "calc", "args": {"x": 1, "y": 2}},
        ]
    }
    result = langchain_mcp_utils.extract_tool_history(agent_response)
    assert "ツール名: search, 入力: {'q': 'test'}" in result
    assert "ツール名: calc, 入力: {'x': 1, 'y': 2}" in result
    assert len(result) == 2


def test_extract_tool_history_empty():
    """
    tool_callsフィールドからツール履歴を抽出するテスト。
    """
    assert langchain_mcp_utils.extract_tool_history({}) == []
    assert langchain_mcp_utils.extract_tool_history({"messages": []}) == []
    assert langchain_mcp_utils.extract_tool_history({"tool_calls": []}) == []


def test_extract_tool_history_mixed():
    """
    messagesとtool_callsの両方からツール履歴を抽出するテスト。
    """
    Msg = lambda calls: type('Msg', (), {"tool_calls": calls})
    agent_response = {
        "messages": [
            Msg([
                {"function": {"name": "search", "arguments": '{"q": "test"}'}},
            ]),
        ],
        "tool_calls": [
            {"tool_name": "search", "input": {"q": "test2"}},
        ],
    }
    result = langchain_mcp_utils.extract_tool_history(agent_response)
    assert any('ツール名: search, 引数: {"q": "test"}' in s for s in result)
    assert any("ツール名: search, 入力: {'q': 'test2'}" in s for s in result)
    assert len(result) == 2

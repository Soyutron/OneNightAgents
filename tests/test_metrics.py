"""drop_past_reasoning: 過去の判断の思考だけを送信時に外す。"""

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from onenightagents.llm.metrics import drop_past_reasoning


async def test_strips_reasoning_before_last_human_only() -> None:
    history = [
        HumanMessage("夜です"),
        AIMessage(content="", additional_kwargs={"reasoning_content": "古い思考"}, tool_calls=[{"name": "seer_look", "args": {"target": "p2"}, "id": "c1"}]),
        ToolMessage(content="受理しました", tool_call_id="c1"),
        AIMessage(content="提出しました", additional_kwargs={"reasoning_content": "古い思考 2"}),
        HumanMessage("発言してください"),
        AIMessage(content="", additional_kwargs={"reasoning_content": "今の思考"}, tool_calls=[{"name": "speak", "args": {"content": "x"}, "id": "c2"}]),
        ToolMessage(content="受理しました", tool_call_id="c2"),
    ]
    seen: list = []

    async def handler(request):
        seen.extend(request.messages)
        return "ok"

    request = ModelRequest(model=None, system_prompt=None, messages=history, tool_choice=None, tools=[], response_format=None, state={}, runtime=None)
    assert await drop_past_reasoning.awrap_model_call(request, handler) == "ok"
    reasoning = [m.additional_kwargs.get("reasoning_content") for m in seen if isinstance(m, AIMessage)]
    assert reasoning == [None, None, "今の思考"]
    # 元のメッセージは変更しない
    assert history[1].additional_kwargs["reasoning_content"] == "古い思考"

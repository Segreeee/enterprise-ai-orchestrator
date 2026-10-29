import pytest
from pydantic import ValidationError

from orchestrator.agents.schemas import AgentSpec, Conversation, Message, Turn


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_message_accepts_valid_roles(role):
    assert Message(role=role, content="hi").role == role


def test_message_rejects_unknown_role():
    with pytest.raises(ValidationError):
        Message(role="tool", content="hi")


@pytest.mark.parametrize("content", ["", "   "])
def test_message_rejects_blank_content(content):
    with pytest.raises(ValidationError):
        Message(role="user", content=content)


@pytest.mark.parametrize("content", [123, b"hi"])
def test_strict_mode_rejects_coercion(content):
    # bytes -> str is accepted in lax mode, so this case only fails because of strict=True.
    with pytest.raises(ValidationError):
        Message(role="user", content=content)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        AgentSpec(name="a", system_prompt="p", temperature=0.5)


def test_agent_spec_requires_name_and_prompt():
    with pytest.raises(ValidationError):
        AgentSpec(name="", system_prompt="p")
    with pytest.raises(ValidationError):
        AgentSpec(name="a", system_prompt="")


def test_turn_links_agent_and_message():
    turn = Turn(agent="planner", message=Message(role="assistant", content="ok"))
    assert turn.agent == "planner"
    assert turn.message.content == "ok"


def test_models_are_frozen():
    msg = Message(role="user", content="hi")
    with pytest.raises(ValidationError):
        msg.content = "changed"


def test_conversation_holds_turns_and_stop_reason():
    turn = Turn(agent="a", message=Message(role="assistant", content="ok"))
    conv = Conversation(task="t", turns=[turn], stop_reason="max_turns")
    assert conv.turns == [turn]


def test_conversation_rejects_unknown_stop_reason():
    with pytest.raises(ValidationError):
        Conversation(task="t", turns=[], stop_reason="timeout")

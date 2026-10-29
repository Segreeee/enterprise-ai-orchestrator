import pytest

from orchestrator.agents.orchestrator import Orchestrator
from orchestrator.agents.schemas import AgentSpec, Conversation, Message
from tests.fakes import FakeRetriever, ScriptedLLM

TASK = "Design a caching layer"


def agents(*names: str) -> list[AgentSpec]:
    return [AgentSpec(name=n, system_prompt=f"You are {n}.") for n in names]


def test_turns_follow_round_robin_and_stop_at_max_turns():
    llm = ScriptedLLM()
    conv = Orchestrator(agents("planner", "coder", "critic"), llm).run(TASK, max_turns=5)

    assert isinstance(conv, Conversation)
    assert [t.agent for t in conv.turns] == ["planner", "coder", "critic", "planner", "coder"]
    assert [t.message.content for t in conv.turns] == [f"reply {i}" for i in range(1, 6)]
    assert conv.stop_reason == "max_turns"
    assert len(llm.calls) == 5


def test_first_agent_sees_its_system_prompt_and_the_task():
    llm = ScriptedLLM()
    Orchestrator(agents("planner", "coder"), llm).run(TASK, max_turns=1)

    assert llm.calls[0] == [
        Message(role="system", content="You are planner."),
        Message(role="user", content=TASK),
    ]


def test_history_is_attributed_and_own_turns_are_assistant_messages():
    llm = ScriptedLLM(["plan v1", "code v1", "plan v2"])
    Orchestrator(agents("planner", "coder"), llm).run(TASK, max_turns=3)

    assert llm.calls[1][1:] == [
        Message(role="user", content=TASK),
        Message(role="user", content="[planner] plan v1"),
    ]
    assert llm.calls[2][1:] == [
        Message(role="user", content=TASK),
        Message(role="assistant", content="plan v1"),
        Message(role="user", content="[coder] code v1"),
    ]


def test_rag_context_is_injected_into_system_prompt_of_agent_on_turn():
    llm = ScriptedLLM()
    retriever = FakeRetriever(["Redis supports TTL.", "LRU evicts oldest."])
    Orchestrator(agents("planner"), llm, retriever=retriever, k=2).run(TASK, max_turns=1)

    system = llm.calls[0][0]
    assert system.role == "system"
    assert system.content.startswith("You are planner.")
    assert "[1] Redis supports TTL." in system.content
    assert "[2] LRU evicts oldest." in system.content


def test_retriever_queried_with_task_then_latest_reply():
    retriever = FakeRetriever(["ctx"])
    Orchestrator(agents("a", "b"), ScriptedLLM(["first answer"]), retriever=retriever, k=3).run(TASK, max_turns=2)

    assert retriever.queries == [(TASK, 3), ("first answer", 3)]


def test_no_context_section_when_retriever_finds_nothing():
    llm = ScriptedLLM()
    Orchestrator(agents("planner"), llm, retriever=FakeRetriever([])).run(TASK, max_turns=1)

    assert llm.calls[0][0] == Message(role="system", content="You are planner.")


def test_stop_token_ends_conversation_early():
    llm = ScriptedLLM(["working", "all good DONE", "never"])
    conv = Orchestrator(agents("a", "b"), llm).run(TASK, max_turns=5, stop_token="DONE")

    assert [t.agent for t in conv.turns] == ["a", "b"]
    assert conv.stop_reason == "stop_token"
    assert len(llm.calls) == 2


def test_each_run_restarts_rotation_from_first_agent():
    orch = Orchestrator(agents("a", "b", "c"), ScriptedLLM())
    orch.run(TASK, max_turns=2)
    assert orch.run(TASK, max_turns=1).turns[0].agent == "a"


@pytest.mark.parametrize("max_turns", [0, -1])
def test_non_positive_max_turns_rejected(max_turns):
    with pytest.raises(ValueError):
        Orchestrator(agents("a"), ScriptedLLM()).run(TASK, max_turns=max_turns)


@pytest.mark.parametrize("task", ["", "   "])
def test_blank_task_rejected(task):
    with pytest.raises(ValueError):
        Orchestrator(agents("a"), ScriptedLLM()).run(task, max_turns=1)


def test_invalid_agent_roster_rejected_at_construction():
    with pytest.raises(ValueError):
        Orchestrator([], ScriptedLLM())
    with pytest.raises(ValueError, match="duplicate"):
        Orchestrator(agents("a", "a"), ScriptedLLM())

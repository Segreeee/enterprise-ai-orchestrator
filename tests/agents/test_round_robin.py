import pytest

from orchestrator.agents.round_robin import RoundRobin
from orchestrator.agents.schemas import AgentSpec


def make_agents(*names: str) -> list[AgentSpec]:
    return [AgentSpec(name=n, system_prompt=f"You are {n}.") for n in names]


def test_next_cycles_in_order_and_wraps_around():
    rr = RoundRobin(make_agents("a", "b", "c"))
    assert [rr.next().name for _ in range(7)] == ["a", "b", "c", "a", "b", "c", "a"]


def test_single_agent_always_returned():
    rr = RoundRobin(make_agents("solo"))
    assert [rr.next().name for _ in range(3)] == ["solo"] * 3


def test_schedule_continues_from_current_position():
    rr = RoundRobin(make_agents("a", "b", "c"))
    rr.next()
    assert [a.name for a in rr.schedule(4)] == ["b", "c", "a", "b"]


def test_schedule_zero_turns_is_empty():
    assert RoundRobin(make_agents("a")).schedule(0) == []


def test_empty_agent_list_rejected():
    with pytest.raises(ValueError, match="at least one agent"):
        RoundRobin([])


def test_duplicate_agent_names_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        RoundRobin(make_agents("a", "b", "a"))


def test_negative_turns_rejected():
    with pytest.raises(ValueError):
        RoundRobin(make_agents("a")).schedule(-1)


def test_external_list_mutation_does_not_affect_rotation():
    agents = make_agents("a", "b")
    rr = RoundRobin(agents)
    agents.clear()
    assert [a.name for a in rr.schedule(3)] == ["a", "b", "a"]

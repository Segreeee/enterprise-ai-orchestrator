from collections.abc import Sequence

from pydantic import NonNegativeInt, validate_call

from orchestrator.agents.schemas import AgentSpec


class RoundRobin:
    """Cycles through agents in fixed order, wrapping around at the end."""

    def __init__(self, agents: Sequence[AgentSpec]) -> None:
        if not agents:
            raise ValueError("RoundRobin requires at least one agent")
        names = [agent.name for agent in agents]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate agent names: {names}")
        self._agents = tuple(agents)
        self._index = 0

    def next(self) -> AgentSpec:
        agent = self._agents[self._index]
        self._index = (self._index + 1) % len(self._agents)
        return agent

    @validate_call
    def schedule(self, n_turns: NonNegativeInt) -> list[AgentSpec]:
        return [self.next() for _ in range(n_turns)]

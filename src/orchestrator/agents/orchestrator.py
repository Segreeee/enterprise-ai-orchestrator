from collections.abc import Sequence
from typing import Annotated, Literal
from uuid import uuid4

from loguru import logger
from pydantic import PositiveInt, StringConstraints, validate_call

from orchestrator.agents.round_robin import RoundRobin
from orchestrator.agents.schemas import AgentSpec, Conversation, Message, Turn
from orchestrator.models.llm import LLMClient
from orchestrator.rag.retriever import Retriever

Task = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Orchestrator:
    """Runs a shared conversation where agents speak in round-robin order."""

    def __init__(
        self,
        agents: Sequence[AgentSpec],
        llm: LLMClient,
        retriever: Retriever | None = None,
        k: int = 4,
    ) -> None:
        RoundRobin(agents)  # fail fast on an empty or duplicate roster
        self._agents = tuple(agents)
        self._llm = llm
        self._retriever = retriever
        self._k = k

    @validate_call
    def run(self, task: Task, max_turns: PositiveInt, stop_token: str | None = None) -> Conversation:
        # contextualize (not bind) so logs from the retriever and LLM calls inside the run carry the id too.
        # Task and reply text stay out of the logs: they can hold user data.
        with logger.contextualize(conversation_id=uuid4().hex):
            logger.info("conversation started", agents=len(self._agents), max_turns=max_turns)
            scheduler = RoundRobin(self._agents)
            turns: list[Turn] = []
            stop_reason: Literal["max_turns", "stop_token"] = "max_turns"
            for _ in range(max_turns):
                agent = scheduler.next()
                query = turns[-1].message.content if turns else task
                reply = self._llm.generate(self._prompt(agent, task, turns, self._context(query)))
                turns.append(Turn(agent=agent.name, message=reply))
                logger.debug("turn completed", agent=agent.name, turn=len(turns))
                if stop_token and stop_token in reply.content:
                    stop_reason = "stop_token"
                    break
            logger.info("conversation finished", stop_reason=stop_reason, turns=len(turns))
            return Conversation(task=task, turns=turns, stop_reason=stop_reason)

    def _context(self, query: str) -> str:
        if self._retriever is None:
            return ""
        results = self._retriever.retrieve(query, k=self._k)
        return "\n".join(f"[{i}] {r.chunk.text}" for i, r in enumerate(results, start=1))

    @staticmethod
    def _prompt(agent: AgentSpec, task: str, turns: list[Turn], context: str) -> list[Message]:
        system = agent.system_prompt + (f"\n\nRelevant context:\n{context}" if context else "")
        messages = [Message(role="system", content=system), Message(role="user", content=task)]
        for turn in turns:
            if turn.agent == agent.name:
                messages.append(Message(role="assistant", content=turn.message.content))
            else:
                messages.append(Message(role="user", content=f"[{turn.agent}] {turn.message.content}"))
        return messages

"""
Membrane OpenEnv data contracts.

Full field list and enums: ../docs/05-types-and-api.md and ../docs/SPEC.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Verb = Literal["QUERY", "SEND", "PLAN", "COMMIT", "REFUSE", "VERIFY"]
MessageSurface = Literal["USER_REPLY", "AGENT_DM", "TEAM_MEMORY", "TOOL_PAYLOAD", "RUN_LOG"]

try:
    from pydantic import Field

    from openenv.core.env_server import Action, Observation, State

    class MembraneAction(Action):
        verb: Verb = "QUERY"
        surface: Optional[MessageSurface] = None
        content: str = ""
        disclosure_tier: Optional[str] = None
        target_agent: Optional[str] = None
        reason: str = ""
        refusal_kind: Optional[str] = None
        acting_as: Optional[str] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class MembraneObservation(Observation):
        episode_goal: str = ""
        message: str = ""
        visible_fact_ids: List[str] = Field(default_factory=list)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class MembraneState(State):
        world_state_hash: str = ""
        surface_audit_json: str = "{}"
        failure_tags: List[str] = Field(default_factory=list)
        metadata: Dict[str, Any] = Field(default_factory=dict)

except ImportError:
    # Minimal dataclasses for tests / docs without openenv-core + pydantic
    @dataclass
    class MembraneAction:
        verb: Verb = "QUERY"
        surface: Optional[MessageSurface] = None
        content: str = ""
        disclosure_tier: Optional[str] = None
        target_agent: Optional[str] = None
        reason: str = ""
        refusal_kind: Optional[str] = None
        acting_as: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class MembraneObservation:
        done: bool = False
        reward: Optional[float] = None
        episode_goal: str = ""
        message: str = ""
        visible_fact_ids: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

        def model_dump(self) -> Dict[str, Any]:
            return {
                "done": self.done,
                "reward": self.reward,
                "episode_goal": self.episode_goal,
                "message": self.message,
                "visible_fact_ids": self.visible_fact_ids,
                "metadata": self.metadata,
            }

    @dataclass
    class MembraneState:
        episode_id: Optional[str] = None
        step_count: int = 0
        world_state_hash: str = ""
        surface_audit_json: str = "{}"
        failure_tags: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

        def model_dump(self) -> Dict[str, Any]:
            return {
                "episode_id": self.episode_id,
                "step_count": self.step_count,
                "world_state_hash": self.world_state_hash,
                "surface_audit_json": self.surface_audit_json,
                "failure_tags": self.failure_tags,
                "metadata": self.metadata,
            }

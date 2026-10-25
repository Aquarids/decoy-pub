from dataclasses import dataclass, field
from uuid import uuid4
from core.models import PrivacyGraph, ShadowGraph

@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    turn_index: int = 0
    session_graph: PrivacyGraph = field(default_factory=PrivacyGraph)
    global_graph: PrivacyGraph = field(default_factory=PrivacyGraph)
    shadows: list[ShadowGraph] = field(default_factory=list)
    provider_traces: list[dict] = field(default_factory=list)
    recent_user_texts: list[str] = field(default_factory=list)
    neutral_identity_map: list[dict] = field(default_factory=list)

    def commit(self, session_graph: PrivacyGraph, global_graph: PrivacyGraph, shadows: list[ShadowGraph], provider_trace: dict, user_text: str) -> None:
        self.session_graph = session_graph
        self.global_graph = global_graph
        self.shadows = shadows
        self.provider_traces.append(provider_trace)
        self.recent_user_texts = (self.recent_user_texts + [user_text])[-3:]
        self.turn_index += 1

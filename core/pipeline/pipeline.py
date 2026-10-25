from __future__ import annotations
from dataclasses import dataclass
import random
from uuid import uuid4
from core.local_model import LocalModel
from core.models import InferenceResult, ObfuscationResult, PrivacyAnalysisResult, ReconstructionResult
from core.privacy.processor import PrivacyProcessor
from core.obfuscation.processor import ShadowObfuscator
from core.inference.client import CloudClient
from core.reconstruction.processor import ResultReconstructor
from core.pipeline.session import SessionState

@dataclass
class TurnResult:
    turn_id: str
    privacy: PrivacyAnalysisResult
    obfuscation: ObfuscationResult
    inference: InferenceResult
    reconstruction: ReconstructionResult
    strategies: list[str]

class DecoyPipeline:

    def __init__(self, local_model: LocalModel, cloud_client: CloudClient, branch_count: int=5, rng: random.Random | None=None):
        self.local_model = local_model
        self.privacy = PrivacyProcessor(local_model)
        self.obfuscation = ShadowObfuscator(local_model, branch_count, rng)
        self.cloud_client = cloud_client
        self.reconstruction = ResultReconstructor(local_model)

    def run_turn(self, session: SessionState, user_text: str, turn_context: dict | None=None) -> TurnResult:
        turn_context = turn_context or {}
        turn_id = turn_context.get('turn_id') or uuid4().hex
        privacy, session_graph, global_graph = self.privacy.process(user_text, session.recent_user_texts, session.session_graph, session.global_graph)
        batch, strategies = self.obfuscation.process(privacy.task, privacy.real_subgraph, privacy.local_subgraph, session.shadows, session.neutral_identity_map, source_user_text=user_text)
        batch.cloud_query = self.local_model.blend_queries([query.query_text for query in batch.queries])
        request_id = turn_context.get('request_id')
        inference = self.cloud_client.infer(turn_id, batch, request_id=request_id)
        final = self.reconstruction.process(user_text, privacy.task, privacy.real_subgraph, session_graph, global_graph, batch, inference)
        trace = {'request': inference.request, 'raw_response': inference.raw_response, 'usage': inference.usage, 'protection_metadata': {'method_revision': batch.method_revision, 'protection_mode': batch.protection_mode, 'effective_case_count': batch.effective_case_count, 'configured_case_count': self.obfuscation.branch_count, 'critical_value_node_ids': list(batch.critical_value_node_ids), 'shadow_value_node_ids': list(batch.shadow_value_node_ids)}}
        session.commit(session_graph, global_graph, batch.shadows, trace, user_text)
        session.neutral_identity_map = batch.private_identity_map
        return TurnResult(turn_id, privacy, batch, inference, final, strategies)

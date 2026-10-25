from core.local_model import LocalModel
from core.models import CloudCaseResponse, InferenceResult, ObfuscationResult, PrivacyGraph, ReconstructionResult, TaskSpec

class ResultReconstructor:

    def __init__(self, local_model: LocalModel):
        self.local_model = local_model

    def process(self, user_text: str, task: TaskSpec, real: PrivacyGraph, session: PrivacyGraph, global_graph: PrivacyGraph, batch: ObfuscationResult, inference: InferenceResult) -> ReconstructionResult:
        selected = self.select_real(batch, inference)
        context = self.retrieve_context(session, global_graph, task)
        subject_alignment = self.align_roles(batch, real, context)
        final = self.local_model.fuse(user_text, task, selected.answer_text, context, selected.missing_information, subject_alignment, batch.private_identity_map)
        return ReconstructionResult(selected.case_id, selected.answer_text, final.strip(), selected.missing_information)

    def select_real(self, batch: ObfuscationResult, inference: InferenceResult) -> CloudCaseResponse:
        try:
            return inference.responses[batch.real_case_id]
        except KeyError as error:
            raise ValueError('Cloud response did not include the genuine task answer') from error

    def retrieve_context(self, session: PrivacyGraph, global_graph: PrivacyGraph, task: TaskSpec) -> PrivacyGraph:
        wanted = set(task.cloud_node_ids + task.local_node_ids)
        nodes = {key: node for key, node in global_graph.nodes.items() if key in wanted}
        nodes.update({key: node for key, node in session.nodes.items() if key in wanted})
        edges = {key: edge for key, edge in global_graph.edges.items() if edge.source_node_id in nodes and edge.target_node_id in nodes}
        edges.update({key: edge for key, edge in session.edges.items() if edge.source_node_id in nodes and edge.target_node_id in nodes})
        return PrivacyGraph(nodes, edges)

    def align_roles(self, batch: ObfuscationResult, real: PrivacyGraph, context: PrivacyGraph) -> dict[str, list[str]]:
        labels = {**batch.subject_labels, **batch.identity_labels}
        grouped: dict[str, list[str]] = {}
        for node_id, label in labels.items():
            grouped.setdefault(label, [])
            if node_id not in grouped[label]:
                grouped[label].append(node_id)
        return grouped

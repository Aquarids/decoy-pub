from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
DECOY_METHOD_REVISION = 'decoy'
DECOY_METHOD_REVISIONS = {'decoy': DECOY_METHOD_REVISION}
NODE_TYPES = {'SUBJECT', 'ATTRIBUTE', 'ENTITY', 'EVENT', 'CONTEXT'}
EDGE_TYPES = {'HAS', 'RELATED_TO', 'INVOLVED_IN', 'RESULT', 'TIME'}

@dataclass
class PrivacyNode:
    node_id: str
    node_type: str
    value: str
    source: str = 'user'
    status: str = 'active'
    aliases: list[str] = field(default_factory=list)

@dataclass
class PrivacyEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    status: str = 'active'

@dataclass
class PrivacyGraph:
    nodes: dict[str, PrivacyNode] = field(default_factory=dict)
    edges: dict[str, PrivacyEdge] = field(default_factory=dict)

    def subgraph(self, node_ids: list[str]) -> PrivacyGraph:
        wanted = set(node_ids)
        return PrivacyGraph(nodes={key: value for key, value in self.nodes.items() if key in wanted}, edges={key: edge for key, edge in self.edges.items() if edge.source_node_id in wanted and edge.target_node_id in wanted})

@dataclass
class TaskSpec:
    question: str
    intent: str
    output_goal: str
    cloud_node_ids: list[str]
    local_node_ids: list[str] = field(default_factory=list)
    private_literals: list[str] = field(default_factory=list)
    neutralizable_identity_node_ids: list[str] = field(default_factory=list)
    critical_value_node_ids: list[str] = field(default_factory=list)
    critical_value_reasons: dict[str, str] = field(default_factory=dict)
    public_concept_node_ids: list[str] = field(default_factory=list)
    neutralization_required_node_ids: list[str] = field(default_factory=list)
    shadow_value_node_ids: list[str] = field(default_factory=list)

def case_value_tokens(task: TaskSpec, graph: PrivacyGraph) -> list[dict]:
    tokens = []
    for prefix, node_ids in (('CRITICAL', task.critical_value_node_ids), ('CONTEXT', task.shadow_value_node_ids)):
        for index, node_id in enumerate(node_ids, start=1):
            if any((item['node_id'] == node_id for item in tokens)):
                continue
            node = graph.nodes[node_id]
            token = f'[[DECOY_CRITICAL_VALUE_{index}]]' if prefix == 'CRITICAL' else f'[[DECOY_NODE_{node_id}]]'
            tokens.append({'node_id': node_id, 'token': token, 'source_forms': list(dict.fromkeys([node.value, *node.aliases]))})
    return tokens

def derive_neutralization_required_node_ids(task: TaskSpec, graph: PrivacyGraph) -> list[str]:
    normalize = lambda value: ' '.join(str(value).casefold().split())
    private_forms = {normalize(value) for value in task.private_literals}
    critical = set(task.critical_value_node_ids)
    public = set(task.public_concept_node_ids)
    required = []
    for node_id in task.cloud_node_ids:
        node = graph.nodes[node_id]
        if node_id in critical or node_id in public:
            continue
        private_value = normalize(node.value) in private_forms or any((normalize(alias) in private_forms for alias in node.aliases))
        if node.node_type == 'SUBJECT' or node.node_type == 'ENTITY' or (node.node_type in {'ATTRIBUTE', 'CONTEXT'} and private_value):
            required.append(node_id)
    return required

@dataclass
class PrivacyAnalysisResult:
    task: TaskSpec
    real_subgraph: PrivacyGraph
    local_subgraph: PrivacyGraph
    graph_delta: dict[str, Any]

@dataclass
class ShadowGraph:
    shadow_id: str
    values: dict[str, str]
    source_values: dict[str, str] = field(default_factory=dict)
    varied_node_ids: list[str] = field(default_factory=list)

@dataclass
class GraphQuery:
    case_id: str
    branch_id: str
    query_text: str

@dataclass
class ObfuscationResult:
    queries: list[GraphQuery]
    shadows: list[ShadowGraph]
    real_case_id: str
    case_to_branch: dict[str, str]
    task_instruction: str
    subject_labels: dict[str, str]
    identity_labels: dict[str, str] = field(default_factory=dict)
    private_identity_map: list[dict[str, Any]] = field(default_factory=list)
    neutral_question: str = ''
    method_revision: str = DECOY_METHOD_REVISION
    protection_mode: str = 'confusion_branches'
    effective_case_count: int = 0
    critical_value_node_ids: list[str] = field(default_factory=list)
    shadow_value_node_ids: list[str] = field(default_factory=list)
    cloud_query: str = ''

@dataclass
class CloudCaseResponse:
    case_id: str
    answer_text: str
    missing_information: list[str] = field(default_factory=list)

@dataclass
class InferenceResult:
    request: dict[str, Any]
    responses: dict[str, CloudCaseResponse]
    raw_response: str
    usage: dict[str, Any]

@dataclass
class ReconstructionResult:
    selected_case_id: str
    selected_answer: str
    final_answer: str
    missing_information: list[str]

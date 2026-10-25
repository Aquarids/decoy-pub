from __future__ import annotations
from core.local_model import LocalModel
from core.models import PrivacyAnalysisResult, PrivacyEdge, PrivacyGraph, PrivacyNode, TaskSpec, derive_neutralization_required_node_ids

class PrivacyProcessor:

    def __init__(self, local_model: LocalModel):
        self.local_model = local_model

    def process(self, user_text: str, recent_user_texts: list[str], session_graph: PrivacyGraph, global_graph: PrivacyGraph) -> tuple[PrivacyAnalysisResult, PrivacyGraph, PrivacyGraph]:
        global_context = self.select_global_context(user_text, global_graph)
        context_graph = PrivacyGraph(nodes={**global_context.nodes, **session_graph.nodes}, edges={**global_context.edges, **session_graph.edges})
        proposal = self.local_model.extract_graph(user_text, recent_user_texts, context_graph)
        updated, delta = self.apply_delta(context_graph, proposal)
        task_data = self.local_model.describe_task(user_text, recent_user_texts, updated)
        task = self.parse_task(task_data)
        self._preserve_private_identity_endpoints(task, updated)
        task.neutralization_required_node_ids = derive_neutralization_required_node_ids(task, updated)
        result = PrivacyAnalysisResult(task, updated.subgraph(task.cloud_node_ids), updated.subgraph(task.local_node_ids), delta)
        durable = self.update_global(global_graph, updated, delta)
        return (result, updated, durable)

    def seed_history_message(self, user_text: str, recent_user_texts: list[str], session_graph: PrivacyGraph, global_graph: PrivacyGraph) -> tuple[PrivacyGraph, PrivacyGraph, dict]:
        global_context = self.select_global_context(user_text, global_graph)
        context_graph = PrivacyGraph(nodes={**global_context.nodes, **session_graph.nodes}, edges={**global_context.edges, **session_graph.edges})
        proposal = self.local_model.extract_graph(user_text, recent_user_texts, context_graph)
        updated, delta = self.apply_delta(context_graph, proposal)
        durable = self.update_global(global_graph, updated, delta)
        return (updated, durable, delta)

    def select_global_context(self, user_text: str, global_graph: PrivacyGraph) -> PrivacyGraph:
        matched = set()
        for node in global_graph.nodes.values():
            if node.node_type != 'SUBJECT' and any((name and name in user_text for name in [node.value, *node.aliases])):
                matched.add(node.node_id)
        for edge in global_graph.edges.values():
            if edge.target_node_id in matched and global_graph.nodes[edge.source_node_id].node_type == 'SUBJECT':
                matched.add(edge.source_node_id)
        return global_graph.subgraph(list(matched))

    def apply_delta(self, graph: PrivacyGraph, proposal: dict) -> tuple[PrivacyGraph, dict]:
        updated = PrivacyGraph(nodes=dict(graph.nodes), edges=dict(graph.edges))
        delta = {key: list(proposal.get(key, [])) for key in ('new_nodes', 'new_edges', 'node_updates', 'edge_updates', 'referenced_node_ids')}
        for item in delta['new_nodes']:
            node_id = item['id']
            updated.nodes[node_id] = PrivacyNode(node_id, item['type'], item['value'].strip(), source=item['source'], aliases=list(dict.fromkeys(item.get('aliases', []))))
        for node_id in delta['referenced_node_ids']:
            updated.nodes[node_id]
        for item in delta['new_edges']:
            edge_id = item['id']
            source, target, kind = (item['source'], item['target'], item['type'])
            updated.nodes[source]
            updated.nodes[target]
            updated.edges[edge_id] = PrivacyEdge(edge_id, source, target, kind)
        for item in delta['node_updates']:
            node_id = item['id']
            old = updated.nodes[node_id]
            updated.nodes[node_id] = PrivacyNode(node_id, old.node_type, item.get('value', old.value), old.source, item.get('status', old.status), list(old.aliases))
        for item in delta['edge_updates']:
            edge_id = item['id']
            old = updated.edges[edge_id]
            updated.edges[edge_id] = PrivacyEdge(edge_id, old.source_node_id, old.target_node_id, old.edge_type, item['status'])
        return (updated, delta)

    @staticmethod
    def parse_task(raw: dict) -> TaskSpec:
        return TaskSpec(question=raw['question'], intent=raw['intent'], output_goal=raw['output_goal'], cloud_node_ids=raw['cloud_node_ids'], local_node_ids=raw['local_node_ids'], private_literals=raw['private_literals'], critical_value_node_ids=raw['critical_value_node_ids'], critical_value_reasons=raw['critical_value_reasons'], public_concept_node_ids=raw['public_concept_node_ids'])

    @staticmethod
    def _preserve_private_identity_endpoints(task: TaskSpec, graph: PrivacyGraph) -> list[str]:

        def norm(value: str) -> str:
            return ' '.join(value.casefold().split())
        cloud = set(task.cloud_node_ids) - set(task.neutralizable_identity_node_ids)
        private_forms = {norm(value) for value in task.private_literals}
        promoted = []
        for node_id in task.local_node_ids:
            node = graph.nodes[node_id]
            forms = {norm(value) for value in [node.value, *node.aliases]}
            if node.node_type in {'SUBJECT', 'ENTITY'} and (node.node_type == 'SUBJECT' or bool(forms & private_forms)):
                if any((edge.source_node_id == node_id and edge.target_node_id in cloud or (edge.target_node_id == node_id and edge.source_node_id in cloud) for edge in graph.edges.values())):
                    promoted.append(node_id)
        if promoted:
            promoted_set = set(promoted)
            task.cloud_node_ids = list(dict.fromkeys([*task.cloud_node_ids, *promoted]))
            task.local_node_ids = [node_id for node_id in task.local_node_ids if node_id not in promoted_set]
            task.neutralizable_identity_node_ids = list(dict.fromkeys([*task.neutralizable_identity_node_ids, *promoted]))
            for node_id in promoted:
                node = graph.nodes[node_id]
                task.private_literals.extend((form for form in [node.value, *node.aliases] if norm(form) not in {norm(existing) for existing in task.private_literals}))
        return promoted

    @staticmethod
    def update_global(global_graph: PrivacyGraph, session_graph: PrivacyGraph, delta: dict) -> PrivacyGraph:
        nodes = dict(global_graph.nodes)
        edges = dict(global_graph.edges)
        durable = {item['id'] for item in delta['new_nodes'] if item.get('long_term') is True and item['source'] == 'user'}
        durable.update((item['id'] for item in delta['node_updates'] if item.get('long_term') is True and session_graph.nodes[item['id']].source == 'user'))
        for edge in session_graph.edges.values():
            if edge.target_node_id in durable:
                owner = session_graph.nodes[edge.source_node_id]
                if owner.node_type == 'SUBJECT':
                    durable.add(owner.node_id)
        for node_id in set(nodes) | durable:
            if node_id in session_graph.nodes:
                nodes[node_id] = session_graph.nodes[node_id]
        for edge in session_graph.edges.values():
            if edge.source_node_id in nodes and edge.target_node_id in nodes:
                edges[edge.edge_id] = edge
        return PrivacyGraph(nodes, edges)

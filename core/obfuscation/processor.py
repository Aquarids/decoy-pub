from __future__ import annotations
import random
from dataclasses import replace
from core.local_model import LocalModel
from core.models import GraphQuery, ObfuscationResult, PrivacyGraph, PrivacyNode, ShadowGraph, TaskSpec, derive_neutralization_required_node_ids, DECOY_METHOD_REVISIONS, case_value_tokens

class ShadowObfuscator:
    LABELS = {'SUBJECT': 'Subject', 'ATTRIBUTE': 'Attribute', 'ENTITY': 'Entity', 'EVENT': 'Event', 'CONTEXT': 'Context'}
    RELATIONS = {'HAS': 'has', 'RELATED_TO': 'relates to', 'INVOLVED_IN': 'participates in', 'RESULT': 'has result', 'TIME': 'has time'}

    def __init__(self, local_model: LocalModel, branch_count: int=5, rng: random.Random | None=None):
        if branch_count < 2 or branch_count > 26:
            raise ValueError('Branch count must be between 2 and 26')
        self.local_model = local_model
        self.branch_count = branch_count
        self.rng = rng or random.SystemRandom()

    def process(self, task: TaskSpec, real: PrivacyGraph, local_only: PrivacyGraph, previous: list[ShadowGraph], existing_identity_map: list[dict] | None=None, source_user_text: str='') -> tuple[ObfuscationResult, list[str]]:
        identities = existing_identity_map or []
        shadow_ids = list(real.nodes)
        if not shadow_ids:
            context_id = 'decoy_task_context'
            while context_id in real.nodes:
                context_id += '_context'
            real = PrivacyGraph(nodes={**real.nodes, context_id: PrivacyNode(context_id, 'CONTEXT', task.intent or task.question, source='inferred')}, edges=dict(real.edges))
            task = replace(task, cloud_node_ids=[*task.cloud_node_ids, context_id])
            shadow_ids = [context_id]
        task = replace(task, shadow_value_node_ids=shadow_ids)
        task.neutralization_required_node_ids = derive_neutralization_required_node_ids(task, real)
        raw = self.local_model.neutralize_graph(task, real, local_only, identities, source_user_text)
        values = raw['values']
        neutral_graph = PrivacyGraph(nodes={key: PrivacyNode(node.node_id, node.node_type, values[key], node.source, node.status, list(node.aliases)) for key, node in real.nodes.items()}, edges=dict(real.edges))
        neutral_task = replace(task, question=raw['question'], intent=raw['intent'], output_goal=raw['output_goal'], private_literals=[])
        identity_map = raw['identity_bindings']
        neutralized_source_text = raw['neutralized_source_text']
        strategies = self.choose_strategies(neutral_graph, bool(previous))
        shadows = self.advance_shadows(neutral_task, neutral_graph, local_only, previous, real)
        subject_labels = {key: node.value for key, node in neutral_graph.nodes.items() if node.node_type == 'SUBJECT'}
        identity_labels = self.identity_labels(identity_map, set(neutral_graph.nodes))
        branches = [('real', {key: node.value for key, node in neutral_graph.nodes.items()})]
        branches.extend(((shadow.shadow_id, shadow.values) for shadow in shadows))
        self.rng.shuffle(branches)
        queries = []
        mapping = {}
        real_case_id = ''
        case_source_tokens = case_value_tokens(task, real)
        for index, (branch_id, branch_values) in enumerate(branches):
            case_id = f"case_{chr(ord('a') + index)}"
            case_source = self.render_source_for_case(neutralized_source_text, case_source_tokens, branch_values)
            case_task = replace(neutral_task, **{field: self.render_source_for_case(getattr(neutral_task, field), case_source_tokens, branch_values) for field in ('question', 'intent', 'output_goal')})
            query_text = self.render_case(neutral_graph, case_task, branch_values, case_source)
            queries.append(GraphQuery(case_id, branch_id, query_text))
            mapping[case_id] = branch_id
            if branch_id == 'real':
                real_case_id = case_id
        instruction = "Help solve the user's task in each independent case, using general knowledge where useful. Use only that case's content for facts about its person; do not import personal facts from another case or invent missing private facts. State conclusions, conditions, uncertainty, and missing information separately for each case."
        protection_mode = 'confusion_branches'
        return (ObfuscationResult(queries, shadows, real_case_id, mapping, instruction, subject_labels, identity_labels, identity_map, neutral_task.question, method_revision=DECOY_METHOD_REVISIONS['decoy'], protection_mode=protection_mode, effective_case_count=len(queries), critical_value_node_ids=list(neutral_task.critical_value_node_ids), shadow_value_node_ids=list(neutral_task.shadow_value_node_ids)), strategies)

    def choose_strategies(self, real: PrivacyGraph, continuing: bool) -> list[str]:
        kinds = {node.node_type for node in real.nodes.values()}
        result = []
        if 'SUBJECT' in kinds:
            result.append('SUBJECT_OBFUSCATION')
        if 'ATTRIBUTE' in kinds or 'ENTITY' in kinds:
            result.append('ATTRIBUTE_OBFUSCATION')
        if 'EVENT' in kinds:
            result.append('EVENT_SUBGRAPH')
        if 'CONTEXT' in kinds:
            result.append('CONTEXT_GENERALIZATION')
        if continuing:
            result.append('TRAJECTORY_CONTINUATION')
        return result

    def advance_shadows(self, task: TaskSpec, real: PrivacyGraph, local_only: PrivacyGraph, previous: list[ShadowGraph], source_graph: PrivacyGraph | None=None) -> list[ShadowGraph]:
        source_graph = source_graph if source_graph is not None else real
        task = replace(task, shadow_value_node_ids=list(real.nodes))
        varied = set(real.nodes)
        if not varied:
            return [ShadowGraph(old.shadow_id, {**{key: node.value for key, node in real.nodes.items()}, **old.values}, {**{key: node.value for key, node in real.nodes.items()}, **old.source_values}, list(old.varied_node_ids)) for old in previous]
        result = []
        for index in range(self.branch_count - 1):
            old = previous[index] if previous else ShadowGraph(f'shadow_{index + 1}', {})
            values = dict(old.values)
            for key, node in real.nodes.items():
                if key not in varied:
                    values[key] = node.value
            needed = {key for key in varied if key not in old.values or old.source_values.get(key) != source_graph.nodes[key].value}
            if needed:
                preserved = {key: value for key, value in values.items() if key not in needed}
                generated = self.local_model.shadow_values(task, real, preserved, index, local_only)
                for key in needed:
                    value = generated[key].strip()
                    values[key] = value
            source_values = {**old.source_values, **{key: node.value for key, node in source_graph.nodes.items()}}
            changed = sorted((key for key in varied if values[key] != real.nodes[key].value))
            result.append(ShadowGraph(old.shadow_id, values, source_values, changed))
        return result

    @staticmethod
    def identity_labels(identity_map: list[dict], node_ids: set[str]) -> dict[str, str]:
        return {node_id: entry['neutral_label'] for entry in identity_map for node_id in entry['node_ids'] if node_id in node_ids}

    @staticmethod
    def render_source_for_case(source_template: str, case_source_tokens: list[dict], values: dict[str, str]) -> str:
        rendered = source_template
        for item in case_source_tokens:
            rendered = rendered.replace(item['token'], values[item['node_id']])
        return rendered

    def render_case(self, real: PrivacyGraph, task: TaskSpec, values: dict[str, str], neutralized_source_text: str) -> str:
        source = neutralized_source_text.strip()
        question = task.question.strip()
        intent = task.intent.strip()
        output_goal = task.output_goal.strip()
        parts = [source] if source else []
        existing = source.casefold()
        for part in (intent, question, output_goal):
            if not part:
                continue
            normalized = part.casefold()
            if normalized in existing or any((normalized in prior.casefold() for prior in parts)):
                continue
            parts.append(part)
        return ' '.join((part for part in parts if part))

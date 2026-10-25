from __future__ import annotations
import json
import re
from typing import Protocol, Any
from core.models import PrivacyGraph, TaskSpec, derive_neutralization_required_node_ids, case_value_tokens
from core.structured_output import tool_object, object_schema, STRING
from core.structured_schemas import GRAPH_DELTA, TASK, NEUTRAL, SHADOW
from core.reconstruction.grounding import INSTRUCTION as CLOUD_GROUNDING_INSTRUCTION
LOCAL_FUSION_PROMPT_REVISION = 'cloud_supported_conditional_context_restoration_v6'

class LocalModel(Protocol):

    def extract_graph(self, user_text: str, recent_user_texts: list[str], graph: PrivacyGraph) -> dict[str, Any]:
        ...

    def describe_task(self, user_text: str, recent_user_texts: list[str], graph: PrivacyGraph) -> dict[str, Any]:
        ...

    def neutralize_graph(self, task: TaskSpec, real: PrivacyGraph, local_only: PrivacyGraph, existing_identity_map: list[dict], source_user_text: str) -> dict[str, Any]:
        ...

    def shadow_values(self, task: TaskSpec, real: PrivacyGraph, existing: dict[str, str], branch_index: int, local_only: PrivacyGraph) -> dict[str, str]:
        ...

    def blend_queries(self, queries: list[str]) -> str:
        ...

    def fuse(self, user_text: str, task: TaskSpec, selected_answer: str, private_context: PrivacyGraph, missing: list[str], subject_alignment: dict[str, list[str]], identity_map: list[dict]) -> str:
        ...

def _graph_payload(graph: PrivacyGraph) -> dict[str, Any]:
    return {'nodes': [{'id': n.node_id, 'type': n.node_type, 'value': n.value, 'source': n.source, 'status': n.status, 'aliases': list(n.aliases)} for n in graph.nodes.values()], 'edges': [{'id': e.edge_id, 'source': e.source_node_id, 'target': e.target_node_id, 'type': e.edge_type, 'status': e.status} for e in graph.edges.values()]}

class JSONLocalModel:

    def __init__(self, wrapper):
        self.wrapper = wrapper

    def _object(self, instruction, payload, name, parameters):
        if name == 'submit_graph_delta':
            fields = dict(parameters['properties'])
            fields['referenced_node_ids'] = {**fields['referenced_node_ids'], 'description': "Node IDs only, from existing_graph.nodes or this result's new_nodes. Never include edge IDs from existing_graph.edges or new_edges."}
            parameters = {**parameters, 'properties': fields}
        elif name == 'submit_neutralization':
            fields = dict(parameters['properties'])
            bindings = fields['identity_bindings']
            item = bindings['items']
            binding_fields = dict(item['properties'])
            binding_fields['source_forms'] = {**binding_fields['source_forms'], 'description': 'A JSON array of exact source strings, including when there is only one form: ["<exact source form>"], not a bare string. Each item must itself be a string. Use only source-supported forms; the example shows shape only.'}
            fields['identity_bindings'] = {**bindings, 'items': {**item, 'properties': binding_fields}}
            parameters = {**parameters, 'properties': fields}
        elif name == 'submit_task':
            ids = [node['id'] for node in payload['updated_graph']['nodes']]
            fields = dict(parameters['properties'])
            for field in ('cloud_node_ids', 'local_node_ids', 'critical_value_node_ids', 'public_concept_node_ids'):
                fields[field] = {**fields[field], 'items': {'type': 'string', 'enum': ids}} if ids else {**fields[field], 'maxItems': 0}
            parameters = {**parameters, 'properties': fields}
        elif name == 'submit_shadow_values':
            parameters = {**parameters, 'properties': {**parameters['properties'], 'values': object_schema({node_id: STRING for node_id in payload['output_template']['values']})}}

        def validate(data):
            if name == 'submit_graph_delta':
                from core.privacy.processor import PrivacyProcessor
                graph_data = payload['existing_graph']
                from core.models import PrivacyNode, PrivacyEdge
                node_ids = {n['id'] for n in graph_data['nodes']} | {n['id'] for n in data['new_nodes']}
                edge_ids = {e['id'] for e in graph_data['edges']} | {e['id'] for e in data['new_edges']}
                for index, node_id in enumerate(data['referenced_node_ids']):
                    if node_id not in node_ids:
                        kind = 'an edge ID, not a node ID' if node_id in edge_ids else 'not a known node ID'
                        raise ValueError(f'$.referenced_node_ids[{index}]: {node_id!r} is {kind}. This field accepts node IDs from existing_graph.nodes or new_nodes, not IDs from existing_graph.edges or new_edges. Correct the references using the supplied source and graph; do not invent nodes or silently discard source-supported facts to pass validation.')
                graph = PrivacyGraph(nodes={n['id']: PrivacyNode(n['id'], n['type'], n['value'], n['source'], n['status'], n['aliases']) for n in graph_data['nodes']}, edges={e['id']: PrivacyEdge(e['id'], e['source'], e['target'], e['type'], e['status']) for e in graph_data['edges']})
                try:
                    PrivacyProcessor(None).apply_delta(graph, data)
                except KeyError as error:
                    raise ValueError(f'Graph delta references missing ID or field: {error}') from error
            elif name == 'submit_task':
                ids = {n['id'] for n in payload['updated_graph']['nodes']}
                for field in ('cloud_node_ids', 'local_node_ids', 'critical_value_node_ids', 'public_concept_node_ids'):
                    missing = set(data[field]) - ids
                    if missing:
                        raise ValueError(f'{field} references unknown graph IDs: {sorted(missing)}')
                selected = set(data['cloud_node_ids'])
                for field in ('critical_value_node_ids', 'public_concept_node_ids'):
                    missing = set(data[field]) - selected
                    if missing:
                        raise ValueError(f'{field} must be selected in cloud_node_ids: {sorted(missing)}. cloud_node_ids selects retained graph structure for later local semantic neutralization, not permission to disclose raw private values. Re-evaluate the original source and task necessity so node selection and classification agree. Do not add or remove IDs merely to pass validation; preserve source-supported facts and the existing private-value necessity criteria.')
                missing = set(data['critical_value_reasons']) - ids
                if missing:
                    raise ValueError(f'critical_value_reasons references unknown graph IDs: {sorted(missing)}')
            elif name == 'submit_neutralization':
                ids = set(payload['retained_graph_node_ids'])
                allowed_tokens = {item['token'] for item in payload['case_source_tokens']}
                token_pattern = '\\[\\[DECOY_[^\\]\\r\\n]+\\]\\]'
                errors = []
                for field in ('question', 'intent', 'output_goal', 'neutralized_source_text'):
                    value = data.get(field)
                    if not isinstance(value, str):
                        continue
                    unknown = set(re.findall(token_pattern, value)) - allowed_tokens
                    if unknown:
                        errors.append(f'$.{field} uses unsupplied case tokens: {sorted(unknown)}. Use the exact node-to-token mapping supplied in case_source_tokens.')
                values = data.get('values')
                if isinstance(values, dict):
                    invalid_keys = [key for key, value in values.items() if isinstance(value, str) and re.search(token_pattern, value)]
                    if invalid_keys:
                        errors.append(f'$.values: token-valued entries at keys {sorted(invalid_keys)}. Node values must be literal rewritten values, not case tokens. Tokens belong in question, intent, output_goal and neutralized_source_text as specified by the existing instructions. For each listed values key, derive its literal rewritten value from the original source under the existing neutralization and necessity criteria; do not substitute raw private values merely to satisfy this check.')
                bindings = data.get('identity_bindings')
                if isinstance(bindings, list):
                    for index, binding in enumerate(bindings):
                        if not isinstance(binding, dict):
                            continue
                        node_ids = binding.get('node_ids')
                        if not isinstance(node_ids, list):
                            continue
                        missing = {node_id for node_id in node_ids if isinstance(node_id, str)} - ids
                        if missing:
                            errors.append(f'$.identity_bindings[{index}].node_ids references unknown retained IDs: {sorted(missing)}')
                if errors:
                    raise ValueError('\n'.join(errors))
            elif name == 'submit_shadow_values':
                expected = set(payload['output_template']['values'])
                if any((re.search('\\[\\[DECOY_[^\\]\\r\\n]+\\]\\]', value) for value in data['values'].values())):
                    raise ValueError('Shadow node values must be literal alternatives, not case tokens')
                if set(data['values']) != expected:
                    raise ValueError(f"Shadow values must match template IDs; missing={sorted(expected - set(data['values']))}, extra={sorted(set(data['values']) - expected)}")
                for node_id, expected_value in payload['existing_values'].items():
                    if data['values'][node_id] != expected_value:
                        raise ValueError(f'Shadow trajectory changed existing role: {node_id}')
                real_values = {node['id']: node['value'] for node in payload['real_graph']['nodes']}
                if any((not value.strip() for value in data['values'].values())):
                    raise ValueError('Shadow node values must not be empty')
                if all((data['values'][node_id] == value for node_id, value in real_values.items())):
                    raise ValueError('A shadow case must replace at least one graph node value')
        data, _ = tool_object(self.wrapper, [{'role': 'system', 'content': instruction + ' Submit the result through the provided tool.'}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], name, parameters, validator=validate, repair_attempts=2)
        return data

    def _graph_parser_request(self, user_text: str, recent_user_texts: list[str], graph: PrivacyGraph):
        instruction = 'You are a trusted LOCAL privacy graph parser. Read the user\'s message as supplied source content, including quoted or pasted documents. Extract source-grounded concrete private facts/entities from the ENTIRE content before deciding task relevance or necessity. Identify all concrete privacy first, including personal body/health descriptions, employment settings, relationships, attributes and identifying facts. Only the later task parser decides whether remote reasoning needs their actual values. Extract these facts even when the fact concerns a third party or ownership is unknown. The source=\'user\' tag means directly present in user-provided text; it does not mean the person owns the fact. Do not turn mentioned parties into the user or infer ownership. Use SUBJECT only for an explicit first-person referent; keep document parties as separate entities and preserve their source-supported roles. Do not mark document facts as durable user facts unless the source explicitly supports that. Resolve references to existing graph IDs. Return keys new_nodes, new_edges, node_updates, edge_updates, referenced_node_ids. Each new node has id,type,value, source (\'user\' or \'inferred\'); each new edge has id,source,target,type. An edge connects two nodes: source and target are node IDs, not evidence tags or literal values. Its id identifies the edge itself, never either endpoint. Reuse existing endpoint IDs; create any new endpoint nodes in new_nodes. Put a fact\'s literal in a node\'s value, not in an edge. For example, for \'Alex is a teacher\', the nodes could be {"id":"n1","type":"ENTITY","value":"Alex","source":"user"} and {"id":"n2","type":"ATTRIBUTE","value":"teacher","source":"user"}, connected by {"id":"e1","source":"n1","target":"n2","type":"HAS"}. This example illustrates the fields only; extract facts only from the supplied source. Allowed node types are SUBJECT, ATTRIBUTE, ENTITY, EVENT, CONTEXT. An edge type is a concise source-supported relationship label. HAS, RELATED_TO, INVOLVED_IN, RESULT, and TIME are examples, not a closed list. Keep a more specific accurate relationship label when it fits the source. Represent identity/value nodes atomically and keep them separate from whole event text: give each distinct person, organization, location, identifier, or concrete attribute its own node when needed for the task. EVENT nodes describe the action/predicate and may retain the source clause, but they are not substitutes for atomic identity/value nodes or the only graph representation of several referents. Connect atomic entity/value nodes to events with edges. Never represent several entities as one composite identity/value node or attach several distinct identity labels to the same atomic identity/value node; keep each referent in its own node. EVENT nodes may contain several source-supported referents for co-reference but are not identity/value nodes. Represent every explicit named-person role or attribute assignment as a connected fact: keep the person as an ENTITY (use SUBJECT only for the user\'s first-person referent), keep the role/attribute node, and link them with HAS. For relational roles such as \'X is Y\'s manager\', preserve who holds the role and whom/which department it relates to. Never leave an explicit role node detached or assign it to the user by assumption. A named person is not automatically the user. TIME is an edge type only. Represent a time expression as a CONTEXT node connected to its event by a TIME edge. Represent a first-person subject separately from the person\'s name. Reuse IDs for existing facts and mark confirmed durable user facts with long_term=true. Preserve historical states and do not confirm inferred facts. For each new node, optionally include aliases as exact source-supported surface forms referring to that same identity; never invent aliases. For a SUBJECT node representing an explicit first-person referent, include each first-person pronoun that actually appears in the current source and refers to that same subject (for example I, me, my, we, us, our) as an alias. These aliases preserve local co-reference for later neutralization; they do not assert that document facts belong to the user. Do not add unstated pronouns or attach a document narrator to another person. Use recent user messages only to resolve references; extract the current turn\'s delta and do not re-add historical facts. Do not output a task or an answer. Empty delta arrays are allowed.'
        payload = {'user_text': user_text, 'recent_user_texts': recent_user_texts, 'existing_graph': _graph_payload(graph)}
        return (instruction, payload)

    def extract_graph(self, user_text: str, recent_user_texts: list[str], graph: PrivacyGraph) -> dict[str, Any]:
        instruction, payload = self._graph_parser_request(user_text, recent_user_texts, graph)
        return self._object(instruction, payload, 'submit_graph_delta', GRAPH_DELTA)

    def describe_task(self, user_text: str, recent_user_texts: list[str], graph: PrivacyGraph) -> dict[str, Any]:
        data = self._object('You are a trusted local task parser. Return question, intent, output_goal, cloud_node_ids, local_node_ids, private_literals, public_concept_node_ids, critical_value_node_ids, critical_value_reasons. Identify what response the user wants, not the privacy parser\'s analysis task. question must be the user\'s request in neutral wording; do not turn the source into a new question about what concerns or experiences it contains. For a request to share feelings or ask whether others relate, preserve that request for empathy/support rather than asking the cloud to extract entities or analyze a graph. private_literals is an array of exact source-supported concrete personal/private values identified BEFORE the necessity decision, including BOTH critical and noncritical private values. Task relevance never makes a private value public. Include names, personal identifiers, private attributes, body/health descriptions, personal employment settings and specific places/organizations. A source-attributed personal occupation may be private even if its role word is common. Do not include an abstract relation label solely for being a relation, generic task verbs, or whole event sentences as private_literals. public_concept_node_ids is an array only for ENTITY nodes selected in cloud_node_ids whose value is a generic non-identifying category or type (for example, \'financial services account\'), not a named person, named organization, specific account/product, address, identifier, or private attribute. A public concept must not contain a private literal or prior private identity. Question is one neutral task shared by every case; it must not contain a private value or first-person wording. The full original user_text will also be locally rewritten and included in each case, so ordinary document clauses do not need graph nodes. Select a connected graph that preserves structural role relations needed for critical-value branch coherence and task references. A private named entity whose source-supported role or relation matters must remain as a graph endpoint in cloud_node_ids; its identifying value will be abstracted by a later trusted-local semantic neutralizer before any cloud request. Do not drop or sever edges merely because an endpoint value is private. cloud_node_ids and local_node_ids must contain updated_graph node IDs only. critical_value_node_ids and public_concept_node_ids must each be subsets of cloud_node_ids. Selecting a node in cloud_node_ids retains its graph structure; it does not authorize sending its raw private value. Noncritical private values still require local semantic neutralization. Keep the original task intent/output. After identifying ALL private values, decide the minimum concrete information needed for the current requested operation. Apply a counterfactual abstraction test to each candidate critical value: replace it with the least specific faithful category, role, relation or neutral placeholder that preserves the source-supported task constraints. If the same sound response can use an abstract placeholder, keep the true value local. Mark the exact value critical only when this abstraction prevents the cloud from satisfying an explicit user requirement or essential task constraint with a correct, useful response. Assess usefulness against the actual requested output, not the most personalized or detailed answer imaginable. Mere relevance, richer context, more tailored wording, professional background, emotional importance, or the possibility of giving more specific advice does not establish necessity. Keep question, intent and output_goal faithful to the source; do not invent a specialist task or add precision requirements to justify releasing a private value. For empathetic conversation, general safety, coping, stress management or work-life balance, consider whether task-relevant categories such as physically demanding outdoor work or healthcare work suffice before retaining a person\'s exact occupation, employer or diagnosis. Preserve the physical demands, relevant relationships and expressed constraints supported by the source at the minimum specificity needed for the response. These abstractions depend on the task; an explicit role-specific comparison can require the actual role. A location required to find venues within a stated distance, or numerical inputs needed for a requested calculation, can remain critical when their abstraction would leave the cloud unable to determine a valid result. Evaluate abstraction with local processing limited to selecting the genuine cloud answer, restoring private bindings and formatting supported content. Do not assume that the local model will perform missing calculations, geographic search or substantive task reasoning. Reassess necessity for the current turn. A private value\'s presence in the accumulated graph or earlier use does not establish current necessity; a later general follow-up does not by itself require reintroducing an exact value that was previously abstracted. Return critical_value_node_ids and critical_value_reasons. A value needed merely to appear in, be copied to, or be translated in the final answer can use local placeholders and restoration. Split composite events into atomic facts; never mark a whole sentence or event critical. Each critical ID must be selected in cloud_node_ids and identify an ATTRIBUTE, ENTITY or CONTEXT whose full concrete value exactly matches a private_literal. critical_value_node_ids is an array of node IDs. Every critical value must also appear verbatim in private_literals; if private_literals is empty, critical_value_node_ids must be empty. critical_value_reasons maps each critical node ID to the explicit task requirement that fails under a faithful abstraction and why local binding restoration cannot recover the missing cloud reasoning. Reasons such as \'relevant context\', \'affects advice\' or \'needed to tailor recommendations\' alone are insufficient. For example, {"n3": "The request asks for venues within 2 km of this site; a generic location cannot determine which venues satisfy that distance, and local name restoration cannot perform the missing geographic search."}. If no specific task failure can be identified, leave the value noncritical. If none are necessary, return critical_value_node_ids=[] and critical_value_reasons={}. Do not invent IDs.', {'user_text': user_text, 'recent_user_texts': recent_user_texts, 'updated_graph': _graph_payload(graph)}, 'submit_task', TASK)
        return data

    @staticmethod
    def _neutralization_instruction() -> str:
        return 'You are a trusted local semantic rewriter. Rewrite the complete current user request and task fields so the cloud can solve the same task without receiving private values. Preserve propositions, roles, relationships, constraints, and requested output. Use the graph and identity history to keep references consistent, replacing private identities and values with stable neutral labels. For values marked critical, keep their graph values available for the real or shadow case: copy each critical node\'s real_graph value exactly into values, and use its supplied critical token wherever that value is needed in neutralized_source_text, question, intent, and output_goal. All four fields are per-case templates. Do not put critical or other case tokens into values or neutral labels. Copy every token exactly from case_source_tokens; its node ID determines the token, not an inferred numeric position. Never invent a token. Fill the tool fields question, intent, output_goal, neutralized_source_text, values, and identity_bindings. Rewrite ALL four text fields semantically, including privacy expressed by paraphrase, negation, or relational descriptions. Moving a fact into private_local_graph does not remove it from the source text: abstract that fact in every outgoing field even when the retained graph is empty. Keep the emotional intent, logical task conditions and support request without restating unnecessary concrete private attributes. For task-necessary values, make the common template compatible with all coherent branches: abstract dependent noncritical descriptions and use the linked critical tokens for any dependent specifics needed for reasoning. Never keep the real setting in one field while another field describes a shadow setting. values maps every retained graph node ID to its rewritten case value, including unchanged public concepts. Fill each required ID key; values cannot be empty when nodes are retained. identity_bindings is a list of objects with all four fields identity_id, neutral_label, source_forms, and node_ids for local restoration. node_ids is an array of retained_graph_node_ids represented by that binding, required for every binding, including a parent, location, or other non-SUBJECT identity. Use only retained_graph_node_ids in bindings; private_local_graph nodes are not retained case nodes. For example, if retained node n2 has the source value Alex and values[n2] is Person A, its binding is {"identity_id":"identity_n2","neutral_label":"Person A","source_forms":["Alex"],"node_ids":["n2"]}. This is a field example only. source_forms must use exact forms in the supplied source, without invented synonyms. For every entry in case_source_tokens, use its token for that node wherever it appears in the four text templates, including noncritical contextual nodes. For noncritical nodes, values must contain the rewritten neutral or generalized value, never restore the original private value merely because a token exists. Context tokens allow alternate occupations, events, emotions, and task settings to vary coherently. Keep the requested operation, language, output format, and graph roles intact. Subject and entity mentions also use their supplied case tokens so replacement labels agree across graph values, source text, and task fields. Do not answer the task.'

    def neutralize_graph(self, task: TaskSpec, real: PrivacyGraph, local_only: PrivacyGraph, existing_identity_map: list[dict], source_user_text: str) -> dict[str, Any]:
        task.neutralization_required_node_ids = derive_neutralization_required_node_ids(task, real)
        case_source_tokens = case_value_tokens(task, real)
        parameters = {**NEUTRAL, 'properties': {**NEUTRAL['properties'], 'values': object_schema({node_id: STRING for node_id in real.nodes})}}
        return self._object(self._neutralization_instruction(), {'source_user_text': source_user_text, 'task': {'question': task.question, 'intent': task.intent, 'output_goal': task.output_goal, 'critical_value_node_ids': list(task.critical_value_node_ids), 'critical_value_reasons': dict(task.critical_value_reasons), 'shadow_value_node_ids': list(task.shadow_value_node_ids), 'public_concept_node_ids': list(task.public_concept_node_ids), 'neutralization_required_node_ids': list(task.neutralization_required_node_ids)}, 'case_source_tokens': case_source_tokens, 'real_graph': _graph_payload(real), 'private_local_graph': _graph_payload(local_only), 'private_literals': list(task.private_literals), 'existing_identity_map': existing_identity_map, 'retained_graph_node_ids': list(real.nodes)}, 'submit_neutralization', parameters)

    def shadow_values(self, task: TaskSpec, real: PrivacyGraph, existing: dict[str, str], branch_index: int, local_only: PrivacyGraph) -> dict[str, str]:
        data = self._object("You are a trusted LOCAL shadow graph generator. Fill every empty string in output_template and submit it through the tool. The values object must contain every template key exactly once, with no other keys. Each key is an ID from real_graph or a previously preserved ID in existing_values; do not introduce IDs. Copy every existing_values entry exactly. For empty template strings, jointly choose coherent node replacements from the union of task.critical_value_node_ids and task.shadow_value_node_ids. These include generalized occupations, events, emotional states, interests, constraints, and task contexts even when no original private value is needed for cloud reasoning. Vary the case setting while preserving the requested operation, target language, output format, and graph roles. Every retained node can be replaced, including SUBJECT and ENTITY labels; keep their graph IDs, semantic roles, and edge endpoints unchanged. Generate neutral shadow identity labels, and reuse each label consistently within its branch. Copy nodes outside that union exactly. Related disease, medicine, event and result values must form one coherent case. Vary the concrete private attribute or setting itself, not merely the tone, caution, or advice around an unchanged fact. Generate linked contextual and critical values jointly so each branch is a plausible internally consistent alternative for the same task. Keep graph roles, task type, and who-did-what relations. A replacement subject or entity remains the same role connected by the same edges, with a different neutral label in this shadow branch. Use neutral SUBJECT values. Never use any excluded_local_values in a generated value. No real user's name or unique identifier. Replace substantive case facts where available, rather than merely renaming a subject. Nodes may keep their neutral source values when needed for coherence; at least one retained node must change. All retained nodes are eligible, not all nodes have to change in every branch.", {'task': task.__dict__, 'real_graph': _graph_payload(real), 'existing_values': existing, 'branch_index': branch_index, 'excluded_local_values': [node.value for node in local_only.nodes.values()], 'output_template': {'values': {**existing, **{key: existing.get(key, '') if key in {*task.critical_value_node_ids, *task.shadow_value_node_ids} else node.value for key, node in real.nodes.items()}}}}, 'submit_shadow_values', SHADOW)
        return data['values']

    def blend_queries(self, queries: list[str]) -> str:
        task_count = len(queries)
        data = self._object(f"Compose one natural cloud request from {task_count} complete task representations. Share a greeting, speaker perspective, and common background once, then present the {task_count} distinct tasks as concise paragraphs with natural transitions. Do not repeat a greeting or common sentence {task_count} times. Do not copy graph headings, Subject, Attribute, Entity, Event, Relations, Task intent, Required output, or Original source labels into the request. Do not use numeric labels, case IDs, branch IDs, or artificial bookkeeping. Keep every task independently recoverable from the combined request. Preserve the full source material for each translation, transformation, extraction, drafting or summarization task, including all sentences, quantities, languages, requested formats and domain constraints. Preserve each task's people, relations and point of view. Shared clauses may appear once when they have the same meaning in all tasks; task-specific clauses must stay with their task. The cloud will answer the {task_count} tasks in their original order. Write natural prose with paragraph breaks, without numbering the tasks. Submit only the fused request.", {'request_representations': queries}, 'submit_cloud_query', object_schema({'query': STRING}))
        query = data['query'].strip()
        if not query:
            raise ValueError('The fused cloud request must not be empty')
        return query

    def fuse(self, user_text: str, task: TaskSpec, selected_answer: str, private_context: PrivacyGraph, missing: list[str], subject_alignment: dict[str, list[str]], identity_map: list[dict]) -> str:
        response = self.wrapper.dialogue([{'role': 'system', 'content': "You are a trusted LOCAL response composer. Reconstruct the user's answer from the cloud response using private values and bindings from the trusted local graph. The cloud response can discuss several possible settings in one answer. Select its conditions and passages that match the original request and private graph, and omit material specific to the shadow settings. Treat an alternative-specific activity, place, qualification, or event in the cloud answer as a condition, not as a neutral name slot. Restore only source-supported neutral bindings or shared general references. Keep common guidance and conditions matching the genuine context. If the cloud answers only a different setting, omit its setting-specific content; never rename that setting to make it fit the user. " + CLOUD_GROUNDING_INSTRUCTION + 'Restore names and other neutral labels consistently from subject_alignment and private_identity_map, preserving co-reference across every sentence. Resolve Person A/B/C and similar neutral labels into the source-supported names, pronouns, forms of address, or relationship descriptions appropriate to the requested answer. Preserve which person speaks, who is addressed, and whose relative/partner/colleague is meant. Do not leave internal case labels in the user-facing answer or merge distinct people. A missing personal name is not permission to invent one; use the supported relationship/pronoun. Preserve negation as well as affirmative facts. Before drafting, internally check the task operation, target language, requested output format, main actions and their actors/recipients, explicit relationships, chronology, dates, numbers and other constraints stated in the task. Use source-supported graph nodes and edges to restore private bindings and relationships expressed in the cloud answer; do not use graph facts to supply task content absent from that answer. Preserve the relationships between people and departments/organizations, and who does what to whom, when expressed in the selected cloud answer. Preserve its task content; do not silently drop an explicit relationship just because the cloud paraphrase sounds plausible without it. Do not add unsupported facts or turn inferred graph content into certainty. Private facts may come only from this case and its trusted local facts; never use another case. Keep valid conditions and uncertainty from the cloud answer. Return only the user-facing answer, not your checklist or analysis. Cloud analysis is not a confirmed user fact.'}, {'role': 'user', 'content': json.dumps({'local_fusion_prompt_revision': LOCAL_FUSION_PROMPT_REVISION, 'user_text': user_text, 'task': task.__dict__, 'selected_answer': selected_answer, 'private_context': _graph_payload(private_context), 'missing_information': missing, 'subject_alignment': subject_alignment, 'private_identity_map': identity_map}, ensure_ascii=False)}])
        return response.strip()

from __future__ import annotations
import json
from typing import Protocol
from uuid import uuid4
from core.models import CloudCaseResponse, InferenceResult, ObfuscationResult
from core.structured_output import tool_object, object_schema, STRING, STRINGS

class CloudClient(Protocol):

    def infer(self, turn_id: str, batch: ObfuscationResult, request_id: str | None=None) -> InferenceResult:
        ...
TASK_HELPFULNESS_PROMPT_REVISION = 'natural_five_task_fusion_v2'
ANSWER = object_schema({'answers': {'type': 'array', 'items': object_schema({'answer_text': STRING, 'missing_information': STRINGS})}})

class PivotCloudClient:

    def __init__(self, wrapper):
        self.wrapper = wrapper

    def infer(self, turn_id: str, batch: ObfuscationResult, request_id: str | None=None) -> InferenceResult:
        request = self.build_request(turn_id, batch, request_id=request_id)
        expected = len(batch.queries)

        def validate(data):
            answers = data['answers']
            if len(answers) != expected:
                raise ValueError(f'Return exactly {expected} task answers in source order')
            if any((not item['answer_text'].strip() for item in answers)):
                raise ValueError('Every task answer must be nonempty')
        data, response = tool_object(self.wrapper, self.messages(request), 'submit_task_answers', ANSWER, validator=validate, repair_attempts=2)
        raw = json.dumps(data, ensure_ascii=False)
        return InferenceResult(request, self.parse_response(raw, request), raw, response.get('usage') or {})

    def build_request(self, turn_id: str, batch: ObfuscationResult, request_id: str | None=None) -> dict:
        if not batch.cloud_query.strip() or not batch.queries:
            raise ValueError('Compose the task cloud request before inference')
        return {'request_id': request_id or uuid4().hex, 'turn_id': turn_id, 'prompt_revision': TASK_HELPFULNESS_PROMPT_REVISION, 'decoy_method_revision': batch.method_revision, 'query': batch.cloud_query, 'task_count': len(batch.queries), 'case_ids': [query.case_id for query in batch.queries]}

    def messages(self, request: dict) -> list[dict[str, str]]:
        task_count = request['task_count']
        return [{'role': 'system', 'content': f'Complete all task passages in the request and return one answer for each passage in the same order. The request combines {task_count} related tasks with a shared opening or background. Shared wording appears once; preserve the distinct operation, source material, language, output format, constraints and people for each task. Do not merge tasks, answer only their common part, or claim that supplied source text is missing. If a task contains a full document for translation or transformation, use that document. Return exactly the requested number of answers through submit_task_answers; the array order is the task order and needs no public task labels.'}, {'role': 'user', 'content': request['query']}]

    def parse_response(self, raw: str, request: dict) -> dict[str, CloudCaseResponse]:
        parsed = json.loads(raw)
        ids = request.get('case_ids') or []
        if len(ids) != len(parsed['answers']):
            raise ValueError('Cloud answer count differs from task count')
        return {case_id: CloudCaseResponse(case_id, item['answer_text'].strip(), [str(value) for value in item['missing_information']]) for case_id, item in zip(ids, parsed['answers'])}

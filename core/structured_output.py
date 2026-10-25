from __future__ import annotations
import json
STRUCTURED_OUTPUT_REVISION = 'function_arguments_v1'

def object_schema(properties, required=None):
    return {'type': 'object', 'properties': properties, 'required': list(properties) if required is None else required, 'additionalProperties': False}
STRING = {'type': 'string'}
STRINGS = {'type': 'array', 'items': STRING}

class _ToolOutputError(ValueError):

    def __init__(self, error, rejected_arguments, rejected_tool_calls=None):
        super().__init__(str(error))
        self.error = error
        self.rejected_arguments = rejected_arguments
        self.rejected_tool_calls = rejected_tool_calls

def _tool_object_once(wrapper, messages, name, parameters, *, strict=False, **kwargs):
    response = wrapper.generate(messages, tools=[{'type': 'function', 'function': {'name': name, 'description': 'Submit the requested structured result.', 'parameters': parameters, 'strict': strict}}], tool_choice='required', parallel_tool_calls=False, **kwargs)
    data = None
    rejected_tool_calls = None
    try:
        calls = response.get('tool_calls') or []
        rejected_tool_calls = calls
        if len(calls) != 1:
            raise ValueError(f'Expected exactly one {name} tool call; got {len(calls)}')
        call = calls[0]
        function = call.get('function', call) if isinstance(call, dict) else call.function
        actual_name = function.get('name') if isinstance(function, dict) else function.name
        if actual_name != name:
            raise ValueError(f'Expected tool {name}; got {actual_name}')
        rejected_tool_calls = None
        arguments = function.get('arguments') if isinstance(function, dict) else function.arguments
        data = arguments
        data = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(data, dict):
            raise ValueError(f'Tool {name} arguments must be an object')
    except ValueError as error:
        raise _ToolOutputError(error, data, rejected_tool_calls) from error
    return (data, response)

def validate_tool_value(value, schema, path='$ '.strip()):
    if 'anyOf' in schema:
        for option in schema['anyOf']:
            try:
                validate_tool_value(value, option, path)
                return
            except ValueError:
                pass
        raise ValueError(f'{path}: value does not match any allowed type')
    kind = schema.get('type')
    types = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'integer': int, 'number': (int, float)}
    if kind in types and (not isinstance(value, types[kind]) or (kind in {'integer', 'number'} and isinstance(value, bool))):
        if kind == 'array' and path.endswith('.source_forms') and (schema.get('items', {}).get('type') == 'string'):
            raise ValueError(f'{path}: expected array of strings, got {type(value).__name__}. JSON shape: ["<source-supported string>"], even for one item; not a bare string or object. Supply the correct result from the source; the example shows shape only.')
        raise ValueError(f'{path}: expected {kind}')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: value outside allowed enum')
    if kind == 'object':
        for key in schema.get('required', []):
            if key not in value:
                raise ValueError(f'{path}: missing required field {key}')
        properties = schema.get('properties', {})
        extra = schema.get('additionalProperties', True)
        for key, item in value.items():
            if key in properties:
                validate_tool_value(item, properties[key], f'{path}.{key}')
            elif extra is False:
                raise ValueError(f'{path}: unexpected field {key}')
            elif isinstance(extra, dict):
                validate_tool_value(item, extra, f'{path}.{key}')
    elif kind == 'array':
        for index, item in enumerate(value):
            validate_tool_value(item, schema.get('items', {}), f'{path}[{index}]')

def collect_tool_value_errors(value, schema, path='$ '.strip()):
    if 'anyOf' in schema:
        try:
            validate_tool_value(value, schema, path)
        except ValueError as error:
            return [str(error)]
        return []
    shallow = {key: item for key, item in schema.items() if key not in {'properties', 'required', 'additionalProperties', 'items'}}
    if schema.get('type') == 'array':
        shallow['items'] = schema.get('items', {})
        if isinstance(value, list):
            shallow['items'] = {}
    try:
        validate_tool_value(value, shallow, path)
    except ValueError as error:
        return [str(error)]
    errors = []
    if schema.get('type') == 'object':
        for key in schema.get('required', []):
            if key not in value:
                errors.append(f'{path}: missing required field {key}')
        properties = schema.get('properties', {})
        extra = schema.get('additionalProperties', True)
        for key, item in value.items():
            if key in properties:
                errors.extend(collect_tool_value_errors(item, properties[key], f'{path}.{key}'))
            elif extra is False:
                errors.append(f'{path}: unexpected field {key}')
            elif isinstance(extra, dict):
                errors.extend(collect_tool_value_errors(item, extra, f'{path}.{key}'))
    elif schema.get('type') == 'array':
        for index, item in enumerate(value):
            errors.extend(collect_tool_value_errors(item, schema.get('items', {}), f'{path}[{index}]'))
    return errors

def tool_object(wrapper, messages, name, parameters, *, validator=None, repair_attempts=0, strict=False, **kwargs):
    current = messages
    for attempt in range(repair_attempts + 1):
        error = None
        rejected_tool_calls = None
        try:
            data, response = _tool_object_once(wrapper, current, name, parameters, strict=strict, **kwargs)
        except _ToolOutputError as rejected:
            if not repair_attempts:
                raise rejected.error from None
            data, error = (rejected.rejected_arguments, rejected.error)
            rejected_tool_calls = rejected.rejected_tool_calls
        else:
            try:
                if name == 'submit_neutralization' and validator is not None:
                    errors = collect_tool_value_errors(data, parameters)
                    try:
                        validator(data)
                    except ValueError as rejected:
                        errors.append(str(rejected))
                    if errors:
                        raise ValueError('\n'.join(errors))
                else:
                    if strict or repair_attempts or validator is not None:
                        validate_tool_value(data, parameters)
                    if validator is not None:
                        validator(data)
            except ValueError as rejected:
                error = rejected
        if error is None:
            wrapper.last_structured_correction_count = attempt
            return (data, response)
        if attempt == repair_attempts:
            raise ValueError(f'{name} output invalid after {attempt} corrective calls: {error}') from error
        correction = {'instruction': 'Correct the rejected tool arguments using the original source and schema. Return a complete corrected tool result. Do not invent source facts, supply placeholder values, or silently remove source-supported facts to bypass the validation error.', 'rejected_arguments': data, 'validation_error': str(error), 'corrective_attempt': attempt + 1}
        if rejected_tool_calls is not None:
            correction['rejected_tool_calls'] = [call.model_dump(mode='json') if hasattr(call, 'model_dump') else call for call in rejected_tool_calls]
        current = [*messages, {'role': 'user', 'content': json.dumps(correction, ensure_ascii=False)}]

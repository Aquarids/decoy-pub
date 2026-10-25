from openai import OpenAI

class LLMWrapper:
    def __init__(self, model_id, base_url, api_key):
        self.model_id = model_id
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, messages, **kwargs):
        response = self.client.chat.completions.create(model=self.model_id, messages=messages, **kwargs)
        result = response.choices[0].message.model_dump()
        result["usage"] = response.usage.model_dump() if response.usage else {}
        return result

    def dialogue(self, messages, **kwargs):
        return self.generate(messages, **kwargs).get("content") or ""

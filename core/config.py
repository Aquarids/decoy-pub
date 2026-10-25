import os
from pathlib import Path
from core.helper.llm_wrapper import LLMWrapper

class LLMSettings:
    def __init__(self, path=".env"):
        values = {}
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
        self.values = {**values, **os.environ}

    def wrapper(self, prefix):
        return LLMWrapper(self.values[prefix + "_MODEL_ID"], self.values[prefix + "_BASE_URL"], self.values[prefix + "_API_KEY"])

    def local_wrapper(self):
        return self.wrapper("LOCAL_LLM")

    def cloud_wrapper(self):
        return self.wrapper("LLM")

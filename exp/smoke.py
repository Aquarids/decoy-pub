import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import LLMSettings
from core.local_model import JSONLocalModel
from core.inference.client import PivotCloudClient
from core.pipeline.pipeline import DecoyPipeline
from core.pipeline.session import SessionState

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".env")
    parser.add_argument("--query", default="My name is Alex. I am moving to a new city for work. Help me draft a short message to my friends.")
    args = parser.parse_args()
    settings = LLMSettings(args.env)
    pipeline = DecoyPipeline(JSONLocalModel(settings.local_wrapper()), PivotCloudClient(settings.cloud_wrapper()), branch_count=5)
    result = pipeline.run_turn(SessionState(), args.query)
    print(json.dumps({"cloud_query": result.inference.request["query"], "cloud_answers": [answer.answer_text for answer in result.inference.responses.values()], "restored_answer": result.reconstruction.final_answer}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

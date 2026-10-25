from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from core.models import PrivacyEdge, PrivacyGraph, PrivacyNode

class GlobalGraphStore:

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> PrivacyGraph:
        if not self.path.exists():
            return PrivacyGraph()
        payload = json.loads(self.path.read_text(encoding='utf-8'))
        nodes = {item['node_id']: PrivacyNode(**item) for item in payload['nodes']}
        edges = {item['edge_id']: PrivacyEdge(**item) for item in payload['edges']}
        return PrivacyGraph(nodes, edges)

    def save(self, graph: PrivacyGraph) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'nodes': [asdict(node) for node in graph.nodes.values()], 'edges': [asdict(edge) for edge in graph.edges.values()]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

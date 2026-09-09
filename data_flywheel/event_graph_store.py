"""Lightweight event graph store for text->structure persistence.

This is a minimal, local alternative inspired by GraphRAG-style pipelines.
It stores event/topic/entity/sector relations for tracing impact paths.
"""

from __future__ import annotations

from typing import Any, Dict, List
import os

import networkx as nx

from core.runtime_paths import resolve_runtime_path


class EventGraphStore:
    def __init__(self, graph_path: str = "data/event_graph.graphml"):
        if str(graph_path) == "data/event_graph.graphml":
            self.graph_path = str(resolve_runtime_path(graph_path, env_var="CIVITAS_EVENT_GRAPH_PATH"))
        else:
            self.graph_path = graph_path
        graph_dir = os.path.dirname(self.graph_path)
        if graph_dir:
            os.makedirs(graph_dir, exist_ok=True)

        self.graph = nx.DiGraph()
        if os.path.exists(self.graph_path):
            try:
                self.graph = nx.read_graphml(self.graph_path)
            except Exception:
                self.graph = nx.DiGraph()

    def ingest(self, event: Any) -> None:
        event_id = getattr(event, "event_id", "") or ""
        if not event_id:
            return

        event_node = f"event::{event_id}"
        title = str(getattr(event, "title", "") or "")
        source = str(getattr(event, "source", "") or "")
        self.graph.add_node(event_node, kind="event", title=title, source=source)

        text_factors: Dict[str, Any] = getattr(event, "text_factors", {}) or {}

        dominant_topic = str(text_factors.get("dominant_topic", "uncategorized"))
        topic_node = f"topic::{dominant_topic}"
        self.graph.add_node(topic_node, kind="topic", label=dominant_topic)
        self.graph.add_edge(event_node, topic_node, relation="has_topic", weight=1.0)

        for sig in text_factors.get("topic_signals", []) or []:
            if not isinstance(sig, dict):
                continue
            topic = str(sig.get("topic", "")).strip()
            if not topic:
                continue
            score = self._safe_float(sig.get("score"), 0.5)
            tnode = f"topic::{topic}"
            self.graph.add_node(tnode, kind="topic", label=topic)
            self.graph.add_edge(event_node, tnode, relation="has_topic", weight=score)

        for ent in getattr(event, "entities", []) or []:
            name = str(getattr(ent, "name", "")).strip()
            etype = str(getattr(ent, "entity_type", "entity")).strip() or "entity"
            conf = self._safe_float(getattr(ent, "confidence", 0.7), 0.7)
            if not name:
                continue
            enode = f"entity::{etype}:{name}"
            self.graph.add_node(enode, kind="entity", label=name, entity_type=etype)
            self.graph.add_edge(event_node, enode, relation="mentions_entity", weight=conf)

        for sector in getattr(event, "affected_sectors", []) or []:
            if not isinstance(sector, str) or not sector.strip():
                continue
            snode = f"sector::{sector.strip()}"
            self.graph.add_node(snode, kind="sector", label=sector.strip())
            self.graph.add_edge(event_node, snode, relation="impacts_sector", weight=0.8)
            self.graph.add_edge(topic_node, snode, relation="impacts_sector", weight=0.6)

        for path in text_factors.get("impact_paths", []) or []:
            if not isinstance(path, dict):
                continue
            src = str(path.get("source", "")).strip()
            rel = str(path.get("relation", "relates_to")).strip() or "relates_to"
            tgt = str(path.get("target", "")).strip()
            w = self._safe_float(path.get("weight"), 0.5)
            if not src or not tgt:
                continue
            src_node = f"factor::{src}"
            tgt_node = f"factor::{tgt}"
            self.graph.add_node(src_node, kind="factor", label=src)
            self.graph.add_node(tgt_node, kind="factor", label=tgt)
            self.graph.add_edge(src_node, tgt_node, relation=rel, weight=w)

        self._save()

    def query_neighbors(self, label: str, limit: int = 12) -> List[Dict[str, Any]]:
        key = str(label or "").strip()
        if not key:
            return []
        result: List[Dict[str, Any]] = []
        for node in self.graph.nodes:
            if key.lower() not in str(node).lower():
                continue
            for _, dst, edge in self.graph.out_edges(node, data=True):
                result.append(
                    {
                        "source": node,
                        "target": dst,
                        "relation": edge.get("relation", "relates_to"),
                        "weight": self._safe_float(edge.get("weight"), 0.0),
                    }
                )
                if len(result) >= limit:
                    return result
        return result

    def _save(self) -> None:
        try:
            nx.write_graphml(self.graph, self.graph_path)
        except Exception:

            pass

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

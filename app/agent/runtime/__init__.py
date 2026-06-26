"""Production-oriented Agent runtime subgraph.

The runtime package owns the model/tool observation loop. The outer Agent graph
continues to own request routing, memory lifecycle, safety branches and final
turn persistence.
"""

from app.agent.runtime.graph import build_agent_runtime_subgraph

__all__ = ["build_agent_runtime_subgraph"]

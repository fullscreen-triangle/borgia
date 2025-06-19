GRAPH_SOLVERS = {
    "networkx": {
        "algorithms": ["graph_isomorphism", "subgraph_matching", "centrality"],
        "use_case": "molecular_graph_analysis"
    },
    "igraph": {
        "algorithms": ["community_detection", "graph_clustering"],
        "use_case": "molecular_fragment_analysis"
    },
    "graph_tool": {
        "algorithms": ["statistical_inference", "graph_models"],
        "use_case": "probabilistic_graph_matching"
    }
}

from unittest.mock import MagicMock

import pytest

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import Document, NodeRelationship, NodeWithScore
from llama_index.core.storage import StorageContext


@pytest.mark.parametrize("case", ["plain", "metadata", "stale", "different", "missing"])
def test_gap_fill_uses_neighbor_identity(case: str) -> None:
    document = Document(
        text="Alpha.\nBeta.\nGamma.",
        metadata={"file_name": "example.txt"} if case == "metadata" else {},
    )
    nodes = SentenceWindowNodeParser(
        sentence_splitter=str.splitlines
    ).get_nodes_from_documents([document])
    next_info = nodes[0].next_node
    assert next_info is not None
    if case == "stale":
        nodes[0].relationships[NodeRelationship.NEXT] = next_info.model_copy(
            update={"hash": "older-content-hash"}
        )
    elif case == "different":
        nodes[2].relationships[NodeRelationship.PREVIOUS] = next_info.model_copy(
            update={"node_id": "different-node"}
        )
    elif case == "missing":
        nodes[2].relationships.pop(NodeRelationship.PREVIOUS)

    context = StorageContext.from_defaults()
    context.docstore.add_documents(nodes)
    backend = MagicMock()
    backend.retrieve.return_value = [
        NodeWithScore(node=nodes[0], score=0.8),
        NodeWithScore(node=nodes[2], score=0.4),
    ]
    retriever = AutoMergingRetriever(backend, context)

    results = retriever.retrieve("query")

    if case in ("different", "missing"):
        assert [result.node.node_id for result in results] == [
            nodes[0].node_id,
            nodes[2].node_id,
        ]
    else:
        assert [result.node.node_id for result in results] == [
            node.node_id for node in nodes
        ]
        assert results[1].get_content() == "Beta."
        assert results[1].score == pytest.approx(0.6)
    backend.retrieve.assert_called_once()

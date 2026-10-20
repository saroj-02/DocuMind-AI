import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import ai_service
from embeddings import embed_texts
from retriever import build_index, retrieve


class RagPipelineTests(unittest.TestCase):
    def test_embeddings_are_requested_in_batches(self):
        def embed_content(**request):
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[1.0, 0.0])
                    for _ in request["contents"]
                ]
            )

        client = SimpleNamespace(
            models=SimpleNamespace(embed_content=embed_content)
        )
        texts = [f"chunk {number}" for number in range(33)]

        with patch("embeddings.get_client", return_value=client):
            vectors = embed_texts(texts)

        self.assertEqual(len(vectors), 33)
        self.assertTrue(all(np.array_equal(vector, [1.0, 0.0]) for vector in vectors))

    def test_index_retrieves_the_most_similar_chunk(self):
        chunks = [
            {"text": "first", "source": "one.txt"},
            {"text": "second", "source": "two.txt"},
        ]
        with patch("retriever.embed_texts", return_value=[
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ]):
            index = build_index(chunks)
        with patch("retriever.embed_query", return_value=np.array([1.0, 0.0])):
            results = retrieve(index, "question", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "one.txt")
        self.assertAlmostEqual(results[0]["score"], 1.0)

    def test_answer_generation_uses_retrieved_evidence(self):
        create_interaction = unittest.mock.Mock(
            return_value=SimpleNamespace(output_text="The text supports this [S1].")
        )
        client = SimpleNamespace(
            interactions=SimpleNamespace(create=create_interaction)
        )
        sources = [{"source": "notes.txt", "page": None, "text": "Evidence"}]

        with patch("ai_service.get_client", return_value=client):
            answer = ai_service.generate_answer("What is stated?", sources)

        self.assertEqual(answer, "The text supports this [S1].")
        request_prompt = create_interaction.call_args.kwargs["input"]
        self.assertIn("Evidence", request_prompt)
        self.assertIn("What is stated?", request_prompt)

    def test_answer_generation_abstains_without_sources(self):
        with patch("ai_service.get_client") as get_client:
            answer = ai_service.generate_answer("Question", [])

        get_client.assert_not_called()
        self.assertIn("could not find relevant information", answer)


if __name__ == "__main__":
    unittest.main()

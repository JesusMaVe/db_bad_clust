"""
test_bert_embedder.py — Tests for the BERTEmbedder mean-pooling logic.

Verifies (with mocked tokenizer/model, no network):
  - Mean pooling weights token embeddings by the attention mask
  - L2 normalization of the output vectors
  - Batch concatenation preserves order
  - Empty input returns an empty array
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import ClassVar

import numpy as np
import pytest
import torch

from db_bad_clust.features.bert_embedder import BERTEmbedder


class MockModel:
    """Returns fixed last_hidden_state so pooling math is verifiable."""

    config: ClassVar[dict] = {"hidden_size": 4}

    def eval(self):
        pass

    def to(self, device):
        pass

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        # token_embeddings[i, t] encodes (sample, token) so the expected
        # mean pooling result is easy to compute by hand
        token_embeddings = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]],
                [[4.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]],
            ]
        )
        return type("Output", (), {"last_hidden_state": token_embeddings})()


class MockTokenizer:
    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        # Two texts, three token slots each; second token masked out for text 2
        return {
            "input_ids": torch.ones(len(texts), 3, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
        }


@pytest.fixture
def embedder():
    return BERTEmbedder(tokenizer=MockTokenizer(), model=MockModel(), device="cpu")


class TestMeanPooling:
    def test_attention_mask_excludes_padding(self, embedder):
        """Text 2 has tokens 2-3 masked: mean must use token 1 only."""
        vectors = embedder.encode(["text one", "text two"])
        # Text 2: mean of [4,0,0,0] -> normalized [1,0,0,0]
        assert vectors.shape == (2, 4)
        np.testing.assert_allclose(vectors[1], [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_mean_over_unmasked_tokens(self, embedder):
        """Text 1 has all 3 tokens: mean = [1/3, 2/3, 1, 0], L2-normalized."""
        vectors = embedder.encode(["text one", "text two"])
        expected = np.array([1 / 3, 2 / 3, 1.0, 0.0])
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(vectors[0], expected, atol=1e-6)

    def test_l2_normalized(self, embedder):
        vectors = embedder.encode(["text one", "text two"])
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_not_cls_token(self, embedder):
        """CLS pooling would return token 0 verbatim; mean pooling must not."""
        vectors = embedder.encode(["text one", "text two"])
        assert not np.allclose(vectors[0], [1.0, 0.0, 0.0, 0.0])

    def test_batching_preserves_order(self, embedder):
        vectors = embedder.encode(["a", "b"])
        assert vectors.shape[0] == 2

    def test_empty_input(self, embedder):
        result = embedder.encode([])
        assert result.size == 0

    def test_default_model_is_sentence_transformer(self):
        assert "paraphrase-multilingual" in BERTEmbedder().model_name

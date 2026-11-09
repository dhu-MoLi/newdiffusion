from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DefectVocabulary:
    category: str
    labels: tuple[str, ...]
    sha256: str

    @classmethod
    def from_config(cls, category: str, defect_types: Sequence[str]) -> DefectVocabulary:
        if not isinstance(category, str) or not category:
            raise ValueError("category must be a non-empty string")
        if isinstance(defect_types, str) or not defect_types:
            raise ValueError("defect_types must be a non-empty label sequence")
        labels = tuple(defect_types)
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("defect_types must contain non-empty strings")
        if "normal" in labels:
            raise ValueError("normal has no trainable embedding")
        if len(set(labels)) != len(labels):
            raise ValueError("defect_types must not contain duplicate labels")
        if labels != tuple(sorted(labels)):
            raise ValueError("defect_types must be lexicographically sorted")
        payload = json.dumps(
            {"category": category, "labels": list(labels)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(category=category, labels=labels, sha256=hashlib.sha256(payload).hexdigest())

    def index(self, label: str) -> int:
        if label == "normal":
            raise ValueError("normal has no trainable embedding")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string")
        try:
            return self.labels.index(label)
        except ValueError as error:
            raise ValueError(f"unknown defect label: {label}") from error


class DefectConditioner(nn.Module):
    def __init__(self, vocabulary: DefectVocabulary, embedding_dim: int) -> None:
        super().__init__()
        if not isinstance(vocabulary, DefectVocabulary):
            raise TypeError("vocabulary must be a DefectVocabulary")
        if type(embedding_dim) is not int or embedding_dim != 128:
            raise ValueError("embedding_dim must be 128")
        self.vocabulary = vocabulary
        self.embedding = nn.Embedding(len(vocabulary.labels), embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(
        self, labels: Sequence[str], *, device: torch.device | str, dtype: torch.dtype
    ) -> Tensor:
        if isinstance(labels, str) or not labels:
            raise ValueError("labels must be a non-empty sequence")
        indices = [self.vocabulary.index(label) for label in labels]
        index_tensor = torch.tensor(indices, device=self.embedding.weight.device, dtype=torch.long)
        return self.embedding(index_tensor).to(device=device, dtype=dtype)

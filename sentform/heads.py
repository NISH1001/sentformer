#!/usr/bin/env python3

from typing import List, Optional, Tuple

import torch


class NetworkHead(torch.nn.Module):
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class ClassificationHead(NetworkHead):
    """
    Classifies whole sentence into label/s.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        labels: Optional[Tuple[str]] = None,
        multi_label: bool = False,
    ):
        super(ClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.labels = labels or tuple()
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc(embeddings)

    def _logits_to_labels(
        self,
        logits: torch.Tensor,
        threshold: float = 0.5,
        **kwargs,
    ) -> List[List[str]]:
        res = []
        if self.multi_label:
            probabilities = torch.sigmoid(logits)
            predicted_indices = probabilities >= threshold
            for pred in predicted_indices:
                active_labels = [
                    self.labels[idx] for idx, active in enumerate(pred) if active.item()
                ]
                res.append(active_labels)
        else:
            probabilities = torch.softmax(logits, dim=-1)
            predicted_indices = torch.argmax(probabilities, dim=-1)
            res = [self.labels[idx] for idx in predicted_indices]
        return res


class NERHead(ClassificationHead):
    """
    This classifies each token into a NER tag/label.
    """

    def __init__(
        self,
        input_dim: int,
        num_tags: int,
        ner_tags: Optional[Tuple[str]] = None,
        multi_label: bool = False,
    ):
        super(NERHead, self).__init__(
            input_dim=input_dim,
            num_classes=num_tags,
            labels=ner_tags,
            multi_label=multi_label,
        )

    @property
    def ner_tags(self) -> List[str]:
        return self.labels

    def _logits_to_labels(
        self, logits: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> List[List[str]]:
        res = []
        probabilities = torch.softmax(logits, dim=-1)
        predicted_indices = torch.argmax(probabilities, dim=-1)

        for batch_idx, mask in zip(predicted_indices, attention_mask):
            sentence_length = mask.sum().item()
            # ignore padding tokens
            batch_labels = [
                self.ner_tags[idx.item()] for idx in batch_idx[:sentence_length]
            ]
            res.append(batch_labels)
        return res


def main():
    pass


if __name__ == "__main__":
    main()

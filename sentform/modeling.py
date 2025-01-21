#!/usr/bin/env python3

from typing import Union, Optional, List, Dict, Tuple

import torch
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .pooling import PoolingLayer, MeanPooling
from .heads import ClassificationHead, NERHead, NetworkHead


class SentenceTransformer(torch.nn.Module):
    def __init__(
        self,
        backbone: Optional[Union[str, PreTrainedModel]] = "bert-base-uncased",
        pooling_layer: Optional[PoolingLayer] = None,
        device: str = "cpu",
    ):
        super(SentenceTransformer, self).__init__()
        if isinstance(backbone, str):
            backbone = AutoModel.from_pretrained(backbone)
        self.device = device
        self.backbone = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone.name_or_path)

        self.pooling = pooling_layer or MeanPooling()
        assert isinstance(self.pooling, PoolingLayer)

    @staticmethod
    def _texts_to_list(texts: Union[str, List[str]]) -> List[str]:
        return [texts] if isinstance(texts, str) else texts

    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, Tensor]:
        texts = self._texts_to_list(texts)
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    def encode(self, texts: Union[str, List[str]]) -> Tensor:
        self.backbone.eval()
        texts = self._texts_to_list(texts)
        inputs = self.tokenize(texts)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.backbone(**inputs)

        token_embeddings = outputs.last_hidden_state
        sentence_embeddings = self.pooling(token_embeddings, inputs["attention_mask"])
        return sentence_embeddings

    def _encode(self, texts: Union[str, List[str]]) -> Tensor:
        self.backbone.eval()
        texts = self._texts_to_list(texts)
        inputs = self.tokenize(texts)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.backbone(**inputs)

        token_embeddings = outputs.last_hidden_state
        sentence_embeddings = self.pooling(token_embeddings, inputs["attention_mask"])
        return (inputs, token_embeddings, sentence_embeddings)

    @property
    def hidden_size(self) -> int:
        return self.backbone.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self.hidden_size


class MultiTaskFormer(SentenceTransformer):
    def __init__(
        self,
        heads: List[NetworkHead],
        backbone: Optional[Union[str, PreTrainedModel]] = "bert-base-uncased",
        pooling_layer: Optional[PoolingLayer] = None,
        device: str = "cpu",
    ) -> None:
        super(MultiTaskFormer, self).__init__(backbone, pooling_layer, device)
        self.heads = heads

    def forward(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        # use the _encocde instead of encode to access both the sentence and token embeddings
        inputs, token_embeddings, sentence_embeddings = self._encode(texts)

        outputs = {}
        for i, head in enumerate(self.heads):
            logits = torch.Tensor([])
            with torch.no_grad():
                if isinstance(head, NERHead):
                    logits = head(token_embeddings)
                else:
                    logits = head(sentence_embeddings)
            outputs[f"head_{i}"] = dict(
                logits=logits,
                predicted_labels=head._logits_to_labels(
                    logits, attention_mask=inputs["attention_mask"]
                ),
            )
        return outputs


def main():
    pass


if __name__ == "__main__":
    main()

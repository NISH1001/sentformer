#!/usr/bin/env python3

import torch


class PoolingLayer(torch.nn.Module):
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class MeanPooling(PoolingLayer):
    def forward(self, token_embeddings, attention_mask) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        return summed / summed_mask

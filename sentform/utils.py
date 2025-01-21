#!/usr/bin/env python3

import random
import numpy as np
import torch


def pairwise_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return torch.matmul(normalized_embeddings, normalized_embeddings.T)


def set_seed(seed=42):
    """
    Set seed for more determinism
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    pass


if __name__ == "__main__":
    main()

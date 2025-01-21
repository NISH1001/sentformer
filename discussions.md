Please refer to `notebooks/test-sentence-transformers.ipynb` notebook for some implementation tests.

# Step-1

##  Sentence Transformer Implementation

We implement `sentform.modeling.SentenceTransformer` which can take in a transformer backbone as mentioned. Plus, any`sentform.pooling.PoolingLayer`implementation.

```python
from sentform.modeling import SentenceTransformer
from sentform.pooling import MeanPooling

sentences = [
    "I love cats.",
    "I don't like mangoes.",
    "They are using NLP in the company Fetch."
]

backbone = AutoModel.from_pretrained("bert-base-uncased")
sentformer = SentenceTransformer(
    backbone=backbone,
    pooling_layer=MeanPooling()
)
embeddings = sentformer.encode(sentences)
print(embeddings.shape)
```

```
torch.Size([3, 768])

tensor([[ 0.5344,  0.3247, -0.1033,  ..., -0.0295,  0.2302,  0.2154],
        [ 0.2443,  0.2077, -0.2987,  ...,  0.1340,  0.0335, -0.0820],
        [ 0.0744, -0.1423,  0.2127,  ..., -0.4782,  0.1212,  0.1719]])
```

embeddings similarities:
```python
from sentform.utils import pairwise_cosine_similarity
import seaborn as sns
import matplotlib.pyplt as plt

sims = pairwise_cosine_similarity(embeddings)

sns.heatmap(
    sims,
    annot=True,
    cmap="coolwarm",
    cbar=True,
    square=True,
    xticklabels=sentences,
    yticklabels=sentences,
)
```

![image](https://github.com/user-attachments/assets/f52669e9-4d06-47d8-9f68-2bff14b84466)



> Note: tokenizer are automatically initialized based on the backbone type. We can also access `SentenceTransformer.tokenize(...)` method for only tokenization.

## Pooling Discussions

One of the choices that was made is adding a pooling mechanism to for aggregation to get the final embeddings. This is needed because most of the backbones such as BERT or BERT-like models/variants don't directly give us sentence-level embeddings. Rather, they generate **token-level** embeddings in the final hidden layer. We can use these token embeddings in any downstream implementation to generate sentence-level embeddings. The simplest mechanism is pooling. We implement `sent form.pooling.MeanPooling` that computes the average of all these token embeddings to generate a fixed shape vector/embedding. The MeanPooling takes accounts for only non-padded tokens because padded tokens contribute no meaningful information and only are noisy in representing the final sentence embedding. These non-padded tokens are the actual valid tokens to consider.

For example, in case of `bert-base-uncased`, if a sentence/text has 5 tokens, we'd be getting:

- (5, 768) shape in the hidden layer mentioned
- From here, we aggregate and generate (1, 768) -> Simply 768-dim vector



Additionally, we can implement several other standard pooling mechanism such as `MaxPooling` (not implemented in the codebase, left out for brevity). Unlike `MeanPooling`, we just take max value across all the tokens in the sequence in `MaxPooling`. Although this can capture the most "*important*" or highly activate feature, it can lose information from less prominent features. The possible implementation  in the codebase could be `sentform.pooling.MaxPooling`.

Similarly, ``CLS` token pooling is also a widely-used pooling mechanism in BERT-based models. In these models the `CLS` token is supposed to encode all the information (meaning) of the entire sequence. As a result, we can directly use the embedding of `CLS` token as sentence embedding. Again, possible implementation in the codebase could be `sentform.pooling.CLSTokenPooling`.

There can be a lot of ways to add more complexity to generate the embeddings such as adding more projection layer and reprojections and maybe attention mechanisms. The codebase allows for modularity for all these.

---

# Step-2

We implement `sentform.modeling.MultiTaskFormer` which is an extension to `SentenceTransformer` that allows to add any downstream heads `sentform.heads.NetworkHead` for any downtream tasks. We implement two such heads:

- `sentform.heads.ClassificationHead` for general multi-class and multi-label classification, working at sequence-level classification.
- `sentform.heads.NERHead`, an extension to the `ClassificationHead` which works at token-level classification.

```python
from sentform.modeling import MultiTaskFormer
from sentform.heads import ClassificationHead, NERHead

 Needs fine-tuning of these heads
# left the tuning part for brevity as per assignment
multi_tasker = MultiTaskFormer(
    heads=[
        ClassificationHead(
            backbone.config.hidden_size,
            num_classes=3,
            labels=["Positive", "Neutral", "Negative"],
            multi_label=False
        ),
        NERHead(
            backbone.config.hidden_size,
            num_tags=3,
            ner_tags=["Person", "Organization", "Location"],
            multi_label=False
        )
    ],
    backbone=backbone,
)

outputs = multi_tasker(sentences)
print(outputs)
# print(outputs["head_0"])
# print(outputs["head_1"])
```

```python
{'head_0': {'logits': tensor([[-0.0638, -0.1983,  0.0074],
          [ 0.0208,  0.1266,  0.2032],
          [-0.0508,  0.1157,  0.0638]]),
  'predicted_labels': [['Negative'],
   ['Positive', 'Neutral', 'Negative'],
   ['Neutral', 'Negative']]},
 'head_1': {'logits': tensor([[[-0.3314, -0.5311,  0.1053],
           [-0.2897, -0.6716,  0.2112],
           [-0.1475, -0.8265, -0.1555],
           [ 0.1923, -0.0331,  0.1125],
           [-0.6644, -0.6219, -0.0615],
           [-0.2435,  0.4752, -0.0369],
           [-0.0547, -0.5965, -0.1171],
           [-0.0953, -0.5887, -0.1721],
           [ 0.0295, -0.5246, -0.1318],
           [-0.1297, -0.5958, -0.2530],
           [-0.0790, -0.5677, -0.1527],
           [-0.0513, -0.5438, -0.0992]],
  
          [[-0.3009, -0.4976,  0.1998],
           [-0.2441, -0.5311,  0.2400],
           [-0.3107, -0.2357, -0.3953],
           [-0.2541,  0.2635,  0.0801],
           [-0.1912, -0.3858, -0.3256],
           [-0.0159, -0.2862, -0.0302],
           [-0.3075, -0.0702,  0.3962],
           [-0.2758,  0.1982,  0.1446],
           [-0.4386, -0.3423, -0.0923],
           [-0.1159,  0.0474, -0.0023],
           [-0.1556, -0.5621, -0.0545],
           [-0.2751, -0.7029, -0.1213]],
  
          [[-0.3553, -0.4153,  0.2557],
           [-0.5644, -0.2095,  0.1845],
           [-0.5335, -0.1052,  0.2065],
           [-0.3325, -0.0078, -0.1156],
           [-0.5451, -0.2389,  0.2108],
           [-0.1052, -0.0777, -0.4317],
           [-0.0275, -0.4303,  0.0639],
           [-0.4370, -0.3666,  0.2235],
           [-0.0432, -0.1246, -0.3537],
           [-0.7474, -0.3369,  0.0270],
           [-0.3570,  0.3229, -0.0076],
           [-0.0948,  0.2846,  0.1393]]]),
  'predicted_labels': [['Location',
    'Location',
    'Person',
    'Person',
    'Location',
    'Organization'],
   ['Location',
    'Location',
    'Organization',
    'Organization',
    'Person',
    'Person',
    'Location',
    'Organization',
    'Location',
    'Organization'],
   ['Location',
    'Location',
    'Location',
    'Organization',
    'Location',
    'Organization',
    'Location',
    'Location',
    'Person',
    'Location',
    'Organization',
    'Organization']]}}
```

> Note: Since no training mechanism is implemented, the multi_tasker output in the code snippet and the adjoined notebook will output random things because the downstream heads aren't trained.



The architecture changes here are addition of the downtream heads which are iterated over during forward passes. The classificaiton heads would require only sentence embeddings whereas for NER it'd require to have token-level embeddings as we're assigning some named entity to each word (token actually) in the sequence. This iteration also required us to implement `SentenceTransformer._encode(...)` method which wasn't initially present in the initial iteration of the `SentenceTransformer` implementation. Additionally, we also implement `ClassificationHead._logits_to_labels` utility method to convert output logits from these heads to proper label values. Also note that for `NERHead`, we make use of `attention_mask` to remove classification/tagging for padded tokens which carry no meaning in the final sequence. However, since tokenization in BERT-based models happen at sub-word level, we don't necessarily map the logits to NER tag for original actual words (left out for brevity).



---

# Step-3

## 1) Training Strategy

**When to freeze the backbone?**

Freezing is effective when we are dealing with pre-trained models (easily available, open sourced) and we are limited in data for fine-tuning. Backbone exhibiting a superset of the task abstraction would be beneficial directly as it would have already learnt a lot of text representation internally and there's less need to train the backbone as well. General rule of thumb is:

- Backbone is highly capable in meaningful internal representation of the language (same)
- We are limited in data size (few dozens or maybe few hundreds of samples)
- We are limited in computational resource to train the backbone. For instance, for backprop, we'd be getting more gradients to backward pass which stresses out the memory axis of the training.
- In some cases, training the backbone will only make the fine-tuning worse as the network would have to *unlearn* initial representation of the language and *re-learn* new associations/representations.

> In this, we might assume that the downtream tasks are similar in nature.

**When to freeze a one head?**

We might benefit in freezing one head while fine-tuning another in cases where one task is already performing well and optimized beforehand. We could also freeze one head to avoid further overfitting and sometimes prevent catastrophic forgetting which can happend if the network is being trained longer.



## 2) Multi-task Modeling vs Separate Modeling

**Multi-task**

We use multi-task modeling when tasks are related and shared learning helps better. In this case, the backbone would have meaningful *share representation*; something like a shared encoder that learns meanignful represntation for the tasks. This is the classic transfer learning approach in shared tasks settings. It's also efficient memory+compute wise while training in this setting.

**Separate Modeling**

When tasks are fundamentally different and don't share much semantic overlap, it's better we model the tasks separately as separate models with their own backbones. In some cases, tasks could have conflicting objectives leading to worser performance when done in share multi-task manner.

## 3) Task A abundant data than Task B

In cases where the task A has abundant data than task B, there are different strategies we can employ for training:

- Simplest approach is data augmentation where we generate more task B samples and equalizing the training sample distribution for both the tasks. This way, each tasks would get similar number of training samples during training.
- Instead of data augmentation, we could try weighted training where the objective for task B has more weight than task B.
- Another useful approach could be: first train only on task A and then start training task B in conjunction later. This way the network could benefit in learning more and better representation via task A's abundant samples which can later be useful for task B. This is also more like curriculum learning.
- We could also try alternate training by using task-specific learning rates so that task A can get updates in smaller steps while task B can get in larger frequent steps. Additionally, task specific early stopping could also be added.

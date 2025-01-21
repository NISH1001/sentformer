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
print(outputs["head_0"])
print(outputs["head_1"])
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

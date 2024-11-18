# Active Learning
## Multi-label Uncertainty Sampling
From modAL.multilabel [docs](https://modal-python.readthedocs.io/en/latest/content/apireference/multilabel.html)

### SVM Binary Minimum
SVM binary minimum multilabel active learning strategy. For details see the paper Klaus Brinker, On Active Learning in Multi-label Classification ([paper](https://link.springer.com/chapter/10.1007%2F3-540-31314-1_24)).

*Not applicable - must be an SVM model.*

### Multi-label Average Confidence
AvgConfidence query strategy for multilabel classification.

For more details on this query strategy, see Esuli and Sebastiani., Active Learning Strategies for Multi-Label Text Classification ([paper](http://dx.doi.org/10.1007/978-3-642-00958-7_12)).

### Multi-label Average Score
AvgScore query strategy for multilabel classification.

For more details on this query strategy, see Esuli and Sebastiani., Active Learning Strategies for Multi-Label Text Classification ([paper](http://dx.doi.org/10.1007/978-3-642-00958-7_12))

### Max Loss
Max Loss query strategy for SVM multilabel classification.

For more details on this query strategy, see Li et al., Multilabel SVM active learning for image classification ([paper](http://dx.doi.org/10.1109/ICIP.2004.1421535))
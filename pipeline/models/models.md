## Models
Models of interest - BirdNet, **Perch**, AST, PaSST.\
*Perch (Google) is essentially BirdNet but trained only on bird vocalisations (BirdNet contains some non-bird sounds), [read here](https://www.nature.com/articles/s41598-023-49989-z.epdf)*.
*PaSST is based on AST (Audio Spectrogram Transformer) but applies patchout which reduces computation cost [read here](https://arxiv.org/pdf/2110.05069)*

AvesEcho models - fine-tuned for european-based species: BirdNet - EfficientNet (CNN-based), PaSST - Vision tranformer (ViT) (Transformer-based). [read here](https://arxiv.org/html/2409.15383v1)

Model Name | Architecture | Train Data | Embedding Size | Window Size | Stack |
--- | --- | --- | --- | --- | --- |
BirdNet | EfficientNet B1  | XC+ML+Custom | 1024 | 3s | TensorFlow
Perch | EfficientNet B1  | XC | 1280 | 5s | TensorFlow 
PaSST (AvesEcho) | Vision Transformer (ViT) | XC | 527 | 3s | PyTorch
More...

*Xeno-Canto (XC), Macaulay Library (ML)*


### Next Step(s)
- Create wrappers for PyTorch and Tensorflow models to run inference and model fine-tuning.
    - [X] PaSST (AvesEcho) model setup and running
    - [X] fc (AvesEcho) model

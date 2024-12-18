# TABMON Data Pipeline
This is a data and classification pipeline repository for the [TABMON](https://www.biodiversa.eu/2024/04/15/tabmon/) (Transnational Acoustic Biodiversity Monitoring Network). The purpose of this repository is to provide an end-to-end framework from raw field data to inference of essential biodiversity variables (EBVs).

Author: Ben McEwen \
Source: AvesEcho (author: Burooj Ghani) [paper](https://arxiv.org/abs/2409.15383) \

### Getting Started

*Set Up:*\
Create conda or virtualenv environment and specify python version 3.11.10 i.e. `conda create --name tabmon python=3.11.10`.
Install dependencies `pip install -r requirements.txt`

*To Run Inference:*\
Place audio into `/audio` directory and specify directory to analyze `--i` relative to `analyze.py` i.e. `../audio/<my-audio>/`.
After installing dependencies navigate to `/original` then run:\
```
python analyze.py --i '../audio/<my-audio>' --model_name 'fc' --add_csv --add_filtering
```
Both 'fc' and 'passt' are setup.

It is recommended to pre-generate sample embeddings before inference.
*To generate embeddings* navigate to `/original` then run:
```
python embeddings.py --i '../audio/subset' --model_name 'fc'
```

### Next Steps
- [ ] Documentation for model training and evaluation
- [ ] Embedding based pre-trained model evaluation (bacpipe)
- [ ] Evaluation of PaSST on Sounds of Norway dataset
- [ ] Visualise uncertainty sampling and sample ranking

### Structure
`loader.py` - Downloads audio from Google Cloud storage, start and end dates as well as download limit can be specified. \
`original/analyze.py` - Current file for model inference \
`original/embeddings.py` - Pre-generate embeddings for audio samples 

#### Research Questions
This is just a list of research questions I am interested in:
- Spatiotemporal fine-tuning for acoustic species distribution modelling - adapting to specific domains (new locations and specific times)
- Mutliscale audio classification
- Anomaly or out-of-distribution detection.

Fine-tuning for **Domain Adaption**
- Spatiotemporal active learning (human-in-the-loop). Related sampling issue ->
- Can self-supervised approaches be used to fine-tune generic models for specific locations
- Test-time adaption or in-context learning

**Domain Specific Evaluation**
- Given a lack of labelled data for specific locations and times, how do we evaluate model performance
- Domain invariance - We want to use model probability for EBV inference. Model confidence != probability of presence and generally requires additional calibration. Calibration may differ between locations and across time (requiring calibration) especially when fine-tuning. [Overview](https://scikit-learn.org/1.5/modules/calibration.html)
    - Should we be optimising for model performance or calibration?
    - How do you calibrate a model with limited labelled data
- More general evaluation issue when applying fine-tuning

Uncertainty Sampling
- What is the difference between uncertainty sampling (as used for active learning) and anomaly detection?
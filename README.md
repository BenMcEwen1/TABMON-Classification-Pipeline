# TABMON Classification Pipeline
This is a data and classification pipeline repository for the [TABMON](https://www.biodiversa.eu/2024/04/15/tabmon/) (Transnational Acoustic Biodiversity Monitoring Network). The purpose of this repository is to provide an end-to-end framework from raw field data to inference of essential biodiversity variables (EBVs).

Author: Ben McEwen \
**Adapted from the AvesEcho pipeline (Ghani et al. 2024) - [paper](https://arxiv.org/abs/2409.15383)*

### Getting Started 🌱
*Set Up:*\
Create conda or virtualenv environment and specify python version 3.11.10 i.e. `conda create --name tabmon python=3.11.10`.
Install dependencies `pip install -r requirements.txt`

*To Run Inference:*\
**Note that you must specify module `python -m` and run from the base directory as shown.*

Place audio into `audio/` directory and specify directory to analyze `--i` relative to `analyze.py` i.e. `audio/<data>/`.
After installing dependencies run:
```
python -m pipeline.analyze --i 'audio/<data>' --model_name 'birdnet' --device_id '<id>'
```
Both 'birdnet', 'fc' and 'passt' models are setup.

For efficient testing it is recommended to pre-generate sample embeddings before inference.
*To generate embeddings* run:
```
python -m pipeline.embeddings --i 'audio/<data>' --model_name 'birdnet' --device_id '<id>'
```

*Model fine-tuning:*\
The config file `config.yml` specifies file paths and hyperparameters required for training. You will need to set the following:
Pretrained weights path within `/inputs/checkpoints/<weights.pt>`
Annotations path (within data directory) `audio/<data>/<annotations.csv>`
Data path for both training and evaluation sets `audio/<data>/Train/` and `audio/<data>/Test/`

```
python -m pipeline.training
```

*Note: the dataloader is specific to Sounds of Norway annotation protocol, if annotation column names/labels are different you will need to adjust this within `training.py`*

---

### Database Setup
The pipeline can also be run using a FastAPI server which stores predictions in an SQL database.

To *start the server* run:
```
fastapi dev app/main.py
```
The database can be queried using the FastAPI /docs at http://localhost:8000/docs.

---

### Next Steps
- [X] Setup predictions with sql database for efficient querying
- [x] Location-based filtering
- [ ] Specify device information
- [ ] Tidy up redundant AvesEcho/Warbler code

### Structure
`audio/loader.py` - Downloads audio from Google Cloud storage, start and end dates as well as download limit can be specified. \
`pipeline/analyze.py` - Current file for model inference \
`pipeline/embeddings.py` - Pre-generate embeddings for audio samples \

---

### Research Questions 🚀
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
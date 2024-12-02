# TABMON Data Pipeline
This is a data and classification pipeline repository for the TABMON (Transnational Acoustic Biodiversity Monitoring Network). The purpose of this repository is to provide an end-to-end framework from raw field data to inference of essential biodiversity variables (EBVs).

Author: Ben McEwen \
Source: AvesEcho (author: Burooj Ghani) \
Date: 28/11/24

**IMPORTANT!** \
Work in the `/original` directory for now, this is the original set of AvesEcho models with some minor changes. It is more stable than `/pipeline`. Upcoming and recent changes:
- [ ] Remove redundant features from `/original`
- [ ] Transfer uncertainty sampling and custom classification head and *remove* old `/pipeline` 
- [X] Add weights and biases for tracking on server
- [X] Adjusted file selector so that it works with nested directories
- [X] Adjustments for Windows, removed multiprocessing and temp files for now (check if these run on linux server because significantly faster)

*To Run:*\
After installing dependencies navigate to `/original` then run:
`python .\analyze.py --i audio --model_name 'fc' --add_csv --add_filtering`\
Both 'fc' and 'passt' are setup.

## Structure
`loader.py` - Downloads audio from Google Cloud storage, start and end dates as well as download limit can be specified. \
`original/analyze.py` - Current file for model inference \
`pipeline/main.py` - Previous file to run model inference.

## Research Outputs/Questions
This is just a list of research questions I am interested in:
- Spatiotemporal fine-tuning for acoustic species distribution modelling - adapting to specific domains (new locations and specific times)
- Mutliscale audio classification
- Anomaly or out-of-distribution detection.

### Fine-tuning for **Domain Adaption**
- Spatiotemporal active learning (human-in-the-loop). Related sampling issue ->
- Can self-supervised approaches be used to fine-tune generic models for specific locations
- Test-time adaption or in-context learning

### **Domain Specific Evaluation**
- Given a lack of labelled data for specific locations and times, how do we evaluate model performance
- Domain invariance - We want to use model probability for EBV inference. Model confidence != probability of presence and generally requires additional calibration. Calibration may differ between locations and across time (requiring calibration) especially when fine-tuning. [Overview](https://scikit-learn.org/1.5/modules/calibration.html)
    - Should we be optimising for model performance or calibration?
    - How do you calibrate a model with limited labelled data
- More general evaluation issue when applying fine-tuning

### Uncertainty Sampling
- What is the difference between uncertainty sampling (as used for active learning) and anomaly detection?
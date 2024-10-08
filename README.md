# TABMON 
This is a data and classification pipeline repository for the TABMON (Transnational Acoustic Biodiversity Monitoring Network). The purpose of this repository is to provide an end-to-end framework from raw field data to inference of essential biodiversity variables (EBVs).

Author: Ben McEwen \
Date: 08/10/24

## Structure
- Data pipeline - Continuous field data *(24 hour)* is collected using Bugg automatic recording devices that upload data to Google Cloud storage. Data is seperated into *5 minute* segments (*sampled at 44.1 kHz*) for processing. Included in the data pipeline is the following.
    - Optional - Segmentation step to filter empty audio before the classification stage - work by Ghani et al. (2024) previously compared performance with and without segmentation on weakly labelled (Xeno-canto) data demonstrating no meaningful differences in performance. **Question - Discuss potential additional testing.**
- Classification pipeline - Building upon previous works by Ghani et al. (2024) the pre-trained AvesEcho European bird classification model provides the foundation for sound event detection and classification. **Question - An important first step is understand where previous projects (Sounds of Norway) were successful and where improvements could be made.** 
    - Continuous (active learning) pipeline - Due to the spatial and temporal range of sites it is likely that fine-tuning of site-specific models would benefit performance. *How can we integrate new site-specific data into the model in a data and time efficient way while maintaining results that a verifiable and interpretable (i.e. comparing results between sites)*
    - Data and training efficiency - An expert is available for data annotation but due to the amount of data being collected it is likely that only a small proportion of data can be labelled. *How can unlabelled data be effectively ranked in order of highest impact on train efficiency (build upon previous works McEwen et al. (2024))*
    - Data augmentation and supplimentation - A major consideration of this project are class imbalances. *How can augmentation of samples and the addition of alternative data sources be used to improve performance for underrepresented classes (Task 1.3)*
    - **Optional - rare event detection (anomaly detection) which could include rare/at-risk species identification, invasive species identification. This could also include anomalous spatiotemporal distributions - identification of an animal outside of "normal" locations or at unusual times of the year (i.e. Shifts in migratory phenology due to climate change?)**
- Estimation of Essential Biodiversity Variables (EBVs). 
    - Inferring EBVs from sound event detection and classification. Alternative approaches: 1) Treat classified events the same as observations, treating the classification and biodiversity estimation as two separate stages (Task 2.2). 2) Combined approach that incorporates covariate data into classification model and attempt to directly infer biodiversity variables (Task 2.3). *Quantifying and incorporating false detections into estimations.*
    - **Questions - Unified list of target species, short list of relevant EBVs, estimation of biodiversity metrics (distribution, abundance) and the inclusion of covariates?**

## Next Steps
- [ ] Download TABMON data from Google Cloud Storage.
- [ ] Testing of pre-trained AvesEcho models 
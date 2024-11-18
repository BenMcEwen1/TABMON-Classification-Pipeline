## Pipeline
Standardised audio pipeline for downloading and preparing audio samples for inference and model training.

Method based on [AvesEcho](https://arxiv.org/html/2409.15383v1) implementation which applies 3 second chunking of audio recordings applying padding as necessary.

### Other functions
This is where active learning implementation will go and will be used to identify samples for annotation and fine-tune the model on annotated samples.

### Next Steps
- [X] Helper function to split signal into 3s chunks
- [X] PyTorch Dataset and dataloader for preparing inputs
- [ ] Add method for ranking samples in order of highest uncertainty
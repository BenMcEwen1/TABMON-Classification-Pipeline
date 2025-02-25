from pipeline.config import *
from pipeline.util import *

current_dir = os.path.dirname(os.path.abspath(__file__))

@display_time
def renormalizedEntropy(outputs:torch.tensor, top_k:int=3):
    uncertainty = []

    for confidences in outputs:
        if isinstance(top_k, int):
            confidences = torch.topk(confidences, k=top_k)
            confidences = confidences[0]
        
        # Normalize the confidence scores to sum to 1 (if not already normalized)
        confidences = confidences / torch.sum(confidences)
        entropy = -torch.sum(confidences * torch.log(confidences + 1e-10))
        num_classes = len(confidences)
        max_entropy = torch.log(torch.tensor(num_classes))
        
        # Normalize the entropy
        normalized_entropy = entropy / max_entropy
        uncertainty.append(normalized_entropy.item())
    return uncertainty

def uncertaintySTD(outputs:torch.tensor, normalise:bool=True, top_k:int=5):
    """
    Calculate the standard deviation of the output probabilities as a measure of uncertainty.
    Parameters
    ----------
    outputs : tensor
        Output of the model
    normalise : bool, optional
        If set to True, normalise the uncertainty values between 0 and 1, by default True
    top_k : int, optional
        If set to an integer, only consider the top k values of the output probabilities, by default 5
    Returns
    -------
    List of uncertainties per batch
    """
    uncertainty = []

    for p in outputs:
        if isinstance(top_k, int):
            p = torch.topk(p, k=top_k)
            p = p[0]
        uncertainty.append(torch.std(p).item())

    if normalise:
        max_entropy = torch.log(torch.tensor(len(p)))
        uncertainty = [1-(x/max_entropy.item()) for x in uncertainty]
    return uncertainty


def uncertaintyEntropy(outputs:torch.tensor, threshold:float=0.0, normalise:bool=True, top_k:int=3):
    """
    Calculate aggregated uncertainty (entropy) for a batch of multi-label classes.
    Binary Entropy - https://en.wikipedia.org/wiki/Binary_entropy_function

    Parameters
    ----------
    outputs : tensor
        Output of the model
    threshold : float, optional
        Threshold for filtering low uncertainty values, by default 0.1

    Returns: List of uncertainties per batch
    """
    uncertainty = []

    for p in outputs:
        if isinstance(top_k, int):
            p = torch.topk(p, k=top_k)
            p = p[0]
    
        uncertainty_array = -(p*torch.log2(p))-(torch.subtract(1, p))*torch.log2(torch.subtract(1, p))
        uncertainty_array = torch.nan_to_num(uncertainty_array)
        if threshold:
            uncertainty_array = [x for x in uncertainty_array if x > threshold]
        av_uncertainty = sum(uncertainty_array)/len(uncertainty_array)
        uncertainty.append(av_uncertainty.item())

    if normalise:
        uncertainty = [x/max(uncertainty) for x in uncertainty]
    return uncertainty


@display_time
def k_predictions(confidence_batch, energy_scores, filename, species_list, predictions:dict={}, k:int=3, length:int=3, confidence_threshold:float=0.0, energy_metric:str="ROItotal", energy_threshold:float=0.0, filter_list:list=None):
    species_name = load_species_list(species_list)

    if filter_list:
        species_to_index = {species.split(',')[0]: index  for index, species in enumerate(species_name)}
        filtered_indices = [species_to_index[species] for species in filter_list if species in species_to_index]

        confidence_batch = confidence_batch[:, torch.tensor(filtered_indices, dtype=torch.long)]
        species_name = [species_name[index] for index in filtered_indices]

    uncertainty = renormalizedEntropy(confidence_batch)
    
    # Return top k predictions
    results = []
    for i, confidence in enumerate(confidence_batch):
        # Append top k predictions per segment
        top_scores, top_indices = torch.topk(confidence, k=k)
        energy_score = energy_scores[i]
        if max(top_scores) > confidence_threshold:
            pred = []
            for rank, index in enumerate(top_indices):
                scientific_name, common_name = species_name[index].split(',')
                predict = {
                    "rank": rank,
                    "confidence": confidence[index].item(),
                    "scientific name": scientific_name,
                    "common name": common_name,
                }
                pred.append(predict)

            results.append({
                "start time": length*i,
                "uncertainty": uncertainty[i],
                "energy": energy_score,
                "predictions": pred
            })

    predictions["files"].append({
        filename: results
        })
    
    # save predictions to .json
    with open(f'{current_dir}/outputs/predictions.json', 'w', encoding ='utf8') as json_file:
        json.dump(predictions, json_file, indent=4)

    if len(results) > 0:
        predictions = convert_ranked_to_tabular(predictions)
    else: 
        predictions = pd.DataFrame(columns=["filename","start time","uncertainty","energy","rank","confidence","scientific name","common name","device_id","lat","lng","datetime","model","model_checkpoint"])
    
    print(predictions)
    return predictions

def convert_ranked_to_tabular(predictions):
    # Function to convert json object as created by prediction function to table
    rows = []
    metadata = predictions["metadata"]
    files = predictions["files"]

    for file in files:
        filename = list(file)

        for result in file[filename[0]]:
            segment = {
                "filename": filename[0],
                "start time": result["start time"],
                "uncertainty": result["uncertainty"],
                "energy": result["energy"]
            }
            for pred in result["predictions"]:
                row = {
                    **segment,
                    **pred,
                    **metadata
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(f"{current_dir}/outputs/top_k_predictions.csv", index=False)
    return df

@display_time
def energy_metrics(audio_batch, sr):
    energy_scores = []
    for audio in audio_batch:
        Sxx_power,tn,fn,ext = maad.sound.spectrogram(audio, sr.item())  
        df_indices,_ = maad.features.all_spectral_alpha_indices(Sxx_power,tn,fn,extent=ext)
        energy_scores.append(df_indices.to_dict(orient="index")[0])
    return energy_scores

@display_time
def inference(model, data_loader, device, predictions:dict={}, save:bool=True, energy:bool=True, filter_list:list=None):
    '''
    Perform inference on data in directory, outputs prediction results in .json and .csv formats
    Model details:
        BirdNet (v2.4) - logits [1, 6522], emb [B, 1, 1024]
        fc (v2.2) - logits [1, 585], emb [B, 1, 320]
        PaSST - logits [1, 585], emb [B, 1, 768]

    Returns:
    emb (torch.Tensor): The embeddings of the last batch
    '''
    model.eval()  # Set the model to evaluation mode
    torch.set_grad_enabled(False)

    model_name = model.__class__.__name__
    species_list = model.species_list

    # start_time = time.time()
    for i, inputs in enumerate(data_loader):
        audio = inputs['inputs'].to(device)
        sr = inputs['sr'][0]
        emb = inputs['emb'].to(device) # Same for both 'birdnet - v2.4' and 'fc - v2.2'
        filename = inputs['file'][0]

        if model_name == 'birdnet':
            logits = inputs['logits'].to(device)
            confidence_scores = model(logits)
        else:
            outputs = model(audio, emb) # PaSST only require spectrogram (audio), emb is empty
            emb = outputs["emb"]
            confidence_scores = F.sigmoid(outputs['logits'])

    if energy:
        energy_scores = energy_metrics(audio.cpu(), sr)

    if save:
        pred = k_predictions(confidence_scores, 
                             energy_scores, 
                             filename, 
                             species_list, 
                             predictions, 
                             confidence_threshold=0.2, 
                             energy_metric="ROItotal", 
                             energy_threshold=0.0, 
                             filter_list=filter_list)
    return emb, pred
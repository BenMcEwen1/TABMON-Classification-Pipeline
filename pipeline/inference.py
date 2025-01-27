from pipeline.config import *
from pipeline.util import *

current_dir = os.path.dirname(os.path.abspath(__file__))

def renormalizedEntropy(outputs:torch.tensor, top_k:int=3):
    uncertainty = []

    for confidences in outputs:
        if isinstance(top_k, int):
            confidences = torch.topk(confidences, k=top_k)
            confidences = confidences[0]
            
        # Normalize the confidence scores to sum to 1 (if not already normalized)
        confidences = confidences / torch.sum(confidences)
        shannon_entropy = -torch.sum(confidences * torch.log(confidences + 1e-10))
        num_classes = len(confidences)
        max_entropy = torch.log(torch.tensor(num_classes))
        
        # Normalize the entropy
        normalized_entropy = shannon_entropy / max_entropy
        uncertainty.append(normalized_entropy.item())
    return uncertainty

# # Sanity check
# x1 = [0.2, 0.2, 0.2] # Max uncertainty = 1.0
# x2 = [0.299, 0.291, 0.018] # expected high uncertainty
# x3 = [0.3, 0.1, 0.1] # expected medium
# x4 = [0.812, 0.025, 0.021] # expected low uncertainty
# print(renormalizedEntropy(torch.tensor([x1, x2, x3, x4]), top_k=3))


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

def prediction(confidence_batch, filename, species_list, predictions:dict={}, length:int=3, threshold:float=0.0):
    species_name = load_species_list(species_list) # BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.csv
    uncertainty = uncertaintyEntropy(confidence_batch)

    results = []
    for i, confidence in enumerate(confidence_batch):
        # Return maximum confidence prediction
        index = torch.argmax(confidence)
        if confidence[index] > threshold:
            scientific_name, common_name = species_name[index].split(',')
            pred = {
                "start time": length*i,
                "uncertainty": uncertainty[i],
                "confidence": confidence[index].item(),
                "energy": None,
                "scientific name": scientific_name,
                "common name": common_name,
            }
            results.append(pred)
    
    predictions["files"].append({filename: results})

    with open(f"{current_dir}/outputs/prediction.json", "w") as json_file:
        json.dump(predictions, json_file, indent=4)

    convert_to_tabular(predictions)
    return predictions

def k_predictions(confidence_batch, filename, species_list, predictions:dict={}, k:int=3, length:int=3, threshold:float=0.0):
    species_name = load_species_list(species_list)
    uncertainty = renormalizedEntropy(confidence_batch)
    
    # Return top k predictions
    results = []
    for i, confidence in enumerate(confidence_batch):
        # Append top k predictions per segment
        top_scores, top_indices = torch.topk(confidence, k=k)
        if max(top_scores) > threshold:
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
                "energy": None,
                "predictions": pred
            })

    predictions["files"].append({filename: results})

    with open(f"{current_dir}/outputs/top_k_prediction.json", "w") as json_file:
        json.dump(predictions, json_file, indent=4)
    return predictions


def convert_to_tabular(predictions):
    # Function to convert json object as created by prediction function to table
    rows = []
    metadata = predictions["metadata"]
    files = predictions["files"]

    for file in files:
        filename = list(file)
        for result in file[filename[0]]:
            row = {
                "filename": filename[0],
                **result,
                **metadata
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(f"{current_dir}/outputs/predictions.csv", index=False)
    return df

def inference(model, data_loader, device, predictions:dict={}, save:bool=True):
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
    all_filtered_outputs = torch.Tensor()
    all_filtered_outputs = all_filtered_outputs.to(device)

    model_name = model.__class__.__name__
    species_list = model.species_list

    for i, inputs in enumerate(data_loader):
        print(i)
        images = inputs['inputs'].to(device)
        emb = inputs['emb'].to(device) # Same for both 'birdnet - v2.4' and 'fc - v2.2'
        filename = inputs['file'][0]

        if model_name == 'birdnet':
            logits = inputs['logits'].to(device)
            confidence_scores = model(logits)
        else:
            outputs = model(images, emb) # PaSST only require spectrogram (image), emb is empty
            emb = outputs["emb"]
            confidence_scores = F.sigmoid(outputs['logits'])

    if save:
        # prediction(confidence_scores, filename, species_list, predictions, threshold=0.1)
        k_predictions(confidence_scores, filename, species_list, predictions, threshold=0.1)
    return emb


def process_audio(file_path):
    # Load the audio file. Open the file with librosa (limited to the first certain number of seconds)
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, offset=0.0, res_type='kaiser_fast')
    except:
        audio, sr = [], SAMPLE_RATE
    return audio


def max_pool(predictions, scores):
    final_species = []
    final_scores = []
    # Iterate through each chunk
    for chunk_species, chunk_scores in zip(predictions, scores):
        for species, score in zip(chunk_species, chunk_scores):
            if species in final_species:
                # Get the index of the existing species in the final list
                index = final_species.index(species)
                # Compare and keep the higher score
                final_scores[index] = max(final_scores[index], score)
            else:
                # Add new species and its score to the final lists
                final_species.append(species)
                final_scores.append(score)
    return final_species, final_scores            


def T_scaling(logits, temperature):
  return torch.div(logits, temperature)


# Define the function that takes Count and Threshold as input and applies the specified formula
def adjusted_threshold(count, threshold):
    c = count / (count + 100)
    return threshold * c + 0.1 * (1 - c)
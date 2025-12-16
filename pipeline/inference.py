from pipeline.util import *

current_dir = os.path.dirname(os.path.abspath(__file__))

def binaryEntropy(outputs: torch.Tensor, eps=1e-8):
    entropy = -(outputs * torch.log2(outputs + eps) + (1 - outputs) * torch.log2(1 - outputs + eps))
    entropy = torch.nan_to_num(entropy) 
    per_sample_entropy = entropy.max(dim=1).values
    return per_sample_entropy.tolist()


@display_time
def k_predictions(confidence_batch, filename, species_list, predictions:dict={}, k:int=5, length:int=3, confidence_threshold:float=0.0, filter_list:list=None):
    species_name = load_species_list(species_list)

    if filter_list:
        species_to_index = {species.split(',')[0]: index  for index, species in enumerate(species_name)}
        filtered_indices = [species_to_index[species] for species in filter_list if species in species_to_index]

        confidence_batch = confidence_batch[:, torch.tensor(filtered_indices, dtype=torch.long)]
        species_name = [species_name[index] for index in filtered_indices]

    uncertainty = binaryEntropy(confidence_batch)
    
    # Return top k predictions
    results = []
    for i, confidence in enumerate(confidence_batch):
        top_indices = (confidence > confidence_threshold).nonzero(as_tuple=True)[0]

        if len(top_indices) > 5:
            top_indices = top_indices[:4]

        if len(top_indices) > 0:
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
        predictions = pd.DataFrame(columns=["filename","start time","uncertainty","rank","confidence","scientific name","common name","device_id","country","lat","lng","datetime","model","model_checkpoint"])
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
            }
            for pred in result["predictions"]:
                row = {
                    **segment,
                    **pred,
                    **metadata
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    # df.to_parquet(f"{current_dir}/outputs/{filename[0].split('.')[0]}_top_k.parquet", index=False)
    return df



@display_time
def inference(model, data_loader, device, predictions:dict={}, save:bool=True, filter_list:list=None):
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


    if save:
        pred = k_predictions(confidence_scores, 
                             filename, 
                             species_list, 
                             predictions, 
                             confidence_threshold=0.01, 
                             filter_list=filter_list)
    return emb, pred
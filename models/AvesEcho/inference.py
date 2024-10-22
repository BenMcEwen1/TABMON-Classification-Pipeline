import torch
import torch.nn.functional as F
import pandas as pd
import librosa

SAMPLE_RATE = 48000


def inference(model, data_loader, device, mapping, allowed_species, threshold):
    model.eval()  # Set the model to evaluation mode
    torch.set_grad_enabled(False)  # Disable gradient calculation
    all_predictions = []
    all_scores = []
    all_predictions_filtered = []  # Store all predictions
    all_scores_filtered = []
    all_images = []
    all_filtered_outputs = torch.Tensor()
    all_filtered_outputs = all_filtered_outputs.to(device)
    temperature = 0.0149 

    # print(model)

    if allowed_species is not None:
        allowed_indices = [mapping.index(species) for species in allowed_species if species in mapping]

    for i, inputs in enumerate(data_loader):  # Ignore labels
        images = inputs['inputs'].to(device)
        emb = inputs['emb'].to(device)
        files = inputs['file']
        outputs = model(images, emb)  # Forward pass
        # Temperature scaling
        #outputs = T_scaling(outputs, temperature)
        outputs = F.sigmoid(outputs)
        print("Output: ", outputs.shape)


        if allowed_species is None:

            if threshold == None:
                df_thresholds = pd.read_csv('AvesEcho/inputs/species_thresholds_AvesEcho_1.csv')
                sorted_df_thresholds = df_thresholds.sort_values(by='Scientific Name')
                # Apply the function to each row in the sorted DataFrame and create a tensor of adjusted thresholds
                adjusted_thresholds = sorted_df_thresholds.apply(lambda row: adjusted_threshold(row['Count'], row['Threshold']), axis=1)
                adjusted_thresholds_tensor = torch.tensor(adjusted_thresholds.values)

                # Apply the threshold to get predictions mask for the entire output
                predicted_mask = outputs > adjusted_thresholds_tensor.to(device) 
            else:
                predicted_mask = outputs > threshold

            
            # Get indices and scores for non-filtered predictions
            predicted_indices = [torch.nonzero(predicted_mask[i], as_tuple=False).squeeze() for i in range(predicted_mask.shape[0])]
            # Convert tensors in 'predicted' to lists
            predicted_indices = [[tensor.item()] if tensor.nelement() == 1 else tensor.tolist() for tensor in predicted_indices]


            predicted_scores = [outputs[i, predicted_mask[i]] for i in range(outputs.shape[0])]
            predicted_scores = [[tensor.item()] if tensor.nelement() == 1 else tensor.tolist() for tensor in predicted_scores]


            # Replace indices in 'predicted_lists' with species names
            predicted_species = [[mapping[index] for index in sublist] for sublist in predicted_indices]

        else:     
            # Filter the outputs to consider only allowed indices
            # Make sure 'allowed_indices' is correctly defined as per your earlier snippet
            filtered_outputs = outputs[:, allowed_indices]

            if threshold == None:
                df_thresholds = pd.read_csv('inputs/species_thresholds_AvesEcho_1.csv')
                sorted_df_thresholds = df_thresholds.sort_values(by='Scientific Name')
                # Apply the function to each row in the sorted DataFrame and create a tensor of adjusted thresholds
                adjusted_thresholds = sorted_df_thresholds.apply(lambda row: adjusted_threshold(row['Count'], row['Threshold']), axis=1)
                adjusted_thresholds_tensor = torch.tensor(adjusted_thresholds.values)

                adjusted_thresholds_filtered = adjusted_thresholds_tensor[allowed_indices]
                # Apply the threshold to get predictions mask for the filtered output
                predicted_mask_filtered = filtered_outputs > adjusted_thresholds_filtered.to(device)
            else:
                predicted_mask_filtered = filtered_outputs > threshold

            # Get indices and scores for filtered predictions
            predicted_indices_filtered = [torch.nonzero(predicted_mask_filtered[i], as_tuple=False).squeeze() for i in range(predicted_mask_filtered.shape[0])]
            predicted_scores_filtered = [filtered_outputs[i, predicted_mask_filtered[i]] for i in range(filtered_outputs.shape[0])]
            predicted_scores = [[tensor.item()] if tensor.nelement() == 1 else tensor.tolist() for tensor in predicted_scores_filtered]

            # Adjust the extraction of predicted indices for filtered predictions to handle single-element cases
            # Adjusted code to handle tensors with 0 elements in predicted_indices_filtered
            predicted_indices_filtered_adjusted = []
            # Convert the predicted indices for filtered predictions back to original indices

            for indices in predicted_indices_filtered:
                if indices.nelement() == 0:  # Check if the tensor is empty
                    # Handle empty tensor case, e.g., by appending an empty list
                    predicted_indices_filtered_adjusted.append([])
                else:
                    # Non-empty tensor, process as before
                    indices_list = indices.tolist() if indices.nelement() > 1 else [indices.item()]
                    adjusted_indices = [allowed_indices[index] for index in indices_list]
                    predicted_indices_filtered_adjusted.append(adjusted_indices)


            # Replace indices in 'predicted_filtered_lists' with species names
            predicted_species = [[mapping[index] for index in sublist] for sublist in predicted_indices_filtered_adjusted]


        all_predictions.extend(predicted_species)
        all_scores.extend(predicted_scores)
        all_images.extend(files)

    return all_predictions, all_scores, all_images


# def inference_maxpool(model, data_loader, device, mapping, allowed_species, threshold):
#     model.eval()  # Set the model to evaluation mode
#     torch.set_grad_enabled(False)  # Disable gradient calculation
#     all_predictions = []
#     all_scores = []
#     all_predictions_filtered = []  # Store all predictions
#     all_scores_filtered = []
#     all_images = []
#     all_filtered_outputs = torch.Tensor()
#     all_filtered_outputs = all_filtered_outputs.to(device)
#     temperature = 0.0149 

#     if allowed_species is not None:
#         allowed_indices = [mapping.index(species) for species in allowed_species if species in mapping]
#         allowed_species_names = [species for species in allowed_species if species in mapping]

#     for i, inputs in enumerate(data_loader):  # Ignore labels
#         images = inputs['inputs'].to(device)
#         emb = inputs['emb'].to(device)
#         files = inputs['file']
#         outputs = model(images, emb)  # Forward pass
#         # Temperature scaling
#         #outputs = T_scaling(outputs, temperature)
#         outputs = F.sigmoid(outputs)

#         if allowed_species is None:

#             if threshold == None:
#                 df_thresholds = pd.read_csv('AvesEcho/inputs/species_thresholds_AvesEcho_1.csv')
#                 sorted_df_thresholds = df_thresholds.sort_values(by='Scientific Name')
#                 # Apply the function to each row in the sorted DataFrame and create a tensor of adjusted thresholds
#                 adjusted_thresholds = sorted_df_thresholds.apply(lambda row: adjusted_threshold(row['Count'], row['Threshold']), axis=1)
#                 adjusted_thresholds_tensor = torch.tensor(adjusted_thresholds.values)

#                 # Apply the threshold to get predictions mask for the entire output
#                 predicted_mask = outputs > adjusted_thresholds_tensor.to(device) 
#             else:
#                 predicted_mask = outputs > threshold
#             # Apply the threshold to get predictions mask for the entire output
#             #predicted_mask = outputs > threshold

#             # Get indices and scores for non-filtered predictions
#             predicted_indices = [torch.nonzero(predicted_mask[i], as_tuple=False).squeeze() for i in range(predicted_mask.shape[0])]
#             # Convert tensors in 'predicted' to lists
#             predicted_indices = [[tensor.item()] if tensor.nelement() == 1 else tensor.tolist() for tensor in predicted_indices]
#             predicted_scores = [outputs[i, predicted_mask[i]] for i in range(outputs.shape[0])]
#             predicted_scores = [[tensor.item()] if tensor.nelement() == 1 else tensor.tolist() for tensor in predicted_scores]
#             # Replace indices in 'predicted_lists' with species names
#             predicted_species = [[mapping[index] for index in sublist] for sublist in predicted_indices]
#         else:
        
#             # Filter the outputs to consider only allowed indices
#             # Make sure 'allowed_indices' is correctly defined as per your earlier snippet
#             filtered_outputs = outputs[:, allowed_indices]
#             # Apply the threshold to get predictions mask for the filtered output

#             if threshold == None:
#                 df_thresholds = pd.read_csv('inputs/species_thresholds_AvesEcho_1.csv')
#                 sorted_df_thresholds = df_thresholds.sort_values(by='Scientific Name')
#                 # Apply the function to each row in the sorted DataFrame and create a tensor of adjusted thresholds
#                 adjusted_thresholds = sorted_df_thresholds.apply(lambda row: adjusted_threshold(row['Count'], row['Threshold']), axis=1)
#                 adjusted_thresholds_tensor = torch.tensor(adjusted_thresholds.values)

#                 adjusted_thresholds_tensor = adjusted_thresholds_tensor[allowed_indices]
#                 # Apply the threshold to get predictions mask for the entire output
#                 predicted_mask_filtered = filtered_outputs > adjusted_thresholds_tensor.to(device) 
#             else:
#                 predicted_mask_filtered = filtered_outputs > threshold

#             #predicted_mask_filtered = filtered_outputs > threshold
#             # Get indices and scores for filtered predictions
#             predicted_indices_filtered = [torch.nonzero(predicted_mask_filtered[i], as_tuple=False).squeeze() for i in range(predicted_mask_filtered.shape[0])]
#             predicted_scores_filtered = [filtered_outputs[i, predicted_mask_filtered[i]] for i in range(filtered_outputs.shape[0])]
#             predicted_scores = [[tensor.item()] if tensor.nelement() == 1 else tensor.tolist() for tensor in predicted_scores_filtered]
#             # Adjust the extraction of predicted indices for filtered predictions to handle single-element cases
#             # Adjusted code to handle tensors with 0 elements in predicted_indices_filtered
#             predicted_indices_filtered_adjusted = []
#             # Convert the predicted indices for filtered predictions back to original indices
#             for indices in predicted_indices_filtered:
#                 if indices.nelement() == 0:  # Check if the tensor is empty
#                     # Handle empty tensor case, e.g., by appending an empty list
#                     predicted_indices_filtered_adjusted.append([])
#                 else:
#                     # Non-empty tensor, process as before
#                     indices_list = indices.tolist() if indices.nelement() > 1 else [indices.item()]
#                     adjusted_indices = [allowed_indices[index] for index in indices_list]
#                     predicted_indices_filtered_adjusted.append(adjusted_indices)

            
#             # Replace indices in 'predicted_filtered_lists' with species names
#             predicted_species = [[mapping[index] for index in sublist] for sublist in predicted_indices_filtered_adjusted]
        

#         all_predictions.extend(predicted_species)
#         all_scores.extend(predicted_scores)
#         all_images.extend(files)

#     species_maxpool, scores_maxpool = max_pool(all_predictions, all_scores)
   

#     return species_maxpool, scores_maxpool, all_images


def process_audio(file_path):
    # Load the audio file
    # Open the file with librosa (limited to the first certain number of seconds)
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
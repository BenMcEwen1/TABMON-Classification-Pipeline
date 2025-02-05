# Parts of this code are taken/adapted from: https://github.com/google-research/chirp
# Define some utility functions
import os.path
import os
from types import SimpleNamespace
import builtins

from pipeline.algorithm_mode import AlgorithmMode
from pipeline.config import *
from pipeline.species_list import get_species_list

current_dir = os.path.dirname(os.path.abspath(__file__))


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary):
        # Convert all dictionary items to attributes, recursively handling nested dictionaries
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = NestedNamespace(value)  # Recursively convert nested dictionaries
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__dict__}"


@dataclasses.dataclass    
class EmbeddingModel:
  """Wrapper for a model which produces audio embeddings.

  Attributes:
    sample_rate: Sample rate in hz.
  """

  sample_rate: int

  def embed(self, audio_array: np.ndarray) -> np.ndarray:
    """Create evenly-spaced embeddings for an audio array.

    Args:
      audio_array: An array with shape [Time] containing unit-scaled audio.

    Returns:
      An InferenceOutputs object.
    """
    raise NotImplementedError

  def batch_embed(self, audio_batch: np.ndarray) -> np.ndarray:
    """Embed a batch of audio."""
    outputs = []
    for audio in audio_batch:
      outputs.append(self.embed(audio))
    if outputs[0].embeddings is not None:
      embeddings = np.stack([x.embeddings for x in outputs], axis=0)
    else:
      embeddings = None

    return embeddings
    
  def frame_audio(
      self,
      audio_array: np.ndarray,
      window_size_s: "float | None",
      hop_size_s: float,
  ) -> np.ndarray:
    """Helper function for framing audio for inference."""
    if window_size_s is None or window_size_s < 0:
      return audio_array[np.newaxis, :]
    frame_length = int(window_size_s * self.sample_rate)
    hop_length = int(hop_size_s * self.sample_rate)
    # Librosa frames as [frame_length, batch], so need a transpose.
    framed_audio = librosa.util.frame(audio_array, frame_length=frame_length, hop_length=hop_length).T
    return framed_audio

@dataclasses.dataclass
class BirdNet(EmbeddingModel):
  """Wrapper for BirdNet models.

  Attributes:
    model_path: Path to the saved model checkpoint or TFLite file.
    class_list_name: Name of the BirdNet class list.
    window_size_s: Window size for framing audio in samples.
    hop_size_s: Hop size for inference.
    num_tflite_threads: Number of threads to use with TFLite model.
    target_class_list: If provided, restricts logits to this ClassList.
    model: The TF SavedModel or TFLite interpreter.
    tflite: Whether the model is a TFLite model.
    class_list: The loaded class list.
  """

  model_path: str
  class_list_name: str = 'birdnet_v2_1'
  window_size_s: float = 3.0
  hop_size_s: float = 3.0
  num_tflite_threads: int = 16
  target_class_list: "namespace.ClassList | None" = None
  # The following are populated during init.
  model: "Any | None" = None
  tflite: bool = False
  class_list: "namespace.ClassList | None" = None

  def __post_init__(self):
    # logging.info('Loading BirdNet model...')
    if self.model_path.endswith('.tflite'):
      self.tflite = True
      self.model = tf.lite.Interpreter(
          self.model_path, num_threads=self.num_tflite_threads
      )
      self.model.allocate_tensors()
    else:
      self.tflite = False
      

  def embed_tflite(self, audio_array: np.ndarray) -> np.ndarray:
    """Create an embedding and logits using the BirdNet TFLite model."""
    input_details = self.model.get_input_details()[0]
    output_details = self.model.get_output_details()[0]
    embedding_idx = output_details['index'] - 1
    embeddings = []
    logits = []
    for audio in audio_array:
      self.model.set_tensor(
          input_details['index'], np.float32(audio)[np.newaxis, :]
      )
      self.model.invoke()
    
      embeddings.append(self.model.get_tensor(embedding_idx))
      logits.append(self.model.get_tensor(output_details['index']))
    # Create [Batch, 1, Features]
    embeddings = np.array(embeddings)
    logits = np.array(logits)
    
    return embeddings, logits
    

  def embed(self, audio_array: np.ndarray) -> np.ndarray:
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
    
    return self.embed_tflite(framed_audio)
    



def embed_sample(
    embedding_model: EmbeddingModel,
    sample: np.ndarray,
    data_sample_rate: int,
) -> np.ndarray:
  
  """Compute embeddings for an audio sample.

  Args:
    embedding_model: Inference model.
    sample: audio example.
    data_sample_rate: Sample rate of dataset audio.

  Returns:
    Numpy array containing the embeddeding.
  """
  
  
  try:
        if data_sample_rate > 0 and data_sample_rate != embedding_model.sample_rate:
            sample = librosa.resample(
                sample,
                data_sample_rate,
                embedding_model.sample_rate,
                res_type='polyphase',
            )

        audio_size = sample.shape[0]
        if hasattr(embedding_model, 'window_size_s'):
            window_size = int(
                embedding_model.window_size_s * embedding_model.sample_rate
            )
        if window_size > audio_size:
            pad_amount = window_size - audio_size
            front = pad_amount // 2
            back = pad_amount - front + pad_amount % 2
            sample = np.pad(sample, [(front, back)], 'constant')

        outputs = embedding_model.embed(sample)
        
        if outputs is not None:
        #embeds = outputs.embeddings.mean(axis=1).squeeze()
            embed = outputs[0].mean(axis=0).squeeze()
            logits = outputs[1].squeeze().squeeze()

        return embed, logits
        
  except:
        return None


def split_data(filenames):
    # create a dictionary to store filenames grouped by prefix
    prefix_dict = {}
    for filename in filenames:
        prefix = filename.split("/")[-1].split("_")[0]
        if prefix in prefix_dict:
            prefix_dict[prefix].append(filename)
        else:
            prefix_dict[prefix] = [filename]
    # split filenames for each prefix
    train_filenames = []
    test_filenames = []
    for prefix in prefix_dict:
        h = hashlib.sha256(prefix.encode())
        n = int(h.hexdigest(), base=16)
        prefix_filenames = prefix_dict[prefix]
        if n % 4 < 3:
            train_filenames += prefix_filenames
        else:
            test_filenames += prefix_filenames

    return train_filenames, test_filenames


def generate_sampling_weights(labels, list_IDs):
        class_counts = {}
        for key, value in labels.items():
            if value[0] in class_counts:
                class_counts[value[0]] += 1 
            else:
                class_counts[value[0]] = 1
                

        """ class_counts_arr = np.array(list(class_counts.values()))

        # Create a mask for values greater than 2999
        mask = class_counts_arr > 2999
        # Use numpy's interpolation function to scale the values to the new range
        scaled_counts = class_counts_arr.copy()
        scaled_counts[mask] = np.interp(class_counts_arr[mask], (3000, class_counts_arr[mask].max()), (3000, 4000))
        #class_counts_arr[class_counts_arr > 2999] = 3000
        class_counts = dict(zip(class_counts.keys(), scaled_counts)) """        

        sample_weights = np.zeros(len(list_IDs))
        for i, filename in enumerate(list_IDs):
            sample_weights[i] += 1 / class_counts[labels[filename][0]]

        return sample_weights, class_counts    



def save_chunk(args):
    #Function to save a single chunk of audio data to a file.
    chunk, save_path, rate = args
    sf.write(save_path, chunk, rate)


def split_signals(filepath, output_dir, signal_length=15, n_processes=None):
    """
    Function to split an audio signal into chunks and save them using multiprocessing.
    
    Args:
    - filepath: Path to the input audio file.
    - output_dir: Directory where the output chunks will be saved.
    - signal_length: Length of each audio chunk in seconds.
    - n_processes: Number of processes to use in multiprocessing. If None, the number will be determined automatically.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Configure logging
    logging.basicConfig(filename=f'{current_dir}/outputs/audio_errors.log', level=logging.ERROR, # NOTE: Changed
                    format='%(asctime)s:%(levelname)s:%(message)s')

    try:
        # Load the signal
        sig, rate = librosa.load(filepath, sr=SAMPLE_RATE, offset=0.0, duration=None, res_type='kaiser_fast')
    except Exception as e:
        logging.error(f"Error loading audio from {filepath}: {e}")
        return []

    # Split signal into chunks
    sig_splits = [sig[i:i + int(signal_length * rate)] for i in range(0, len(sig), int(signal_length * rate)) if len(sig[i:i + int(signal_length * rate)]) == int(signal_length * rate)]

    # Prepare multiprocessing
    with Pool(processes=n_processes) as pool:
        args_list = []
        for s_cnt, chunk in enumerate(sig_splits):
            save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}_{s_cnt}.wav")
            args_list.append((chunk, save_path, rate))

        # Save each chunk in parallel
        pool.map(save_chunk, args_list)


""" def split_signals(filepath, output_dir, signal_length=15):
    
    # Open the file with librosa (limited to the first certain number of seconds)
    
    try:
        sig, rate = librosa.load(filepath, sr=SAMPLE_RATE, offset=0.0, res_type='kaiser_fast')
         
    except:
        sig, rate = [], SAMPLE_RATE

       
    
    #os.remove(filepath_wav)
    #sig, rate = librosa.load(filepath, sr=SAMPLE_RATE, offset=None, duration=60)
    #sig, rate = librosa.load(filepath, sr=SAMPLE_RATE, offset=None)
     
    # Split signal into five second chunks
    sig_splits = []
    for i in range(0, len(sig), int(signal_length * SAMPLE_RATE)):
        split = sig[i:i + int(signal_length * SAMPLE_RATE)]

        # End of signal?
        if len(split) < int(signal_length * SAMPLE_RATE):
            break
        
        sig_splits.append(split)

    # Extract npy array for each audio chunk and save it in its respective labeled folder
    s_cnt = 0
    saved_samples = []
    
    for chunk in sig_splits:
        
        save_path = os.path.join(output_dir, filepath.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] + 
                                 '_' + str(s_cnt) + '.wav')
        #librosa.output.write_wav(save_path, chunk, SAMPLE_RATE) 
        sf.write(save_path, chunk, SAMPLE_RATE)                        

        
        saved_samples.append(save_path)
        s_cnt += 1   """


def compute_dataset_stats(partition):
    total_mean = 0

    for file_path in tqdm(partition['train'], desc="Processing audio files"):
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None, offset=0.0, res_type='kaiser_fast')  # sr=None ensures original sampling rate is used

        # Compute and accumulate the mean for this file
        file_mean = np.mean(audio)
        total_mean += file_mean
        #total_files += 1

    # Calculate the global mean
    global_mean = total_mean / len(partition['train'])

    # For standard deviation, a second pass is needed
    total_var = 0
    for file_path in tqdm(partition['train'], desc="Processing audio files for std dev"):
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None, offset=0.0, res_type='kaiser_fast')

        # Accumulate the variance
        total_var += np.sum((audio - global_mean) ** 2)

    # Calculate the global standard deviation
    global_std = np.sqrt(total_var / (sum(len(librosa.load(fp, sr=None, offset=0.0, res_type='kaiser_fast')[0]) for fp in partition['train'])))

    return global_mean, global_std


def extract_species_recording_ids(file_paths):
    species_to_ids = {}
    for path in file_paths:
        parts = path.split('/')
        species_name = parts[-2]
        recording_id = parts[-1].split('_')[0]
        if species_name in species_to_ids:
            species_to_ids[species_name].append(recording_id)
        else:
            species_to_ids[species_name] = [recording_id] 
        
    return species_to_ids    


# Function to convert seconds to "minutes:seconds" format
def format_time(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    # Format seconds with leading zero if necessary
    return f"{minutes}:{seconds:02d}"

def get_base_name(filename):
    return filename.split('/')[-1].split('.')[0]

def create_json(algorithm_mode, analysis_results, predictions, scores, files, args, df, add_csv, fname, m_conf, filtered=False):

     # Create a predictions file name
    filename_without_ext = fname.split('.')[0]  
    filename = filename_without_ext.split('\\')[-1]
    pred_name = './outputs/predictions_' + filename

    if add_csv:
        with open(f'{pred_name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Begin Time", "End Time", "File", "Prediction", "Score"])  # write header

    # Include the input directory when running in directories mode and args.i is referring to a directory.
    file_path = f'{args.i}/{fname}' if algorithm_mode == AlgorithmMode.DIRECTORIES and os.path.isdir(args.i) else fname

    # Add each file to the 'media' list in analysis_results
    analysis_results['media'].append({"filename": file_path, "id": fname})
    
    for i, prediction in enumerate(predictions):
        prediction_sp = []
        begin_time = i * 3
        end_time = begin_time + 3
        formatted_begin_time = format_time(begin_time)
        formatted_end_time = format_time(end_time)
        
        # Set a threshold for scores, 0.1 for unfiltered and 0.2 for filtered
        threshold = m_conf
        for name, score in zip(prediction, scores[i]):

            row = df[df['ScientificName'] == name]
            # Extract the ScientificName from the matched row
            common_name = row['CommonName'].values[0] if not row.empty else 'Not found'
            prediction_sp.append(common_name)
            
            if args.add_csv:
                with open(f'{pred_name}.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    #writer.writerow([begin_time, end_time, files[i], name, score]) # uncomment for time in seconds in csv
                    writer.writerow([formatted_begin_time, formatted_end_time, fname, f'{common_name}_{name}', score]) # uncomment for time in minutes:seconds in csv
        
        region_group_id = f"{file_path}?region={i}"

        add_predictions(region_group_id, fname, begin_time, end_time, prediction, scores[i], analysis_results)

    # Determine the output file name based on filtering
    # json_name = f'{pred_name}.json'
    
    # Write the analysis results to a JSON file
    # with open(json_name, 'w') as json_file:
    #     json.dump(analysis_results, json_file, indent=4)


def create_json_maxpool(algorithm_mode, analysis_results, predictions, scores, files, args, df, add_csv, fname, m_conf, length, filtered=False):

    # Create a predictions file name
    filename_without_ext = fname.split('.')[0]  
    pred_name = 'outputs/predictions_' + filename_without_ext
    

    if add_csv:
        with open(f'{pred_name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File", "Prediction", "Score"])  # write header
    
    begin_time = 0
    end_time = length*3

    # Include the input directory when running in directories mode and args.i is referring to a directory.
    file_path = f'{args.i}/{fname}' if algorithm_mode == AlgorithmMode.DIRECTORIES and os.path.isdir(args.i) else fname

    # Add each file to the 'media' list in analysis_results
    analysis_results['media'].append({"filename": file_path, "id": fname})

    prediction_sp = []
        
        
    # Set a threshold for scores, 0.1 for unfiltered and 0.2 for filtered
    threshold = m_conf
    for name, score in zip(predictions, scores):

        row = df[df['ScientificName'] == name]
        # Extract the ScientificName from the matched row
        common_name = row['CommonName'].values[0] if not row.empty else 'Not found'
        prediction_sp.append(common_name)
        
        if args.add_csv:
            with open(f'{pred_name}.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                #writer.writerow([begin_time, end_time, files[i], name, score]) # uncomment for time in seconds in csv
                writer.writerow([fname, common_name, score]) # uncomment for time in minutes:seconds in csv
    
    region_group_id = f"{file_path}?region={0}"

    add_predictions(region_group_id, fname, begin_time, end_time, predictions, scores, analysis_results)


def add_predictions(
    region_group_id: str,
    media_id: str,
    begin_time: float,
    end_time: float,
    scientific_names: list[str],
    probabilities: list[float],
    analysis_results: dict[str, Any]
) -> None:
    assert len(scientific_names) == len(probabilities)

    if scientific_names:
        analysis_results["region_groups"].append(
            {
                "id": region_group_id,
                "regions": [
                    {
                        "media_id": media_id,
                        "box": {
                            "t1": float(begin_time),
                            "t2": float(end_time)
                        }
                    }
                ]
            }
        )

    for prediction_index, scientific_name in enumerate(scientific_names):
        analysis_results["predictions"].append(
            {
                "region_group_id": region_group_id,
                "taxa": {
                    "type": "multilabel",
                    "items": [
                        {
                            "scientific_name": scientific_name,
                            "probability": probabilities[prediction_index],
                            # Fill in taxon_id if available: "taxon_id": "",
                        }
                    ]
                }
            }
        )


def load_species_list(path):
    """Load the species list from a file."""
    species_list = []
    with open(path) as file:
        for line in file:
            species_list.append(line.strip())
    return sorted(species_list)


""" def setup_filtering(args, species_list):
    # Setup filtering based on geographic location.
    if args.lat != None and args.lon != None:
        filtering_list_series = get_species_list(args.lat, args.lon)
        filtering_list = filtering_list_series['birdlife_scientific_name'].tolist()
    elif not args.add_filtering:
        filtering_list = species_list
    else: 
        filtering_list = []
        with open(args.flist) as f:
            for line in f:
                filtering_list.append(line.strip())
            #filtering_list.append('Noise')
    return filtering_list  """


def setup_filtering(lat, lon, add_filtering, flist, species_list):
    """Setup filtering based on geographic location."""
    if lat != None and lon != None:
        filtering_list_series = get_species_list(lat, lon)
        filtering_list = filtering_list_series['birdlife_scientific_name'].tolist()
    elif not add_filtering:
        filtering_list = None
    else: 
        filtering_list = []
        with open(flist) as f:
            for line in f:
                filtering_list.append(line.strip())
    return filtering_list 

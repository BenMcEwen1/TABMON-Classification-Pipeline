from pipeline.config import *

ENABLE_PROFILING = False  # Toggle this to enable/disable timing

current_dir = os.path.dirname(os.path.abspath(__file__))

def get_species_list(recording_lat, recording_long):
    "Concert WGS84 lat-lon to MGRS UTM coordinates"

    # Load occurances data from csv file
    file_path = "./pipeline/inputs/ebba2_data_occurrence_50km.csv"
    df = pd.read_csv(file_path, delimiter=';')
    
    m = mgrs.MGRS()
    c = m.toMGRS(recording_lat, recording_long)

    input_code = c[0:3] 
    filtered_df = df[df['cell50x50'].str.startswith(input_code)]
    species_list = pd.DataFrame(filtered_df['birdlife_scientific_name'].drop_duplicates())
    return species_list


def display_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if ENABLE_PROFILING:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"[{func.__name__}] took {end_time - start_time:.4f} seconds")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

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

def save_chunk(args):
    # Saves a single chunk of audio data to a file.
    chunk, save_path, rate = args
    sf.write(save_path, chunk, rate, subtype='PCM_16')

@display_time
def ROIfilter(audio, sr, threshold:float=0.01):
    EPS = np.finfo(float).eps
    Sxx_power,tn,fn,_ = maad.sound.spectrogram(audio, sr)
    Sxx_power[fn > 18000,:] = EPS # remove aliasing in high frequencies due to upsampling
    Sxx_power[fn < 200,:] = EPS # remove low frequencies
    if Sxx_power.max() > 0 :
        Sxx_noNoise= maad.sound.median_equalizer(Sxx_power) 
        Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)
        _, ROIcover = maad.features.region_of_interest_index(Sxx_dB_noNoise, tn, fn)
        if ROIcover < threshold:
            return True
    return False

@display_time
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
    logging.basicConfig(filename=f'{current_dir}/outputs/audio_errors.log', level=logging.ERROR, # NOTE: Changed
                    format='%(asctime)s:%(levelname)s:%(message)s')
    try:
        sig, original_rate = librosa.load(filepath, sr=None)
        sig = librosa.resample(sig, orig_sr=original_rate, target_sr=SAMPLE_RATE)
        if len(sig) < (signal_length * SAMPLE_RATE):
            raise Exception(f"Audio {filepath} is too short (min. length 3 seconds)")
    except Exception as error:
        logging.error(error)
        raise Exception(error)

    roicover = ROIfilter(sig, SAMPLE_RATE)
    if roicover:
        print("skipping audio - ROI Filter")
        return None # Skip

    sig_splits = [sig[i:i + int(signal_length * SAMPLE_RATE)] for i in range(0, len(sig), int(signal_length * SAMPLE_RATE)) if len(sig[i:i + int(signal_length * SAMPLE_RATE)]) == int(signal_length * SAMPLE_RATE)]

    # Prepare multiprocessing
    with ThreadPoolExecutor(max_workers=None) as executor:
        args_list = []
        for s_cnt, chunk in enumerate(sig_splits):
            save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filepath))[0].lower()}_{s_cnt}.wav")
            args_list.append((chunk, save_path, SAMPLE_RATE))
        executor.map(save_chunk, args_list)
    return True


def load_species_list(path):
    """Load the species list from a file."""
    species_list = []
    with open(path) as file:
        for line in file:
            species_list.append(line.strip())
    return sorted(species_list)


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
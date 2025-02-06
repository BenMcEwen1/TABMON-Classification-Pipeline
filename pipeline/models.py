import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import flask

from werkzeug.datastructures import FileStorage
from enum import Enum

from pipeline.config import *
from pipeline.dataset import *
from pipeline.inference import *
from pipeline.util import *

# from pipeline.algorithm_mode import AlgorithmMode
from pipeline.passt.base import get_basic_model, get_model_passt

current_dir = os.path.dirname(os.path.abspath(__file__))

classifier: "AvesEcho"


# TODO: Things to Test
# - Try different training dataset sizes
# - Plot attention scores for target and context
# - Sample selection with redundant point removal (or SIFT)
# - Dimensionity reduction + PCA of embedding space
# - Quantify uncertainty and estimate uncertainty reduction
# - Compare offline context sampling i.e. compute all the embeddings for both train and test sets

# Uncertainty Sampling 
# - 

# Other Datasets
# - Norway secondary dataset
# - WABAD dataset
# - BEANS benchmark

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, emb):
        x = self.fc1(emb.squeeze(1))
        return x
    

class ContextAwareHead(nn.Module):
    # Context aware classification head for any generic feature extractor
    def __init__(self, n_classes:int, externalEmbeddingSize:int=320, heads:int=2, context:int=0):
        super(ContextAwareHead, self).__init__()
        self.target_dim = 1 + context
        self.externalEmbeddingSize = externalEmbeddingSize
        self.multihead = nn.MultiheadAttention(self.externalEmbeddingSize, heads, batch_first=True)
        self.head = nn.Linear(externalEmbeddingSize*self.target_dim, n_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

    def add_context(self, target:torch.Tensor, embeddings:torch.Tensor):
        # Pad context embeddings to target dimension
        combined = torch.cat((target, embeddings), dim=1)
        current_dim = combined.shape[1]
        if current_dim > self.target_dim:
            cropped = combined[:,:self.target_dim,:]
            return cropped
        else:
            padding = torch.zeros(target.shape[0], self.target_dim-current_dim, self.externalEmbeddingSize) # B, padding, embedding
            padded = torch.cat((combined, padding), dim=1)
            return padded

    def forward(self, target:torch.Tensor, context:torch.Tensor):
        combined = self.add_context(target, context)
        x = combined / combined.norm(dim=-1, keepdim=True) # Normalised

        x, _ = self.multihead(x, x, x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.head(x)
        return x

class avesecho(nn.Module):
    def __init__(self, NumClasses=585, pretrain=True, ExternalEmbeddingSize=320, hidden_layer_size=100):
        super(avesecho, self).__init__()
        self.fc1 = nn.Linear(ExternalEmbeddingSize, NumClasses)
        self.species_list = f"{current_dir}/inputs/list_AvesEcho.csv"
        
    def forward(self, x, emb):
        x = self.fc1(emb.squeeze(1))
        return {"emb": emb, "logits": x}    
    
class birdnet(nn.Module):
    """
        This class is an empty wrapper for the BirdNET model required for pipeline consistency
        NOTE: we are using the birdnet model prediction directly hence why the logits are carried forwards. 
    """
    def __init__(self):
        super(birdnet, self).__init__()
        self.species_list = f"{current_dir}/inputs/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.csv"
        
    def forward(self, logits):
        return F.sigmoid(logits)


def run_algorithm(args):
    global classifier

    classifier = AvesEcho(args=args)

    results = classifier.analyze(audio_input=args.i, lat=args.lat, lng=args.lng)
    return results


class AlgorithmMode(Enum):
    DIRECTORIES = "directories"
    ENDPOINT = "endpoint"


class AvesEcho:
    def __init__(self, args):
        self.slist = args.slist
        self.flist = args.flist
        self.model_name = args.model_name

        if args.flist or args.lat and args.lng:
            self.add_filtering = True # Necessary when we get a species list :/
        else:
            self.add_filtering = False

        self.outputd = f"{current_dir}/outputs/temp"
        self.species_list = load_species_list(self.slist)
        self.n_classes = len(self.species_list)
        self.split_signals = split_signals
        self.args = args 
        self.algorithm_mode = AlgorithmMode("directories")

        print(f"Running AvesEcho-v1 in {self.algorithm_mode.value} mode, model: {self.model_name}, device: {device}.")

        # Load the model
        if self.model_name == 'passt':
            self.model = get_basic_model(mode = 'logits', arch="passt_s_kd_p16_128_ap486") # "passt_s_kd_p16_128_ap486"
            self.model.net =  get_model_passt(arch="passt_s_kd_p16_128_ap486", n_classes=self.n_classes)
            self.model = self.model.to(device)
            self.model.load_state_dict(torch.load(f'{current_dir}/inputs/checkpoints/best_model_passt.pt', map_location=device))
        if self.model_name == 'fc':
            self.model = avesecho(NumClasses=self.n_classes)
            self.model = self.model.to(device)
            self.model.load_state_dict(torch.load(f'{current_dir}/inputs/checkpoints/best_model_fc_1.pt', map_location=device))
        if self.model_name == 'birdnet':
            self.model = birdnet()
            self.model = self.model.to(device)


    def analyze(self, audio_input: str, lat: Optional[Any]=None, lng: Optional[Any]=None):
        if self.algorithm_mode == AlgorithmMode.DIRECTORIES:
            results = self.analyze_directories(audio_input, lat, lng)
        return results

    def generate_embeddings(self, audio_path: str, regenerate: bool, save: bool):
        def emb(audio_file_path, filename):
            # Load soundfile and split signal into 3s chunks
            self.split_signals(audio_file_path, self.outputd, signal_length=3, n_processes=10)

            # Load a list of files for in a dir
            inference_dir = self.outputd
            inference_data = [
                os.path.join(inference_dir, f)
                for f in sorted(os.listdir(inference_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))]

            # Inference
            inference_set = InferenceDataset(inference_data, filename, model=self.model_name)
            params_inf = {'batch_size': 124, 'shuffle': False}
            inference_generator = torch.utils.data.DataLoader(inference_set, **params_inf)
            embeddings,_ = inference(self.model, inference_generator, device, save=False)
            return embeddings
        
        # Get File list
        if os.path.isfile(audio_path):
            audio_files = [audio_path]
        else:
            audio_files = []
            for path, _, files in os.walk(audio_path):
                file_path = [os.path.join(path, filename) for filename in files
                             if not self.ignore_filesystem_object(audio_path, filename)]
                audio_files += file_path
        supported_file_extensions = ['.wav', '.mp3', '.ogg', '.flac']

        # Iterate through each audio file
        with tempfile.TemporaryDirectory() as temporary_directory:
            print(f"Generating embeddings for {len(audio_files)} files.")
            for audio_file in tqdm(audio_files, leave=True, dynamic_ncols=True):
                filename = os.path.basename(audio_file) if self.algorithm_mode == AlgorithmMode.DIRECTORIES else audio_file.filename
                file_extension = os.path.splitext(filename)[1].lower()

                if file_extension in supported_file_extensions:
                    if self.algorithm_mode == AlgorithmMode.DIRECTORIES:
                        audio_file_path = audio_file
                    else:
                        audio_file_path = os.path.join(temporary_directory, filename)
                        with open(audio_file_path, "wb") as output_file:
                            output_file.write(audio_file.read())

                    embedding_path = os.path.join(os.path.dirname(audio_file_path), os.path.splitext(filename)[0].lower() + f"_{self.model_name}.pt")

                    # Check if embedding path exists
                    if (not os.path.exists(embedding_path)) or regenerate:
                        _embeddings, _ = emb(audio_file_path, filename)
                        if save:
                            torch.save(_embeddings, embedding_path)
                        try:
                            shutil.rmtree(self.outputd)
                        except:
                            pass
                    else:
                        _embeddings = torch.load(embedding_path)
        return _embeddings

    def analyze_directories(self, audio_input: str, lat: Optional[Any]=None, lng: Optional[Any]=None):
        if os.path.isfile(audio_input):
            audio_files = [audio_input]
        else:
            audio_files = []
            for path, _, files in os.walk(audio_input):
                file_path = [os.path.join(path, filename) for filename in files
                             if not self.ignore_filesystem_object(audio_input, filename)]
                audio_files += file_path

        pred,_ = self.analyze_audio_files(audio_files, lat, lng)
        return pred

    def ignore_filesystem_object(self, directory: str, filename: str) -> bool:
        return os.path.isdir(os.path.join(directory, filename)) or filename.startswith(".")

    def analyze_audio_files(self, audio_files: list[Union[str, FileStorage]], lat: Optional[Any]=None, lng: Optional[Any]=None):
        # Running the model to get predictions, and then returning the results.

        filtering_list = setup_filtering(lat, lng, self.add_filtering, self.flist, self.slist)

        predictions = {
            "metadata":  {
                "device_id": self.args.device_id,
                "lat": self.args.lat,
                "lng": self.args.lng,
                "datetime": str(datetime.now()),
                "model": self.model_name,
                "model_checkpoint": self.args.model_checkpoint,
            },
            "files": []
        }

        supported_file_extensions = ['.wav', '.mp3', '.ogg', '.flac']

        # For endpoint mode: write the audio files to a temporary directory.
        with tempfile.TemporaryDirectory() as temporary_directory:
            for audio_file in tqdm(audio_files):
                filename = os.path.basename(audio_file) if self.algorithm_mode == AlgorithmMode.DIRECTORIES else audio_file.filename
                file_extension = os.path.splitext(filename)[1].lower()

                if file_extension in supported_file_extensions:
                    if self.algorithm_mode == AlgorithmMode.DIRECTORIES:
                        audio_file_path = audio_file
                    else:
                        audio_file_path = os.path.join(temporary_directory, filename)
                        with open(audio_file_path, "wb") as output_file:
                            output_file.write(audio_file.read())
                    try:
                        pred = self.analyze_audio_file(audio_file_path, filename, filtering_list, predictions)
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        pass
            return pred, 200

    def analyze_audio_file(self, audio_file_path: str, filename:str, filtering_list: list[str], predictions: dict):
        if not os.path.exists(self.outputd):
            os.makedirs(self.outputd)

        # Load soundfile and split signal into 3s chunks
        self.split_signals(audio_file_path, self.outputd, signal_length=3, n_processes=10)

        # Load a list of files for in a dir
        inference_data = [
            os.path.join(self.outputd, f)
            for f in sorted(os.listdir(self.outputd), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        ]

        # Inference
        inference_set = InferenceDataset(inference_data, filename, model=self.model_name)
        params_inf = {'batch_size': 64, 'shuffle': False} # , 'num_workers': 5
        inference_generator = torch.utils.data.DataLoader(inference_set, **params_inf)

        audio_path = "./audio/"
        embeddings, pred = inference(self.model, inference_generator, device, predictions, filter_list=filtering_list)
        embedding_dir = os.path.join(os.path.dirname(audio_path), "embeddings/")
        embedding_filename = os.path.splitext(filename)[0].lower() + ".pt"
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)

        torch.save(embeddings, os.path.join(embedding_dir, embedding_filename))

        # Filter segments
        segment_dir = os.path.join(os.path.dirname(audio_path), "segments/")
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir)

        filtered = pred[["filename", "start time"]].drop_duplicates()
        for _,row in filtered.iterrows():
            obj_dict = row.to_dict()
            index = int(obj_dict['start time']/3)
            segment_filename = os.path.splitext(obj_dict["filename"])[0].lower() + f"_{index}.wav"
            
            path = os.path.join(self.outputd, segment_filename)
            if os.path.exists(path):
                shutil.copy2(path, segment_dir)

        try:
            shutil.rmtree(self.outputd)
        except:
            pass

        return pred

 
    def determine_algorithm_mode(self, algorithm_mode: str) -> AlgorithmMode:
        """Determine the mode ("directories" or "endpoint") the algorithm will run in. First the "--algorithm-mode"
           command-line argument is used to determine the algorithm mode. If that is not specified, the "ALGORITHM_MODE"
           environment variable determines the mode. If both are not specified, algorithm mode "directories" is used by
           default."""
        try:
            result = AlgorithmMode(algorithm_mode)

        except ValueError:
            print(f"Unknown algorithm mode '{algorithm_mode}' - defaulting to 'directories' mode.")
            result = AlgorithmMode.DIRECTORIES

        return result

    def str2bool(self, value) -> bool:
        """
        Converts a string to boolean by matching it with some well known representations of bool.
        {"yes", "true", "t", "y", "1"} --> true
        If value is not one of these representations it returns False.
        :param value: a string representing a boolean value
        :return: a boolean value
        """
        return str(value).lower() in {"yes", "true", "t", "y", "1"}

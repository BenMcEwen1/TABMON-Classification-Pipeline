from pipeline.config import *
from pipeline.dataset import *
from pipeline.inference import *
from pipeline.util import *

from pipeline.passt.base import get_basic_model, get_model_passt

current_dir = os.path.dirname(os.path.abspath(__file__))

classifier: "TABMON"

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

@display_time
def run_algorithm(args, id=None):
    global classifier
    classifier = AvesEcho(args=args, id=id)
    embeddings, results = classifier.analyze_directories(audio_input=args.i, lat=args.lat, lng=args.lng)
    print("predictions generated")
    return embeddings, results


class AvesEcho:
    def __init__(self, args, id):
        self.slist = args.slist
        self.flist = args.flist
        self.model_name = args.model_name
        self.outputd = f"{current_dir}/outputs/temp/{id}"
        self.species_list = load_species_list(self.slist)
        self.n_classes = len(self.species_list)
        self.split_signals = split_signals
        self.args = args 

        if args.flist or args.lat and args.lng:
            self.add_filtering = True # Necessary when we get a species list :/
        else:
            self.add_filtering = False

        print(f"Running TABMON-v1 - model: {self.model_name}, device: {device}.")

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

    def analyze_directories(self, audio_input:str, lat:Optional[Any]=None, lng:Optional[Any]=None):
        if os.path.isfile(audio_input):
            audio_files = [audio_input]
        else:
            audio_files = []
            for path, _, files in os.walk(audio_input):
                file_path = [os.path.join(path, filename) for filename in files
                             if not self.ignore_filesystem_object(audio_input, filename)]
                audio_files += file_path

        embeddings, pred = self.analyze_audio_files(audio_files, lat, lng)
        return embeddings, pred

    def ignore_filesystem_object(self, directory: str, filename: str) -> bool:
        return os.path.isdir(os.path.join(directory, filename)) or filename.startswith(".")

    def analyze_audio_files(self, audio_files: list, lat: Optional[Any]=None, lng: Optional[Any]=None):
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
            for audio_file in tqdm(audio_files, disable=True):
                filename = os.path.basename(audio_file) 
                file_extension = os.path.splitext(filename)[1].lower()

                if file_extension in supported_file_extensions:
                    audio_file_path = audio_file
                    embeddings, pred = self.analyze_audio_file(audio_file_path, filename, filtering_list, predictions)
            return embeddings, pred

    def analyze_audio_file(self, audio_file_path: str, filename:str, filtering_list: list[str], predictions: dict):
        if not os.path.exists(self.outputd):
            os.makedirs(self.outputd)

        # Load soundfile and split signal into 3s chunks
        status = self.split_signals(audio_file_path, self.outputd, signal_length=3, n_processes=None)
        if status == None:
            print("Skipping file")
            return None, None # Skip
        
        # Load a list of files for in a dir
        inference_data = [
            os.path.join(self.outputd, f)
            for f in sorted(os.listdir(self.outputd), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        ]

        print("Running inference")

        # Inference
        inference_set = InferenceDataset(inference_data, filename, model=self.model_name)
        params_inf = {'batch_size': 124, 'shuffle': False} # , 'num_workers': 5
        inference_generator = torch.utils.data.DataLoader(inference_set, **params_inf)

        audio_path = "./audio/"
        embeddings, pred = inference(self.model, inference_generator, device, predictions, filter_list=filtering_list)
        
        # Make embeddings directory
        embedding_dir = os.path.join(os.path.dirname(audio_path), "embeddings/")
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)

        # Make segments directory
        segment_dir = os.path.join(os.path.dirname(audio_path), "segments/")
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir)

        # Save segments and embeddings
        filtered = pred[["filename", "start time"]].drop_duplicates()
        for _,row in filtered.iterrows():
            obj_dict = row.to_dict()
            index = int(obj_dict['start time']/3)
            segment_filename = os.path.splitext(obj_dict["filename"])[0].lower() + f"_{index}.wav"

            # Save embedding
            embedding_filename = os.path.splitext(filename)[0].lower() + f"_{index}.pt"
            torch.save(embeddings[index], os.path.join(embedding_dir, embedding_filename))
            
            path = os.path.join(self.outputd, segment_filename)
            try:
                if os.path.exists(path):
                    shutil.copy2(path, segment_dir)
            except Exception as e:
                print(f"Error copying {path} to {segment_dir}: {e}")
        try:
            shutil.rmtree(self.outputd)
        except:
            pass
        return embeddings, pred
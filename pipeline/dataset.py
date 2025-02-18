from pipeline.config import *
from pipeline.util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

current_dir = os.path.dirname(os.path.abspath(__file__))

           
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, filename, model):
        # Initialization
        self.list_IDs = list_IDs
        self.model = model
        self.sampleRate = 48000
        self.filename = filename

        if self.model == "birdnet":
            self.embedding_model = BirdNet(self.sampleRate, f'{current_dir}/inputs/checkpoints/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite')
        else:
            self.embedding_model = BirdNet(self.sampleRate, f'{current_dir}/inputs/checkpoints/BirdNET_GLOBAL_3K_V2.2_Model_FP32.tflite')

        with open(f'{current_dir}/inputs/global_parameters.json', 'r') as json_file:
            parameters = json.load(json_file)

        self.global_mean = parameters['global_mean']
        self.global_std = parameters['global_std']

    def __len__(self):
        # Denotes the total number of samples
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates one sample of data
        ID = self.list_IDs[index]

        # Open the file with librosa (limited to the first certain number of seconds)
        try:
            x, rate = librosa.load(ID, sr=SAMPLE_RATE, offset=0.0, res_type='kaiser_fast')
        except:
            x, rate = [], SAMPLE_RATE

        birdnet_embedding = np.zeros(320)
        
        x = (x - self.global_mean) / self.global_std
        # convert mixed to tensor
        x = torch.from_numpy(x).float() 

        if self.model == 'passt':
            # Resample the audio from 48k to 32k for PaSST
            resampler = T.Resample(SAMPLE_RATE, SAMPLE_RATE_AST, dtype=x.dtype)
            x = resampler(x)  
            # Create dummy embedding 
            birdnet_embedding = np.zeros(320) 
            logits = np.zeros(585)
        if self.model == 'fc':
            # Compute BirdNETv2.2 embedding
            try:
                outputs, logits = embed_sample(self.embedding_model, x.numpy(), SAMPLE_RATE)
                birdnet_embedding = np.expand_dims(outputs, axis=0)
            except:
                print("fc embedding failed")
                birdnet_embedding = np.zeros(320, dtype=np.float32)
        if self.model == 'birdnet':
            try:
                outputs, logits = embed_sample(self.embedding_model, x.numpy(), SAMPLE_RATE)
                birdnet_embedding = np.expand_dims(outputs, axis=0)
            except:
                print("BirdNET embedding failed")
                birdnet_embedding = np.zeros(1024, dtype=np.float32)
       
        return {'inputs': x, 'sr': rate, 'emb': birdnet_embedding, 'file': self.filename, 'logits': logits} 
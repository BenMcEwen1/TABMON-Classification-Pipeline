from config import *
from util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

           
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, NumClasses, model):
        # Initialization
        self.list_IDs = list_IDs
        self.NumClasses = NumClasses
        self.embedding_model = BirdNet(48000, 'checkpoints/BirdNET_GLOBAL_3K_V2.2_Model_FP32.tflite')
        self.model = model

        with open('inputs/global_parameters.json', 'r') as json_file:
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

        
        x = (x - self.global_mean) / self.global_std
        #convert mixed to tensor
        x = torch.from_numpy(x).float() 


        if self.model == 'passt':
            #Resample the audio from 48k to 32k for PaSST
            resampler = T.Resample(SAMPLE_RATE, SAMPLE_RATE_AST, dtype=x.dtype)
            x = resampler(x)  
            # Create dummy embedding 
            birdnet_embedding = np.zeros(320) 
        else:
            # Compute BirdNET embedding
            try:
                outputs, logits = embed_sample(self.embedding_model, x.numpy(),SAMPLE_RATE)
                birdnet_embedding = np.expand_dims(outputs, axis=0)
            except:
                print("BirdNET embedding failed")
                birdnet_embedding = np.zeros(320)  

         
        return {'inputs': x, 'emb': birdnet_embedding, 'file': ID} 
    
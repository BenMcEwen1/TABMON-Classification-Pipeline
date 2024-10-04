# Example demonstrating the use of AST on TABMON, single 10 second segment

from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
import librosa
import torch.nn as nn

audio, sr = librosa.load('./audio/proj_sound-of-norway_bugg_RPiID-10000000cc849698_conf_6f40914_2022-09-24T02_03_08.447Z.mp3', sr=16000)

# Extract 10 second segment
sample = audio[0:10*16000]
print(sample)

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# audio file is decoded on the fly
inputs = feature_extractor(sample, sampling_rate=sr, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

sm = nn.Softmax(dim=-1)
print(sm(logits))

predicted_class_ids = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_ids]
print(predicted_label)

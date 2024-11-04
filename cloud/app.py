#!/usr/env/bin python3

# NOTE: I currently lack permissions for cloud deployment
# Information about cloud deployment - https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service
# Example using FastAPI - https://dev.to/0xnari/deploying-fastapi-app-with-google-cloud-run-13f3
# Google cloud monthly cost estimate - https://cloud.google.com/products/calculator/?utm_source=google&utm_medium=cpc&utm_campaign=emea-none-all-en-dr-sitelink-all-all-trial-b-gcp-1011340&utm_content=text-ad-none-any-DEV_c-CRE_671802840933-ADGP_Hybrid%20%7C%20BKWS%20-%20BRO%20%7C%20Txt%20-%20GCP%20-%20General%20-%20v2-KWID_43700077735951054-kwd-4406040420-userloc_1010526&utm_term=KW_google%20cloud-ST_google%20cloud-NET_g-PLAC_&&gad_source=1&gclid=Cj0KCQjwmt24BhDPARIsAJFYKk2ZEY-qzxa1_cuXwAknb_qyChpNdcPZx0U3k3jtpdmolUgB4H-IXsgaAq86EALw_wcB&gclsrc=aw.ds&hl=en&dl=CjhDaVF6WkRRMFpqQXpOUzFtWW1FNUxUUmpOamN0WVRBNU55MDBORFExTmpObVlqazNaR0lRQVE9PRA3GiQwM0FEMDdFMi0wREIwLTREMzktOUFDRS0zMzA1OTdCNjY4RUI

from flask import Flask, request, jsonify, render_template
app = Flask(__name__)

import os
from dotenv import load_dotenv

import argparse
import numpy as np

import smtplib
from email.message import EmailMessage

import torch
from torch.utils.data import DataLoader

# from utils.utils import AudioList
# from utils.audio_signal import AudioSignal

from google.cloud import storage
from google.oauth2 import service_account
import datetime

# from pydub import AudioSegment
import io

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Get the environment variables
load_dotenv(dotenv_path=".env")

GMAIL_USER = os.environ.get('GMAIL_USER')
GMAIL_PASS = os.environ.get('GMAIL_PASS')
RECEIVER_EMAIL = os.environ.get('RECEIVER_EMAIL')


def fetch_audio_data(bucket_name, blob_name): # Already have
    """
    Fetches audio data from Google Cloud Storage.

    Parameters:
        bucket_name (str): The name of the GCS bucket.
        blob_name (str): The name of the blob (file) in the GCS bucket.

    Returns:
        BytesIO: An in-memory file object of the audio data.
    """
    # Create a GCS client
    from google.oauth2 import service_account
    import google.auth

    credentials = service_account.Credentials.from_service_account_file(
        '/app/cloud_analysis/key-file.json'
)

    storage_client = storage.Client(credentials=credentials)

    # Get the GCS bucket and blob
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the file into an in-memory file object
    audio_file_object = io.BytesIO()
    blob.download_to_file(audio_file_object)
    audio_file_object.seek(0)  # Move file pointer to the start

    # Convert MP3 to WAV
    #wav_file_object = convert_mp3_to_wav(audio_file_object)

    return audio_file_object

def generate_signed_url(bucket_name, blob_name, expiration_time=86400): # Link to file for email alert
    """
    Generates a signed URL for a GCS object.
    
    Parameters:
        bucket_name (str): Name of the GCS bucket.
        blob_name (str): Name of the blob (file) in the GCS bucket.
        expiration_time (int): Time, in seconds, until the signed URL expires.

    Returns:
        str: A signed URL to download the object.
    """
    # Path to the service account key file
    # key_path = '/app/cloud_analysis/key-file.json'

    # Initialize a GCS client
    # credentials = service_account.Credentials.from_service_account_file(key_path)
    client = storage.Client()
    
    # Get the bucket and blob objects
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Generate the signed URL
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(seconds=expiration_time),
        method="GET"
    )
    
    return url

def initModel(model_path, device):
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    return model

def compute_hr(array):

    signal = AudioSignal(samples=array, fs=44100)

    signal.apply_butterworth_filter(order=18, Wn=np.asarray([1, 600]) / (signal.fs / 2))
    signal_hr = signal.harmonic_ratio(
        win_length=int(1 * signal.fs),
        hop_length=int(0.1 * signal.fs),
        window="hamming",
    )
    hr = np.mean(signal_hr)

    return hr

def predict(testLoader, model, device):

    proba_list = []
    hr_list = []

    for array in testLoader:

        # Compute confidence for the DL model
        if device == "cpu":
            tensor = torch.tensor(array)
        else:
            tensor = array

        tensor = tensor.to(device)
        output = model(tensor)
        output = np.exp(output.cpu().detach().numpy())
        proba_list.append(output[0])

        # Compute HR if label=snowmobile
        label = np.argmax(output[0], axis=0)

        if label == 1:
            hr = compute_hr(np.array(array))
            hr_list.append(hr)
        else:
            hr_list.append(0)

    return proba_list, hr_list

def analyseAudioFile(
        audio_file_object, min_hr, min_conf, batch_size=1, num_workers=2,
):

    # Initiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/app/audioclip/assets/snowmobile_model.pth"
    model = initModel(model_path=model_path, device=device)

    # Run the predictions
    list_preds = AudioList().get_processed_list(audio_file_object)
    predLoader = DataLoader(list_preds, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    prob_audioclip_array, hr_array = predict(predLoader, model, device)

    # Set index of the array to 0 (beginning)
    idx_begin = 0

    # List of results
    results = []

    # List of model confidence and HR
    conf_arr = []
    hr_arr = []

    for item_audioclip, item_hr in zip(prob_audioclip_array, hr_array):

        # Get the properties of the detection (start, end, label, confidence and harmonic ratio)
        idx_end = idx_begin + 3
        conf = np.array(item_audioclip)
        label = np.argmax(conf, axis=0)
        confidence = conf.max()
        hr = np.array(item_hr)

        # Append the conf and hr of each segment
        conf_arr.append(conf[1])
        hr_arr.append(hr)

        # If the label is not "soundscape" then write the row:
        if hr > min_hr and confidence > min_conf:
            item_properties = [idx_begin, idx_end, confidence, hr]
            results.append(item_properties)

        # Update the start time of the detection
        idx_begin = idx_end

    # Get the max confidence and hr for the file analyzed
    maxes = [conf_arr, hr_arr]

    return results, maxes

def on_process_audio(audio_id: str, audio_rec: dict, bucket_name: str, blob_name: str, hr: float, conf:float):
    
    print(f"PROCESSING audioId={audio_id}")
    location = audio_rec["location"]

    # A call out to your code here. Optionally we can pass on the recorder coordinates 
    audio_file_object = fetch_audio_data(bucket_name, blob_name)
    results, maxes = analyseAudioFile(audio_file_object, hr, conf)
    # The object results is a list containing detections in the form:
    # [start, end, confidence, harmonic ratio]

    # After processing we record each detection in the database. 
    # Each detection should have a start and end time which is used to create an audio clip later. 
    count = len(results)
    detections = []

    for r in results: 
        start, end, confidence, harmonic_ratio = r

        # create the detections dataset
        detections.append({
            u"start": start,
            u"end": end,
            u"tags": ["snowmobile"],
            u"confidence": confidence,
            u"harmonic_ratio": harmonic_ratio,
            #u"analysisId": analysis_id,
            # Add any other information you want to record here
        })
    
    return count, maxes

@app.route('/')
def map():
    return render_template('map.html')

@app.route('/test')
def home(bucket_name="tabmon_data", prefix="proj_sound-of-norway/bugg_RPiID-10000000cc849698/conf_6f40914/", delimiter="/", max_results=1):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter, max_results=max_results)
    
    # Test blob loading
    for blob in blobs:
        print(blob.name)

    return jsonify({"message": blob.name})

@app.route('/process-audio', methods=['POST'])
def process_audio_endpoint():
    data = request.json
    bucket_name = "tabmon_data" # data['bucket_name']
    blob_name = data['blob_name']
    audio_id = data['audio_id']
    audio_rec = data['audio_rec']
    hr = data['hr']
    conf = data['conf']
    
    results, maxes = on_process_audio(audio_id, audio_rec, bucket_name, blob_name, hr, conf)
    max_conf, max_hr = maxes

    email_response = "Not sent"
    if results > 0:
        # Create a signed URL
        download_url = generate_signed_url(bucket_name, blob_name)

        # Extract folder name (location) from the blob name
        location = blob_name.split("/")[0]

        # Write and send the email
        email_body = f"{results} snowmobile detections were made in the audio file!\n"
        email_body += f"Detections come from: {location}\n"
        email_body += f"Download the audio file here: {download_url}"
        email_response = send_email("Snowmobile Detection Alert", email_body)

    return jsonify({"message": f"file {blob_name} processed. CONF SNOWMOBILE = {max_conf}, HR = {max_hr}, DET COUNT = {results}, E-MAIL = {email_response}"})


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8080)
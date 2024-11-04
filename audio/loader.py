from google.cloud import storage
from datetime import datetime

def download_audio(bucket_name: str, 
                  prefix: str, 
                  delimiter: str = "/", 
                  output_folder: str = "./main/audio", 
                  max_results: int = 1000, 
                  start_date: datetime = datetime.min, 
                  end_date: datetime = datetime.max) -> None:

    """
    Downloads audio files from a Google Cloud Storage bucket, filtering by date range.

    Parameters
    ----------
    bucket_name : str
        The name of the Google Cloud Storage bucket to download from.
    prefix : str
        The prefix of the files to download, e.g. a folder name.
    delimiter : str, optional
        The delimiter to use when listing files, by default "/".
    output_folder : str, optional
        The folder to download the files to, by default "./audio".
    max_results : int, optional
        The maximum number of files to download, by default 1000.
    start_date : datetime, optional
        The start date of the date range to filter by, by default datetime.min.
    end_date : datetime, optional
        The end date of the date range to filter by, by default datetime.max.
    """
    
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter, max_results=max_results)

    # Filtering
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        date_string = filename.split(".")[0]+"."+filename.split(".")[1]
        time_object = datetime.strptime(date_string, "%Y-%m-%dT%H_%M_%S.%fZ")

        if (start_date <= time_object <= end_date):
            blob.download_to_filename(f"{output_folder}/{filename}")

## Example of converting filename to datetime
# start = datetime.strptime("2022-09-24T02:03:07.447Z", "%Y-%m-%dT%H:%M:%S.%fZ")
# end = datetime.strptime("2022-09-24T02:03:09.447Z", "%Y-%m-%dT%H:%M:%S.%fZ")

download_audio("tabmon_data", 
               prefix="proj_sound-of-norway/bugg_RPiID-10000000cc849698/conf_6f40914/", 
               max_results=5)
import pandas as pd
import json
import os
from datetime import datetime

##merge parqet files per bugg per month

data_path = "pipeline/outputs/predictions"
output_path = "pipeline/outputs/merged_predictions"

MONTH_SELECTION = ["2025-01"]


META_DATA_PATH = "/DYNI/tabmon/tabmon_data/site_info.csv"
META_DATA_DF = pd.read_csv( META_DATA_PATH , encoding='utf-8')
META_DATA_DF = META_DATA_DF.fillna("")



def get_month(fname):
    date = fname.split("T")[0].split("-")
    return f"{date[0]}-{date[1]}"


def get_file_date(bugg_file_name):

    date = bugg_file_name.replace('.mp3', '')
    date = date.replace('_', ':')
    date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ")

    return date


ERROR_LIST = []

def get_deploymentID(bugg_id, file_name):

    global ERROR_LIST

    file_date = get_file_date(file_name)

    deployment_candidate_list = [i for i, deviceID in enumerate(META_DATA_DF["DeviceID"]) if deviceID==bugg_id]

    deploymentIDs = []

    for deployment_candidate_idx in deployment_candidate_list:

        meta_data_row = META_DATA_DF.iloc[deployment_candidate_idx]

        start_date = meta_data_row["deploymentBeginDate"]
        start_time = meta_data_row["deploymentBeginTime"]
        end_date = meta_data_row["deploymentEndDate"]
        end_time = meta_data_row["deploymentEndTime"]

        deployment_start = datetime.strptime(f"{start_date} {start_time}", "%d/%m/%Y %H:%M:%S")

        if end_date == "" and end_time == "" :
            deployment_end = datetime(3000, 1, 1)
        else :
            deployment_end = datetime.strptime(f"{end_date} {end_time}", "%d/%m/%Y %H:%M:%S")


        if file_date > deployment_start and file_date < deployment_end :
            deploymentIDs.append(meta_data_row["DeploymentID"])
            

    if len(deploymentIDs) == 1:
        deploymentID = deploymentIDs[0]
    elif len(deploymentIDs) > 1:
        deploymentID = "Error"
        ERROR_LIST.append(f"Found multiples deploymentID for {bugg_id} , {file_date.month}-{file_date.day} ")
    else:
        deploymentID = "0"
        ERROR_LIST.append(f"Found 0 deploymentID for {bugg_id} , {file_date.month}-{file_date.day} ")

    return deploymentID


def merge_parquet_files(bugg_id, bugg_path, file_list, bugg_output_path, output_file):
    #try:
        # Read and concatenate all Parquet files
        dataframes = [pd.read_parquet(os.path.join(bugg_path, file_name), engine='pyarrow') for file_name in file_list]
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df.drop_duplicates(subset = ["filename", "start time", "scientific name", "confidence"] )

        deploymentID_list = [ get_deploymentID(bugg_id, file_name) for file_name in merged_df['filename'] ]

        merged_df.insert(1, "deployementID", deploymentID_list)

        # Save to a new Parquet file
        merged_df.to_parquet(os.path.join(bugg_output_path, output_file), index=False)
        
        print(f"Merged {len(file_list)} files into: {output_file}")
    #except Exception as e:
    #    print(f"Error: {e}")




if __name__ == "__main__":

    country_folder_list = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

    print(country_folder_list)

    for country_folder in country_folder_list:
        
        print(country_folder)
        bugg_folder_list = [f for f in os.listdir(os.path.join(data_path, country_folder)) if os.path.isdir(os.path.join(data_path, country_folder, f))]

        for bugg_folder in bugg_folder_list:

            bugg_id = bugg_folder.split("=")[1]
            file_list = sorted(os.listdir(os.path.join(data_path, country_folder,bugg_folder )))
            file_list = [ fname for fname in file_list if fname.endswith('.parquet')]
            month_list =  [ get_month(fname) for fname in file_list ]
            month_list = list(set(month_list))

            for month in month_list:

                if month in MONTH_SELECTION:

                    file_list_month = [ fname for fname in file_list if fname.startswith(month)]
                    bugg_path = os.path.join(data_path, country_folder,bugg_folder )
                    bugg_output_path = os.path.join(output_path, country_folder, bugg_folder)
                    os.makedirs(bugg_output_path, exist_ok=True)
                    output_file = f"{month}_{bugg_id}.parquet"

                    merge_parquet_files(bugg_id, bugg_path, file_list_month, bugg_output_path, output_file)

    print(set(ERROR_LIST))



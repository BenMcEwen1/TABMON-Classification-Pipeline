import pandas as pd
import json
import os


##merge parqet files per bugg per month

data_path = "pipeline/outputs/predictions"
output_path = "pipeline/outputs/merged_predictions"

MONTH_SELECTION = ["2025-03"]



def merge_parquet_files(bugg_path, file_list, bugg_output_path, output_file):
    try:
        # Read and concatenate all Parquet files
        dataframes = [pd.read_parquet(os.path.join(bugg_path, file_name), engine='pyarrow') for file_name in file_list]
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df.drop_duplicates(subset = ["filename", "start time", "scientific name", "confidence", "datetime"] )

        # Save to a new Parquet file
        merged_df.to_parquet(os.path.join(bugg_output_path, output_file), index=False)
        
        print(f"Merged {len(file_list)} files into: {output_file}")
    except Exception as e:
        print(f"Error: {e}")


def get_month(fname):
    date = fname.split("T")[0].split("-")
    return f"{date[0]}-{date[1]}"


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
                merge_parquet_files(bugg_path, file_list_month, bugg_output_path, output_file)




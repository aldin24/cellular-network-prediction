import os
import pandas as pd
import kagglehub


def _check_existing_csv_files(path):
    csv_files = []
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))
    return csv_files


def load_kaggle_dataset(dataset_name, add_source_file=True, force_download=False):
    path = kagglehub.dataset_download(dataset_name, force_download=force_download)
    csv_files = _check_existing_csv_files(path)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset: {dataset_name}")
    
    all_dfs = []
    for csv_file in csv_files:
        f = os.path.basename(csv_file)
        try:
            df_part = pd.read_csv(csv_file)
            if add_source_file:
                df_part["source_file"] = f
            all_dfs.append(df_part)
        except Exception as e:
            print(f"Error loading {f}: {str(e)}")
            continue
    
    if not all_dfs:
        raise FileNotFoundError(f"No CSV files could be loaded from dataset: {dataset_name}")
    
    return pd.concat(all_dfs, ignore_index=True)


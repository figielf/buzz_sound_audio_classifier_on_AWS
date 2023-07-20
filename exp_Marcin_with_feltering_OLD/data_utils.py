import os
import boto3
import pandas as pd
import json

from pathlib import Path


DATA_FOLDER = '/root/data/data'


def filter_data(data, map_filter, should_reset_index=True):
    filter_keys = set(map_filter.keys())
    filtered = data[data['label'].isin(filter_keys)][['file_name', 'label']].copy()
    filtered['label'] = filtered['label'].map(map_filter)
    if should_reset_index:
        filtered.reset_index(drop=True, inplace=True)
    return filtered


def filter_labels(labels, map_filter):
    inv_labels = {v:k for k, v in labels.items()}
    print(inv_labels)
    
    filtered_labels_inv = {}
    for k, v in map_filter.items():
        if v in filtered_labels_inv.keys():
            filtered_labels_inv[v] = '_'.join([filtered_labels_inv[v], inv_labels[k]])
        else:
            filtered_labels_inv[v] = inv_labels[k]
    
    return {v:k for k, v in filtered_labels_inv.items()}


def copy_data_to_s3(source_folder, subfolders, target_folder, data_map_filter, s3_client, bucket_name, bucket_subfolder='data_new'):
    source_folder_full = os.path.join(DATA_FOLDER, source_folder)
    
    # copy labels.json
    local_labels_source_file_path = os.path.join(source_folder_full, 'labels.json')
    local_labels_target_file_path = os.path.join(DATA_FOLDER, target_folder, 'labels.json')
    
    # Load json file with label2id mapping
    with open(local_labels_source_file_path, 'r') as f:
        labels = json.load(f)    
    print(f'Loaded base labels from {local_labels_source_file_path} and start filtering')
    
    labels_filtered = filter_labels(labels, data_map_filter)
    print('labels filtered:\n', labels_filtered)

    Path(os.path.join(DATA_FOLDER, target_folder)).mkdir(parents=True, exist_ok=True)
    with open(local_labels_target_file_path, "w") as outfile:
        json.dump(labels_filtered, outfile)
    
    print(f'Saved labels in {local_labels_target_file_path}')
    
    s3_target_labels_file_path = os.path.join(bucket_subfolder, target_folder, 'labels.json')
    s3_client.upload_file(local_labels_target_file_path, bucket_name, s3_target_labels_file_path)
    print(f"Copied '{local_labels_target_file_path}' to '{s3_target_labels_file_path}' in S3 bucket '{bucket_name}'.")
    
    for subfolder in subfolders:
        print(f'\nStart copying {subfolder} data to s3')
        copy_data_subfolder_to_s3(source_folder, subfolder, target_folder, data_map_filter, s3_client, bucket_name, bucket_subfolder=bucket_subfolder)
        print(f'Copy finished\n')
    

def copy_data_subfolder_to_s3(source_folder, subfolder, target_folder, data_map_filter, s3_client, bucket_name, bucket_subfolder='data_new'):
    source_folder_full = os.path.join(DATA_FOLDER, source_folder)
    source_path = os.path.join(source_folder_full, subfolder)
    target_path = os.path.join(DATA_FOLDER, target_folder, subfolder)
    
    Path(target_path).mkdir(parents=True, exist_ok=True)
    
    # copy metadata
    base_metadata_file_path = os.path.join(source_path, 'metadata.csv')
    matadata = pd.read_csv(base_metadata_file_path, index_col = False)
    print(f'Loaded base metadata from {base_metadata_file_path} and start filtering')
    local_metadata_target_file_path = os.path.join(target_path, 'metadata.csv')
    metadata_entries_count = 0
    print(matadata.columns)
    if 'label' in matadata.columns:
        filtered_metadata = filter_data(matadata, data_map_filter)
        filtered_metadata.to_csv(local_metadata_target_file_path, index = False)
        metadata_entries_count = filtered_metadata.shape[0]
        print(f'Saved metadata in {local_metadata_target_file_path}')
        files_to_move = filtered_metadata['file_name'].tolist()
    else:
        matadata.to_csv(local_metadata_target_file_path, index = False)
        metadata_entries_count = matadata.shape[0]
        print(f'Saved metadata in {local_metadata_target_file_path}')
        files_to_move = matadata['file_name'].tolist()
    
    s3_target_metadata_file_path = os.path.join(bucket_subfolder, target_folder, subfolder, 'metadata.csv')
    s3_client.upload_file(local_metadata_target_file_path, bucket_name, s3_target_metadata_file_path)
    print(f"Copied '{local_metadata_target_file_path}' to '{s3_target_metadata_file_path}' in S3 bucket '{bucket_name}'.")
    
    # copy wav files
    files_copied = 0
    for file_name in files_to_move:
        if file_name.endswith('.wav'):            
            source_file_path = os.path.join(source_path, file_name)
            target_file_path = os.path.join(bucket_subfolder, target_folder, subfolder, file_name)
                                   
            s3_client.upload_file(source_file_path, bucket_name, target_file_path)
            files_copied += 1
            print(f"Copied '{source_file_path}' to '{target_file_path}' in S3 bucket '{bucket_name}'.")
        else:
            print(f'copy wav files warning - found {file_name} that will not be copied')
    
    print(f'Saved {metadata_entries_count} entries in metadata')
    print(f'Copied {files_copied} wav files from {source_path} to {target_path}')

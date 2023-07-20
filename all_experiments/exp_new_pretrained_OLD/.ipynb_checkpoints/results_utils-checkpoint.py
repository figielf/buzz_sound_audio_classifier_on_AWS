import os
import datetime
import zipfile

import numpy as np


RESULTS_FOLDER = '/root/data/experiments/exp_runs'


def archive_solution_files(folder_full_path, solution_name_suffix=''):
    # zips only files within folder_full_path folder, not indludes subfolders
    
    current_folder = os.path.basename(folder_full_path)
    zip_filename = f'{current_folder}_{solution_name_suffix}.zip'
    zip_file_full_path = os.path.join(RESULTS_FOLDER, zip_filename)

    files_to_archive = os.listdir(folder_full_path)
    with zipfile.ZipFile(zip_file_full_path, 'w') as zipf:
        for file_name in files_to_archive:
            if not file_name.endswith('.ipynb_checkpoints'):  # exclude .ipynb_checkpoints files
                file_path = os.path.join(folder_full_path, file_name)
                zipf.write(file_path, arcname=file_name)
    
    print(f'Saved solution files to {zip_file_full_path}')

    
def persist_confusion_matrix(y_true: np.ndarray, 
                             y_preds: np.ndarray, 
                             normalize: str = "true",
                             file_prefix: str = '') -> None:
    cm = confusion_matrix(y_true, y_preds, normalize=normalize)
    
    file_path = os.path.join(RESULTS_FOLDER, '_'.join([file_prefix, 'confusion_matrix.csv']))
    test_dataset_encoded_df.to_csv(file_path, index = False)  # saving the file with test predictions
    print(f'Confucion matrix persisted in {file_path}')
    
    
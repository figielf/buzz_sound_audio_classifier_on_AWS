import numpy as np
import pandas as pd                        
import torch

from datasets import Dataset

from preprocessing import preprocess_audio_arrays
from gdsc_eval import make_chunked_predictions


def predict_with_chunking(model, feature_extractor, test_dataset, chunk_size, min_chunk_size):
    chunks = []
    total = 0
    for i, x in enumerate(test_dataset):
        one_file_dataset = test_dataset.select([i])
        
        file_size = len(x['audio']['array'])
        sampling_rate = x['audio']['sampling_rate']
        
        n = int(file_size / chunk_size)
        num = 0
        for i_n in range(n+1):
            if i_n > 0 and len(x['audio']['array'][i_n * chunk_size : (i_n + 1) * chunk_size]) < min_chunk_size:
                break
                
            x_chunk = copy.deepcopy(x)
            x_chunk['audio']['array'] = x_chunk['audio']['array'][i_n * chunk_size : (i_n + 1) * chunk_size]
            
            one_file_dataset.add_item(x_chunk)
            chunks.append(x_chunk)
            num += 1
            total += 1
        
        print(f'{x["file_name"]} - file_size:{file_size} - splitted into {num} chunks')
        
    print(f'finished chunking, total number of chunks: {total}')
    chunked_one_file_dataset = Dataset.from_pandas(pd.DataFrame(chunks))
        
    print('\npreprocessing of chunks...')
    chunked_one_file_dataset_encoded = chunked_one_file_dataset.map(lambda x: preprocess_audio_arrays(x, 'audio', 'array', feature_extractor), remove_columns="audio", batched=True, batch_size = 16)
    chunked_one_file_dataset_encoded.set_format(type='torch', columns=['input_values'])
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('\ncalculating predictions of chunkeds ...')
    chunked_one_file_dataset_encoded = chunked_one_file_dataset_encoded.map(lambda x: make_chunked_predictions(x['input_values'], model, device), batched=True, batch_size=16, remove_columns="input_values")
    
    return get_agregated_results(chunked_one_file_dataset_encoded.to_pandas())
    

def get_agregated_results(predictions_df):
    # logits
    logits = predictions_df.groupby('file_name').logits.sum().apply(lambda x: np.argmax(x)).reset_index()
    logits.columns = ['file_name', 'predicted_class_id']

    # probs
    predictions_df['probs'] = predictions_df['logits'].apply(lambda x: np.array(np.exp(x) / np.sum(np.exp(x))))
    probits = predictions_df.groupby('file_name').probs.sum().apply(lambda x: np.argmax(x)).reset_index()
    probits.columns = ['file_name', 'predicted_class_id']
 
    # voting
    unique_files = sorted(predictions_df['file_name'].unique(), key=lambda x: int(x.strip('.wav')))
    voting_results = []
    for file in unique_files:
        temp_df = predictions_df[predictions_df['file_name'] == file]
        best_class_id = temp_df['predicted_class_id'].value_counts().idxmax()
        voting_results.append([file, best_class_id])
    voting = pd.DataFrame(voting_results)
    voting.columns = ['file_name', 'predicted_class_id']
    
    return logits, probits, voting
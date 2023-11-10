from dataclasses import dataclass
from extract_metadata import AudioFeatureExtractor
from audio_data_processor import AudioDataProcessor
import time
import multiprocessing
import os
import pandas as pd
import warnings
import psutil  # Import the psutil module for CPU usage monitoring
warnings.filterwarnings("ignore")

@dataclass
class Configuration:
    folder_path: str
    model_pkl_file: str

def extract_features_and_process_data(file_path, model_pkl_file):
    start_time = time.time()  # Record the start time for each file
    audio_feature_extractor = AudioFeatureExtractor(file_path)
    audio_metadata = audio_feature_extractor.extract_audio_features(file_path)

    data_processor = AudioDataProcessor(model_pkl_file)
    encoded_labels, _ = data_processor.process_audio_data(pd.DataFrame([audio_metadata]))

    # Add 'file_name' and 'file_path' to the audio_metadata dictionary
    audio_metadata['file_name'] = os.path.basename(file_path)
    audio_metadata['file_path'] = file_path
    audio_metadata['encoded_label'] = encoded_labels[0]

    end_time = time.time()  # Record the end time for each file
    print(f"File: {os.path.basename(file_path)} | Time taken: {end_time - start_time} seconds")

    return audio_metadata

if __name__ == "__main__":
    start_time = time.time()  # Record the start time

    config = Configuration(
        folder_path=r"C:\Users\TusharPatil\Desktop\Audio Analysis\4_files",
        model_pkl_file="Linear_Discriminant_Analysis.pkl"
    )

    num_processes = 4
    pool = multiprocessing.Pool(processes=num_processes)

    extract_start_time = time.time()

    audio_files = [os.path.join(root, file_name) for root, _, files in os.walk(config.folder_path) for file_name in files if file_name.endswith(('.wav', '.mp3'))]

    output_excel_filename = "Audio_Metadatasample4files.xlsx"

    results = pool.starmap(extract_features_and_process_data, [(file_path, config.model_pkl_file) for file_path in audio_files])

    audio_metadata_df = pd.DataFrame(results)

    reduced_audio_metadata_df = audio_metadata_df[['file_name', 'file_path', 'encoded_label']]

    reduced_output_excel_filename = "Reduced_Audio_Metadatasample4files.xlsx"
    reduced_audio_metadata_df.to_excel(reduced_output_excel_filename, index=False)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")
    print("Successful")

    # Get CPU usage during execution
    cpu_usage = psutil.cpu_percent(interval=None)
    print(f"Average CPU Usage: {cpu_usage}%")

    pool.close()
    pool.join()

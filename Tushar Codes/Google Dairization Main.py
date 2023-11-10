#!/usr/bin/env python
# coding: utf-8

# # For more than 1 Min file

# In[16]:


import os
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
import pandas as pd
from google.cloud import storage

# Set the path to your service account key JSON file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\TusharPatil\Downloads\crucial-citizen-399705-a80d6ed79874.json"

# Instantiate the SpeechClient
client = speech.SpeechClient()

# Set your Google Cloud Storage (GCS) bucket name
bucket_name = 'bucket-1232'  # Replace with your actual GCS bucket name

def upload_file_to_gcs(local_file_path, bucket_name, gcs_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_name)

    blob.upload_from_filename(local_file_path)

    return f'gs://{bucket_name}/{gcs_file_name}'

# Define a function to convert audio to mono
def convert_to_mono(input_audio_path, output_audio_path):
    audio = AudioSegment.from_wav(input_audio_path)
    audio = audio.set_channels(1)  # Convert to mono (single channel)
    audio.export(output_audio_path, format="wav")

# Define a function to transcribe audio with diarization using GCS URI
def transcribe_with_diarization_gcs(gcs_uri, file_name):
    # Define recognition audio and configuration
    audio = speech.RecognitionAudio(uri=gcs_uri)

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=2,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
        diarization_config=diarization_config,
    )

    # Perform asynchronous recognition with speaker diarization
    operation = client.long_running_recognize(config=config, audio=audio)

    # Wait for the operation to complete
    response = operation.result()

    # Process the results and extract transcriptions
    transcriptions = []
    for result in response.results:
        words_info = result.alternatives[0].words
        current_speaker = None
        start_time = None
        transcript = ''
        for word_info in words_info:
            if current_speaker != word_info.speaker_tag:
                if current_speaker is not None:
                    transcriptions.append({
                        'file_name': file_name,
                        'speaker': current_speaker,
                        'transcript': transcript,
                        'start_time': start_time,
                        'end_time': word_info.start_time.total_seconds()
                    })
                current_speaker = word_info.speaker_tag
                start_time = word_info.start_time.total_seconds()
                transcript = word_info.word
            else:
                transcript += " " + word_info.word

        # Add the last segment to the transcriptions list
        transcriptions.append({
            'file_name': file_name,
            'speaker': current_speaker,
            'transcript': transcript,
            'start_time': start_time,
            'end_time': words_info[-1].end_time.total_seconds()
        })

    # Create a DataFrame from the transcriptions
    df = pd.DataFrame(transcriptions)

    return df

# Path to the folder containing audio files (replace with your folder path)
audio_folder_path = r"C:\Users\TusharPatil\Desktop\Audio Analysis\random"

# Initialize an empty list to store DataFrames for each file
all_dfs = []

# Iterate through the files in the folder
for filename in os.listdir(audio_folder_path):
    if filename.endswith(".wav"):
        audio_file_path = os.path.join(audio_folder_path, filename)
        print(f"Transcribing audio file: {audio_file_path}")

        # Convert the audio to mono
        mono_audio_file_path = f"/tmp/mono_{filename}"
        convert_to_mono(audio_file_path, mono_audio_file_path)

        # function call for uploading to GCS
        gcs_uri = upload_file_to_gcs(mono_audio_file_path, bucket_name, f"mono_{filename}")

        # Extract file name
        file_name = os.path.basename(audio_file_path)

        # Transcribe audio with speaker diarization using GCS URI and get the DataFrame
        transcription_df = transcribe_with_diarization_gcs(gcs_uri, file_name)

        # Append the DataFrame to the list
        all_dfs.append(transcription_df)

# Concatenate all DataFrames into one
final_df = pd.concat(all_dfs, ignore_index=True)

# Display the final DataFrame
print(final_df)





# In[17]:


# Path to save the Excel file
excel_file_path =r"C:\Users\TusharPatil\Desktop\dairization.xlsx"

# Save the final DataFrame to an Excel file
final_df.to_excel(excel_file_path, index=False)

# Print a message indicating the file has been saved
print(f"DataFrame saved to Excel file: {excel_file_path}")



# In[18]:


final_df


# In[ ]:





# In[ ]:





# In[ ]:





# # For  less than 1 Min File

# In[35]:


import os
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
import pandas as pd

# Set the path to your service account key JSON file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\TusharPatil\Downloads\crucial-citizen-399705-a80d6ed79874.json"

# Instantiate the SpeechClient
client = speech.SpeechClient()

def convert_to_mono(input_audio_path, output_audio_path):
    # Load the audio file and convert to mono
    audio = AudioSegment.from_wav(input_audio_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio.export(output_audio_path, format="wav")

def transcribe_with_diarization(audio_file_path, file_name):
    # Convert the audio to mono
    temp_mono_file = "temp_mono.wav"
    convert_to_mono(audio_file_path, temp_mono_file)

    # Load the audio file content
    with open(temp_mono_file, "rb") as audio_file:
        content = audio_file.read()

    # Define recognition audio and configuration
    audio = speech.RecognitionAudio(content=content)

    # Clean up temporary mono audio file
    os.remove(temp_mono_file)

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=2,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
        diarization_config=diarization_config,
    )

    # Perform recognition with speaker diarization
    response = client.recognize(config=config, audio=audio)

    # Get words, speaker tags, and timestamps
    result = response.results[-1]
    words_info = result.alternatives[0].words

    # Initialize variables to store speaker and start time
    current_speaker = None
    start_time = None
    transcript = ''

    # Initialize a list to store the transcription segments
    transcriptions = []

    # Print the output with timestamps
    for word_info in words_info:
        if current_speaker != word_info.speaker_tag:
            if current_speaker is not None:
                transcriptions.append({
                    'file_name': file_name,
                    'speaker': current_speaker,
                    'transcript': transcript,
                    'start_time': start_time,
                    'end_time': word_info.start_time.total_seconds()
                })
            current_speaker = word_info.speaker_tag
            start_time = word_info.start_time.total_seconds()
            transcript = word_info.word
        else:
            transcript += " " + word_info.word

    # Add the last segment to the transcriptions list
    transcriptions.append({
        'file_name': file_name,
        'speaker': current_speaker,
        'transcript': transcript,
        'start_time': start_time,
        'end_time': words_info[-1].end_time.total_seconds()
    })

    # Create a DataFrame from the transcriptions
    df = pd.DataFrame(transcriptions)

    return df

# Path to the folder containing audio files (replace with your folder path)
audio_folder_path = r"C:\Users\TusharPatil\OneDrive - Agivant Technlogies India Pvt. Ltd\Audio Analysis\new_audios_10_wav"

# Initialize an empty list to store DataFrames for each file
all_dfs = []

# Iterate through the files in the folder
for filename in os.listdir(audio_folder_path):
    if filename.endswith(".wav"):
        audio_file_path = os.path.join(audio_folder_path, filename)
        print(f"Transcribing audio file: {audio_file_path}")

        # Extract file name
        file_name = os.path.basename(audio_file_path)

        # Transcribe audio with speaker diarization and get the DataFrame
        transcription_df = transcribe_with_diarization(audio_file_path, file_name)

        # Append the DataFrame to the list
        all_dfs.append(transcription_df)

# Concatenate all DataFrames into one
final_df = pd.concat(all_dfs, ignore_index=True)

# Display the final DataFrame
print(final_df)


# In[36]:


final_df


# In[37]:


# Path to save the Excel file
excel_file_path =r"C:\Users\TusharPatil\Desktop\dairization2.xlsx"

# Save the final DataFrame to an Excel file
final_df.to_excel(excel_file_path, index=False)

# Print a message indicating the file has been saved
print(f"DataFrame saved to Excel file: {excel_file_path}")


# In[ ]:





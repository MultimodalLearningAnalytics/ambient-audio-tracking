# Ambient Audio Tracking

Multiple scripts to record and process audio to detect distracting sounds in the ambient space.

## Folders
### audio
The audio files used during the experiment to distract the participants.
### AudioProcessingTool
PSI/C# tool used to do various post processing steps like reprocessing audio, ingesting csv files and applying multimodal joines
### config
Contains two config files for the audio_processor and play_audio_vlc scripts
### tensor_flow_scripts
Scripts used to train the tensorflow/kerbal models.
### utils
Contains a utility used by the audio_processor and play_audio_vlc

## Scripts
### tensor_flow_scripts/audio_recognition_tensorflow
Old script used to train a kerbal model.
### tensor_flow_scripts/audio_recognition_tensorflow_clustered
Same as audio_recogintion_tensorflow, but clusters 1 second audio fragments from the same 1o second source fragment.
### utils/psi_connection
Utility used to connect the python scripts with the C# psi instance.
### audio_predictor
Script that ingests an audio file, splits it into 1 second fragments uses a defined and trained model to predict if the fragment is distracting or not. Outputs a csv file that can be converted into a PSI stream.
### audio_processor
Script that connects to the PSI instance, records audio, calculates the peak frequency and sends this to the PSI instance, over network if needed. Can also sync time if necessary.
### play_audio_vlc
Audio player used to play the distracting sounds and notify the PSI instance about these sounds played.




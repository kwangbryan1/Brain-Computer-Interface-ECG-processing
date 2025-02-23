This is a BCI project aimed at developing a ML/DL model to classify motor imagery tasks. 
This trained model will then take in live data from real patients in the form of ECG signals
and be classified into 1 of 5 imagined-movement intent. This will be used to control a virtual
keyboard and allow patients with speech disorders to communicate with their surroundings.

**Data source**
The EEG data, sourced from the PhysioNet eeg-mmidb database, was collected using the BCI200 EEG system with 64 channels at 160 Hz. The dataset includes 20 subjects performing 5 motor imagery (MI) tasks, labeled 0 to 4, for model training.

Please download the raw data from **https://archive.physionet.org/pn4/eegmmidb/**
Store all 109 patient folders under a main folder. Edit the name of the main folder in the temp.py code as neccessary.


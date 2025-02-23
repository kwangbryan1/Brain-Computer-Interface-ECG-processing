# ---------------------------- IMPORT LIBRARIES -----------------------------#
import numpy as np
import os
import pyedflib
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Import MNE
import mne
from mne.io.edf import read_raw_edf

# Import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, SpatialDropout2D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorboard

# Import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

# Function to extract last 4 digits from filr name
def last_4chars(x):
    return(x[-6:])

# -----------------------------------MNE FILTER -----------------------------#
''' 
This function converts EEG data into a Raw array and uses MNE Python to apply 
bandpass filter. Default low and high pass values are set to 8 Hz and 30 Hz. 

'''

def filter(data, fs):
    ch_names = [
        "FC3", "FC4", "FC1", "FC2", "FC5", "FC6", "C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6"
    ]    # Define channel names
    
    # Create a custom montage
    custom_montage_file = r"C:\Users\kwang\Chanel_positions.txt"  
    custom_montage = mne.channels.read_custom_montage(
        fname=custom_montage_file,
        head_size=0.085
    )
    
    
    # Filtering
    sfreq=fs
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=['eeg'] * len(ch_names),  
        sfreq=sfreq   
    )
    # Set the custom montage for the info object
    info.set_montage(custom_montage)
    
    # Filter settings
    low_cut = 8
    hi_cut  = 30
    
    # Raw object
    raw = mne.io.RawArray(data.T, info=info)
    
    # Apply filter
    raw_filt = raw.copy().filter(low_cut, hi_cut, method='iir') # Band pass 
    filtered = pd.DataFrame(raw_filt.get_data()).T
    
    return filtered

# -------------------------- CHANNEL EXTRACTION ----------------------------#
''' 
This function reads the EDF files and events in the main folder, according to 
the subject count specified by the user. 

input: Folder path
output: Data stored in dictionary
    
Main_sub description:
    dictionary 
        key - subject number
        values - 14 Dictionaries (2 from eye closed, 3 from the remaining four respectively)
        
            key - unique event
            values - DataFrame [9 electrode pairs (columns) across 28.7s (rows)]
                *the time duration can vary for certain subjects
                
'''
main_folder = r"C:\Users\kwang\files" # Folder path

# Main dictionary to contain extracted data
main_sub = {}

# Trials of interest (1 -- eye open 2 -- eye closed 4,8,12 -- task2,  6,10,14 -- task4)
tasks = [4,6,8,10,12,14]

# Subject folder count
sub_count = 1
sub_folders = os.listdir(main_folder) # List out files in folder
    
# Get user input for subject count
while True:
    try:
        subject_count = int(input("Enter the number of subject files to be processed: "))
        break  # Exit the loop if the user provides a valid integer input
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

print("Subjects entered:", subject_count)

# Loop through each subject folder Sxxx
for sub_folder in sorted(sub_folders, key= last_4chars):
    if sub_folder[0] != 'S'  : # Only choose files that are subject files
        continue
    else:
        # Instantiate sub-dict to contain all data of subject x
        sub1 = {}
        # Use count to create unique sub_dict keys
        count = 1
        print("Opening subject folder ",sub_folder)
        start_time = time.time() # Record the start time

    
        # Loop through each trial per subject
        for files in sorted(os.listdir(os.path.join(main_folder,sub_folder)), key = last_4chars): # Sort in running subject order
            if "event" in files: # Skip the event files
                pass
            # Extract experiment number
            else:
                task = int(files[-6:-4])
                
                # Check if experiment is what we want
                if task not in tasks:
                    pass
                
                # Eye open (baseline)
                elif task ==1:
                    edf = read_raw_edf(os.path.join(main_folder,sub_folder,files))
                    edf_data, times = edf[:, :]
                    edf_data_df = pd.DataFrame(edf_data.T) # Time x channel
                    edf_data_df = edf_data_df[[0,1,2,4,5,6,7,8,9,11,12,13,14,15,16,18,19,20]] # Select channels
                    baseline_1 = edf_data_df.mean(axis=0) # Baseline for each channel
                    
                # Eye closed task
                elif task ==2:
                    # Open EDF and convert to df
                    edf = read_raw_edf(os.path.join(main_folder,sub_folder,files))
                    edf_data, times = edf[:, :]
                    edf_data_df = pd.DataFrame(edf_data.T) # Time x channel
                    edf_data_df = edf_data_df[[0,1,2,4,5,6,7,8,9,11,12,13,14,15,16,18,19,20]]  # Select channels

                    # Baseline correction
                    edf_data_df = edf_data_df.subtract(baseline_1, axis=1) # Subtract task 1 (eye open)

                    # Add event labels back
                    edf_data_df['label']=1
                    
                    # Add to sub1
                    name1 = "1" + str(count) # Create unique event key
                    sub1[name1]=edf_data_df.iloc[0:4592]
                    name1v2 = "1v" + str(count) # Create unique event key
                    sub1[name1v2]=edf_data_df.iloc[4592:9184]
                    
                # MI tasks    
                else:                       
                    # Open EDF and convert to df
                    edf = read_raw_edf(os.path.join(main_folder,sub_folder,files))
                    edf_data, times = edf[:, :]
                    edf_data_df = pd.DataFrame(edf_data.T) # Time x channel
                    edf_data_df = edf_data_df[[0,1,2,4,5,6,7,8,9,11,12,13,14,15,16,18,19,20]] # Select channels
                    # Timestamps x channels
                    edf_data_df['time']=times
                    
                    # Open events file
                    file = pyedflib.EdfReader(os.path.join(main_folder,sub_folder,files)) 
                    annotations = file.readAnnotations()  
                    
                    # Separate annotations
                    (time_stamp, _, label) = annotations
                    time_s = pd.DataFrame(time_stamp)
                    
                    # Combine time and event labels
                    time_s['label']=label
                    time_s['label'] = time_s['label'].str[-1]
                    # Group the events T0, T1, T2 together
                    sorts = time_s.sort_values('label')
                    
                    # Get the time index of each event
                    T0 =[]
                    T1 =[]
                    T2 =[]
                    
                    for i in range(0,len(sorts)):
                        if int(sorts.iloc[i,1]) == 0:
                            T0.append(sorts.iloc[i,0])
                        elif int(sorts.iloc[i,1]) == 1:
                            T1.append(sorts.iloc[i,0])
                        else:
                            T2.append(sorts.iloc[i,0])
                    
                    # Add events column to extracted EEG data
                    edf_data_df['event'] = np.nan # Create an empty column
                    
                    for i in range(len(edf_data_df)): # Fill in events trigger at corresponding time points
                        if edf_data_df.iloc[i, -2] in T0:
                            edf_data_df.at[i, 'event'] = 0
                        elif edf_data_df.iloc[i, -2] in T1:
                            edf_data_df.at[i, 'event'] = 1
                        elif edf_data_df.iloc[i, -2] in T2:
                            edf_data_df.at[i, 'event'] = 2
                    
                    # Filldown to populate NAN
                    filldown = edf_data_df.ffill(axis=0)
                    
                    # Get resting states data
                    rest = filldown[filldown['event'] == 0]
                    rest_data = rest.drop(columns=['event', 'time'])
                    rest_mean = rest_data.mean(axis=0)
                    
                    # Drop resting states 
                    filldown_nozero = filldown[filldown['event'] != 0]
                    
                    # Convert events to labels  2,3,4,5
                    for i in tqdm(range(len(filldown_nozero)),desc=("Loading data from subject {}, trial {}".format(sub_folder, task))):
                        # Trials with left / right hand
                        if task%4 ==0:
                            if filldown_nozero.iloc[i,-1] ==1:
                                filldown_nozero.iloc[i,-1] = 2        
                            elif filldown_nozero.iloc[i,-1] ==2:
                                filldown_nozero.iloc[i,-1] = 3
                        # Trials with both hands / feet  
                        else:
                            if filldown_nozero.iloc[i,-1] ==1:
                                filldown_nozero.iloc[i,-1] = 4
                            elif filldown_nozero.iloc[i,-1] ==2:
                                filldown_nozero.iloc[i,-1] = 5   
                    if task%4 ==0:       
                        # Separate into two distinct events
                        event2 = filldown_nozero[filldown_nozero['event']==2]
                        event3 = filldown_nozero[filldown_nozero['event']==3]
                        
                        # Trim the columns
                        event2_trim = event2.drop(columns=['time', 'event'])
                        event3_trim = event3.drop(columns=['time', 'event'])
                        
                        # Baseline correction
                        event2_trim = event2_trim.subtract(rest_mean, axis=1)
                        event3_trim = event3_trim.subtract(rest_mean, axis=1)
                        
                        # Reset index
                        event2_trim.reset_index(drop=True, inplace=True)
                        event3_trim.reset_index(drop=True, inplace=True)

                        # Add labels back
                        event2_trim['label']=2
                        event3_trim['label']=3
                        
                        # Add dict key
                        name2 = "2" + str(count)
                        name3 = "3" + str(count)
                        #4592
                        sub1[name2]=event2_trim.iloc[0:2623]
                        sub1[name3]=event3_trim.iloc[0:2623]
                    else:   
                        # Separate into two distinct events
                        event4 = filldown_nozero[filldown_nozero['event']==4]
                        event5 = filldown_nozero[filldown_nozero['event']==5]
                        
                        # Trim the columns
                        event4_trim = event4.drop(columns=['time', 'event'])
                        event5_trim = event5.drop(columns=['time', 'event'])
                        
                        # Baseline correction
                        event4_trim = event4_trim.subtract(rest_mean, axis=1)
                        event5_trim = event5_trim.subtract(rest_mean, axis=1)
                        
                        # Reset index
                        event4_trim.reset_index(drop=True, inplace=True)
                        event5_trim.reset_index(drop=True, inplace=True)

                        # Add labels back
                        event4_trim['label']=4
                        event5_trim['label']=5
    
                        # Add dict key
                        name4 = "4" + str(count)
                        name5 = "5" + str(count)
                        sub1[name4]=event4_trim.iloc[0:2623]
                        sub1[name5]=event5_trim.iloc[0:2623]
                             
                    count+=1
                    # Return a dict with unique labels for each trial
         
        # Record the end time
        end_time = time.time()
        # Calculate processing time
        processing_time = end_time - start_time
        print(f"Processing time for subject {sub_count}: {processing_time:.2f} seconds")
                    
        current_subject = "Subject" + str(sub_count)
        
        # If data array is empty, remove
        lists = []
        for i in sub1:
            if sub1[i].shape[0] ==0:
               lists.append(i)
               
        for list in lists:
            sub1.pop(list)
                
        main_sub[current_subject]= sub1
        sub_count += 1
        
        # Terminate data extraction once specified subject number is reached
        if len(main_sub) == subject_count:
            break
        else:
            continue
#--------------------------NORMALIZATION -------- ---------------------------#
'''
This function applies z-score normalization to each subject's data after splitting the data 
into columns of 9 electrode pairs. Each subject now has 18 DataFrames for eyes closed, 
and 27 DataFrames for each of the other events (total 126 df)

input: main_sub
output: Dictionary of expanded DataFrames

'''

# New dictionary to contain normalized data
main_sub_new = {}
pairs = [[2,4], [1,5], [0,6], [9,11], [8,12], [7,13], [16,18], [15,19], [14,20]] # 9 electrode pairs mapping


for i in main_sub:
    # Extract unique event key
    event_key = []
    main_sub_new[i] = {}
    for j in main_sub[i]:
        event_key.append(j)
        
        key = 1
        # Extract each set of electrode pairs
        for pair in pairs:
            KEY = str(j) + str(key) 
            temp1 = main_sub[i][j][[pair[0],pair[1],'label']] # We are trying to normalize each electrode pair
            temp2 = temp1.iloc[:,0:2] # Remove event labels
            temp_mean = temp2.mean(axis=0) # Find mean
            temp_std = temp2.std(axis=0) # Find std
            temp_norm = temp2.subtract(temp_mean, axis=1) # Normalize
            temp_norm = temp_norm.divide(temp_std, axis=1) # Normalize
            #Filter
            temp_norm['Label'] = temp1.iloc[:,2] # Add label back
            main_sub_new[i][KEY] = temp_norm
            key+= 1
      
#--------------------------------SEGMENT ---------------------------#
'''
This function splits each DataFrame into segments of 4.1s long, which is the 
duration length for each event. It also groups all DataFrames with the same event
labels together after shuffling, so that each subject's dictionary only has 5 keys

input: normalized data
output: list of subject dictionaries, each event key has a different list size
        of DataFrames
        
        List of data labels according to the count for each event
        
'''
# Initialise lists to separate label from data, used for test train split
Data_arr = []
Data_label = []

for i in main_sub_new: # Loop through subject
    mu, sigma = 0, 0.01 # Parameters for gaussian noise
    subject = {}
    data_label = {}
    subject[i] = {}
    data_label[i] = {}

    for j in main_sub_new[i]: # Loop through experiment to create empty arrays for events 2,3,4,5
        j = str(j)
        subject[i][j[0]] = []
        data_label[i][j[0]] = []        

    for j in main_sub_new[i]: # Each event split into 7 segments, each 4.1s long
        length = len(main_sub_new[i][j]) # Get length of event to know how many splits we can make
        count = 0
        start = 0
        segment = (round(length / 655)) 
        J = str(j)
        
        while count < segment:
            noise = np.random.normal(mu, sigma, [655,2]) 
            data_w_noise = main_sub_new[i][j][start:start+655].iloc[:,0:2] + noise
            data_w_noise['Label'] = main_sub_new[i][j].iloc[0,-1]
            # subject[i][J[0]].append(main_sub_new[i][j][start:start+655])
            subject[i][J[0]].append(data_w_noise)
            start += 656
            count += 1
            sigma += 0.005
            
            Data_arr.append(data_w_noise)


# Shuffle 

np.random.shuffle(Data_arr)
                

for label in Data_arr:
    Data_label.append(label.iloc[0,-1])

# ---------------------------TEST TRAIN SPLIT ---------------------------#
'''
This function performs a test-train split for each event in each subject's data 
to ensure equal contributions of each event towards the training and testing data

The data is also shuffled so that the data labels are in a jumbled order

input: Data array and labels
output: Test and train data, consisting of data and label

'''

# Initialize lists to store the test train splits
Train_X = []
Test_X = []
Train_Y = []
Test_Y= []
Val_X = []
Val_Y= []

X_train, X_test, y_train, y_test = train_test_split(Data_arr, Data_label,test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Initialise list for shuffling
all_Train_X = []
all_Val_X = []
all_Test_X = []

for i in X_train:
    all_Train_X.append(i) 
    
for i in X_val:
    all_Val_X.append(i) 
    
for i in X_test:
    all_Test_X.append(i) 


 # Convert to array
all_Train_X = np.array(all_Train_X)
all_Val_X = np.array(all_Val_X)
all_Test_X = np.array(all_Test_X)

# # Shuffle order
np.random.seed(42)
np.random.shuffle(all_Train_X)
np.random.shuffle(all_Val_X)
np.random.shuffle(all_Test_X)

# Extract labels from shuffled array
all_Train_Y = []
all_Val_Y = []
all_Test_Y = []

for i in all_Train_X:
    all_Train_Y.append(int(i[0,2]))
    
for i in all_Val_X:
    all_Val_Y.append(int(i[0,2]))

for i in all_Test_X:
    all_Test_Y.append(int(i[0,2]))

# Drop labels column in shuffled array
all_Train_X = np.delete(all_Train_X,2,axis=2)
all_Val_X = np.delete(all_Val_X,2,axis=2)
all_Test_X = np.delete(all_Test_X,2,axis=2)

# Convert to float
all_Train_X = (all_Train_X).astype(float)
all_Val_X = (all_Val_X).astype(float)
all_Test_X = (all_Test_X).astype(float)



# ----------------------------MODEL ARCHITECTURE ----------------------------#
'''
This function defines the model architecture used. It is a convolutional neural network
with 10 layers, consisting of 5 conv layers, 4 pooling layers and 1 output layer.

The filter is moved along 1 dimension in a given layer, with once being in the spatial 
direction and the other layers applying concolution along the temporal domain.
These serve as the feature extraction steps, while max pooling acts to reduce the dimension

Batch normalization and spatial dropout are used to reduce overiftting

'''
def CNN_model(cnn_input_shape, num_classes, learning_rate, dropout_rate):
    
    # Input
    input_layer = Input(shape=cnn_input_shape)

    # Conv1
    conv1 = Conv2D(filters=25, kernel_size=(11,1), strides=(1,1), padding="valid")(input_layer) 
    leaky_relu1 = LeakyReLU(alpha=0.2)(conv1)  
    spatial_dropout1 = SpatialDropout2D(rate=dropout_rate)(leaky_relu1) 
  
    # Conv2
    conv2 = Conv2D(filters=25, kernel_size=(1,2), strides=(1,1), kernel_regularizer=l2(0.2), padding="valid")(spatial_dropout1)
    bn1 = BatchNormalization(axis=-1, center=True, scale=True)(conv2)  
    leaky_relu2 = LeakyReLU(alpha=0.2)(bn1)  

    # Pool1
    pool1 = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding="valid")(leaky_relu2)
    
    # Conv3
    conv3 = Conv2D(filters=50, kernel_size=(11,1), strides=(1,1), padding="valid")(pool1)
    leaky_relu3 = LeakyReLU(alpha=0.2)(conv3)  
    spatial_dropout2 = SpatialDropout2D(rate=dropout_rate)(leaky_relu3) 

    # Pool2
    pool2 = MaxPooling2D(pool_size=(3,1), strides=(3,1), padding="valid")(spatial_dropout2)
    
    # Conv4
    conv4 = Conv2D(filters=100, kernel_size=(11,1), strides=(1,1), padding="valid")(pool2)
    bn2 = BatchNormalization(axis=-1, center=True, scale=True)(conv4)  
    leaky_relu4 = LeakyReLU(alpha=0.2)(bn2)  
    spatial_dropout3 = SpatialDropout2D(rate=dropout_rate)(leaky_relu4) 

    # Pool3
    pool3 =MaxPooling2D(pool_size=(3,1), strides=(3,1), padding="valid")(spatial_dropout3)
    
    # Conv5
    conv5 = Conv2D(filters=200, kernel_size=(11,1), strides=(1,1), kernel_regularizer=l2(0.2), padding="valid")(pool3)
    bn3 = BatchNormalization(axis=-1, center=True, scale=True)(conv5)  
    leaky_relu5 = LeakyReLU(alpha=0.2)(bn3)  
    
    # Pool4
    pool4 = MaxPooling2D(pool_size=(2,1), strides=(2,1), padding="valid")(leaky_relu5)

    # Flatten
    flatten = Flatten()(pool4)

    # Output 
    output_layer = Dense(n_classes, activation='softmax')(flatten)

    # Create the model
    CNN = Model(inputs=input_layer, outputs=output_layer)
    custom_optimizer = Adam(learning_rate=learning_rate)
                                         
    # Compile the model
    CNN.compile(optimizer=custom_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return CNN
    
# -------------------------------CNN EVALUATE ----------------------------#
# Define parameters
learning_rate = 0.00001
n_classes = 4
batch_size = 90
dropout_rate = 0.5 
patience = 20
fs = 160


# Defining input shape
cnn_input_shape = (all_Train_X.shape[1], all_Train_X.shape[2], 1)  

# One hot encode
label_binarizer = LabelBinarizer()
all_Train_Y = label_binarizer.fit_transform(all_Train_Y)
all_Val_Y = label_binarizer.fit_transform(all_Val_Y)
all_Test_Y = label_binarizer.fit_transform(all_Test_Y)


# Callbacks
my_callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    ]


# Create and train your model 
model = CNN_model(cnn_input_shape, num_classes=n_classes, learning_rate=learning_rate, dropout_rate=dropout_rate)
history = model.fit(all_Train_X, all_Train_Y, epochs=1000, verbose=1, batch_size=batch_size, validation_data=(all_Val_X,all_Val_Y), callbacks = my_callbacks)
train_loss = history.history['loss']


# Evaluate the model's performance 
CNN_test_loss, CNN_test_acc = model.evaluate(all_Test_X, all_Test_Y, verbose=0)
print(f'Test accuracy for CNN: {CNN_test_acc}')
print(f'Test loss for CNN: {CNN_test_loss}')

    
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


# Generating classification report
y_pred = model.predict(all_Test_X)
y_pred_labels = (y_pred > 0.5).astype(int)

report = classification_report(all_Test_Y, y_pred_labels)
print(report)
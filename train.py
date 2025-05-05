from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Flatten
import numpy as np
import argparse
from model_utils import get_model_conv_rp, sp, dumpModelH5
import shutil
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=7800)]
    )

logical_gpus = tf.config.list_logical_devices('GPU')

parser = argparse.ArgumentParser()
parser.add_argument('--iet', type=int, required=True, help='iET value')
parser.add_argument('--ieta', type=int, required=True, help='iETA value')
parser.add_argument('--n_splits', type=int, default=10, help='Number of splits for cross-validation')
args = parser.parse_args()

# Loading dataset
iet, ieta = args.iet, args.ieta
data_path = 'dataset/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et%i_eta%i.npz'
samples = np.load(data_path %(iet, ieta))

data = samples['data']
features = samples['features']
target = samples['target']
ksamples_Sig = np.max(np.argwhere(target==1)) + 1
dataset = np.concatenate(((data[0:ksamples_Sig,1:101]),data[ksamples_Sig:,1:101]),axis=0)

# Building neural network
models = get_model_conv_rp(100)

# Preparing stratified k-fold cross-validation
random_state_cv = 7
skf = StratifiedKFold(n_splits = args.n_splits, random_state = random_state_cv, shuffle = True)
numInit = 10
history_model = []
model_mlp_cv = []

batch = 1024
num_epochs = 50

# Tunning models
total_folds = skf.get_n_splits(dataset, target)
for i, (train_index, test_index) in enumerate(skf.split(dataset, target)):
    for model in models:
        callback = sp() #customized callback
        callback.set_validation_data((dataset[test_index], target[test_index])) # Set validation data here

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(dataset[train_index],target[train_index],batch_size = batch,epochs=num_epochs,verbose=2,validation_data=(dataset[test_index], target[test_index]),callbacks=[callback])
        history_model.append(history)
        model_mlp_cv.append(model)
    percent = (i + 1) / total_folds * 100
    print(f"Progress: {i + 1} / {total_folds} ({percent:.1f}%)")

# Extract results and properties of the tuned models - saving it
filename = 'models/model_summary_cnn_rpringer_et%i_eta%i.h5' % (iet, ieta)
dumpModelH5(models, history_model, skf, dataset, target, seed_cv=random_state_cv, numInit=numInit, output_file=filename)
destination = "cernbox/TCC_CNN/data2017"
shutil.copy(
    filename, destination
)

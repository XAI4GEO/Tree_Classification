# Import libraries
import keras
import tensorflow as tf
import os
from keras import backend as k
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
from pathlib import Path
import xarray as xr
import json
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, Callback

# Set Keras backend for image data format
keras.backend.set_image_data_format("channels_last")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import backbones
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16

# import data agumentation layers
from keras.layers import RandomCrop, RandomFlip, RandomTranslation, RandomRotation, RandomZoom, RandomContrast 
import numpy as np

from tensorflow.keras.models import save_model

rng = np.random.default_rng(seed=42)


class TrainingAugmentationLayers(keras.layers.Layer):
    # def __init__(self):
    #     super().__init__()
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.random_flip = RandomFlip("horizontal_and_vertical")
        self.random_zoom = RandomZoom(0.1, fill_mode='reflect')
        self.random_rotate = RandomRotation(0.1, fill_mode='reflect')
        self.random_translation = RandomTranslation(0.3, 0.3, fill_mode='reflect')
        self.random_contrast = RandomContrast(0.1)
        
    def build(self, input_shape):
        # This method is called once the input shape is known
        super().build(input_shape)
        
    def call(self, inputs):
        p = np.random.uniform(0, 1, 5)
        x = inputs
        if p[0]>0.5:
            x = self.random_flip(inputs)
        if p[1]>0.5:
            x = self.random_zoom(x)
        if p[2]>0.5:
            x = self.random_rotate(x)
        # if p[3]>0.5:
        #     x = self.random_translation(x)
        # if p[4]>0.5:       
        #     x = self.random_contrast(x)
        return x
    
    
# data_augmentation = tf.keras.Sequential([
#     RandomFlip("horizontal_and_vertical"),
#     RandomRotation(0.2),
#     RandomZoom(0.2)
# ])


# use backbones modle
def _backbone_model(nneurons, inputshape, backbone="mobilenet"):
    inputs = keras.Input(shape=inputshape,name = 'image_input')
    if backbone=="mobilenet":    
        base_model = MobileNet(input_shape=inputshape, weights='imagenet', pooling=max, include_top = False) 
    elif backbone=="resnet50":
        base_model = ResNet50(input_shape=inputshape, weights='imagenet', pooling=max, include_top = False)
    elif backbone=="vgg16":
        base_model = VGG16(input_shape=inputshape, weights='imagenet', pooling=max, include_top = False) 
          
    x = base_model(inputs)
    pooledOutput = keras.layers.GlobalAveragePooling2D()(x)
    pooledOutput = keras.layers.Dense(nneurons[4])(pooledOutput)
    pooledOutput = keras.layers.Dropout(0.3)(pooledOutput)
    outputs = keras.layers.Dense(nneurons[5])(pooledOutput)
    model = keras.Model(inputs, outputs)
    print(model.summary())
    return model


# Model creation
def _create_model(nneurons, nfilters, ndropout, npool):
    
    inputs = keras.Input((128, 128, 3))
    
    # 4 layers
    x = keras.layers.Conv2D(nneurons[0], (nfilters[0], nfilters[0]), padding="same", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(npool[0], npool[0]), data_format='channels_last')(x)

    x = keras.layers.Conv2D(nneurons[1], (nfilters[1], nfilters[1]), padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(npool[1], npool[1]), data_format='channels_last')(x)

    x = keras.layers.Conv2D(nneurons[2], (nfilters[2], nfilters[2]), padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(npool[2], npool[2]), data_format='channels_last')(x)

    x = keras.layers.Conv2D(nneurons[3], (nfilters[3], nfilters[3]), padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(npool[3], npool[3]), data_format='channels_last')(x)
    x = keras.layers.Dropout(ndropout[0])(x)

    pooledOutput = keras.layers.GlobalAveragePooling2D()(x)
    pooledOutput = keras.layers.Dense(nneurons[4])(pooledOutput)
    outputs = keras.layers.Dense(nneurons[5])(pooledOutput)

    model = keras.Model(inputs, outputs)
    print(model.summary())
    return model

@keras.saving.register_keras_serializable(package="MyLayers")
class euclidean_lambda(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(euclidean_lambda, self).__init__(**kwargs)
        self.name = 'euclidean_lambda'

    def call(self, featA, featB):
        squared = keras.ops.square(featA-featB)
        return squared

def siamese_model(nneurons, nfilters, ndropout, npool):
    # feature_extractor_model = _create_model(nneurons, nfilters, ndropout, npool)
    inputshape = (128, 128, 3)
    # change the backbone to one of these: "mobilenet", "resnet50", "vgg16"
    feature_extractor_model = _backbone_model(nneurons, inputshape, backbone="mobilenet")
    imgA = keras.Input(shape=inputshape)
    imgB = keras.Input(shape=inputshape)
    
    agumentation = TrainingAugmentationLayers()
    P = np.random.uniform(0, 1, 2)
    if P[0]>0.5:
        imgA = agumentation(imgA)
    if P[1]>0.5:
        imgB = agumentation(imgB)
            
    featA = feature_extractor_model(imgA)
    featB = feature_extractor_model(imgB)
    distance = euclidean_lambda()(featA, featB)
    outputs = keras.layers.Dense(1, activation="sigmoid")(distance)
    model = keras.Model(inputs=[imgA, imgB], outputs=outputs)
    return model

def compile_model(model, lr, metrics):
    opt = keras.optimizers.Adam(learning_rate=lr)
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = metrics
    model.compile(loss=loss, optimizer=opt, metrics=metrics)


def plot_history(history, metrics):
    """
    Plot the training history

    Args:
        history (keras History object that is returned by model.fit())
        metrics (str, list): Metric or a list of metrics to plot
    """
    history_df = pd.DataFrame.from_dict(history.history)
    fig = plt.figure(figsize=(10, 5))
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("metric")
    plt.ylim(0, 1)


def plot_prediction(labels_pred, labels_pair):
    fig = plt.figure(figsize=(10, 5))
    
    sub = fig.add_subplot(1, 2, 1)
    sub.plot(labels_pred[labels_pair==1], '.', markersize=0.5)
    sub.plot(labels_pair[labels_pair==1])

    sub = fig.add_subplot(1, 2, 2)
    sub.plot(labels_pred[labels_pair==0], '.', markersize=0.5)
    sub.plot(labels_pair[labels_pair==0],)


def plot_prediction_hist(labels_pred):
    fig = plt.figure(figsize=(10, 5))

    counts, bins = np.histogram(labels_pred[labels_pair==1])
    sub = fig.add_subplot(1, 2, 1)
    sub.stairs(counts, bins)

    sub = fig.add_subplot(1, 2, 2)
    counts, bins = np.histogram(labels_pred[labels_pair==0])
    sub.stairs(counts, bins)

def lr_schedule(epoch):
        if epoch<=10:
            return 0.001
        elif epoch<=35:
            return 0.0002
        elif epoch<=60:
            return 0.00005
        else:
            return 0.00001
        
# Custom callback to plot predictions every 10 epochs
class PlotPredictionsCallback(Callback):
    def __init__(self, x_val, y_val, dir_training):
        super(PlotPredictionsCallback, self).__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.dir_training = dir_training
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:
            
            # Make predictions on the validation data
            images_pair = self.x_val
            labels_pred = self.model.predict([images_pair[:,0], images_pair[:,1]])
            
            dir_epoch = self.dir_training / f'epoch_{epoch+1}'
            dir_epoch.mkdir(exist_ok=True)
            
            # Plots
            plot_prediction(labels_pred, self.y_val)
            plt.savefig(dir_epoch/'prediction.png')
            plot_prediction_hist(labels_pred)
            plt.savefig(dir_epoch/'prediction_hist.png')
            plt.close('all')

# class LoadHistoryCallback(Callback):
#     def __init__(self, pickle_file):
#         super(LoadHistoryCallback, self).__init__()
#         self.pickle_file = pickle_file

#     def on_train_begin(self, logs=None):
#         with open(self.pickle_file, 'rb') as f:
#             self.history = pickle.load(f)

#     def on_epoch_end(self, epoch, logs=None):
#         for key in self.history.history.keys():
#             if key in logs:
#                 logs[key] = self.history.history[key][:]

            
if __name__ == "__main__":
    
    # train_name = f'More_dissimilar_pairs_{Path(__file__).stem}'
    train_name = f'Agu_pairs_{Path(__file__).stem}'
    nneurons = [32, 64, 64, 128, 256, 128]
    nfilters = [3, 3, 5, 5]
    ndropout = [0.4]
    npool = [2, 2, 2, 2]
    lr_init = 0.25e-04 # initial learning rate

    # data = xr.open_zarr("/data/Projects/2024_Invasive_species/Tree_Classification/notebooks/step0_data_preparation_examples/traing_pairs_balanced.zarr")
    # data = xr.open_zarr("/data/Projects/2024_Invasive_species/Tree_Classification/notebooks/step0_data_preparation_examples/training_pairs_agu.zarr")
    data = xr.open_zarr("/data/Projects/2024_Invasive_species/Tree_Classification/notebooks/step0_data_preparation_examples/training_pairs_20000.zarr")
    
    # ### select 20000 pairs to avoid memory problem
    # NN = max(data.sizes["sample"], 20000)
    # idx_select = rng.integers(0, data.sizes["sample"], size=NN)
    # data = data.isel(sample=idx_select, drop=True)    
    
    images_pair = data["X"].to_numpy()
    labels_pair = data["Y"].to_numpy()

    # Scale to [0, 1]
    images_pair = images_pair/255.

    model = siamese_model(nneurons, nfilters, ndropout, npool)

    #Compile model
    metrics = [keras.metrics.BinaryAccuracy(threshold=0.5)]    
    opt = keras.optimizers.Adam(learning_rate=lr_init)
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    # Configure training    
    dir_training = Path("/data/Projects/2024_Invasive_species/Tree_Classification/optimized_models/results_training/") / train_name
    dir_training.mkdir(exist_ok=True)

    # Set callbacks
    # learning_rate_scheduler = LearningRateScheduler(lr_schedule)
    learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    plot_pred_callback = PlotPredictionsCallback(images_pair, labels_pair, dir_training)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, start_from_epoch=5, restore_best_weights=True)
    callbacks = [learning_rate_scheduler, plot_pred_callback, earlystop]

    history = model.fit([images_pair[:,0], images_pair[:,1]], labels_pair[:], batch_size=4, epochs=10, validation_split=0.2, callbacks=callbacks)
   
    # Save training results
    metadata ={ 
        "nneurons": nneurons,
        "nfilters": nfilters,
        "ndropout": ndropout,
        "npool":npool,
    } 
    with open(dir_training /"hyperparams.json", "w") as file:
        json.dump(metadata, file)
 
    keras.saving.save_model(model, dir_training / 'siamese_model.keras', overwrite=True)
    model.save_weights(dir_training / "optimized_weights.weights.h5")
    
    # save_model(model, dir_training / 'siamese_base_model.h5')
    
    with open(dir_training / "history.pkl", "wb") as file_pi:
        pickle.dump(history, file_pi)
    
    # Model evaluation plots
    model_loaded = keras.saving.load_model(dir_training / 'siamese_model.keras', compile=False)
    labels_pred = model.predict([images_pair[:, 0], images_pair[:, 1]])
    plot_history(history, ['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])
    plt.savefig(dir_training / "history.png")
    plot_prediction(labels_pred, labels_pair)
    plt.savefig(dir_training / "prediction.png")
    plot_prediction_hist(labels_pred)
    plt.savefig(dir_training / "prediction_hist.png")

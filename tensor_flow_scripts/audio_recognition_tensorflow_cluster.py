import os
import pathlib
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.python.ops.linalg_ops import norm

commands = []
AUTOTUNE = None

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this 
    # to work in a TensorFlow graph.
    return parts[-3]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_spectrogram(waveform):
    # Padding for files with less than 520000 samples
    zero_padding = tf.zeros([48000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the 
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # spectrogram = tf.signal.rfft(waveform, fft_length=500)

    spectrogram = tf.abs(spectrogram)

    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
    return output_ds

def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

def stringify_list(l):
    temp = np.array(l)
    fin = list(dict.fromkeys([re.search('[dn]-p\d-..-..-..', x.decode()).group() for x in temp]))
    # fin.sort()

    return '\n'.join(fin)

def main(batch, epoch):
    global commands
    global AUTOTUNE
    # Set seed for experiment reproducibility
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # set train, validate and test size
    ds_size = 274 * 2
    train_size = int(ds_size * 0.8)
    val_size = int(ds_size * 0.1)
    test_size = ds_size - train_size - val_size

    data_dir = pathlib.Path('E:\\wav_files\\fragments_clustered')

    # extract "command" names
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    print('Commands:', commands)

    # load audio files and shuffle them, maybe look into canceling shuffeling
    foldernames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    foldernames = tf.random.shuffle(foldernames)

    filenames_temp = tf.io.gfile.glob(np.array(foldernames)[0].decode() + '/*')
    for fol in np.array(foldernames)[1:]:
        filenames_temp = tf.concat([filenames_temp, tf.io.gfile.glob(fol.decode() + '/*')], 0)

    filenames = tf.convert_to_tensor(filenames_temp)

    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    print('Number of examples per label:',
        len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
    print('Example file tensor:', filenames[0])

    # split into training, validating and testing files
    train_files = tf.random.shuffle(filenames[:train_size])
    val_files = tf.random.shuffle(filenames[train_size: train_size + val_size])
    test_files = tf.random.shuffle(filenames[-test_size:])

    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))

    print('Training set: ', train_files)
    print('Validation set: ', val_files)
    print('Test set: ', test_files)

    # print("===================================================")
    # print("training files: ", train_files)
    # print("validation files: ", val_files)
    # print("test files", test_files)

    AUTOTUNE = tf.data.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    for waveform, label in waveform_ds.take(1):
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(waveform)

        print('Label:', label)
        print('Waveform shape:', waveform.shape)
        print('Spectrogram shape:', spectrogram.shape)

    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 48000])
    plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    # plt.show()


    spectrogram_ds = waveform_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

    rows = 3
    cols = 3
    n = rows*cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
        ax.set_title(commands[label_id.numpy()])
        ax.axis('off')

    # plt.show()


    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)

    batch_size = batch
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
    print('Input shape:', input_shape)
    num_labels = len(commands)

    norm_layer = preprocessing.Normalization()
    temp = spectrogram_ds.map(lambda x, _: x)
    norm_layer.adapt(temp)

    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32), 
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = epoch
    history = model.fit(
        train_ds, 
        validation_data=val_ds,  
        epochs=EPOCHS,
        # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    # plt.show()

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    pred = model.predict(test_audio)
    print(">>>>>> input shape of test_audio", tf.shape(test_audio))
    y_pred = np.argmax(pred, axis=1)
    y_true = test_labels

    print(pred)
    print("Length of pred: ", len(pred))

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, 
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    start = round(time.time() * 1000)
    model_name = "a" + str(int(test_acc * 100)) + "_" + str(start) + "_e" + str(EPOCHS) + "_b" + str(batch_size)
    
    model.save("E:/models/" + model_name)

    f = open("E:/modelset/" + model_name + ".txt", "w")
    f.write(">> Training set:\r\n")
    f.write(stringify_list(train_files) + "\r\n")
    f.write("\r\n\r\n>> Validation set:\r\n ")
    f.write(stringify_list(val_files) + "\r\n")
    f.write("\r\n\r\n>> Test set:\r\n ")
    f.write(stringify_list(test_files) + "\r\n")
    f.close()

    return (metrics['accuracy'], metrics['val_accuracy'], test_acc)


# acc = [[]]
# for x in range(3,21):
#     acc_s = []
#     for i in range(0,6):
#         acc_s.append(main(19, x))
#     acc.append(acc_s)

# print('Operation done.')
# x = 1
# for a in acc:
#     print('Overall accuracies of epochs (', x, ') with accuracies: ', a)
#     x += 1

out = main(19, 3)
print("Model generated with accuracy ", out)




















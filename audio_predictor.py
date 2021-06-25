import csv
import json
import os
import pathlib
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.python.ops.linalg_ops import norm
from utils.psi_connection import PsiConnection

class AudioPredictor:

    def __init__(self):
        self.data_dir = pathlib.Path('E:\\wav_files\\fragments_split')
        # extract "command" names
        self.commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))
        print('Commands:', self.commands)

        self.model = tf.keras.models.load_model('E:\models\\a60_1623867156315_e3_b19')
        self.model.summary()
        self.AUTOTUNE = tf.data.AUTOTUNE

        # self.conn = PsiConnection(pub_ip="127.0.0.1:12346", sub_ip="127.0.0.1:12345", sync=False)


    def decode_audio(self, audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)

        # Note: You'll use indexing here instead of tuple unpacking to enable this 
        # to work in a TensorFlow graph.
        return parts[-2]

    def get_waveform_and_label(self, file_path):
        label = self.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = self.decode_audio(audio_binary)
        return waveform, label

    def get_spectrogram(self, waveform):
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

    def get_spectrogram_and_label_id(self, audio, label):
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == self.commands)
        return spectrogram, label_id

    def preprocess_dataset(self, files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=self.AUTOTUNE)
        output_ds = output_ds.map(
            self.get_spectrogram_and_label_id,  num_parallel_calls=self.AUTOTUNE)
        return output_ds

    def test_func(self):
        files = tf.io.gfile.glob([str(self.data_dir) + '/distracting/d-p2-01-04-02.wav'])
        ds = self.preprocess_dataset(files)

        audio_set = []
        label_set = []
        for audio, label in ds:
            audio_set.append(audio.numpy())
            label_set.append(label.numpy())

        audio_set = np.array(audio_set)
        label_set = np.array(label_set)

        print("\r\n\r\n >> amount of files: ", len(files))
        print("\r\n >> files: ", files)
        prediction = self.model.predict(audio_set)
        print(">>>>>> input shape of test_audio", tf.shape(audio_set))
        print("Prediction: ", prediction.tolist())
        print("\r\n Length: ", len(prediction.tolist()))

        # print(tf.io.read_file(files[0]))

    def pred_distraction(self, frag):

        spectrogram = self.get_spectrogram(tf.convert_to_tensor(frag))
        spectrogram = tf.expand_dims(spectrogram, -1)

        in_array = np.array([spectrogram])

        pred = self.model.predict(in_array)
        return pred

    def loop(self):
        # c = 0
        # curr = np.array([])

        # while True:
        #     [topic, message] = self.conn.sub_sock.recv_multipart()
        #     c += 1
        #     j = json.loads(message)
        #     np.insert(curr, j['message']['Data'])
        #     # np.concatenate((curr, res))

        #     if(c == 10):
        #         c = 0
        #         pred = self.pred_distraction(curr)
        #         curr = np.array([])
        #         self.conn.publish("LavAudioPrediction", j['originatingTime'], pred)

        outf = open("E:/derived_csv/p3-01-lav-filter.csv", 'w', newline='')
        csvwrite = csv.writer(outf, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        files = tf.io.gfile.glob(['E:/wav_files/p3-01/21_06_08_17_49_03_p3-01_lav.wav'])
        print(">>>>>>>>>> Files", files)
        audio_binary = tf.io.read_file(np.array(files)[0])
        waveform = np.array(self.decode_audio(audio_binary))
        size = int(len(waveform) / 44100.0)

        filter = True

        for x in range(0, size):
            start = x * 44100
            end = (x+1) * 44100
            frag = waveform[start:end]
            pred = self.pred_distraction(frag)
            pred_prc = np.array(tf.nn.softmax(pred))
            print("Fragment: %d | start: %d | end %d | fragment: %d | Prediction: %s | Derived: %s"%(x, start, end, len(frag), pred_prc[0], pred_prc[0][0]))

            if(pred_prc[0][0] <= 0.5 and filter):
                csvwrite.writerow([x, 0.0])
            else:
                csvwrite.writerow([x, pred_prc[0][0]])

if __name__ == "__main__":
    audio_predictor = AudioPredictor()
    audio_predictor.test_func()
    audio_predictor.loop()


#!/usr/bin/python

""" AudioProcessor: A python script that records microphone data, and processes it for use in PSI. Will output the raw audio data and the calculated/estimated frequency over time.

***************************************************************************************
Developer: Jeffrey Pronk
Github: jeffthestig
Email: j.s.pronk@student.tudelft.nl

Year: 2021

Developed at the request of Yoon Lee as part of the CSE BSc Research Project (CSE3000).
***************************************************************************************

A python script that records microphone data, and processes it for use in PSI. Will output the raw audio data and the calculated/estimated frequency over time.
Needs to be accompanied by a audio_setup.json file. If not present, will be created!
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import struct
import time
import wave

from datetime import datetime
from sys import exit
from utils.psi_connection import PsiConnection


class AudioProcessor:
    """ AudioProcessor: Class responsible for the audio capture, processing and publishing.
    """

    def __init__(self):
        # collect setup
        self.collect_settings()

        # init pyaudio and setup datastream for mic
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.BLOCK)

        # init socket 
        if(self.online):
            self.conn = PsiConnection(pub_ip=self.PUB_IP, sub_ip=self.SUB_IP, sync=self.SYNC_TIME)

        # init matplotlib
        if(self.matplotlib_use):
            plt.ion()
            self.fig = plt.figure()
            plt.xlim(0,60)
            plt.ylim(0,33000)

            # self.canvas = np.zeros((4800,6400))
            # self.screen = pf.screen(self.canvas, 'Sound-level')

        # init wave file
        if(self.wav_use):
            start = round(time.time() * 1000)
            self.wav = wave.open("./recordings/test_wave_" + str(start) + ".wav", "w")
            self.wav.setparams((self.CHANNELS, 2, self.RATE, 600, "NONE", "not compressed"))
            

    def collect_settings(self):
        try:
            f = open("./config/audio_setup.json", "r")
            js = json.loads(f.read())

            self.BLOCK = js["audio"]["block"]
            self.RATE = js["audio"]["rate"]
            self.CHANNELS = js["audio"]["channels"]

            self.SUB_IP = js["socket"]["sub_ip"]
            self.PUB_IP = js["socket"]["pub_ip"]
            self.RAW_DATA_TOPIC_NAME = js["socket"]["raw_data_topic"]
            self.FREQ_TOPIC_NAME = js["socket"]["freq_topic"]
            self.SYNC_TIME = js["socket"]["sync_time"]

            self.matplotlib_use = js["usage"]["matplotlib_use"]
            self.wav_use = js["usage"]["wav_use"]
            self.online = js["usage"]["online"]
        except Exception as e:
            print("EXCEPTION: " + str(e))
            print("Please make sure the file ./audio_setup.json is created and that the file is correct.\r\n" + 
            "The following JSON string should be present in the json file: \r\n{\r\n\t\"audio\": {\r\n\t\t\"block\": <block size>, \r\n\t\t\"rate\": <sample rate mic>, \r\n\t\t\"channels\": 1\r\n\t},\r\n\t\"socket\": {\r\n\t\t\"sub_ip\": \"<subscribe ip, connect ip>\",\r\n\t\t\"pub_ip\": \"<publish ip, bind ip>\"\r\n\t\t\"raw_data_topic\": \"<name of raw data topic>\", \r\n\t\t\"freq_topic\": \"<name of freq data topic>\", \r\n\t\t\"sync_topic\": \"<true or false, if audio should be synced>\"\r\n\t},\r\n\t\"usage\": {\r\n\t\t\"matplotlib_use\": <true if matplotlib should be used!>, \r\n\t\t\"wav_use\": <true if wave file should be created!>, \r\n\t\t\"online\": <true if socket should be enabled (should be default), false if socket is disabled: no comms with PSI!>\r\n\t}\r\n}")

            exit()
    
    def record_loop(self):
        try:
            # create array and store recorded data, print it and plot it
            print("Starting record loop: \r\n" + 
                "rate: %d"%self.RATE + "\r\n" + 
                "block: %d"%self.BLOCK + "\r\n" + 
                "channels: %d"%self.CHANNELS)
            # for i in range(60 * int(self.RATE / self.BLOCK)):
            i = 0
            while True:
                recb = np.frombuffer(self.stream.read(self.BLOCK), dtype=np.int16)
                # ts = datetime.datetime.utcnow().isoformat()
                ts = datetime.utcnow()
                # ts = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]
                s = ( (i-1) * self.BLOCK ) / self.RATE
                e = ( i * self.BLOCK ) / self.RATE
                x = np.linspace(s, e, self.BLOCK)

                # print(recb)

                ## FFT to obtain peak freq, working great! Just needs some filtering
                # Data improved by disabling windows audio processing
                data = recb * np.hanning(len(recb)) # smooth the FFT by windowing data
                fft = abs(np.fft.fft(data).real)
                fft = fft[:int(len(fft)/2)] # keep only first half
                freq = np.fft.fftfreq(self.BLOCK,1.0/self.RATE)
                freq = freq[:int(len(freq)/2)] # keep only first half
                freqPeak = freq[np.where(fft==np.max(fft))[0][0]]+1
                print("peak frequency: %d Hz"%freqPeak)

                if(self.online):
                    # TODO: refactor to seperate method!
                    # rdp = {}
                    # rdp['message'] = recb
                    # rdp['originatingTime'] = ts
                    # self.sock.send_multipart([self.RAW_DATA_TOPIC_NAME, json.dump(rdp).encode('utf-8')])

                    # self.conn.publish(self.RAW_DATA_TOPIC_NAME, ts, recb.tolist())
                    # for c in range(self.BLOCK):
                    #     try:
                    #         delta = (-self.BLOCK + c + 1) * 1000 / self.RATE
                    #         ts_d = datetime.timedelta(milliseconds=delta)


                    #         self.conn.publish(self.RAW_DATA_TOPIC_NAME, (ts + ts_d).isoformat(), int(recb[c]))
                    #     except KeyboardInterrupt:
                    #         print("AudioProcessor was interrupted. Shutting down python script!")
                    #         self.close()

                    self.conn.publish(self.RAW_DATA_TOPIC_NAME, ts, str(recb.tolist()))

                    # fp = {}
                    # fp['message'] = freqPeak
                    # fp['originatingTime'] = ts
                    # self.sock.send_multipart([self.FREQ_TOPIC_NAME, json.dump(fp).encode('utf-8')])
                    self.conn.publish(self.FREQ_TOPIC_NAME, ts, freqPeak)

                if(self.matplotlib_use):
                    plt.plot(x, abs(recb)) # Audio signal
                    # self.fig.canvas.draw()

                    # image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.int16, sep='')
                    # image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

                    # self.screen.update(image)
                    # plt.show()
                    # plt.draw()
                    plt.pause(0.001)

                    # drawnow(self.plot_draw, False, False, False, x, abs(recb))
                    

                if(self.wav_use):
                    for x in recb:
                        self.wav.writeframes(struct.pack('h', x))
                
                i += 1
        except KeyboardInterrupt:
            print("AudioProcessor was interrupted. Shutting down python script!")
        except Exception as e:
            print("LOOP EXCEPTION: " + str(e))

    def plot_draw(self, x, y):
        plt.plot(x, y, c='black')

        # self.plot_sem = False

    def close(self):
            
        if(self.wav_use):
            self.wav.close()

        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

        if(self.matplotlib_use):
            plt.show(block=True)

        print("Closed down audio processing script!")
        exit()

if __name__ == "__main__":
    audio_processor = AudioProcessor()
    audio_processor.record_loop()
    audio_processor.close()


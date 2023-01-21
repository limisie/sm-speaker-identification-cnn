import tkinter
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox
import pyaudio
import wave

from audio_processing import process_audio_sample
from consts import CLASS_NAMES, TEST_DATA_ROOT
from ai_utils import analyze


class App:
    def __init__(self, models=CLASS_NAMES, data_path=TEST_DATA_ROOT, chunk=3024, frmat=pyaudio.paInt16, channels=1,
                 rate=44100, py=pyaudio.PyAudio()):
        self.main = tkinter.Tk()
        self.collections = []
        self.main.geometry('900x800')
        self.main.title('Record')
        self.CHUNK = chunk
        self.FORMAT = frmat
        self.CHANNELS = channels
        self.RATE = rate
        self.p = py
        self.frames = []
        self.st = 1
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True,
                                  frames_per_buffer=self.CHUNK)
        self.results = []
        self.listOfModels = models

        self.audio_file = None
        self.model_path = None
        self.data_path = data_path

        self.buttons = tkinter.Frame(self.main, padx=50, pady=20)

        self.buttons.pack(fill=tk.BOTH)

        self.listbox = tkinter.Listbox(self.buttons, width=30, height=5, selectmode=tkinter.SINGLE)
        self.listbox.grid(row=0, column=0, columnspan=1, padx=10, pady=5)

        for item in self.listOfModels:
            self.listbox.insert(tkinter.END, item)

        self.line = tkinter.Label(self.buttons, text='_____________________________________________________________')

        self.strt_rec = tkinter.Button(self.buttons, width=10, padx=10, pady=5, text='Start Recording',
                                       command=lambda: self.start_record())
        self.strt_rec.grid(row=1, column=0, padx=50, pady=5)
        self.stop_rec = tkinter.Button(self.buttons, width=10, padx=10, pady=5, text='Stop Recording',
                                       command=lambda: self.stop())
        self.stop_rec.grid(row=2, column=0, columnspan=1, padx=50, pady=5)
        self.select_file = tk.Button(self.buttons, width=10, padx=10, pady=5, text="Select Audio File",
                                     command=lambda: self.select_audio_file())
        self.select_file.grid(row=3, column=0, columnspan=1, padx=50, pady=5)
        self.select_model = tk.Button(self.buttons, width=10, padx=10, pady=5, text="Select Model File",
                                      command=lambda: self.select_model_file())
        self.select_model.grid(row=4, column=0, columnspan=1, padx=50, pady=5)

        self.file_string = tkinter.StringVar()
        self.file_string.set('No file selected')
        self.file_label = tkinter.Label(self.buttons, textvariable=self.file_string)
        self.file_label.grid(row=5, column=0, columnspan=1, padx=50, pady=5)

        self.model_string = tkinter.StringVar()
        self.model_string.set('No file selected')
        self.model_label = tkinter.Label(self.buttons, textvariable=self.model_string)
        self.model_label.grid(row=6, column=0, columnspan=1, padx=50, pady=5)

        self.label = tkinter.Label(self.buttons,
                                   text='Press Process Audio File and then Analyze '
                                        'to start authentication process on selected audio file')
        self.label.grid(row=7, column=0, columnspan=1, padx=50, pady=5)

        self.create_spectrograms = tk.Button(self.buttons, width=10, padx=10, pady=5, text="Process Audio File",
                                             command=lambda: self.process_audio())
        self.create_spectrograms.grid(row=8, column=0, columnspan=1, padx=50, pady=5)

        self.analyze = tkinter.Button(self.buttons, width=10, padx=10, pady=5, text='Analyze',
                                      command=lambda: self.analyze_audio_file())
        self.analyze.grid(row=9, column=0, columnspan=1, padx=50, pady=5)

        self.message_box = tkinter.Text(self.buttons, width=100, height=20, padx=0, pady=5)
        self.message_box.grid(row=10, column=0, columnspan=1, padx=50, pady=5)

        self.line = tkinter.Label(self.buttons, text='_____________________________________________________________')

        self.main.mainloop()

    def select_model_file(self):
        self.model_path = filedialog.askopenfilename(initialdir="../", title="Select Trained Model",
                                                     filetypes=(("pt files", "*.pt"), ("all files", "*.*")))
        self.model_string.set(self.model_path)
        self.message_box.insert(tkinter.END, 'Model selected: ' + self.model_path + ' \n\n')

    def select_audio_file(self):
        self.audio_file = filedialog.askopenfilename(initialdir="../", title="Select Audio File",
                                                     filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
        self.file_string.set(self.audio_file)
        self.message_box.insert(tkinter.END, 'File selected: ' + self.audio_file+ ' \n\n')

    def process_audio(self):
        process_audio_sample(self.audio_file, self.data_path)
        self.message_box.insert(tkinter.END, 'Spectrograms created! \n\n')

    def analyze_audio_file(self):
        self.check_data()
        self.results = analyze(self.model_path, self.data_path)
        self.show_results()

    def check_data(self):
        if self.model_path is None or self.audio_file is None:
            self.message_box.insert(tkinter.END, 'No model or audio selected! \n\n')

    def show_results(self):
        for line in self.results:
            self.message_box.insert(tkinter.END, line)
        self.message_box.insert(tkinter.END, '\n\n')

    def start_record(self):
        self.st = 1
        self.frames = []
        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True,
                             frames_per_buffer=self.CHUNK)
        self.message_box.insert(tkinter.END, 'Recording started! \n')

        while self.st == 1:
            data = stream.read(self.CHUNK)
            self.frames.append(data)
            print("* recording")
            self.main.update()

        stream.close()

        wf = wave.open('../data/tests/test_recording.wav', 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.audio_file = '../data/tests/test_recording.wav'
        self.file_string.set(self.audio_file)

    def stop(self):
        self.message_box.insert(tkinter.END, 'Recording stopped!  \n\n')
        self.st = 0


app = App()

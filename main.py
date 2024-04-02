import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave
import json
import time
import torch

from scipy.io import wavfile
from scipy.fft import fft
from scipy import signal
from mpl_toolkits import mplot3d


def get_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data



def add_song_hash_json(file_path, spectrogram_data):
    with open(file_path, 'r') as file:
        existing_data = json.load(file)

    new_element = {
        "id": len(existing_data) + 1,
        "hash": spectrogram_data.tolist()
    }

    existing_data.append(new_element)

    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)


def get_hash_tables_for_songs(filename, interval):
    data = []
    for i in range(0, 109):
        dt = dict()

        song_path = 'songs\\' + str(i + 1) + '.wav'
        print('Currently hashing: ', song_path)
        _, x = get_signal(song_path)

        sample_spectrogram = get_signal_spectrogram(x, interval)
        sample_hash = get_signal_hash_table(sample_spectrogram)

        dt['id'] = str(i + 1)
        dt['hash_table'] = sample_hash.tolist()

        data.append(dt)

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def add_song_duration_to_songs_info(sample_rate: int):
    data = get_json_data('songs_info.json')
    for i in range(len(data)):
        song_path = 'songs\\' + str(i + 1) + '.wav'
        _, x = get_signal(song_path)
        data[i]['duration'] = x.shape[0] // sample_rate
    with open('songs_info.json', 'w') as file:
        json.dump(data, file, indent=4)


def seconds_into_minutes_seconds(seconds):
    _minutes = seconds // 60
    if seconds % 60 < 10:
        return f'{seconds // 60}.0{seconds % 60}'
    return f'{seconds // 60}.{seconds % 60}'


def record_sound(file_path: str, chunk: int, channels: int, sample_rate: int, record_seconds: int):
    p = pyaudio.PyAudio()
    sample_format = pyaudio.paInt16

    frames = []

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=sample_rate,
                    frames_per_buffer=chunk,
                    input=True)

    for i in range(0, int(sample_rate / chunk * record_seconds), 1):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    wf = wave.open(file_path, "wb")

    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))

    p.terminate()

    wf.close()


def get_signal(file_path: str):
    samplerate, x = wavfile.read(file_path)
    return samplerate, x


def get_signal_spectrogram(x, interval: int):
    epsilon = 1e-10

    if x.ndim > 1:
        x = x.mean(axis=1)

    # mean_value = np.mean(x)
    # print(mean_value)
    # x = x - mean_value

    extra_space = interval - x.shape[0] % interval
    # x = x.copy()
    x = np.append(x, np.zeros(extra_space))

    # x = np.concatenate((x, np.zeros(extra_space)))
    x_size = x.shape[0]

    slice_db = np.empty((x_size // interval, interval // 2))

    for i in range(0, x_size // interval, 1):
        to_fft = x[interval * i: interval * (i + 1)].copy()  # * signal.windows.blackman(interval)
        to_fft *= signal.windows.blackman(interval)
        transformed_frame = np.abs(fft(to_fft, interval))[0:interval // 2] + epsilon
        transformed_frame_db = 20 * np.log10(transformed_frame)
        slice_db[i] = transformed_frame_db

    return slice_db


def get_signal_hash_table(spectrogram):
    # Behavior of peaks in frequency ranges [40, 79], [80, 119], [120, 159], [160, 199]

    frequency_lowest = 40
    frequency_highest = 200

    slice_size = frequency_highest - frequency_lowest
    step_size = 40
    amount_of_steps = slice_size // step_size

    to_hash = spectrogram[:, frequency_lowest: frequency_highest]

    spectrogram_length = to_hash.shape[0]

    # Getting hash table of the signal using peaks in frequency ranges
    key_points = np.empty((spectrogram_length, amount_of_steps))

    for slice_index in range(0, spectrogram_length, 1):
        to_hash_slice = to_hash[slice_index]
        for step in range(0, slice_size, step_size):
            max_index = np.argmax(to_hash_slice[step:step + step_size - 1])
            key_points[slice_index][step // step_size] = max_index + step

    hash_table = np.empty(spectrogram_length)

    for slice_index in range(to_hash.shape[0]):
        hash_number = 2
        for i in range(amount_of_steps):
            hash_number = 173 * hash_number + key_points[slice_index][i]
        hash_table[slice_index] = hash_number

    return hash_table


# Returns the match score between the sample and the song using hash tables
def identify_song(sample_hash_table, song_hash_table):
    sample_hash_table = np.round(sample_hash_table / 1e6)
    song_hash_table = np.round(song_hash_table / 1e6)

    matches = []

    sample_hash_table_size = sample_hash_table.shape[0]
    song_hash_table_size = song_hash_table.shape[0]

    # Finding matches (beginning of the song)
    for i in range(0, sample_hash_table_size, 1):
        for j in range(0, song_hash_table_size, 1):
            if sample_hash_table[i] == song_hash_table[j]:
                matches.append(j - i)

    if len(matches) == 0:
        return 0, 0

    # Finding index of the mode in matches array
    unique_matches, counts = np.unique(matches, return_counts=True)

    number_of_mode = np.max(counts)
    mode = unique_matches[np.argmax(counts)]

    ratio = number_of_mode / sample_hash_table.shape[0]

    return ratio, mode / song_hash_table_size


def __shazam(sample_file_path, hash_table_file_path, songs_info_file_path, interval, chunk, channels, sample_rate,
             record_seconds):
    print("Recording...")

    record_sound(sample_file_path, chunk, channels, sample_rate, record_seconds)

    print('Finished recording.')

    _, x = get_signal(sample_file_path)

    sample_spectrogram = get_signal_spectrogram(x, interval)
    sample_hash_table = get_signal_hash_table(sample_spectrogram)

    hash_data = get_json_data(hash_table_file_path)

    ratios = []
    values = []

    print("Finding song...")

    for dt in hash_data:
        song_hash_table = np.array(dt['hash_table'])
        ratio, value = identify_song(sample_hash_table, song_hash_table)

        ratios.append(ratio)
        values.append(value)

    song_max = np.max(ratios)
    song_index = np.argmax(ratios)

    song_max_count = np.count_nonzero(ratios == song_max)

    for i in range(len(ratios)):
        print(i + 1, ' ', ratios[i])

    songs_data = get_json_data(songs_info_file_path)

    if round(np.max(ratios), 2) < 0.12:
        print('Song is not found.')
        if song_max_count == 1:
            print('Most likely song: ')
            for i in range(len(songs_data)):
                if ratios[i] == song_max:
                    print('Song:', songs_data[i]['artist'] + ' - ' + songs_data[i]['title'])
                    current_time = int(values[song_index] * songs_data[i]['duration'])
                    print(f'Song time: {seconds_into_minutes_seconds(current_time)}\n')
    else:
        if song_max_count == 1:
            for i in range(len(songs_data)):
                if ratios[i] == song_max:
                    print(i + 1, end=') ')
                    print('Song:', songs_data[i]['artist'] + ' - ' + songs_data[i]['title'])
                    current_time = int(values[song_index] * songs_data[i]['duration'])
                    print(f'Song time: {seconds_into_minutes_seconds(current_time)}\n')
                    break
        else:
            print('Almost there. Try on more time.')
            for dt in songs_data:
                if dt.get('id') == str(song_index + 1):
                    print('Song:', dt['artist'] + ' - ' + dt['title'])
                    current_time = int(values[song_index] * dt['duration'])
                    print(f'Song time: {seconds_into_minutes_seconds(current_time)}\n')


def shazam():
    sample_path = 'sample.wav'
    song_hash_tables_path = 'songs_hash_table.json'
    songs_info_path = 'songs_info.json'

    chunk = 2205
    channels = 2
    sample_rate = 44100
    record_seconds = 5

    interval = 2205

    __shazam(sample_path, song_hash_tables_path, songs_info_path, interval, chunk, channels, sample_rate,
             record_seconds)


def plot_spectrogram3D(spectrogram, interval, start, finish):
    sample_rate = 44100
    song_spectrogram = spectrogram[start * sample_rate // interval:finish * sample_rate // interval]

    x = np.linspace(0, sample_rate // 2, num=(song_spectrogram.shape[1]))
    y = np.linspace(start, finish, num=(song_spectrogram.shape[0]))

    X, Y = np.meshgrid(x, y)

    ax = plt.axes(projection='3d')
    ax.invert_xaxis()
    ax.plot_surface(X, Y, song_spectrogram, cmap='magma')

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Time')
    ax.set_zlabel('Magnitude')


def plot_spectrogram():
    chunk = 2025
    sample_rate = 44100

    # song_path = 'sample.wav'
    song_path = 'songs\\' + '99' + '.wav'
    _, song_signal = get_signal(song_path)
    #
    song_spectrogram = get_signal_spectrogram_overlap(song_signal, chunk, chunk // 2)

    start = 66
    finish = 67

    plot_spectrogram3D(song_spectrogram, chunk, start, finish)


def plot_spectrogram2D(spectrogram, interval, start, finish, is_log=False):
    sample_rate = 44100
    song_spectrogram = spectrogram[start * sample_rate // interval:finish * sample_rate // interval]
    if is_log:
        plt.ylim(20, 22050)
        plt.yscale("log")
    plt.imshow(song_spectrogram.T, aspect='auto', origin='lower', cmap='magma',
               extent=[start, finish, 0, sample_rate // 2])


def get_signal_spectrogram_overlap(x, interval: int, overlap: int):
    epsilon = 1e-10

    if x.ndim > 1:
        x = x.mean(axis=1)

    extra_space = interval - x.shape[0] % interval

    x = np.concatenate((x, np.zeros(extra_space)))
    x_size = x.shape[0]

    hop_size = interval - overlap
    num_slices = (x_size - interval) // hop_size + 1

    slice_db = np.empty((num_slices, interval // 2))

    for i in range(num_slices):
        start = i * hop_size
        end = start + interval

        to_fft = x[start:end]
        transformed_frame = np.abs(np.fft.fft(to_fft, interval))[0:interval // 2] + epsilon
        transformed_frame_db = 20 * np.log10(transformed_frame)
        slice_db[i] = transformed_frame_db

    return slice_db


if __name__ == "__main__":

    print(torch.cuda.is_available())
    # x = get_signal(file_path='songs/110.wav')[1]
    # spectrogram = get_signal_spectrogram(x=x, interval=2205)
    #
    # plot_spectrogram3D(spectrogram=spectrogram, interval=2205, start=0, finish=1)
    # plt.show()
    # start_time = time.time()
    # add_song_duration_to_songs_info(44100)
    # shazam()
    # get_hash_tables_for_songs('songs_hash_table.json', 2205)
    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print("\nProcessing for: {:.2f} seconds".format(elapsed_time))

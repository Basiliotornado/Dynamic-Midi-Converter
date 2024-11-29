from scipy.signal import get_window
from scipy.io.wavfile import read
import scipy.fft
from numpy import rot90, flipud, frombuffer, absolute, array, exp, clip
import mido
from multiprocessing import Pool
from subprocess import run
from tqdm import tqdm

from arguments import * # best practice? probably not

file_list = file.split(".")
file_name = "".join(file_list[:-1])

if file_list[-1].lower() != "wav":
    test = run(["ffmpeg",
                "-i", file,
                "-ar", "48000",
                "-c:v", "none",
                "-c:a", "pcm_s16le",
                "-f", "s16le",
                "-"], capture_output=True)
    data = frombuffer(test.stdout, dtype="int16")
    data = data.reshape((len(data)//2,2))
    samplerate = 48000

else:
    samplerate, data = read(f"{file_name}.wav")

if len(data) == 0:
    quit("Invalid input!")

dataR = data[:,1]
dataL = data[:,0]

def frequency_from_key(key):
    return 2 ** ((key - 69) / 12) * 440

def minimum_r_line(note):
    return (m * (128 - note)) + b

def spectrogram(data, samplerate, window, nperseg, overlap):

    step = round(nperseg - nperseg * overlap)
    spec = []
    for i in range(0, len(data), step):
        sample = data[i:i+nperseg]
        if len(sample) != nperseg: continue

        sample = sample * window
        fft = absolute(scipy.fft.fft(sample))
        fft = fft[:round(len(fft) / 2)]
        spec.append((fft / nperseg).astype('float32'))

    dt = step / samplerate
    return dt, array(spec)


def odd_symmetric(length):
    b = 2.5
    out = []
    for t in range(length,-length,-2):
        t /= length
        out.append(t * exp(-b**2 * t**2))

    return array(out) / max(out)

def generate_spectrograms(note):
    mult = multiplier
    track = []

    binSize = round(samplerate / frequency_from_key(note) * mult)
    minimum_bin_size = minimum_r_line(note)
    while binSize < minimum_bin_size:
        mult += multiplier
        binSize = round(samplerate / frequency_from_key(note) * mult)

    dt, spectrogramL = spectrogram(dataL, samplerate, window=get_window("hann", binSize), nperseg=binSize, overlap=overlap)
    __, spectrogramR = spectrogram(dataR, samplerate, window=get_window("hann", binSize), nperseg=binSize, overlap=overlap)

    if do_odd:
        w = odd_symmetric(binSize) * odd_amp

        dt, spectrogramL1 = spectrogram(dataL, samplerate, window=w, nperseg=binSize, overlap=overlap)
        __, spectrogramR1 = spectrogram(dataR, samplerate, window=w, nperseg=binSize, overlap=overlap)

        spectrogramL -= spectrogramL1
        spectrogramR -= spectrogramR1

    L = spectrogramL[:,mult]
    R = spectrogramR[:,mult]

    L = clip(L, 0, None)
    R = clip(R, 0, None)

    largest = max([max(L),max(R)])

    return {"dt":dt, "max":largest, "L":L, "R":R}

def get_velocity(amp):
    vel = int((amp ** 0.5) * 128)

    if vel < 0:   vel = 0
    if vel > 127: vel = 127

    return vel

def get_time(time):
    return round(time * ppqn * (bpm / 60))

if visualize_mininum_bin_size:
    from matplotlib import pyplot as plt

    debug_graph_target = []
    debug_graph_actual = []

    mult = multiplier
    for i in range(note_count):
        binSize = round(samplerate / frequency_from_key(i) * mult)
        minimum_bin_size = minimum_r_line(i)
        while binSize < minimum_bin_size:
            mult += multiplier
            binSize = round(samplerate / frequency_from_key(i) * mult)

        debug_graph_target.append(minimum_bin_size)
        debug_graph_actual.append(binSize)

    plt.plot(debug_graph_actual)
    plt.plot(debug_graph_target)
    plt.show()


if __name__ == "__main__":
    midi = mido.MidiFile(type = 1)

    midi.ticks_per_beat = ppqn

    for i in range(tracks + 1):
        midi.tracks.append(mido.MidiTrack())

    midi.tracks[0].append(mido.MetaMessage('set_tempo', tempo = mido.bpm2tempo(bpm)))

    midi.tracks[0].append(mido.Message('control_change', control = 121))

    midi.tracks[0].append(mido.Message('control_change', channel = 0, control = 10, value = 0))
    midi.tracks[0].append(mido.Message('control_change', channel = 1, control = 10, value = 127))

    with Pool(round(threads)) as p:
        spectrograms = list(tqdm(p.imap(generate_spectrograms, range(note_count)), desc='Generating spectrograms', total=note_count, smoothing=0))

    largest = 0
    delta_times = [0] * note_count
    for i in range(len(spectrograms)):
        spectrogram = spectrograms[i]
        delta_times[i] = spectrogram["dt"]


        if spectrogram["max"] > largest:
            largest = spectrogram["max"]

    for spectrogram in tqdm(range(len(spectrograms)),desc='Normalizing spectrogram'):
        for i in range(len(spectrograms[spectrogram]['L'])):
            spectrograms[spectrogram]['L'][i] /= largest
            spectrograms[spectrogram]['R'][i] /= largest

    next_times = [(i / (1 - overlap)) / 2 for i in delta_times]
    track_offset = [0] * tracks
    done = [0] * note_count
    note_index = [0] * note_count

    note_data = [[] for _ in range(tracks + 1)]

    for note in range(note_count):
        time = delta_times[note]
        total_time = next_times[note]

        spectrogram = spectrograms[note]
        for i in range(len(spectrogram['L'])):
            vel_l = get_velocity(spectrogram['L'][i])
            vel_r = get_velocity(spectrogram['R'][i])

            track = int(vel_l/(82/tracks)) + 1
            if track > tracks: track = tracks

            if vel_l > minimum_velocity:
                note_data[track].append({'velocity':vel_l, 'note': note, 'time': total_time})
                note_data[track].append({'velocity':0,     'note': note, 'time': total_time + time})
            if vel_r > minimum_velocity:
                note_data[0].append({'velocity': vel_r, 'note': note, 'time': total_time})
                note_data[0].append({'velocity': 0,     'note': note, 'time': total_time + time})

            total_time += time

    note_data = [sorted(i, key=lambda x: x['time']) for i in note_data] # thanks pon (idea stolen)

    for track in tqdm(range(len(note_data)), desc="Placing notes"):
        channel = 1 if track == 0 else 0

        total_time = 0
        t = midi.tracks[track]
        for i in note_data[track]:
            time = get_time(i['time']) - total_time
            if i['velocity'] <= 0:
                t.append(mido.Message('note_off', channel=channel,                         note=i['note'], time=time))
            else:
                t.append(mido.Message('note_on',  channel=channel, velocity=i['velocity'], note=i['note'], time=time))

            total_time += time
        midi.tracks[track] = t


    print("\nExporting")
    midi.save(file_name + ".mid")
    print("\nDone!")

from scipy.signal import spectrogram, get_window
from scipy.io.wavfile import read
import scipy.fft
from numpy import rot90, flipud, frombuffer, absolute, array
import mido
import argparse
from multiprocessing import Pool, cpu_count
from subprocess import run

parser = argparse.ArgumentParser(description='Generate a MIDI file from WAV')

parser.add_argument("i",
                    type=str,
                    help="File name ")
parser.add_argument("-r",
                    type=int,
                    help="Minimum bin size",
                    default=2048)
parser.add_argument("-o",
                    type=float,
                    help="Overlap between fft bins. Increases notes per second with higher numbers. range 0.0 to 0.99.",
                    default=0.00)
parser.add_argument("-t",
                    type=int,
                    help="Number of midi tracks.",
                    default=31)
parser.add_argument("-m",
                    type=int,
                    help="How much to add to the multiplier when the minimum bin size is reached",
                    default=8)
parser.add_argument("-n",
                    type=int,
                    help="Note count.",
                    default=128)
parser.add_argument("--threads",
                    type=int,
                    help="How many threads the script uses to process. More threads use more ram.",
                    default=(cpu_count() / 1.5))
parser.add_argument("--ppqn",
                    type=int,
                    help="PPQN of the midi file.",
                    default=30000)
parser.add_argument("--bpm",
                    type=int,
                    help="Bpm of the midi",
                    default=120)

args = parser.parse_args()

if args.i: file = args.i
else: file = input("File name: ")

minimum_bin_size = args.r
overlap = args.o
tracks = args.t
multiplier = args.m
note_count = args.n
threads = args.threads
ppqn = args.ppqn
bpm = args.bpm


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

def spectrogram(data, samplerate, window, nperseg, overlap):

    step = round(nperseg - nperseg * overlap)
    spec = []
    for i in range(0, len(data), step):
        sample = data[i:i+nperseg]
        if len(sample) != nperseg: continue

        sample = sample * window
        fft = absolute(scipy.fft.fft(sample))
        fft = fft[:round(len(fft) / 2)]
        spec.append(fft / nperseg)

    dt = step / samplerate
    return  dt, spec

def generate_spectrograms(note):
    mult = multiplier
    track = []

    binSize = round(samplerate / frequency_from_key(note) * mult)
    while binSize < minimum_bin_size:
        mult += multiplier
        binSize = round(samplerate / frequency_from_key(note) * mult)

    dt, spectrogramL = spectrogram(dataL, samplerate, window=get_window("hann", binSize), nperseg=binSize, overlap=overlap)
    __, spectrogramR = spectrogram(dataR, samplerate, window=get_window("hann", binSize), nperseg=binSize, overlap=overlap)

    L = [spectrogramL[i][mult] for i in range(len(spectrogramL))]
    R = [spectrogramR[i][mult] for i in range(len(spectrogramR))]

    largest = max([max(L),max(R)])

    print(f"Spectrogram {note} done")
    return {"dt":dt, "max":largest, "L":L, "R":R}

def get_velocity(amp):
    vel = int((amp ** 0.5) * 128)

    if vel < 0:   vel = 0
    if vel > 127: vel = 127

    return vel

if __name__ == "__main__":
    midi = mido.MidiFile(type = 1)

    midi.ticks_per_beat = ppqn

    for i in range(tracks):
        midi.tracks.append(mido.MidiTrack())

    midi.tracks[0].append(mido.MetaMessage('set_tempo', tempo = mido.bpm2tempo(bpm)))

    midi.tracks[0].append(mido.Message('control_change', control = 121))

    midi.tracks[0].append(mido.Message('control_change', channel = 0, control = 10, value = 0))
    midi.tracks[0].append(mido.Message('control_change', channel = 1, control = 10, value = 127))

    delta_times = [[0] for i in range(note_count)]


    with Pool(round(threads)) as p:
        spectrograms = p.map(generate_spectrograms, range(note_count))

    largest = 0
    prev_track = [0 for i in range(note_count)]
    for i in range(len(spectrograms)):
        delta_times[i] = spectrograms[i]["dt"]

        if spectrograms[i]["max"] > largest:
            largest = spectrograms[i]["max"]

    for spectrogram in range(len(spectrograms)):
        for i in range(len(spectrograms[spectrogram]['L'])):
            spectrograms[spectrogram]['L'][i] /= largest
            spectrograms[spectrogram]['R'][i] /= largest

    next_times = [0 for i in range(len(delta_times))]
    track_offset = [0 for i in range(tracks)]
    done = [0 for i in range(note_count)]

    prev_note = 0
    while sum(done) < note_count:

        time = min(next_times)
        note = next_times.index(time)

        for i in range(len(next_times)):
            next_times[i] -= time

        next_times[note] = delta_times[note]

        if len(spectrograms[note]["L"]) == 0:
            next_times[note] += 9999
            done[note] = 1
            print(f"{note} Done")
            spectrograms[note]["L"].append(0)
            spectrograms[note]["R"].append(0) # make sure both channels finish

        time = round(time * ppqn * (bpm / 60))

        velL = get_velocity(spectrograms[note]["L"].pop(0))
        velR = get_velocity(spectrograms[note]["R"].pop(0))

        track = int(velL/(82/tracks) + 1)
        if track > tracks-1: track = tracks-1

        midi.tracks[0].append(mido.Message('note_off', channel = 1, note=note, time=time))
        midi.tracks[track].append(mido.Message('note_off', channel = 0, note=note, time=time + track_offset[track]))

        midi.tracks[0].append(mido.Message('note_on', channel = 1, note=note, velocity=velR))
        midi.tracks[track].append(mido.Message('note_on', channel = 0, note=note, velocity=velL))




        for i in range(tracks):
            track_offset[i] += time
        track_offset[track] = 0

        prev_track[note] = track
        prev_note = note

    print("\nExporting")
    midi.save(file_name + ".mid")
    print("\nDone!")

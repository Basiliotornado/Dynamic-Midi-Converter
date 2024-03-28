from scipy.signal import spectrogram, get_window
from scipy.io.wavfile import read
from numpy import rot90, flipud, frombuffer
import mido
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from subprocess import run

parser = argparse.ArgumentParser(description='Generate a MIDI file from WAV')

parser.add_argument("i",
                    type=str,
                    help="File name ")
parser.add_argument("-r",
                    type=int,
                    help="Minimum bin size",
                    default=4096)
parser.add_argument("-o",
                    type=float,
                    help="Overlap between fft bins. Increases notes per second with higher numbers. range 0.0 to 0.99.",
                    default=0.70)
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

args = parser.parse_args()

if args.i: file = args.i
else: file = input("File name: ")

minimum_bin_size = args.r
overlap = args.o
tracks = args.t
multiplier = args.m
note_count = args.n

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

def generate_spectrograms(note):
    mult = multiplier if multiplier > 1 else 1#multiplier if multiplier > 1 else 1
    track = []

    binSize = round(samplerate / frequency_from_key(note) * mult)
    while binSize < minimum_bin_size:
        mult += 8
        binSize = round(samplerate / frequency_from_key(note) * mult)

    f, t, spectrogramL = spectrogram(dataL, samplerate, window=get_window("hann", binSize), nperseg=binSize, noverlap=round(binSize*overlap), mode='magnitude')
    _, _, spectrogramR = spectrogram(dataR, samplerate, window=get_window("hann", binSize), nperseg=binSize, noverlap=round(binSize*overlap), mode='magnitude')

    spectrogramL = flipud(rot90(spectrogramL))
    spectrogramR = flipud(rot90(spectrogramR))

    L = [spectrogramL[i][mult] for i in range(len(spectrogramL))]
    R = [spectrogramR[i][mult] for i in range(len(spectrogramR))]

    dt = t[1] - t[0]

    largest = max([max(L),max(R)])

    print(f"Spectrogram {note} done")
    return {"f":f, "dt":dt, "max":largest, "L":L, "R":R}

def get_velocity(amp,mult,largest):
    vel = int((amp ** 0.5 / largest)*12800 * mult)

    if vel < 0:   vel = 0
    if vel > 127: vel = 127

    return vel

if __name__ == "__main__":
    
    midi = mido.MidiFile(type = 1)

    midi.ticks_per_beat = 30000

    for i in range(tracks):
        midi.tracks.append(mido.MidiTrack())
    
    midi.tracks[0].append(mido.Message('control_change', control = 121))

    midi.tracks[0].append(mido.Message('control_change', channel = 0, control = 10, value = 0))
    midi.tracks[0].append(mido.Message('control_change', channel = 1, control = 10, value = 127))

    delta_times = [[0] for i in range(note_count)]


    with Pool(round(cpu_count() / 1.5)) as p:
        spectrograms = p.map(generate_spectrograms, range(note_count))

    largest = 0
    prev_track = []
    for i in range(len(spectrograms)):
        delta_times[i] = spectrograms[i]["dt"]

        if spectrograms[i]["max"] > largest:
            largest = spectrograms[i]["max"]

        note_mapped = i / 128
        note_mult = (note_mapped ** 1.8 * (3 - 2 * note_mapped)) / 1.25 + 0.2

        velL = get_velocity(spectrograms[i]["L"].pop(0), note_mult, 3564.1987)
        velR = get_velocity(spectrograms[i]["R"].pop(0), note_mult, 3564.1987) #

        track = int(velR/(96/tracks) + 1)
        if track > tracks-1: track = tracks-1

        prev_track.append(track)
        midi.tracks[track].append(mido.Message('note_on', channel = 0, note=i, velocity=velL))
        midi.tracks[0].append(mido.Message('note_on', channel = 1, note=i, velocity=velR))

    next_times = [delta_times[i] for i in range(len(delta_times))]
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
            print(note)
            spectrograms[note]["L"].append(0)
            spectrograms[note]["R"].append(0) # make sure both channels finish

        time = int(time*60000)
        note_mult = (-(note / 128 * 0.75 - 1) ** 2 + 1.06)
        if note_mult < 0.4: note_mult = 0.4

        note_mapped = note / 128
        note_mult = (note_mapped ** 1.8 * (3 - 2 * note_mapped)) / 1.25 + 0.2

        velL = get_velocity(spectrograms[note]["L"].pop(0), note_mult, largest)
        velR = get_velocity(spectrograms[note]["R"].pop(0), note_mult, largest)

        track = int(velL/(64/tracks) + 1)
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

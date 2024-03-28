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
                    help="Increases notes per second with higher numbers. range 0.01 to 0.99",
                    default=0.70)
parser.add_argument("-m",
                    type=int,
                    help="How much to add to the multiplier when the minimum bin size is reached",
                    default=8)
parser.add_argument("--threads",
                    type=int,
                    help="How many threads the script uses to process. More threads use more ram.",
                    default=(cpu_count() / 1.5))

args = parser.parse_args()

if args.i: file = args.i
else: file = input("File name: ")

minimum_bin_size = args.r
overlap = args.o
multiplier = args.m
threads = args.threads

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

def key_from_frequency(freq):
    if freq <= 0: return 0
    return keyFreq[int(freq)]

def frequency_from_key(key):
    return 2 ** ((key - 69) / 12) * 440

def process(note):
    mult = multiplier if multiplier > 1 else 1
    track = []

    binSize = round(samplerate / frequency_from_key(note) * mult)
    while binSize < minimum_bin_size:
        mult += multiplier
        binSize = round(samplerate / frequency_from_key(note) * mult)

    f, t, spectrogramL = spectrogram(dataL, samplerate, window=get_window("hann", binSize), nperseg=binSize, noverlap=round(binSize*overlap), mode='magnitude')
    _, _, spectrogramR = spectrogram(dataR, samplerate, window=get_window("hann", binSize), nperseg=binSize, noverlap=round(binSize*overlap), mode='magnitude')

    #print(note, mult, binSize, f[mult], frequency_from_key(note), f"off by {(f[mult]/frequency_from_key(note) - 1) * 100}%")

    spectrogramL = flipud(rot90(spectrogramL))
    spectrogramR = flipud(rot90(spectrogramR))
    large = 0 # this is wrong but i would have to restructure (not a hard one) to make it work so we're stuck here
    for column in spectrogramL:
        m = max(column)
        if m > large:
            large = m
    for column in spectrogramR:
        m = max(column)
        if m > large:
            large = m

    timer = 0

    note_mult = (-(note / 128 * 0.75 - 1) ** 2 + 1.06)
    note_mult = ((note / 128 / 1.3) ** 2) + 0.4

    note_mapped = note / 128
    note_mult = (note_mapped ** 1.8 * (3 - 2 * note_mapped)) / 1.25 + 0.2

    #note_mult += ((sin(note_mapped * 7 -1.7) + 1) / 10)
    #if note_mult < 0.4: note_mult = 0.4

    for i in range(len(spectrogramL)):
        wait = int(t[i]*60000 - timer)
        timer += wait

        velL = int((spectrogramL[i][mult] ** 0.5 / 3564.1987)*12800 * note_mult)#(note/128) ** 0.75)

        if velL < 0:   velL = 0
        if velL > 127: velL = 127

        velR = int((spectrogramR[i][mult] ** 0.5 / 3564.1987)*12800 * note_mult)#(note/128) ** 0.75)

        if velR < 0:   velR = 0
        if velR > 127: velR = 127

        track.append(mido.Message('note_on', channel = 0, note=note, velocity=velL))
        track.append(mido.Message('note_on', channel = 1, note=note, velocity=velR))

        track.append(mido.Message('note_off', channel = 0, note=note, time=wait))
        track.append(mido.Message('note_off', channel = 1, note=note))

    print(f"Note: {note} done")
    return track

if __name__ == "__main__":
    
    midi = mido.MidiFile(type = 1)



    midi.ticks_per_beat = 30000

    for i in range(127):
        midi.tracks.append(mido.MidiTrack())

    with Pool(round(threads)) as p:
        tracks = p.map(process, range(127))


    tracks[0].insert(0, mido.Message('control_change', control = 121))

    tracks[0].insert(0, mido.Message('control_change', channel = 0, control = 10, value = 0))
    tracks[0].insert(0, mido.Message('control_change', channel = 1, control = 10, value = 127))

    midi.tracks = tracks
    print("\nExporting")
    midi.save(file_name + ".mid")
    print("\nDone!")

import argparse
from multiprocessing import cpu_count
parser = argparse.ArgumentParser(description='Generate a MIDI file from WAV')

parser.add_argument("i",
                    type=str,
                    help="File name ")
parser.add_argument("-b",
                    type=int,
                    help="b in y = mx+b. Works like minimum bin size.",
                    default=128)
parser.add_argument("-m",
                    type=int,
                    help="m in y = mx+b.",
                    default=32)
parser.add_argument("-o",
                    type=float,
                    help="Overlap between fft bins. Increases notes per second with higher numbers. range 0.0 to 0.99.",
                    default=0.50)
parser.add_argument("-t",
                    type=int,
                    help="Number of midi tracks.",
                    default=31)
parser.add_argument("--mult",
                    type=int,
                    help="How much to add to the multiplier when the minimum bin size is reached",
                    default=24)
parser.add_argument("--do-odd",
                    type=bool,
                    help="If an extra spectrogram using an odd function as a window should be calculated and subtracted from the main spectrogram. (slow)",
                    default=True)
parser.add_argument("--odd-amp",
                    type=float,
                    help="Amplitude of the odd function windowed spectrogram.",
                    default=0.166)
parser.add_argument("-n",
                    type=int,
                    help="Note count.",
                    default=128)
parser.add_argument("--threads",
                    type=int,
                    help="How many threads the script uses to process. More threads use more ram.",
                    default=(cpu_count() / 2))
parser.add_argument("--ppqn",
                    type=int,
                    help="PPQN of the midi file.",
                    default=30000)
parser.add_argument("--bpm",
                    type=int,
                    help="Bpm of the midi",
                    default=120)
parser.add_argument("--visualize",
                    type=bool,
                    help="Visualize the minimum bin size.",
                    default=False)

args = parser.parse_args()

if args.i: file = args.i
else: file = input("File name: ")

b = args.b
m = args.m
overlap = args.o
tracks = args.t
multiplier = args.mult
note_count = args.n
threads = args.threads
ppqn = args.ppqn
bpm = args.bpm
do_odd = args.do_odd
odd_amp= args.odd_amp
visualize_mininum_bin_size = args.visualize # lmao

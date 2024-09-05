# Dynamic-Midi-Converter
```
usage: convert.py i [options]

Generate a MIDI file from WAV

positional arguments:
  i                     File name

options:
  -h, --help            show this help message and exit
  -b B                  b in y = mx+b. Works like minimum bin size.
  -m M                  m in y = mx+b.
  -o O                  Overlap between fft bins. Increases notes per second with higher numbers. range 0.0 to 0.99.
  -t T                  Number of midi tracks.
  --mult MULT           How much to add to the multiplier when the minimum bin size is reached
  -n N                  Note count.
  --threads THREADS     How many threads the script uses to process. More threads use more ram.
  --ppqn PPQN           PPQN of the midi file.
  --bpm BPM             Bpm of the midi
  --visualize VISUALIZE
                        Visualize the minimum bin size.
```

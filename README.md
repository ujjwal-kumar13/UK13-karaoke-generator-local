# Karaoke Video Generator (Local File)

Turns a local video/audio file of a song into a scrolling-lyrics karaoke
video: it separates vocals from instrumental, times the lyrics against the
vocal track (word-by-word), and renders a 1080p video with color-changing
scrolling text synced to the music, plus a matching thumbnail.

This is the **local-file** version — you supply a video/audio file already
on your machine. See the companion
[YouTube-source version](../karaoke-generator-youtube) if you'd rather point
it at a YouTube link instead.

## How it works

1. **Extract audio** from the input video (`ffmpeg`).
2. **Separate vocals from instrumental** using [Demucs](https://github.com/facebookresearch/demucs).
3. **Time the lyrics** to the vocals:
   - If you provide a lyrics `.txt` file, it uses **forced alignment**
     ([ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner))
     to match your exact lyrics text to the audio, word by word.
   - If you don't, it falls back to [OpenAI Whisper](https://github.com/openai/whisper)
     to transcribe the vocals automatically.
4. **Renders the video**: a scrolling lyrics panel over a background image,
   with each word changing color as it's sung, a progress bar, a countdown
   badge before verses, and a bottom ticker. Optionally colors lines by
   singer (male/female) for duets.
5. **Generates a thumbnail** image alongside the video.

## Requirements

- **Python 3.9+**
- **ffmpeg** on your PATH
  - macOS: `brew install ffmpeg`
  - Windows: download from ffmpeg.org and add it to your PATH
  - Linux: `sudo apt install ffmpeg`
- A background image for the video canvas (see **Background image**, below)
- ~4-5 GB of disk for one-time model downloads on first run (Demucs,
  Whisper, and the forced-aligner model, all pulled automatically from
  their usual hosts the first time each is used)
- Optional but recommended for Hindi/Devanagari lyrics: a Noto Sans
  Devanagari font installed (e.g. `/Library/Fonts/NotoSansDevanagari-Bold.ttf`
  on macOS) — falls back to Arial/Helvetica otherwise, which won't render
  Devanagari script correctly

## Installation

```bash
git clone https://github.com/<your-username>/karaoke-generator-local.git
cd karaoke-generator-local

python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`torch`/`torchaudio` pull a large download; on Apple Silicon Macs the
pipeline automatically uses the `mps` GPU backend if available, falling
back to CPU otherwise.

## Background image

The renderer draws the lyrics panel, progress bar, and legend over a
background image named in `Config.BACKGROUND_IMAGES` (`UK-Karaoke_Background.png`
by default) and expects it in the **current working directory** when you
run the script. A sample background is included under `assets/` — copy it
next to the script (or update `Config.BACKGROUND_IMAGES` to point wherever
you keep it):

```bash
cp assets/UK-Karaoke_Background.png .
```

If you use your own background art, note that the panel/progress-bar/legend
positions (`Config.PANEL_CENTER_Y` and friends) were tuned by eye against
the included background — a very differently laid-out image may need those
values adjusted so the panel doesn't overlap your artwork.

## Usage

```bash
python karaoke_generator_local.py
```

You'll be walked through a series of prompts:

1. **Path to the input video file** (`.mp4`, or any format ffmpeg reads).
2. **Lyrics file?** — if you have a plain-text lyrics file (one line per
   lyric line, wrap ad-lib/chorus-hum filler lines in parentheses, e.g.
   `(la la la)`), point it there for accurate forced-alignment timing.
   Otherwise Whisper transcribes the vocals automatically.
3. **Song info** — title, film name, artist (used in the bottom ticker and
   thumbnail; all optional).
4. **Is this a duet?** — if yes, lines are colored by singer (pitch-based
   male/female classification), and you're offered the option to load a
   hand-corrected voice-assignment file from a previous run (see below).
5. The pipeline then extracts audio, separates vocals, times the lyrics,
   and renders the video and thumbnail into `karaoke_output/` in the
   current directory (named `<Song Title> Karaoke.mp4` / `<Song Title>
   Thumbnail.png`).

### Correcting duet voice assignment

Pitch-based male/female classification is a heuristic (lower pitch =
"male," higher = "female") — it can't truly tell two singers apart, only
"higher" from "lower," so it can occasionally misclassify a line. After a
run, `karaoke_output/line_timing_preview.txt` shows every line's assigned
voice. Hand-correct any wrong `[MALE]`/`[FEMALE]` tags in a copy of that
file, then answer "yes" at the voice-override prompt on your next run and
point it at your corrected copy — it's applied by line order, so the file
must have the same number of lines as the current run's lyrics.

## Known limitations

- Voice classification is pitch-based only (not real speaker
  identification) — it works well for a genuine two-voice duet with
  clearly separated vocal ranges, but can't reliably resolve songs with
  more than two singers, or songs with heavy pitch/tempo variation.
- Ad-lib/filler detection (`(...)`) only recognizes a line that opens and
  closes its own parentheses — a filler passage spanning multiple lyric
  lines needs each line individually wrapped in parentheses.
- Long instrumental intros/outros with no matching lyrics text can still
  occasionally cause minor local timing drift in forced alignment; the
  console prints warnings (`flag_slow_lines`, `cap_anomalous_word_durations`,
  low-confidence line warnings) when it detects this, so check the console
  output after a run.

## Notes

- Only use this on media you have the rights to process. Building a
  karaoke video from copyrighted music is a personal/practice use case;
  sharing or distributing the resulting video further may raise separate
  copyright considerations depending on your jurisdiction and use.
- If a step fails, the console output is verbose on purpose — warnings and
  errors explain what happened and usually suggest a fix (e.g. adding a
  placeholder lyrics line for an un-transcribed hum passage).

## License

MIT — see [LICENSE](LICENSE).

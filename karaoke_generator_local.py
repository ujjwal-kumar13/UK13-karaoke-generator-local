#!/usr/bin/env python3
"""
Karaoke Video Generator v7
============================
Builds on v6. Real duet testing on "Teri Baaton Mein Aisa Uljha Jiya"
(4 actual singers: 2 male, 2 female) showed the pitch-based voice
classifier getting several lines wrong -- but analyzing the diff between
the auto-generated line_timing_preview.txt and the user's hand-corrected
copy showed something important: the errors were NOT random noise, they
consistently hit specific melodic phrases every time those phrases
repeated (e.g. "Rakh loon saja ke" wrong both times it's sung; "Aaja gale
lag jaa" correct both times) while other phrases were reliably correct on
every occurrence, including one that correctly resolved MALE three times
and FEMALE on its one genuinely-female occurrence. That's the fingerprint
of the known, structural limitation of pitch-only classification: it can
only ever split "higher" from "lower," never tell apart two different
people, so a singer's own higher notes on a particular phrase will always
risk crossing into the other cluster -- no amount of threshold-tuning
fixes that, since the signal itself doesn't carry identity information.
With 4 actual singers instead of 2, this is fundamentally out of reach for
a 2-cluster pitch split regardless of tuning.

Rather than keep chasing that with more pitch heuristics, this version
adds a way to correct the classifier's output by hand -- fast and fully
reliable, since a human glancing at a 25-line file can do in a minute what
pitch-only clustering structurally cannot do at all:

  1. New read_voice_overrides() / apply_voice_overrides(). After a run,
     line_timing_preview.txt (already generated automatically) can be
     hand-edited to fix any wrong [MALE]/[FEMALE] tags, then fed back in
     on a subsequent run via a new prompt ("Do you have a hand-corrected
     voice-assignment file..."). Overrides are matched strictly by line
     ORDER, not fuzzy text matching -- a line-count mismatch hard-fails
     (refuses to guess), and a text mismatch on an individual line prints
     a loud warning but doesn't block the run, since minor transliteration
     differences between runs are expected. Applied right after voice
     classification, before the preview file is regenerated, so a
     corrected run's own preview.txt already reflects the fix.
  2. Verified directly against the user's real 25-line correction: loaded
     the original run's output as synthetic lines, applied the user's
     corrected file as an override, and confirmed exactly the 6 genuinely-
     wrong lines changed (matching the user's manual diff) and the other
     19 were left untouched, with zero false text-mismatch warnings.
     Also verified both failure paths (line-count mismatch, malformed
     override line) raise a clear error instead of silently misapplying.

Everything from v6 (flag_slow_lines, ad-lib coloring) and v5 (Both
removed, panel/legend repositioning, single-word duration capping) is
unchanged -- see v6's docstring below.

---- v6 docstring (unchanged, kept for history) ----

Karaoke Video Generator v6
============================
Builds on v5. v5's chorus-hum fix (capping one word's own duration) turned
out not to cover a real-world case: a wordless ~17s chorus/mutter passage
got almost entirely absorbed into a short *neighboring* line instead (a
4-word line stretched to 21+ seconds), while a single-word placeholder like
"(Chorus)" meant to anchor that passage got squeezed to under a second.
Root cause (confirmed by reading the forced-aligner's actual source, not
guessing): it does true Viterbi forced alignment across the ENTIRE song in
one pass -- every character token must be placed somewhere, matched to
wherever the acoustic model finds the strongest signal for that letter's
*sound*. A hum/mutter passage is almost all vowel/nasal sound with very
little hard-consonant content, so a consonant-heavy placeholder word (like
"chorus" itself -- "ch", "r", "s") finds almost no acoustic support in it
and gets starved out in favor of neighboring real words, even though those
words don't match the hum either -- they're just relatively less bad.

Changes in this version:

  1. New flag_slow_lines(), wired in right after cap_anomalous_word_durations().
     Detection-only (prints a console warning, never resizes/moves anything,
     safe for unattended runs): flags any line whose words-per-second pace is
     far below the song's typical pace -- the whole-LINE-level symptom of the
     same root cause cap_anomalous_word_durations() catches at the single-word
     level. On the real case that prompted this, it correctly and only
     flagged the two genuinely broken lines out of 53.
  2. New ad-lib/chorus-filler line support. Any lyrics line wrapped in
     parentheses (the existing convention, e.g. "(excuse me)") now renders in
     a new, distinct muted color (Config.ADLIB_COLOR) instead of normal solo
     or voice-based colors, so it reads at a glance as filler rather than a
     real lyric to sing precisely. Takes priority over ENABLE_VOICE_COLORING.
     Combined with (1)'s diagnosis, the recommended fix for a real chorus/hum
     passage is no longer a single descriptive placeholder word -- it's
     several short, repeated, vowel-heavy lines that actually resemble the
     hum sound (e.g. multiple "(Aa Aa Aa Aa)" lines instead of one "(Chorus)"),
     which both displays correctly AND gives the aligner far more evenly
     distributed acoustic matches to lock onto.

Everything from v5 (Both removed, panel/legend repositioning, single-word
duration capping) is unchanged -- see that version's docstring below.

---- v5 docstring (unchanged, kept for history) ----

Builds on v4. Changes in this version, from real-world duet testing on a
second song:

  1. The "Both" duet category has been removed. v4's Both-detection (pitch
     near the Male/Female midpoint, or lower pyin confidence than the song's
     average) mislabeled genuine Male and Female lines as Both on a real
     duet test, and there's no reliable audio signal in this pipeline for
     "two voices at once" vs. "one voice, ambiguous/low-confidence" -- so
     every line is now assigned strictly to the nearer of the two pitch
     clusters, Male or Female, full stop. Lines close to the midpoint are
     still printed as a borderline spot-check note, but are not rendered
     differently. See classify_voices() docstring for the full rationale.
  2. The lyrics panel, progress bar, countdown badge, and voice legend have
     all been moved up via Config.PANEL_CENTER_Y (600 -> 560), to use spare
     space near the background's title and stop the legend from overlapping
     the microphone graphic near the bottom of the background. Unlike v4,
     this wasn't just a flat "2 line-slots up" arithmetic shift -- it was
     checked directly against the actual UKKaraoke_Background.png (a test
     render was generated and visually inspected) so the panel clears the
     title text and the legend clears the mic with real margin on both. See
     the PANEL_CENTER_Y config comment for a known, pre-existing trade-off
     this didn't attempt to fix (the countdown badge briefly touching the
     title during a long lead-in/instrumental gap).
  3. Fixed a real bug where a wordless chorus hum/ad-lib (with no matching
     text in the lyrics file) got absorbed by the forced aligner into the
     boundary of the nearest real word, stretching that word's measured
     duration -- and therefore its color-fill sweep -- far beyond how long
     it's actually sung. New cap_anomalous_word_durations() caps any word
     whose duration exceeds 4x the song's median word duration and
     recomputes its line's end time, so the display just holds on the
     finished line (like an ordinary instrumental gap) instead of
     stretching the sweep. Flagged words are printed to the console; the
     more robust long-term fix is still adding a placeholder lyrics line
     (e.g. "(humming)") for any such passage, the same way ad-lib lines
     like "(excuse me)" are handled.

Run with:  python karaoke_generator_v7.py
See README_SETUP.md for installation (same steps as before).
"""

import os
import re
import sys
import json
import wave
import subprocess
import contextlib
from pathlib import Path

import numpy as np

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    OUT_W, OUT_H = 1920, 1080
    FPS = 24
    OUTPUT_DIR = "karaoke_output"

    BACKGROUND_IMAGES = ["UK-Karaoke_Background.png"]

    TICKER_TEXT = "LIKE, SUBSCRIBE, MAKE REQUESTS FOR YOUR FAVORITE KARAOKE SONG ON THE CHANNEL     •     "
    TICKER_SPEED_PX_PER_SEC = 120
    SHOW_TICKER = True

    FONT_CANDIDATES_BOLD = [
        "/Library/Fonts/NotoSansDevanagari-Bold.ttf",
        "/System/Library/Fonts/Supplemental/NotoSansDevanagari-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    FONT_CANDIDATES_REGULAR = [
        "/Library/Fonts/NotoSansDevanagari-Regular.ttf",
        "/System/Library/Fonts/Supplemental/NotoSansDevanagari-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    LYRIC_FONT_SIZE = 38
    TICKER_FONT_SIZE = 26

    # --- Lyrics panel ---
    # VISIBLE_LINES lines are shown on screen at once, scrolling up by one
    # slot at a time. PANEL_H should comfortably fit VISIBLE_LINES rows at
    # LINE_SLOT_HEIGHT spacing -- default here assumes a background with a
    # generously tall cleared middle section; tune PANEL_H / PANEL_CENTER_Y
    # to match your actual background image.
    #
    # Moved up from the original 600 default, per feedback that there was
    # spare room below the background's title and the legend/mic image at
    # the bottom were overlapping. NOTE: a literal 200px (2 line-slot) shift
    # was tried first but made the panel cover the "UK-Karaoke" title text
    # itself -- this default (560) was instead checked directly against the
    # actual UKKaraoke_Background.png (rendered + visually inspected) so the
    # panel clears the title and the legend clears the mic graphic with a
    # comfortable margin on both. One remaining, pre-existing trade-off
    # (present in v4 too, not new here): the countdown badge briefly
    # overlaps the tail of the title text during a long lead-in/instrumental
    # gap, since there isn't enough vertical room on this background to fit
    # badge + bar + panel + legend without the badge or the panel touching
    # something -- clearing the panel/legend (the two things you flagged)
    # was prioritized. Shrink COUNTDOWN_BADGE_SIZE or reposition the badge
    # to a corner if this bothers you.
    VISIBLE_LINES = 4
    PANEL_W, PANEL_H = 1450, 460
    PANEL_CENTER_Y = 560
    LINE_SLOT_HEIGHT = 100
    SCROLL_TRANSITION_SECONDS = 0.6

    # A line whose rendered text (at LYRIC_FONT_SIZE, centered in the panel)
    # would be wider than PANEL_W * MAX_LINE_WIDTH_FRACTION gets split into
    # two or more shorter lines by split_long_lines() -- draw_line() below
    # does not wrap text, so anything wider than the panel just gets
    # silently clipped off both edges at render time otherwise (confirmed
    # real-world case: a Whisper segment spanning several sentences with no
    # natural break clipped "shimmering" -> "himmering" and "the" -> "th").
    # Leaves a bit of margin on each side rather than exactly PANEL_W so a
    # split-to-fit line doesn't look like it's touching the panel's edges.
    MAX_LINE_WIDTH_FRACTION = 0.92

    # --- Word coloring: solo (non-duet) songs ---
    # Not-yet-sung words are a dim/muted gray, already-sung words crisp
    # white, and the currently-singing word the bright gold highlight.
    SUNG_COLOR = (255, 255, 255, 255)
    UPCOMING_COLOR = (115, 115, 128, 255)
    SOLO_ACTIVE_COLOR = (255, 205, 90, 255)

    # --- Duet voice coloring (opt-in -- see the y/n prompt in main()) ---
    # For duets, BOTH the not-yet-sung state and the currently-singing word
    # are colored by voice, so the singer can see who's up before a line
    # starts, not just at the moment it's sung: not-yet-sung words show the
    # "UPCOMING" shade below, the current word brightens to the "ACTIVE"
    # shade of the same hue, and once sung a word always turns SUNG_COLOR
    # (white) above, regardless of voice.
    #
    # Male/Female only for now -- a third "Both" (sung together) category
    # was tried and removed: there's no reliable audio signal in this
    # pipeline for "two voices at once" (a monophonic pitch tracker can't
    # really tell that apart from "one voice, ambiguous note" or "one voice,
    # low tracking confidence"), and on a real duet test it mislabeled
    # genuine male/female lines as "Both." Revisit only if a better
    # detection signal turns up.
    VOICE_COLORS = {
        "MALE":   {"UPCOMING": (35, 105, 205, 255),  "ACTIVE": (120, 190, 255, 255)},   # blue
        "FEMALE": {"UPCOMING": (170, 20, 90, 255),   "ACTIVE": (255, 90, 165, 255)},    # dark pink
    }
    VOICE_LABELS = {"MALE": "Male", "FEMALE": "Female"}
    ENABLE_VOICE_COLORING = False  # set True at runtime only if the song is a confirmed duet
    # Lines within this many semitones of the midpoint between the two
    # voice clusters are still assigned MALE/FEMALE (nearer one wins) but
    # printed as a borderline note for you to spot-check -- informational
    # only, does not change rendering.
    VOICE_BORDERLINE_MARGIN_SEMITONES = 1.0

    # --- Voice legend (shown just below the lyrics panel for duets) ---
    LEGEND_FONT_SIZE = 30
    LEGEND_MARGIN_BELOW_PANEL = 18
    LEGEND_GAP_PX = 60

    # --- Ad-lib / chorus-filler lines ---
    # Any lyrics line wrapped in parentheses, e.g. "(excuse me)" or
    # "(Aa Aa Aa Aa)", is treated as filler -- background chorus muttering,
    # hums, ad-libs -- rather than a line to sing along to precisely. It
    # renders in this distinct muted palette instead of the normal solo
    # (gray/gold/white) or voice-based (blue/pink) colors, so it's visually
    # obvious at a glance that it's not a real lyric. Takes priority over
    # ENABLE_VOICE_COLORING -- an ad-lib line always uses this palette
    # regardless of which voice it got classified as.
    ADLIB_COLOR = {
        "UPCOMING": (95, 95, 108, 255),    # dim neutral gray
        "ACTIVE":   (175, 165, 195, 255),  # muted lavender
        "SUNG":     (150, 150, 160, 255),  # muted light gray -- deliberately NOT
                                            # pure white, so it stays visually
                                            # distinct from real sung lyrics even
                                            # after this line finishes
    }

    PANEL_BG_COLOR = (8, 6, 10, 165)
    PANEL_OUTLINE_COLOR = (200, 160, 90, 120)

    SHOW_PROGRESS_BAR = True
    PROGRESS_BAR_HEIGHT = 8
    PROGRESS_BAR_MARGIN_ABOVE_PANEL = 14
    PROGRESS_BAR_BG_COLOR = (255, 255, 255, 40)
    PROGRESS_BAR_FILL_COLOR = (255, 205, 90, 230)

    COUNTDOWN_GAP_THRESHOLD = 3.0
    COUNTDOWN_LEAD_SECONDS = 3
    COUNTDOWN_COLOR = (255, 205, 90, 255)
    COUNTDOWN_FONT_SIZE = 110
    COUNTDOWN_MARGIN_ABOVE_PANEL = 24
    COUNTDOWN_BADGE_SIZE = 130
    COUNTDOWN_BADGE_BG_COLOR = (8, 6, 10, 190)
    COUNTDOWN_BADGE_OUTLINE_COLOR = (200, 160, 90, 160)

    WHISPER_MODEL_SIZE = "medium"
    WHISPER_LANGUAGE = None
    TRANSLITERATE_NON_LATIN = True

    FORCED_ALIGN_LANGUAGE = "hin"

    # A raw Whisper segment is dropped as "likely instrumental hallucination"
    # only if BOTH of these indicate low confidence -- avg_logprob alone
    # (the model's confidence in the exact words it transcribed) isn't a
    # reliable-enough signal by itself: a real but quietly-sung vocal
    # entrance right after a long instrumental passage can legitimately
    # score a low avg_logprob while still being real singing, not
    # hallucination (confirmed with a real example: Whisper transcribed
    # "On a dark desert highway, cool wind in my hair," -- genuinely
    # correct -- right after a long instrumental intro, yet Whisper's own
    # no_speech_prob for that exact segment was 0.65, i.e. Whisper itself
    # was not at all sure there was speech there despite getting it right).
    # no_speech_prob (Whisper's own estimate of "is there even speech in
    # this segment at all") is the more direct hallucination signal, so
    # BOTH thresholds must be crossed before a segment is discarded --
    # this is what caused a real, correctly-sung opening line to be
    # wrongly dropped as hallucination on a "Hotel California" run (a
    # ~52-second pure-instrumental intro before the first word).
    MIN_AVG_LOGPROB = -3.0
    MAX_NO_SPEECH_PROB_FOR_DROP = 0.85
    MERGE_SEGMENT_GAP = 0.8

    DEMUCS_MODEL = "htdemucs"

    # Advanced: set these directly if you ever need to trim (prompts removed)
    TRIM_START_SEC = 0.0
    TRIM_END_SEC = None

    # --- Thumbnail ---
    THUMB_W, THUMB_H = 1280, 720
    THUMB_TITLE_FONT_SIZE = 92
    THUMB_SUBTITLE_FONT_SIZE = 46
    THUMB_TITLE_COLOR = (255, 205, 90, 255)
    THUMB_SUBTITLE_COLOR = (255, 255, 255, 255)
    THUMB_TEXT_STROKE_WIDTH = 4
    THUMB_TEXT_STROKE_COLOR = (8, 6, 10, 255)


# ============================================================================
# Utilities
# ============================================================================

def run(cmd, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def prompt_path(prompt_text, must_exist=True):
    while True:
        raw = input(prompt_text).strip().strip('"').strip("'")
        if not raw:
            print("  Please enter a path.")
            continue
        p = Path(raw).expanduser()
        if must_exist and not p.exists():
            print(f"  File not found: {p}")
            continue
        return p


def prompt_text(prompt_str, default=""):
    raw = input(prompt_str).strip()
    return raw if raw else default


def prompt_yes_no(prompt_str, default=False):
    raw = input(prompt_str).strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


# ============================================================================
# Step 1: Extract audio (trim removed from prompts; Config knobs remain)
# ============================================================================

def extract_clip(video_path, out_dir, start_sec=0.0, end_sec=None):
    clip_path = out_dir / "clip.mp4"
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if start_sec:
        cmd += ["-ss", str(start_sec)]
    if end_sec:
        cmd += ["-to", str(end_sec)]
    cmd += ["-c", "copy", str(clip_path)]
    run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    audio_path = out_dir / "clip_audio.wav"
    run([
        "ffmpeg", "-y", "-i", str(clip_path), "-vn",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(audio_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return clip_path, audio_path


# ============================================================================
# Step 2: Separate vocals from instrumental using Demucs
# ============================================================================

def separate_audio(audio_path, out_dir, device):
    sep_dir = out_dir / "separated"
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "-n", Config.DEMUCS_MODEL,
        "-d", device,
        "-o", str(sep_dir),
        str(audio_path),
    ]
    run(cmd)

    stem = audio_path.stem
    vocals = sep_dir / Config.DEMUCS_MODEL / stem / "vocals.wav"
    instrumental = sep_dir / Config.DEMUCS_MODEL / stem / "no_vocals.wav"
    if not vocals.exists() or not instrumental.exists():
        raise FileNotFoundError(f"Demucs output not found under {sep_dir}")
    return vocals, instrumental


# ============================================================================
# Step 3: Forced alignment (fixed, same as v2) + Whisper fallback
# ============================================================================

def check_vocab_coverage(full_text, language, model_path="MahmoudAshraf/mms-300m-1130-forced-aligner"):
    from ctc_forced_aligner import preprocess_text
    from transformers import AutoTokenizer

    tokens_starred, text_starred = preprocess_text(full_text, romanize=True, language=language)
    tokens = [t for t in tokens_starred if t != "<star>"]
    words = [t for t in text_starred if t != "<star>"]
    assert len(tokens) == len(words), (
        f"preprocess_text produced {len(tokens)} tokens for {len(words)} words -- "
        f"unexpected internal mismatch, cannot safely proceed."
    )

    tok = AutoTokenizer.from_pretrained(model_path, word_delimiter_token=None)
    dictionary = {k.lower(): v for k, v in tok.get_vocab().items()}

    problems = []
    for w, tk in zip(words, tokens):
        chars = tk.split(" ")
        bad = [c for c in chars if c not in dictionary]
        if bad:
            problems.append((w, tk, bad))
    return problems


_ALIGNMENT_QUOTE_CHARS = '"“”'  # straight " and curly "smart" double quotes only


def _strip_alignment_noise_chars(text):
    """Removes characters that carry no acoustic/singable signal but can
    break the forced-aligner's vocab pre-flight check when they show up as
    their own orphaned token -- confirmed directly against a real user
    lyrics file: a word with a trailing straight double-quote ('1969"')
    and a standalone double-quote used as its own "word" (from a stray
    space before a closing quote, e.g. `"Relax, " said...`) both romanized
    to an empty string and failed the vocab check, aborting forced
    alignment entirely and silently falling back to a far less accurate
    proportional-block timing method for the WHOLE song -- not just the
    quoted words. Ported here from the YouTube-branch v15 fix, since this
    branch's forced_align_lyrics() runs the exact same aligner code and
    would hit the identical failure on any lyrics file containing quotes.

    Deliberately narrow in scope: only straight/curly DOUBLE quotes are
    stripped. Apostrophes/single quotes are left untouched -- confirmed
    from the real lyrics file that prompted this fix that contractions
    using a straight apostrophe ("haven't", "thinkin'", "livin'") were NOT
    flagged as vocab problems, so there's no evidence they need this
    treatment, and no reason to touch them defensively.
    """
    for ch in _ALIGNMENT_QUOTE_CHARS:
        text = text.replace(ch, "")
    return text


def forced_align_lyrics(vocals_path, lyric_lines_text, language, device, out_dir=None, vocal_window=None):
    import torch
    from ctc_forced_aligner import (
        load_audio, load_alignment_model, generate_emissions,
        preprocess_text, get_alignments, get_spans, postprocess_results,
    )

    # Build a cleaned copy of the lyrics used ONLY for the vocab check and
    # the actual alignment call -- quote characters stripped (see
    # _strip_alignment_noise_chars docstring for why). The ORIGINAL
    # lyric_lines_text (with quotes intact) is still used below for every
    # line's displayed "text" field and its ad-lib-parens check --
    # unaffected. Per-line word counts are computed from the SAME cleaned
    # text used for alignment (not the original), so a line where
    # stripping removed an entire orphaned quote "word" doesn't desync
    # downstream line-to-word slicing.
    alignment_lyric_lines = [
        " ".join(_strip_alignment_noise_chars(lt).split()) for lt in lyric_lines_text
    ]
    stripped_count = sum(
        1 for orig, clean in zip(lyric_lines_text, alignment_lyric_lines) if orig != clean
    )
    if stripped_count:
        print(f"  Note: removed quote character(s) from {stripped_count} lyrics line(s) before "
              f"alignment (quotes carry no sung sound and can break the aligner's vocab check) -- "
              f"on-screen captions are unaffected, this only changes what's fed to the aligner.")

    full_text = " ".join(alignment_lyric_lines)

    print("  Pre-flight: checking lyrics text against aligner vocabulary...")
    problems = check_vocab_coverage(full_text, language)
    if problems:
        print(f"  WARNING: {len(problems)} word(s) contain characters outside the aligner's "
              f"vocabulary after romanization -- these WILL desync alignment for this word and "
              f"everything after it. Fix the lyrics text for these words before proceeding:")
        for w, tk, bad in problems:
            print(f"    word='{w}' romanized='{tk}' bad_chars={bad}")
        print("  Common cause: a numeral (e.g. \"1969\") often has no romanized form at all -- try "
              "spelling it out in words (e.g. \"nineteen sixty-nine\") in the lyrics file instead.")
        raise ValueError(f"{len(problems)} word(s) failed the vocab pre-flight check -- see above.")
    print("  Pre-flight OK.")

    # --- Optional: narrow the aligner's input audio to the coarse
    # vocal-activity window detected earlier from raw pre-separation audio
    # (see detect_vocal_window()). This is purely an anti-drift measure for
    # the forced-alignment path: a long silent/instrumental lead-in or
    # ad-lib-heavy outro with no matching lyrics text gives the aligner
    # nothing to anchor to there, which is what produced non-constant
    # timing drift on the YouTube branch's "Hotel California" test (~52s
    # pure instrumental intro) -- ported here since this branch runs the
    # identical forced-alignment code and would hit the same failure mode
    # on any local video with a long instrumental intro/outro. Trimming
    # the input audio to just the padded activity window removes that
    # ambiguous stretch entirely; every resulting timestamp then has the
    # window's start time added back on so it lines up with the original
    # (untrimmed) track. Skipped automatically (falls back to the original
    # untrimmed behavior) if no window was detected, if the window barely
    # trims anything, or if the trim itself fails for any reason -- this
    # must never be able to break alignment outright.
    align_input_path = vocals_path
    window_offset = 0.0
    if vocal_window is not None:
        onset, offset = vocal_window
        try:
            full_duration = get_audio_duration(vocals_path)
        except Exception:
            full_duration = None
        MIN_TRIM_SEC = 3.0
        trims_enough = (
            onset >= MIN_TRIM_SEC
            or (full_duration is not None and (full_duration - offset) >= MIN_TRIM_SEC)
        )
        if trims_enough:
            trim_dir = out_dir if out_dir is not None else vocals_path.parent
            windowed_path = trim_dir / "vocals_for_alignment_windowed.wav"
            dur_str = f"{full_duration:.2f}s" if full_duration is not None else "unknown"
            print(f"  Detected vocal activity window [{onset:.2f}s - {offset:.2f}s] from raw audio "
                  f"(full track: {dur_str}); trimming forced-alignment input to this window to keep "
                  f"the aligner from drifting across the surrounding silence/instrumental...")
            cmd = ["ffmpeg", "-y", "-i", str(vocals_path), "-ss", str(onset)]
            if full_duration is None or offset < full_duration:
                cmd += ["-to", str(offset)]
            cmd += ["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(windowed_path)]
            try:
                run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if windowed_path.exists() and windowed_path.stat().st_size > 0:
                    align_input_path = windowed_path
                    window_offset = onset
                else:
                    print("  Windowed trim produced no output; falling back to the full audio for alignment.")
            except Exception as e:
                print(f"  Failed to trim audio to the detected window ({e}); "
                      f"falling back to the full audio for alignment.")
        else:
            print(f"  Detected vocal window [{onset:.2f}s - {offset:.2f}s] is close to the full track "
                  f"length; skipping windowing (not enough to gain from trimming).")

    print("  Loading forced-alignment model (one-time ~1.2GB download on first run)...")
    try:
        alignment_model, alignment_tokenizer = load_alignment_model(device, dtype=torch.float32)
    except Exception as e:
        print(f"  Could not load on {device} ({e}); falling back to CPU.")
        alignment_model, alignment_tokenizer = load_alignment_model("cpu", dtype=torch.float32)

    audio_waveform = load_audio(str(align_input_path), alignment_model.dtype, alignment_model.device)

    emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=8)
    tokens_starred, text_starred = preprocess_text(full_text, romanize=True, language=language)
    segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)
    word_results = postprocess_results(text_starred, spans, stride, scores)

    # Word counts must be computed from the SAME cleaned text that was
    # actually fed to the aligner above (alignment_lyric_lines), not the
    # original lyric_lines_text -- see the comment above where that list
    # is built.
    per_line_counts = [len(lt.split()) for lt in alignment_lyric_lines]
    expected_total = sum(per_line_counts)
    if len(word_results) != expected_total:
        raise ValueError(
            f"Word-count mismatch after alignment: got {len(word_results)} word results but "
            f"expected {expected_total}. Stopping instead of guessing how to slice lines."
        )

    lines_out = []
    wi = 0
    for line_text, n in zip(lyric_lines_text, per_line_counts):
        word_entries = word_results[wi:wi + n]
        wi += n
        if not word_entries:
            continue
        words = [
            {"text": we["text"], "start": round(we["start"] + window_offset, 3),
             "end": round(we["end"] + window_offset, 3),
             "score": we.get("score", 0.0)}
            for we in word_entries
        ]
        line_score = sum(w["score"] for w in words) / len(words)
        lines_out.append({
            "start": words[0]["start"], "end": words[-1]["end"], "words": words,
            "score": line_score, "text": line_text, "is_adlib": is_adlib_line(line_text),
        })

    flag_low_confidence_lines(lines_out)
    return lines_out


def flag_low_confidence_lines(lines_out, z_threshold=1.5):
    scores = [l["score"] for l in lines_out if "score" in l]
    if len(scores) < 3:
        return
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance ** 0.5
    if std < 1e-6:
        return
    flagged = [(i, l) for i, l in enumerate(lines_out) if (mean - l["score"]) / std >= z_threshold]
    if flagged:
        print(f"\n  WARNING: {len(flagged)} line(s) have unusually low alignment confidence:")
        for i, l in flagged:
            txt = " ".join(w["text"] for w in l["words"])
            print(f"    Line {i+1} [{l['start']:.2f}-{l['end']:.2f}] (score {l['score']:.2f}): {txt}")


def cap_anomalous_word_durations(lines, factor=4.0):
    """Fixes a real bug found on a duet test: a wordless passage (chorus
    hum, ad-lib, instrumental break) with no corresponding entry in the
    lyrics text has nothing for the aligner to attach it to, so the CTC
    aligner sometimes absorbs that extra time into the boundary of the
    nearest real word instead -- usually stretching the LAST word before
    the gap far beyond how long it's actually sung, which visually shows
    up as that word's color-fill crawling in slow motion until the hum
    ends.

    This is a generic, general-purpose safeguard (not specific to any one
    song): any word whose measured duration is more than `factor`x the
    song's own median word duration is treated as having absorbed a gap
    it shouldn't have, and is capped back down to a reasonable length. The
    line's overall end time is recomputed from the (now-capped) last word,
    so the display simply holds on the finished line -- exactly like any
    other ordinary instrumental gap -- instead of stretching the sweep.

    This is a safety net, not a substitute for good input: the cleanest
    fix for a real hum/ad-lib section is adding a placeholder line for it
    in your lyrics text (e.g. "(humming)"), the same way ad-libs like
    "(excuse me)" were handled for the very first song -- that gives the
    aligner something real to anchor to instead of guessing. Flagged
    words are printed either way so you can decide.
    """
    durations = [w["end"] - w["start"] for l in lines for w in l["words"]]
    if len(durations) < 10:
        return
    durations_sorted = sorted(durations)
    n = len(durations_sorted)
    median_dur = durations_sorted[n // 2] if n % 2 else (durations_sorted[n // 2 - 1] + durations_sorted[n // 2]) / 2
    if median_dur <= 0:
        return
    cap = median_dur * factor

    flagged = []
    for l in lines:
        for w in l["words"]:
            dur = w["end"] - w["start"]
            if dur > cap:
                flagged.append((l, dict(w), dur))
                w["end"] = w["start"] + cap
        if l["words"]:
            l["end"] = l["words"][-1]["end"]

    if flagged:
        print(f"\n  WARNING: {len(flagged)} word(s) had anomalously long durations (>{factor:.0f}x the "
              f"song's median word length, ~{median_dur:.2f}s) -- likely a hum/ad-lib/instrumental "
              f"passage with no matching lyrics text got absorbed into that word's timing. Capped to "
              f"{cap:.2f}s for display; for a cleaner fix, add a placeholder line for that passage to "
              f"your lyrics file at the right spot (see this function's docstring):")
        for l, w, dur in flagged:
            print(f"    '{w['text']}' at {w['start']:.2f}s in line \"{l['text']}\" measured {dur:.2f}s")


def flag_slow_lines(lines, factor=0.3):
    """Detection-only safety net (prints a console warning, changes no
    timings -- safe for unattended runs): flags any line whose singing pace
    (words per second, using the line's own start/end and word count) is far
    slower than the song's typical pace.

    This catches a real bug pattern found on a duet test: a wordless chorus
    passage got almost entirely absorbed into a short nearby real line (a
    4-word line stretched to over 20 seconds), while a single-word
    placeholder like "(Chorus)" -- meant to anchor that passage -- got
    squeezed down to under a second instead. Root cause: forced alignment
    matches each letter to wherever the acoustic model finds the strongest
    signal for that letter's *sound*. A hum/mutter passage is almost all
    vowel/nasal sound with very little hard-consonant content, so a
    consonant-heavy placeholder word (like "chorus" itself) can find almost
    no acoustic support anywhere in it and gets starved out in favor of
    neighboring real words, even though those real words don't actually
    match the hum either -- they're just relatively less bad. This is
    cap_anomalous_word_durations()'s blind spot: that function catches one
    WORD's own duration ballooning, but here the whole LINE is short (its
    total word count is normal) while its total time span is what's
    absurd -- a different shape of the same underlying problem.

    This function only reports; it does not resize or move anything. If you
    hit this, the fix is on the lyrics-text side: replace a single
    descriptive placeholder with several short, repeated, vowel-heavy
    syllables that actually resemble the hum/mutter sound (e.g. several
    lines of "(Aa Aa Aa Aa)" rather than one "(Chorus)"), sized roughly to
    the passage's real duration -- vowel-heavy text gives the aligner much
    more evenly-distributed acoustic matches to lock onto instead of one
    word the model has no real signal for.
    """
    rates = []
    for l in lines:
        dur = l["end"] - l["start"]
        n = len(l["words"])
        if dur > 0 and n > 0:
            rates.append(n / dur)
    if len(rates) < 5:
        return
    rates_sorted = sorted(rates)
    m = len(rates_sorted)
    median_rate = rates_sorted[m // 2] if m % 2 else (rates_sorted[m // 2 - 1] + rates_sorted[m // 2]) / 2
    if median_rate <= 0:
        return
    threshold = median_rate * factor

    flagged = []
    for l in lines:
        dur = l["end"] - l["start"]
        n = len(l["words"])
        if dur <= 0 or n == 0:
            continue
        rate = n / dur
        if rate < threshold:
            flagged.append((l, rate))

    if flagged:
        print(f"\n  WARNING: {len(flagged)} line(s) are sung far slower than the song's typical pace "
              f"(~{median_rate:.2f} words/sec) -- likely absorbed a nearby wordless passage (chorus hum, "
              f"ad-lib) into this line's word timings, the same root cause as the anomalous-word-duration "
              f"check above but showing up as a whole slow LINE instead of one stretched word:")
        for l, rate in flagged:
            print(f"    [{l['start']:.2f}-{l['end']:.2f}] ({rate:.2f} words/sec vs ~{median_rate:.2f} "
                  f"typical) \"{l['text']}\" -- if there's an un-transcribed hum/chorus near this "
                  f"timestamp, see this function's docstring for how to placeholder it more effectively.")


# ============================================================================
# Step 3b: Two-voice (duet) pitch classification -- now opt-in
# ============================================================================

def classify_voices(vocals_path, lines):
    """Assign each line to 'MALE' (lower pitch register) or 'FEMALE'
    (higher register) based on the vocal audio during that line's time
    span. This is a heuristic, not real speaker ID -- there's no signal in
    this pipeline that distinguishes "a different person" from "the same
    person on a different note," so it only makes sense for genuine duets
    with two clearly separated vocal ranges. Do not enable for solo songs
    (a single singer's own melodic range would get split by note instead,
    which is misleading).

    2-cluster k-means (1D, on semitones) on each line's median F0 --
    lower cluster is labeled MALE, higher FEMALE (a labeling convention,
    not a guarantee; swap Config.VOICE_COLORS' "MALE"/"FEMALE" keys if your
    song's actual singers are the other way around). Every line gets
    assigned to its nearer cluster, no in-between category.

    (A third "BOTH" category for lines sung together was tried and removed
    -- on a real duet test it mislabeled genuine male/female lines, since
    there's no reliable audio signal for "two voices at once" available
    here. Lines close to the midpoint between the two clusters are still
    printed as a borderline note, in case you want to spot-check them, but
    they're assigned to a side rather than a third color.)
    """
    import librosa

    print("  Classifying voices by pitch (for duet color separation)...")
    y, sr = librosa.load(str(vocals_path), sr=22050, mono=True)

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr, frame_length=2048
    )
    times = librosa.times_like(f0, sr=sr)

    def median_f0_semitones(t0, t1):
        mask = (times >= t0) & (times <= t1) & voiced_flag & ~np.isnan(f0)
        vals = f0[mask]
        if len(vals) < 3:
            return None
        med_hz = float(np.median(vals))
        return 12 * np.log2(med_hz / 440.0)

    line_semis = [median_f0_semitones(l["start"], l["end"]) for l in lines]
    valid = [(i, s) for i, s in enumerate(line_semis) if s is not None]
    if len(valid) < 4:
        print("  Not enough voiced lines to classify -- defaulting all lines to MALE.")
        for l in lines:
            l["voice"] = "MALE"
        return None

    semis = np.array([s for _, s in valid])
    c0, c1 = np.percentile(semis, 20), np.percentile(semis, 80)
    for _ in range(50):
        d0, d1 = np.abs(semis - c0), np.abs(semis - c1)
        assign = (d1 < d0).astype(int)
        new_c0 = semis[assign == 0].mean() if np.any(assign == 0) else c0
        new_c1 = semis[assign == 1].mean() if np.any(assign == 1) else c1
        if abs(new_c0 - c0) < 1e-4 and abs(new_c1 - c1) < 1e-4:
            c0, c1 = new_c0, new_c1
            break
        c0, c1 = new_c0, new_c1
    if c0 > c1:
        c0, c1 = c1, c0

    midpoint = (c0 + c1) / 2
    borderline_margin = Config.VOICE_BORDERLINE_MARGIN_SEMITONES

    for l in lines:
        l["voice"] = "MALE"

    borderline = []
    for i, semi in valid:
        lines[i]["voice"] = "MALE" if semi < midpoint else "FEMALE"
        if abs(semi - midpoint) <= borderline_margin:
            borderline.append(i)

    hz0, hz1 = 440.0 * 2 ** (c0 / 12), 440.0 * 2 ** (c1 / 12)
    print(f"  Voice clusters: MALE ~{hz0:.0f}Hz (lower), FEMALE ~{hz1:.0f}Hz (higher)")
    if borderline:
        print(f"  {len(borderline)} line(s) had pitch close to the midpoint between the two voices -- "
              f"assigned to the nearer one, but worth a spot-check:")
        for i in borderline:
            txt = " ".join(w["text"] for w in lines[i]["words"])
            print(f"    [{lines[i]['start']:.2f}-{lines[i]['end']:.2f}] ({lines[i]['voice']}) {txt}")

    return {"MALE_hz": hz0, "FEMALE_hz": hz1}


def read_voice_overrides(path):
    """Parses a voice-override file: either a raw copy of a previous run's
    line_timing_preview.txt with the [MALE]/[FEMALE] tags hand-corrected, or
    any file matching that same per-line format. Returns an ordered list of
    (voice, text) tuples -- text is kept only for a sanity-check against the
    current run's lines, not used for timing (this does NOT replace forced
    alignment; it only overrides the voice classification, since pitch-based
    voice detection is a heuristic that a human can usually correct faster
    and more reliably than the code can, especially on songs with more than
    two singers).

    Why this exists: pitch-only classification can only ever split "higher"
    from "lower," never tell apart two different people -- it's reliably
    right on some phrases and reliably wrong on others (a singer's own
    higher notes crossing into the other cluster), verified concretely on a
    real 25-line file. Rather than chase that with more pitch heuristics, a
    human spot-check of the auto-generated line_timing_preview.txt is fast
    and fully reliable, and this just wires that correction back in.
    """
    row_re = re.compile(r"^\s*\[\s*(-?[\d.]+)\s*-\s*(-?[\d.]+)\]\s*\[([A-Za-z]+)\]\s*(.*)$")
    overrides = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            m = row_re.match(raw)
            if not m:
                raise ValueError(
                    f"Voice-override file line doesn't match the expected "
                    f"'[start - end] [VOICE] text' format: {raw!r}\n"
                    f"This should be a copy of line_timing_preview.txt (hand-edited "
                    f"voice tags only) from a previous run on this same song."
                )
            voice = m.group(3).strip().upper()
            text = m.group(4).strip()
            overrides.append((voice, text))
    return overrides


def _normalize_for_compare(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def apply_voice_overrides(lines, overrides):
    """Overrides lines[i]['voice'] from a hand-corrected voice-override file,
    matched strictly by line ORDER (index) -- not by fuzzy text matching,
    since silently applying a correction to the wrong line would be worse
    than the bug this is fixing. Hard-fails on a line-count mismatch (most
    likely cause: the override file is from a different song/lyrics-file
    version than this run). Line text is still compared as a softer sanity
    check -- a mismatch there doesn't stop the run, but is printed loudly so
    you can confirm before trusting the render, since minor punctuation/
    transliteration differences between runs are expected and harmless.
    """
    if len(overrides) != len(lines):
        raise ValueError(
            f"Voice-override file has {len(overrides)} line(s) but this run produced "
            f"{len(lines)} line(s) -- refusing to apply overrides by position, since a "
            f"count mismatch means they'd land on the wrong lines. Make sure the override "
            f"file was generated from THIS song with THIS lyrics file (i.e. it's a "
            f"line_timing_preview.txt from a previous run on the exact same input)."
        )

    text_mismatches = []
    changed = 0
    for i, (line, (voice, override_text)) in enumerate(zip(lines, overrides)):
        current_text = line.get("text", " ".join(w["text"] for w in line["words"]))
        if _normalize_for_compare(current_text) != _normalize_for_compare(override_text):
            text_mismatches.append((i, current_text, override_text))
        if line.get("voice") != voice:
            changed += 1
        line["voice"] = voice

    print(f"  Applied voice overrides: {changed}/{len(lines)} line(s) changed from the "
          f"auto-classifier's guess.")
    if text_mismatches:
        print(f"  NOTE: {len(text_mismatches)} line(s)' text didn't exactly match between "
              f"this run and the override file (voice was still applied by position -- "
              f"spot-check these):")
        for i, cur, ov in text_mismatches:
            print(f"    line {i + 1}: this run says \"{cur}\" / override file says \"{ov}\"")


# ============================================================================
# Step 3c: Whisper-fallback timing helpers (unchanged from v2)
# ============================================================================

def split_proportional(start, end, weights):
    total_w = sum(weights) or 1
    dur = end - start
    t = start
    out = []
    for w in weights:
        seg_dur = dur * (w / total_w)
        out.append((t, t + seg_dur))
        t += seg_dur
    return out


def merge_blocks_to_target_count(blocks, target_count):
    blocks = [tuple(b) for b in blocks]
    while len(blocks) > target_count and len(blocks) > 1:
        gaps = [blocks[i + 1][0] - blocks[i][1] for i in range(len(blocks) - 1)]
        merge_at = min(range(len(gaps)), key=lambda i: gaps[i])
        merged = (blocks[merge_at][0], blocks[merge_at + 1][1])
        blocks = blocks[:merge_at] + [merged] + blocks[merge_at + 2:]
    return blocks


def allocate_lines_to_blocks(lyric_lines, blocks):
    n_lines = len(lyric_lines)
    block_durs = [b[1] - b[0] for b in blocks]
    total_dur = sum(block_durs) or 1

    ideal = [n_lines * (d / total_dur) for d in block_durs]
    floor_alloc = [int(np.floor(x)) for x in ideal]
    remainder = n_lines - sum(floor_alloc)

    order = sorted(range(len(blocks)), key=lambda i: (ideal[i] - floor_alloc[i]), reverse=True)
    alloc = floor_alloc[:]
    i = 0
    while remainder > 0 and order:
        alloc[order[i % len(order)]] += 1
        remainder -= 1
        i += 1

    result_lines = []
    idx = 0
    for bi, count in enumerate(alloc):
        if count == 0:
            continue
        assigned_text_lines = lyric_lines[idx:idx + count]
        idx += count
        b_start, b_end = blocks[bi]
        line_weights = [max(len(t), 1) for t in assigned_text_lines]
        line_spans = split_proportional(b_start, b_end, line_weights)
        for text, (lstart, lend) in zip(assigned_text_lines, line_spans):
            words_text = text.split(" ")
            word_weights = [max(len(w), 1) for w in words_text]
            word_spans = split_proportional(lstart, lend, word_weights)
            words = [
                {"text": wt, "start": round(ws, 3), "end": round(we, 3)}
                for wt, (ws, we) in zip(words_text, word_spans)
            ]
            result_lines.append({
                "start": round(lstart, 3), "end": round(lend, 3), "words": words, "text": text,
                "is_adlib": is_adlib_line(text),
            })
    return result_lines


def run_whisper(vocals_path, device, model_size=None, temperature=None):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import whisper

    size = model_size if model_size is not None else Config.WHISPER_MODEL_SIZE

    def _load_and_transcribe(dev):
        print(f"  Loading Whisper model '{size}' on {dev}...")
        model = whisper.load_model(size, device=dev)
        kwargs = dict(
            word_timestamps=True, verbose=False, language=Config.WHISPER_LANGUAGE,
        )
        # temperature stays unset (Whisper's own default temperature-fallback
        # tuple) for every existing caller -- only passed explicitly by
        # callers that need deterministic single-pass decoding (see
        # detect_vocal_window()). Whisper's default fallback (retrying at
        # higher, randomly-sampled temperatures when its own quality checks
        # fail) is valuable for real transcription accuracy and must not be
        # disabled pipeline-wide.
        if temperature is not None:
            kwargs["temperature"] = temperature
        return model.transcribe(str(vocals_path), **kwargs)

    try:
        return _load_and_transcribe(device)
    except Exception as e:
        msg = str(e)
        retriable = (
            device != "cpu" and (
                "float64" in msg or "MPS" in msg or "CERTIFICATE_VERIFY_FAILED" in msg
            )
        )
        if retriable:
            reason = (
                "Whisper's word-timestamp alignment doesn't support Apple's MPS backend"
                if "float64" in msg or "MPS" in msg else "a certificate error occurred"
            )
            print(f"  {device} failed ({reason}); retrying on CPU instead...")
            return _load_and_transcribe("cpu")
        raise


def _is_likely_hallucination(seg):
    """A raw Whisper segment is treated as likely instrumental hallucination
    only when BOTH avg_logprob is very low AND no_speech_prob is very high
    -- see Config.MIN_AVG_LOGPROB / Config.MAX_NO_SPEECH_PROB_FOR_DROP's
    docstring for why avg_logprob alone isn't a reliable enough signal on
    its own (a real, quietly-sung vocal entrance right after a long
    instrumental passage can legitimately score a low avg_logprob while
    still being real singing). Ported here from the YouTube-branch v13
    fix -- this branch's own Whisper-fallback path runs the same
    single-signal check that dropped a real, correctly-sung opening line
    on a real test."""
    return (
        seg.get("avg_logprob", 0) < Config.MIN_AVG_LOGPROB
        and seg.get("no_speech_prob", 0) >= Config.MAX_NO_SPEECH_PROB_FOR_DROP
    )


def clean_whisper_segments(whisper_result):
    kept = [seg for seg in whisper_result["segments"] if not _is_likely_hallucination(seg)]
    if not kept:
        return []

    merged = [{"start": kept[0]["start"], "end": kept[0]["end"], "segments": [kept[0]]}]
    for seg in kept[1:]:
        if seg["start"] - merged[-1]["end"] < Config.MERGE_SEGMENT_GAP:
            merged[-1]["end"] = seg["end"]
            merged[-1]["segments"].append(seg)
        else:
            merged.append({"start": seg["start"], "end": seg["end"], "segments": [seg]})
    return merged


def segments_as_blocks(whisper_result):
    cleaned = clean_whisper_segments(whisper_result)
    dropped_segs = [seg for seg in whisper_result["segments"] if _is_likely_hallucination(seg)]
    if dropped_segs:
        print(f"  Filtered out {len(dropped_segs)} low-confidence segment(s) (likely instrumental hallucination):")
        for seg in dropped_segs:
            print(f"    [{seg['start']:7.2f} - {seg['end']:7.2f}]  "
                  f"avg_logprob={seg.get('avg_logprob', 0):.2f}  "
                  f"no_speech_prob={seg.get('no_speech_prob', 0):.2f}  "
                  f"text={seg.get('text', '').strip()!r}")
    return [(c["start"], c["end"]) for c in cleaned]


def detect_vocal_window(raw_audio_path, pad_before=2.0, pad_after=2.0):
    """Runs a fast, cheap Whisper pass on the RAW (pre-vocal-separation)
    audio purely to find a rough "where is there likely vocal activity"
    window -- NOT for word-level transcription accuracy, that's what
    forced-alignment against the user's own lyrics text is for. Ported
    here from the YouTube-branch v14 fix -- this branch's forced_align_
    lyrics() runs the identical aligner code and is equally exposed to
    drift across a long instrumental intro/outro with no matching lyrics
    text.

    Why raw audio, deliberately, and why this is safe here specifically:
    a real singing voice mixed with instruments is often still detectable
    as speech-like sound to Whisper even when it's too quiet/blended for
    a full transcription to come out clean or correct. Using raw audio
    for full-song transcription accuracy would be a bad trade (background
    instruments generally hurt Whisper's word-level accuracy across a
    whole song, which is why vocal separation is used for that job) --
    but finding a coarse activity window is a much lower bar, and this
    function is ONLY ever used for that, never for word text.

    Reuses the existing run_whisper() + segments_as_blocks() low-
    confidence filtering (Config.MIN_AVG_LOGPROB / MAX_NO_SPEECH_PROB_
    FOR_DROP) rather than a new detector, so a hallucinated segment
    during a loud instrumental doesn't get mistaken for the vocal start.
    Uses the fast 'base' Whisper model on CPU -- this is a coarse
    boundary check, not the final transcript, so the extra accuracy (and
    much longer runtime) of the pipeline's normal 'medium' model isn't
    needed here.

    Returns (onset, offset) in seconds -- the first and last detected
    activity block's timestamps, padded outward by pad_before/pad_after
    (clamped to [0, audio duration]) so the forced-aligner still gets a
    little headroom rather than a razor-exact cut. Returns None if no
    vocal activity was detected at all, or if anything about this
    detection step fails -- callers should treat None as "skip windowing
    entirely, run forced-alignment against the full audio as before,"
    never as license to guess a window.
    """
    try:
        # temperature=0.0 forces single-pass, deterministic decoding for
        # this coarse pass -- Whisper's own default temperature-fallback
        # (randomly-sampled retries when its confidence checks fail) was
        # observed, directly in testing on the YouTube branch, to make
        # this detection non-deterministic run-to-run on identical audio.
        # This only affects this detection call -- run_whisper()'s
        # default behavior (used everywhere else, including the real
        # transcription passes) is unchanged.
        whisper_result = run_whisper(raw_audio_path, "cpu", model_size="base", temperature=0.0)

        # Even with deterministic decoding, Whisper still reliably labels
        # purely-instrumental stretches with filler captions like "Music"
        # or musical-note symbols. These pass the existing avg_logprob/
        # no_speech_prob hallucination filter (they ARE what Whisper
        # "confidently" heard), so they need a second, content-based
        # filter specific to this coarse-detection use -- not applied to
        # clean_whisper_segments()/segments_as_blocks() themselves, so
        # real transcription elsewhere in the pipeline is unaffected.
        real_segments = [
            seg for seg in whisper_result["segments"]
            if not _is_nonlyric_filler_text(seg.get("text", ""))
        ]
        filtered_result = {**whisper_result, "segments": real_segments}

        blocks = segments_as_blocks(filtered_result)
        if not blocks:
            return None
        duration = get_audio_duration(raw_audio_path)
        onset = max(0.0, blocks[0][0] - pad_before)
        offset = min(duration, blocks[-1][1] + pad_after)
        if offset <= onset:
            return None
        return onset, offset
    except Exception as e:
        print(f"  Vocal-activity detection for alignment windowing failed ({e}); "
              f"skipping windowing -- forced alignment will run against the full audio.")
        return None


def _is_nonlyric_filler_text(text):
    """True for Whisper segment text that is a filler/instrumental caption
    rather than actual sung words -- e.g. "Music", "[Music]", or text
    that's empty once punctuation/symbols are stripped. Used only by
    detect_vocal_window()'s coarse activity scan, never for real
    transcription."""
    import re
    stripped = re.sub(r"[^\w]", "", text.strip().lower())
    if not stripped:
        return True
    return stripped in {"music", "instrumental", "instrumentalmusic"}


def build_lines_from_whisper_text(whisper_result):
    def maybe_transliterate(text):
        if Config.TRANSLITERATE_NON_LATIN and any(ord(c) > 127 for c in text):
            from unidecode import unidecode
            return unidecode(text).strip()
        return text.strip()

    cleaned = clean_whisper_segments(whisper_result)
    lines = []
    for block in cleaned:
        for seg in block["segments"]:
            words = []
            for w in seg.get("words", []):
                text = maybe_transliterate(w["word"])
                if text:
                    words.append({"text": text, "start": w["start"], "end": w["end"]})
            if words:
                line_text = " ".join(w["text"] for w in words)
                lines.append({
                    "start": words[0]["start"], "end": words[-1]["end"], "words": words,
                    "text": line_text, "is_adlib": is_adlib_line(line_text),
                })
    return lines


def split_long_lines(lines, max_width_fraction=None):
    """Splits any line whose rendered text would be wider than the lyrics
    panel into two or more shorter lines, so words never get silently
    clipped off the panel's left/right edge at render time. Ported here
    from the YouTube-branch v12 fix -- this branch's draw_line() has the
    identical un-wrapped, fixed-width rendering and is equally exposed.

    Why this is needed: draw_line() (in render_video, below) draws a
    line's full joined text as one un-wrapped string centered in a
    fixed-size panel image -- it never wraps or shrinks to fit. Anything
    drawn outside the panel image's own pixel bounds is just dropped by
    PIL, no error, no visual cue. Real-world case this was built from (on
    the YouTube branch, same rendering code): a Whisper segment (no
    lyrics file provided) spanning several sentences with no natural
    pause-driven break came out ~1620px wide against a 1450px panel,
    silently losing the first letter of "shimmering" and the last two
    letters of "the" off the two edges.

    Split points are chosen at a punctuation mark (comma, period,
    semicolon, colon, question mark, exclamation mark) as close to the
    middle of the line as possible -- the natural place for a lyrics/
    subtitle line break. If a line has no punctuation at all, it's split
    at the midpoint by word count instead so it still fits, at the cost of
    a slightly less natural break point. Recurses on each half, so a line
    spanning 3+ sentences ends up fully within width after multiple
    splits, not just cut once. A single word that alone is still wider
    than the panel can't be split further and is left as-is (this should
    be extremely rare).

    Runs against the exact same font/size render_video() renders with, so
    the fit check matches what will actually appear on screen rather than
    approximating it.

    Any key already on a line dict besides words/start/end/text/is_adlib
    (e.g. "voice", once voice classification has run) is carried over
    unchanged to every piece a line gets split into -- deliberately, so
    this can also be called safely after voice classification without
    losing per-line voice tags. (In this pipeline's actual call order it
    runs BEFORE voice classification/overrides, so every downstream
    per-line step -- word-duration capping, slow-line flagging, voice
    classification, voice overrides, the printed/saved timing preview --
    already sees the final, post-split lines, matching what's rendered.)
    """
    from PIL import Image, ImageDraw
    font = load_font(Config.FONT_CANDIDATES_BOLD, Config.LYRIC_FONT_SIZE)
    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    fraction = max_width_fraction if max_width_fraction is not None else Config.MAX_LINE_WIDTH_FRACTION
    max_width_px = Config.PANEL_W * fraction

    PUNCT_SPLIT_CHARS = {",", ".", "!", "?", ";", ":"}

    def text_width(words):
        txt = " ".join(w["text"] for w in words)
        bbox = dummy_draw.textbbox((0, 0), txt, font=font)
        return bbox[2] - bbox[0]

    def make_line(words, template):
        text = " ".join(w["text"] for w in words)
        new_line = {k: v for k, v in template.items()
                    if k not in ("words", "start", "end", "text", "is_adlib")}
        new_line.update({
            "words": words,
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": text,
            "is_adlib": is_adlib_line(text),
        })
        return new_line

    def split_one(line):
        words = line["words"]
        if len(words) <= 1 or text_width(words) <= max_width_px:
            return [line]

        candidates = [
            i for i, w in enumerate(words[:-1])
            if w["text"] and w["text"][-1] in PUNCT_SPLIT_CHARS
        ]
        if candidates:
            mid = len(words) / 2
            split_i = min(candidates, key=lambda i: abs((i + 1) - mid))
        else:
            split_i = max(0, len(words) // 2 - 1)

        left_words, right_words = words[:split_i + 1], words[split_i + 1:]
        if not left_words or not right_words:
            return [line]  # guard only -- the index math above can't actually hit this

        return split_one(make_line(left_words, line)) + split_one(make_line(right_words, line))

    result = []
    lines_split = 0
    for line in lines:
        pieces = split_one(line)
        if len(pieces) > 1:
            lines_split += 1
        result.extend(pieces)

    if lines_split:
        print(f"  Split {lines_split} line(s) too wide for the display panel into "
              f"{len(result)} total line(s) (was {len(lines)}).")
    return result


# ============================================================================
# Step 4: Render
# ============================================================================

def get_audio_duration(wav_path):
    with contextlib.closing(wave.open(str(wav_path), "r")) as wf:
        return wf.getnframes() / float(wf.getframerate())


def load_font(candidates, size):
    from PIL import ImageFont
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    print(f"  WARNING: no candidate font found, using PIL default (size will be fixed).")
    from PIL import ImageFont
    return ImageFont.load_default()


def build_backgrounds(image_paths):
    from PIL import Image
    bases = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        scale = Config.OUT_H / img.height
        new_w = int(img.width * scale)
        resized = img.resize((new_w, Config.OUT_H), Image.LANCZOS)
        canvas = Image.new("RGB", (Config.OUT_W, Config.OUT_H), (0, 0, 0))
        x_off = (Config.OUT_W - new_w) // 2
        canvas.paste(resized, (x_off, 0))
        bases.append(canvas.convert("RGBA"))
    return bases


def build_ticker_text(song_title, film_name, artist_name):
    parts = []
    if song_title:
        parts.append(song_title)
    meta_bits = [b for b in (film_name, artist_name) if b]
    if meta_bits:
        parts.append(" / ".join(meta_bits))
    song_info = "  •  ".join(parts)
    if song_info:
        return f"{song_info}     •     {Config.TICKER_TEXT}"
    return Config.TICKER_TEXT


def render_video(lines, vocal_blocks, duration, instrumental_path, out_dir, ticker_text,
                  output_filename="karaoke_final.mp4"):
    from PIL import Image, ImageDraw

    lyric_font = load_font(Config.FONT_CANDIDATES_BOLD, Config.LYRIC_FONT_SIZE)
    ticker_font = load_font(Config.FONT_CANDIDATES_REGULAR, Config.TICKER_FONT_SIZE)
    countdown_font = load_font(Config.FONT_CANDIDATES_BOLD, Config.COUNTDOWN_FONT_SIZE)

    backgrounds = build_backgrounds(Config.BACKGROUND_IMAGES)

    PANEL_CX = Config.OUT_W // 2
    PANEL_CY = Config.PANEL_CENTER_Y
    PANEL_W, PANEL_H = Config.PANEL_W, Config.PANEL_H
    N_VIS = Config.VISIBLE_LINES
    SLOT = Config.LINE_SLOT_HEIGHT

    panel_top = PANEL_CY - PANEL_H // 2
    bar_y1 = panel_top - Config.PROGRESS_BAR_MARGIN_ABOVE_PANEL
    bar_y0 = bar_y1 - Config.PROGRESS_BAR_HEIGHT
    bar_x0 = PANEL_CX - PANEL_W // 2
    bar_x1 = PANEL_CX + PANEL_W // 2

    badge_r = Config.COUNTDOWN_BADGE_SIZE // 2
    badge_cy = bar_y0 - Config.COUNTDOWN_MARGIN_ABOVE_PANEL - badge_r
    badge_cx = PANEL_CX

    # row vertical centers within the panel layer, for N_VIS visible rows
    # plus 1 extra (off the bottom) that slides into view during transitions
    top_margin = (PANEL_H - N_VIS * SLOT) / 2

    def row_center(i):
        return top_margin + i * SLOT + SLOT / 2

    TICKER_Y0, TICKER_Y1 = Config.OUT_H - 90, Config.OUT_H - 43
    dummy = Image.new("RGB", (10, 10))
    ticker_bbox = ImageDraw.Draw(dummy).textbbox((0, 0), ticker_text, font=ticker_font)
    ticker_text_w = ticker_bbox[2] - ticker_bbox[0]

    gaps = []
    if vocal_blocks:
        if vocal_blocks[0][0] > 0:
            gaps.append((0.0, vocal_blocks[0][0]))
        for i in range(len(vocal_blocks) - 1):
            gaps.append((vocal_blocks[i][1], vocal_blocks[i + 1][0]))
    else:
        gaps.append((0.0, duration))

    def countdown_number(t):
        for g_start, g_end in gaps:
            if g_end - g_start >= Config.COUNTDOWN_GAP_THRESHOLD:
                lead_start = g_end - Config.COUNTDOWN_LEAD_SECONDS
                if lead_start <= t < g_end:
                    return max(1, int(np.ceil(g_end - t)))
        return None

    def active_line_index(t):
        for i, l in enumerate(lines):
            if l["start"] <= t <= l["end"]:
                return i
        return None

    def next_line_index(t):
        for i, l in enumerate(lines):
            if l["start"] > t:
                return i
        return None

    def _smoothstep(x):
        x = min(max(x, 0.0), 1.0)
        return x * x * (3 - 2 * x)

    def scroll_center_index(t):
        if not lines:
            return 0.0
        idx = active_line_index(t)
        if idx is not None:
            return float(idx)
        nxt = next_line_index(t)
        if nxt is None:
            return float(len(lines) - 1)
        if nxt == 0:
            return 0.0
        prev_end = lines[nxt - 1]["end"]
        next_start = lines[nxt]["start"]
        transition_start = max(prev_end, next_start - Config.SCROLL_TRANSITION_SECONDS)
        if t < transition_start:
            return float(nxt - 1)
        frac = (t - transition_start) / max(next_start - transition_start, 1e-6)
        return (nxt - 1) + _smoothstep(frac)

    def upcoming_active_colors_for_line(line):
        """Returns (upcoming_color, active_color) for a line. Ad-lib/chorus
        filler lines (Config.ADLIB_COLOR) take priority over everything else
        -- they're never voice- or solo-colored, so they read as filler at a
        glance regardless of ENABLE_VOICE_COLORING. Otherwise: for duets,
        both states are colored by voice (Male=blue, Female=dark pink) so the
        singer knows who's up before the line starts; for solo songs,
        upcoming stays the neutral dim gray."""
        if line.get("is_adlib"):
            return Config.ADLIB_COLOR["UPCOMING"], Config.ADLIB_COLOR["ACTIVE"]
        if not Config.ENABLE_VOICE_COLORING:
            return Config.UPCOMING_COLOR, Config.SOLO_ACTIVE_COLOR
        voice = line.get("voice", "MALE")
        pal = Config.VOICE_COLORS.get(voice, Config.VOICE_COLORS["MALE"])
        return pal["UPCOMING"], pal["ACTIVE"]

    def draw_line(draw, y_center, line, t, font, x_center=None):
        if x_center is None:
            x_center = PANEL_CX
        upcoming_color, active_color = upcoming_active_colors_for_line(line)
        sung_color = Config.ADLIB_COLOR["SUNG"] if line.get("is_adlib") else Config.SUNG_COLOR
        txt = " ".join(w["text"] for w in line["words"])
        bbox = draw.textbbox((0, 0), txt, font=font)
        total_w = bbox[2] - bbox[0]
        x = x_center - total_w / 2
        y = y_center - (bbox[3] - bbox[1]) / 2
        for w in line["words"]:
            word_txt = w["text"] + " "
            if t < w["start"]:
                color = upcoming_color
            elif t <= w["end"]:
                color = active_color
            else:
                color = sung_color
            draw.text((x, y), word_txt, font=font, fill=color)
            wbbox = draw.textbbox((0, 0), word_txt, font=font)
            x += (wbbox[2] - wbbox[0])

    def draw_centered(draw, y_center, text, font, color, x_center=None):
        if x_center is None:
            x_center = PANEL_CX
        bbox = draw.textbbox((0, 0), text, font=font)
        x = x_center - (bbox[2] - bbox[0]) / 2
        y = y_center - (bbox[3] - bbox[1]) / 2
        draw.text((x, y), text, font=font, fill=color)

    # --- voice legend (fixed caption just below the panel, duets only) ---
    legend_font = load_font(Config.FONT_CANDIDATES_BOLD, Config.LEGEND_FONT_SIZE) if Config.ENABLE_VOICE_COLORING else None
    legend_entries = []
    if Config.ENABLE_VOICE_COLORING:
        legend_entries = [
            (Config.VOICE_LABELS[v], Config.VOICE_COLORS[v]["ACTIVE"]) for v in ("MALE", "FEMALE")
        ]
    legend_y = PANEL_CY + PANEL_H // 2 + Config.LEGEND_MARGIN_BELOW_PANEL

    def draw_legend(draw):
        if not legend_entries:
            return
        dummy_img = Image.new("RGB", (10, 10))
        d = ImageDraw.Draw(dummy_img)
        widths = []
        for text, _ in legend_entries:
            bbox = d.textbbox((0, 0), text, font=legend_font)
            widths.append(bbox[2] - bbox[0])
        total_w = sum(widths) + Config.LEGEND_GAP_PX * (len(legend_entries) - 1)
        x = Config.OUT_W / 2 - total_w / 2
        for (text, color), w in zip(legend_entries, widths):
            draw_centered(draw, legend_y, text, legend_font, color, x_center=x + w / 2)
            x += w + Config.LEGEND_GAP_PX

    def current_bg_index(t):
        if len(backgrounds) <= 1:
            return 0
        block_idx = 0
        for i, b in enumerate(vocal_blocks):
            if b[0] <= t:
                block_idx = i
        return block_idx % len(backgrounds)

    def build_frame(t):
        frame = backgrounds[current_bg_index(t)].copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        if Config.SHOW_PROGRESS_BAR:
            draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=Config.PROGRESS_BAR_BG_COLOR)
            frac = min(max(t / duration, 0), 1)
            fill_x1 = bar_x0 + (bar_x1 - bar_x0) * frac
            draw.rectangle([bar_x0, bar_y0, fill_x1, bar_y1], fill=Config.PROGRESS_BAR_FILL_COLOR)

        px0, px1 = PANEL_CX - PANEL_W // 2, PANEL_CX + PANEL_W // 2
        py0, py1 = PANEL_CY - PANEL_H // 2, PANEL_CY + PANEL_H // 2
        draw.rounded_rectangle([px0, py0, px1, py1], radius=24,
                                fill=Config.PANEL_BG_COLOR, outline=Config.PANEL_OUTLINE_COLOR, width=2)

        if lines:
            ci = scroll_center_index(t)
            base = int(np.floor(ci))
            frac = ci - base
            panel_layer = Image.new("RGBA", (PANEL_W, PANEL_H), (0, 0, 0, 0))
            panel_draw = ImageDraw.Draw(panel_layer, "RGBA")
            # N_VIS visible rows + 1 extra row sliding in from the bottom
            # during a transition, so no blank row ever appears mid-scroll
            for i in range(N_VIS + 1):
                li = base + i
                if 0 <= li < len(lines):
                    y = row_center(i) - frac * SLOT
                    draw_line(panel_draw, y, lines[li], t, lyric_font, x_center=PANEL_W // 2)
            frame.alpha_composite(panel_layer, (px0, py0))
            draw = ImageDraw.Draw(frame, "RGBA")

        draw_legend(draw)

        cd = countdown_number(t)
        if cd is not None:
            draw.ellipse(
                [badge_cx - badge_r, badge_cy - badge_r, badge_cx + badge_r, badge_cy + badge_r],
                fill=Config.COUNTDOWN_BADGE_BG_COLOR, outline=Config.COUNTDOWN_BADGE_OUTLINE_COLOR, width=3,
            )
            draw_centered(draw, badge_cy, str(cd), countdown_font, Config.COUNTDOWN_COLOR)

        if Config.SHOW_TICKER:
            draw.rectangle([0, TICKER_Y0, Config.OUT_W, TICKER_Y1], fill=(5, 5, 8, 175))
            scroll_x = Config.OUT_W - (t * Config.TICKER_SPEED_PX_PER_SEC) % (ticker_text_w + Config.OUT_W)
            ty = TICKER_Y0 + (TICKER_Y1 - TICKER_Y0) // 2 - 14
            draw.text((scroll_x, ty), ticker_text, font=ticker_font, fill=(230, 190, 110, 255))
            draw.text((scroll_x - ticker_text_w, ty), ticker_text, font=ticker_font, fill=(230, 190, 110, 255))

        return frame.convert("RGB")

    n_total_frames = int(duration * Config.FPS)
    video_only = out_dir / "karaoke_video_only.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{Config.OUT_W}x{Config.OUT_H}", "-r", str(Config.FPS),
        "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(video_only),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    from tqdm import tqdm
    for i in tqdm(range(n_total_frames), desc="Rendering frames"):
        t = i / Config.FPS
        frame = build_frame(t)
        proc.stdin.write(np.array(frame).tobytes())
    proc.stdin.close()
    proc.wait()

    final_out = out_dir / output_filename
    run([
        "ffmpeg", "-y", "-i", str(video_only), "-i", str(instrumental_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final_out)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return final_out


def generate_thumbnail(song_title, film_name, artist_name, out_dir, output_filename="thumbnail.png"):
    """1280x720 thumbnail using the same background as-is (no darkened
    band behind the text) -- large title text and smaller film/artist text
    below, each with a dark outline stroke so they stay readable against
    whatever the background looks like at that spot."""
    from PIL import Image, ImageDraw

    title_font = load_font(Config.FONT_CANDIDATES_BOLD, Config.THUMB_TITLE_FONT_SIZE)
    sub_font = load_font(Config.FONT_CANDIDATES_BOLD, Config.THUMB_SUBTITLE_FONT_SIZE)

    bg_path = Config.BACKGROUND_IMAGES[0]
    img = Image.open(bg_path).convert("RGB")
    W, H = Config.THUMB_W, Config.THUMB_H
    scale = max(W / img.width, H / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    x_off = (img.width - W) // 2
    y_off = (img.height - H) // 2
    img = img.crop((x_off, y_off, x_off + W, y_off + H)).convert("RGBA")

    draw = ImageDraw.Draw(img, "RGBA")

    lines = []
    if song_title:
        lines.append((song_title, title_font, Config.THUMB_TITLE_COLOR))
    meta_bits = [b for b in (film_name, artist_name) if b]
    if meta_bits:
        lines.append((" • ".join(meta_bits), sub_font, Config.THUMB_SUBTITLE_COLOR))

    if not lines:
        img.convert("RGB").save(out_dir / output_filename)
        return out_dir / output_filename

    line_gap = 20
    heights = []
    for text, font, _ in lines:
        bbox = draw.textbbox((0, 0), text, font=font)
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + line_gap * (len(lines) - 1)

    y = (H - total_h) // 2
    for (text, font, color), h in zip(lines, heights):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        draw.text(
            ((W - w) / 2, y), text, font=font, fill=color,
            stroke_width=Config.THUMB_TEXT_STROKE_WIDTH, stroke_fill=Config.THUMB_TEXT_STROKE_COLOR,
        )
        y += h + line_gap

    thumb_path = out_dir / output_filename
    img.convert("RGB").save(thumb_path)
    return thumb_path


# ============================================================================
# Main pipeline
# ============================================================================

def is_adlib_line(text):
    """True if a lyrics line is wrapped in parentheses, e.g. '(excuse me)' or
    '(Aa Aa Aa Aa)' -- the existing convention for ad-libs and chorus/hum
    filler that isn't meant to be sung along to precisely. These lines get
    Config.ADLIB_COLOR instead of normal lyric colors when rendered."""
    t = text.strip()
    return len(t) >= 2 and t.startswith("(") and t.endswith(")")


def sanitize_filename_component(name):
    """Strips characters that are unsafe or awkward in filenames across
    macOS/Windows/Linux (\\ / : * ? " < > |) and collapses/trims
    whitespace, so a song title can be dropped straight into an output
    filename. Returns "" for blank/None input -- deliberately does NOT
    invent a fallback name itself, since different callers may want a
    different fallback (e.g. "Karaoke"); that's the caller's call."""
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def read_lyrics_txt(path):
    path = Path(path)
    if path.suffix.lower() == ".rtf":
        print("  Lyrics file is .rtf, converting via macOS 'textutil'...")
        try:
            result = subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", str(path)],
                check=True, capture_output=True, text=True,
            )
            raw_lines = result.stdout.splitlines()
        except Exception as e:
            raise RuntimeError(
                f"Could not auto-convert {path.name} from RTF ({e}). Run manually:\n"
                f'  textutil -convert txt "{path}"'
            )
    else:
        with open(path, encoding="utf-8") as f:
            raw_lines = f.readlines()
    return [l.strip() for l in raw_lines if l.strip()]


def main():
    print("=" * 70)
    print("  KARAOKE VIDEO GENERATOR v7 (main_v2)")
    print("=" * 70)

    video_path = prompt_path("\nEnter path to the input video file (.mp4): ")

    has_lyrics = input("\nDo you have a lyrics .txt file? (y/n): ").strip().lower().startswith("y")
    lyrics_path = None
    if has_lyrics:
        lyrics_path = prompt_path("Enter path to the lyrics .txt file (one line per lyric line): ")
    else:
        print("No lyrics file -- will attempt automatic transcription with Whisper.")

    print("\nSong info for the bottom ticker and thumbnail (leave blank to skip any):")
    song_title = prompt_text("  Song title: ")
    film_name = prompt_text("  Film name: ")
    artist_name = prompt_text("  Artist name: ")

    is_duet = prompt_yes_no(
        "\nIs this song a genuine duet with two distinct singers/voices? (y/N): ", default=False
    )
    Config.ENABLE_VOICE_COLORING = is_duet

    voice_overrides_path = None
    if is_duet:
        has_overrides = prompt_yes_no(
            "Do you have a hand-corrected voice-assignment file from a previous run's "
            "line_timing_preview.txt to apply (instead of trusting the auto pitch-classifier)? (y/N): ",
            default=False,
        )
        if has_overrides:
            voice_overrides_path = prompt_path(
                "Enter path to the corrected line_timing_preview.txt: "
            )

    out_dir = Path(Config.OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    device = get_device()
    print(f"\nUsing device: {device}")

    print("\n[1/6] Extracting audio clip...")
    clip_path, audio_path = extract_clip(video_path, out_dir, Config.TRIM_START_SEC, Config.TRIM_END_SEC)

    print("\n[2/6] Separating vocals from instrumental (this can take a while)...")
    vocals_path, instrumental_path = separate_audio(audio_path, out_dir, device)

    print("\n[3/6] Building lyric timing...")
    lines = None
    if lyrics_path:
        lyric_lines_text = read_lyrics_txt(lyrics_path)
        print(f"  Loaded {len(lyric_lines_text)} lyric line(s) from file.")
        print("  Scanning the raw (pre-separation) audio for the vocal-activity window, to "
              "anchor forced alignment and avoid drift across leading/trailing instrumental...")
        vocal_window = detect_vocal_window(audio_path)
        print("  Aligning your lyrics directly to the vocal audio (forced alignment)...")
        try:
            lines = forced_align_lyrics(vocals_path, lyric_lines_text, Config.FORCED_ALIGN_LANGUAGE, device,
                                         out_dir, vocal_window=vocal_window)
            print(f"  Forced alignment placed {len(lines)}/{len(lyric_lines_text)} line(s) on the timeline.")
            vocal_blocks = [(l["start"], l["end"]) for l in lines]
        except Exception as e:
            print(f"  Forced alignment failed ({e}); falling back to Whisper-based timing.")
            whisper_result = run_whisper(vocals_path, "cpu")
            vocal_blocks = segments_as_blocks(whisper_result)
            if vocal_blocks:
                allocation_blocks = merge_blocks_to_target_count(vocal_blocks, len(lyric_lines_text))
                lines = allocate_lines_to_blocks(lyric_lines_text, allocation_blocks)
            else:
                duration = get_audio_duration(instrumental_path)
                lines = allocate_lines_to_blocks(lyric_lines_text, [(0.0, duration)])
    else:
        print("  No lyrics file -- using Whisper's transcribed text directly.")
        whisper_result = run_whisper(vocals_path, "cpu")
        vocal_blocks = segments_as_blocks(whisper_result)
        lines = build_lines_from_whisper_text(whisper_result)

    print("  Checking for lines too wide to fit the display panel...")
    lines = split_long_lines(lines)

    print("  Checking for anomalously long word durations (e.g. an unlyriced chorus hum absorbed into a word's timing)...")
    cap_anomalous_word_durations(lines)
    flag_slow_lines(lines)

    print("\n[4/6] Voice classification for duet color separation...")
    if Config.ENABLE_VOICE_COLORING:
        try:
            classify_voices(vocals_path, lines)
        except Exception as e:
            print(f"  Voice classification failed ({e}); all lines will use MALE (single palette).")
            for l in lines:
                l["voice"] = "MALE"
    else:
        print("  Skipped (not marked as a duet) -- all lines use the single default palette.")
        for l in lines:
            l["voice"] = "MALE"

    if voice_overrides_path:
        print(f"\n  Applying hand-corrected voice assignments from {voice_overrides_path}...")
        overrides = read_voice_overrides(voice_overrides_path)
        apply_voice_overrides(lines, overrides)

    print(f"\n  Line timing preview ({len(lines)} line(s)):")
    preview_path = out_dir / "line_timing_preview.txt"
    with open(preview_path, "w", encoding="utf-8") as pf:
        for l in lines:
            voice_tag = f"[{l.get('voice','MALE')}]" if Config.ENABLE_VOICE_COLORING else ""
            row = f"  [{l['start']:7.2f} - {l['end']:7.2f}] {voice_tag:9s} " + " ".join(w["text"] for w in l["words"])
            print(row)
            pf.write(row.strip() + "\n")
    print(f"  (also saved to {preview_path})")

    duration = get_audio_duration(instrumental_path)
    ticker_text = build_ticker_text(song_title, film_name, artist_name)

    # Per-song output filenames -- stops different songs from overwriting
    # each other's kept video/thumbnail. Everything else in out_dir
    # (working files like vocals.wav, line_timing_preview.txt) is still
    # shared/overwritten per run, and re-rendering the SAME song again
    # still overwrites its own prior output -- unchanged, only these two
    # kept deliverables are now per-song-named.
    song_name_for_files = sanitize_filename_component(song_title) or "Karaoke"
    video_filename = f"{song_name_for_files} Karaoke.mp4"
    thumb_filename = f"{song_name_for_files} Thumbnail.png"

    print("\n[5/6] Rendering video...")
    final_out = render_video(lines, vocal_blocks, duration, instrumental_path, out_dir, ticker_text,
                              output_filename=video_filename)

    print("\n[6/6] Generating thumbnail...")
    thumb_out = generate_thumbnail(song_title, film_name, artist_name, out_dir, output_filename=thumb_filename)

    print("\n" + "=" * 70)
    print(f"  DONE! Video saved to:     {final_out.resolve()}")
    print(f"        Thumbnail saved to: {thumb_out.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()

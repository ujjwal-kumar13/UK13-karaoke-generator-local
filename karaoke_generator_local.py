#!/usr/bin/env python3


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

try:
    from num2words import num2words
    _NUM2WORDS_AVAILABLE = True
except ImportError:
    _NUM2WORDS_AVAILABLE = False

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
    # comfortable margin on both. (A prior trade-off note here described
    # the old floating countdown badge briefly overlapping the title text
    # during a long lead-in/instrumental gap -- no longer applicable, since
    # that badge was replaced by the in-panel gap-indicator line; see
    # Config.SHOW_GAP_INDICATOR_LINE.)
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

    # 85% opaque / 15% transparent -- confirmed against a real preview
    # rendered with the actual UKKaraoke background image at the real
    # panel size/position (the user picked this exact level: "the 85%
    # opaque gets a fair amount of artwork show through... I am fine to
    # go back to 85%"). This alpha value alone was never actually the
    # problem -- see build_frame()'s panel-background drawing for the
    # real bug (a direct ImageDraw fill doesn't blend against existing
    # pixels; it was rendering fully opaque regardless of this number).
    PANEL_BG_COLOR = (8, 6, 10, 217)
    PANEL_OUTLINE_COLOR = (200, 160, 90, 120)

    SHOW_PROGRESS_BAR = True
    PROGRESS_BAR_HEIGHT = 8
    PROGRESS_BAR_MARGIN_ABOVE_PANEL = 14
    PROGRESS_BAR_BG_COLOR = (255, 255, 255, 40)
    PROGRESS_BAR_FILL_COLOR = (255, 205, 90, 230)

    # Gap indicator: during a long instrumental/no-lyrics gap (between two
    # real lines, OR before the very first line if there's enough lead-in),
    # a synthetic "line" -- cycling music-note symbols, then a numeric
    # countdown -- is spliced into the rendered line list right where the
    # gap is. It's an ordinary line dict (see _build_gap_indicator_line()),
    # so it scrolls in/out and gets its words colored by render_video()'s
    # EXISTING per-word active/upcoming/sung machinery exactly like any
    # real lyric line -- no separate overlay/badge system with its own
    # timing to keep in sync.
    #
    # This replaces the old floating circular countdown badge (the
    # COUNTDOWN_*/COUNTDOWN_BADGE_* settings), confirmed on a real test to
    # visibly desync from the panel's own scroll -- its gap boundaries came
    # from `vocal_blocks`, which on the forced-alignment path is a
    # SNAPSHOT of the lines taken before later steps (split_long_lines(),
    # cap_anomalous_word_durations()) could still nudge those same lines'
    # timing, so the badge could disappear a fraction of a second out of
    # step with when the panel actually started scrolling. Building this
    # as a real entry in the same `lines` list removes that second data
    # source entirely -- there's only ever one timeline. Ported from the
    # YouTube branch, where this same redesign (and two follow-up
    # revisions: font-glyph filtering so unsupported symbols never render
    # as box characters, and a continuous scrolling color-mask highlight
    # in place of a translucent overlay bar that rendered fully opaque in
    # practice) was built, iterated on with the user via standalone
    # preview clips, and confirmed working on a real render.
    SHOW_GAP_INDICATOR_LINE = True

    # Only bother inserting the indicator for a gap at least this long.
    # Needs to comfortably fit: SCROLL_TRANSITION_SECONDS of real empty
    # space on EACH side (so the normal scroll transition still has room
    # to run smoothly into and out of the indicator, rather than the
    # indicator abutting its neighbor with a zero-length transition
    # window and popping instead of scrolling) plus the full countdown --
    # 2*0.6 + 4 = 5.2s at the current defaults below -- with a little
    # headroom left over for at least one music-note symbol before the
    # countdown starts.
    GAP_INDICATOR_MIN_GAP = 6.0

    # Length of the numeric countdown ("4 3 2 1") right before the next
    # line begins; each digit is shown/highlighted for 1 second.
    GAP_INDICATOR_COUNTDOWN_SECONDS = 4

    # CANDIDATE music-note symbols for whatever's left of the gap before
    # the countdown starts (the "music is still playing" portion) --
    # cycled through in this order. Not every character here is
    # guaranteed to have a glyph in whichever font FONT_CANDIDATES_BOLD
    # actually resolves to on the user's machine (a missing glyph renders
    # as a "tofu" box) -- so this is filtered at render time by
    # _resolve_gap_indicator_symbols() down to whichever of these the
    # ACTUAL loaded font can really render, in this priority order,
    # falling back to "." if none of them can. Never used directly for
    # drawing -- see insert_gap_indicator_lines().
    GAP_INDICATOR_SYMBOLS = "♫♬♪𝄞♩"
    GAP_INDICATOR_SYMBOL_INTERVAL = 1.5  # seconds each symbol stays lit
    GAP_INDICATOR_FALLBACK_SYMBOL = "."  # used if NONE of the above render

    # The gap-indicator line's highlight is a CONTINUOUSLY moving color
    # boundary, not a discrete per-word/per-character flip and not a
    # translucent overlay -- both were tried and rejected by the user on
    # the YouTube branch (per-word/per-character flips read as choppy
    # steps, not a smooth "scrolling" highlight; a semi-transparent bar
    # drawn on top ended up rendering fully opaque in practice and hid the
    # text underneath it). The text is rendered fully opaque, TWICE --
    # once entirely in GAP_LINE_UPCOMING_COLOR, once in either
    # GAP_LINE_ACTIVE_COLOR (the music-note symbols) or
    # GAP_LINE_COUNTDOWN_ACTIVE_COLOR (the "4 3 2 1" digits specifically --
    # drawn word-by-word so the countdown gets its own distinct highlight
    # color from the symbols) -- and a hard-edged mask, whose split
    # x-position slides continuously with time (see
    # draw_gap_indicator_line()), picks which of the two opaque layers
    # shows through on each side of that position. Because the split can
    # land in the middle of a single glyph's pixels, this reads as one
    # continuous sweep rather than any kind of stepped or dimmed effect --
    # and since both layers are fully opaque, nothing can ever look
    # "washed out" or obscure the text the way the overlay bar did.
    GAP_LINE_UPCOMING_COLOR = (170, 170, 180, 255)
    GAP_LINE_ACTIVE_COLOR = (90, 175, 255, 255)
    GAP_LINE_COUNTDOWN_ACTIVE_COLOR = (255, 214, 40, 255)  # bright yellow

    WHISPER_MODEL_SIZE = "medium"
    WHISPER_LANGUAGE = None
    # Pinned to 0.0 (single-pass, deterministic decoding) for the real
    # transcription pass too, not just detect_vocal_window()'s coarse
    # pass -- confirmed directly on the YouTube branch (diagnostic script
    # output diffed against an earlier run on the byte-identical audio
    # file) that leaving Whisper's default temperature-fallback enabled
    # here produces different wording on some lines run-to-run, which
    # made a real bug report ("this run messed up some lines that were
    # fine before") indistinguishable from ordinary Whisper non-
    # determinism. Known, accepted tradeoff: this gives up whatever
    # accuracy benefit the temperature-fallback retries occasionally
    # provide on a low-confidence passage, in exchange for every run on
    # the same audio producing the same transcript -- set back to None to
    # restore the old default-fallback behavior.
    WHISPER_TEMPERATURE = 0.0
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

    # detect_vocal_window()'s coarse pass sometimes turns up an isolated
    # block, at either end of the song, that's separated from the rest of
    # the song's real vocal activity by an unusually large gap -- e.g. a
    # confidently-but-wrongly transcribed instrumental passage (a sax
    # solo) that the existing hallucination/filler-text filters don't
    # catch on their own (see detect_vocal_window()'s docstring for the
    # full story). A leading or trailing block reached only via a gap
    # bigger than this many seconds is treated as untrusted and excluded
    # from the window, falling back to the next block in from that end
    # instead. Calibrated on the YouTube branch against a real song
    # ("Careless Whisper"): every genuine internal gap between real blocks
    # measured <=15.16s, while the confirmed false-positive trailing block
    # (the sax outro, picked up ~24s after real singing ended) was
    # separated by a 23.72s gap -- 20.0s sits with comfortable margin
    # inside that ~8.5s window between the two.
    VOCAL_WINDOW_MAX_EDGE_GAP = 20.0

    # A bare 4-digit numeral in this range is spelled out as a year
    # ("1969" -> "nineteen sixty-nine") rather than a plain cardinal count
    # ("one thousand nine hundred sixty-nine") before the aligner's vocab
    # check -- see _numeral_to_words(). Numbers outside this range (or not
    # 4 digits) are read as plain cardinals.
    NUMERAL_YEAR_RANGE = (1000, 2999)

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


_ORDINAL_SUFFIX_RE = re.compile(r"^(\d+)(st|nd|rd|th)$", re.IGNORECASE)


def _split_edge_punct(word):
    """Splits a word into (leading_punct, core, trailing_punct), where
    leading/trailing punct is any run of non-alphanumeric characters at
    the very start/end of the word only -- internal punctuation is left
    inside `core` untouched. E.g. "8," -> ("", "8", ","), "(mm-mm)" ->
    ("(", "mm-mm", ")"). Used to find the actual numeral inside a word
    without disturbing surrounding punctuation."""
    i = 0
    while i < len(word) and not word[i].isalnum():
        i += 1
    j = len(word)
    while j > i and not word[j - 1].isalnum():
        j -= 1
    return word[:i], word[i:j], word[j:]


def _line_has_bare_numeral(text):
    """True if `text` contains at least one word that
    _convert_numerals_for_alignment() would actually convert (a pure
    numeral, or a numeral with an ordinal suffix) -- used only to decide
    whether it's worth warning that num2words isn't installed (see
    prepare_alignment_text()). Cheap, no dependency on num2words itself,
    safe to call regardless of _NUM2WORDS_AVAILABLE."""
    for word in text.split(" "):
        if not word:
            continue
        _, core, _ = _split_edge_punct(word)
        if _ORDINAL_SUFFIX_RE.match(core) or core.isdigit():
            return True
    return False


def _numeral_to_words(digits, is_ordinal):
    """Spells out a bare digit string for the aligner. 4-digit numbers in
    Config.NUMERAL_YEAR_RANGE are read as years ("1969" -> "nineteen
    sixty-nine") rather than a plain cardinal count, matching the
    guidance already baked into the vocab pre-flight's own error
    message; everything else is a plain cardinal or ordinal reading."""
    n = int(digits)
    if is_ordinal:
        return num2words(n, to="ordinal")
    lo, hi = Config.NUMERAL_YEAR_RANGE
    if len(digits) == 4 and lo <= n <= hi:
        return num2words(n, to="year")
    return num2words(n)


def _convert_numerals_for_alignment(text):
    """Spells out bare numeral words (e.g. "8" -> "eight", "85" ->
    "eighty-five", "1969" -> "nineteen sixty-nine", "21st" ->
    "twenty-first") for the SAME reason _strip_alignment_noise_chars()
    exists: a bare digit sequence typically has no romanized form at all,
    which fails the aligner's vocab pre-flight check and aborts forced
    alignment for the WHOLE song, falling back to the much less accurate
    proportional-timing method -- confirmed directly against two real
    failing songs on the local-batch branch ("Eldest Daughter":
    word='8'/word='9'; "Ruin The Friendship": word='85'/word='50'), both
    of which hit exactly this and nothing else (their lyrics also
    contained parentheses and straight apostrophes, neither of which was
    flagged -- so this fix is scoped to the failure actually observed,
    not a guess). This branch shares the identical forced_align_lyrics()
    vocab-check code, so it's equally exposed.

    Only whole numeral tokens are converted: a word that, once leading/
    trailing punctuation is stripped, is either pure digits or digits
    plus an ordinal suffix (21st, 3rd). A word that mixes letters and
    digits some other way (e.g. "24/7", a stylized word) is left
    untouched -- there's no safe generic way to guess how to romanize
    that, and it isn't the failure mode observed.

    IMPORTANT DISPLAY NOTE: this conversion (and the existing quote-
    stripping) DOES change what's shown on screen for that word once
    forced alignment succeeds -- the per-word captions render_video()
    draws come from the aligner's own word output (built from this
    cleaned text), not from your original raw lyrics line. So a line
    like "I must've been about 8 or 9" will display as "... about eight
    or nine" once real alignment runs, not "8 or 9" as typed. Confirmed
    by reading ctc_forced_aligner's own preprocess_text()/
    postprocess_results() source.

    Silently returns text unchanged if the num2words package isn't
    installed (see _NUM2WORDS_AVAILABLE) -- this must never be able to
    crash the pipeline outright over a missing optional dependency.
    """
    if not _NUM2WORDS_AVAILABLE:
        return text

    out_words = []
    for word in text.split(" "):
        if not word:
            continue
        lead, core, trail = _split_edge_punct(word)
        ordinal_m = _ORDINAL_SUFFIX_RE.match(core)
        if ordinal_m:
            spelled = _numeral_to_words(ordinal_m.group(1), is_ordinal=True)
        elif core.isdigit():
            spelled = _numeral_to_words(core, is_ordinal=False)
        else:
            out_words.append(word)
            continue
        out_words.append(f"{lead}{spelled}{trail}")
    return " ".join(out_words)


def _clean_line_for_alignment(line_text):
    """Returns the version of one lyrics line actually sent to the
    aligner's vocab check/alignment call: quote characters stripped
    (_strip_alignment_noise_chars), then bare numerals spelled out
    (_convert_numerals_for_alignment), with whitespace normalized. The
    original, untouched line_text is still what's used for the LINE-level
    "text" field (is_adlib detection, the whole-line preview) -- see
    prepare_alignment_text()'s docstring for what this does and does not
    affect."""
    cleaned = _strip_alignment_noise_chars(line_text)
    cleaned = _convert_numerals_for_alignment(cleaned)
    return " ".join(cleaned.split())


def prepare_alignment_text(lyric_lines_text, language):
    """Builds the cleaned per-line text sent to the forced aligner and
    checks it against the aligner's vocabulary, raising the same
    ValueError (with the same fix-it guidance) forced_align_lyrics() has
    always given if anything still fails. Called from two places: early,
    right after the lyrics file loads -- purely so a lyrics-text problem
    surfaces in seconds, before Demucs separation and a coarse Whisper
    pass run for nothing -- and again inside forced_align_lyrics()
    itself, right before the real alignment call, which is the one that
    actually matters for correctness. Calling it twice is cheap (text
    cleaning plus one vocabulary lookup, not a model inference) and keeps
    both call sites guaranteed to agree, rather than duplicating the
    cleaning logic in two places that could drift apart.

    Returns (alignment_lyric_lines, full_text). Prints a one-line note
    when any line was actually changed by cleaning, so it's clear from
    the console that this ran and what it affected -- including the
    display-text caveat from _convert_numerals_for_alignment()'s
    docstring, so it isn't a surprise later.

    If num2words isn't importable, numeral conversion silently does
    nothing (see _convert_numerals_for_alignment) -- that was confirmed
    as a real, previously-undetected failure mode on the local-batch
    branch (a re-run showed the exact same vocab failures as before the
    fix, with no adjustment note anywhere in the log, because the
    package was never installed in that environment). So this checks
    explicitly and prints a loud, impossible-to-miss warning naming the
    missing package, instead of quietly doing nothing.
    """
    if not _NUM2WORDS_AVAILABLE and any(_line_has_bare_numeral(lt) for lt in lyric_lines_text):
        print("  " + "!" * 68)
        print("  WARNING: this lyrics file contains bare numeral(s) (e.g. \"8\", \"85\") and the "
              "'num2words' package is NOT installed in this Python environment -- numerals will "
              "NOT be spelled out, and the vocab pre-flight check below will very likely fail "
              "on them, falling back to the less accurate Whisper-based timing method. Fix: run "
              "'pip install num2words' in the SAME environment you run this script with (the "
              "one whose 'python' this is), then re-run.")
        print("  " + "!" * 68)

    alignment_lyric_lines = [_clean_line_for_alignment(lt) for lt in lyric_lines_text]
    changed_count = sum(
        1 for orig, clean in zip(lyric_lines_text, alignment_lyric_lines)
        if " ".join(orig.split()) != clean
    )
    if changed_count:
        print(f"  Note: adjusted {changed_count} lyrics line(s) for the aligner (quote "
              f"character(s) removed and/or bare numeral(s) spelled out, e.g. \"8\" -> "
              f"\"eight\") -- an adjusted word's on-screen caption will show the adjusted "
              f"wording once alignment succeeds, not your original digits/quotes.")

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
    return alignment_lyric_lines, full_text


def forced_align_lyrics(vocals_path, lyric_lines_text, language, device, out_dir=None, vocal_window=None):
    import torch
    from ctc_forced_aligner import (
        load_audio, load_alignment_model, generate_emissions,
        preprocess_text, get_alignments, get_spans, postprocess_results,
    )

    # Build the cleaned copy of the lyrics used ONLY for the vocab check
    # and the actual alignment call -- quote characters stripped and bare
    # numerals spelled out (see prepare_alignment_text() and its helpers
    # for the full rationale; this was already run once earlier, right
    # after the lyrics file loaded, purely to surface a problem before
    # audio processing started -- re-run here since this is the call that
    # actually matters for correctness). The ORIGINAL lyric_lines_text is
    # still used below for every line's LINE-level "text" field and its
    # ad-lib-parens check -- unaffected. Per-line word counts are computed
    # from the SAME cleaned text used for alignment (not the original), so
    # a line whose word count changed during cleaning (a stray quote
    # "word" removed, or a numeral expanding into multiple words) doesn't
    # desync downstream line-to-word slicing.
    alignment_lyric_lines, full_text = prepare_alignment_text(lyric_lines_text, language)

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


def _trim_isolated_edge_blocks(blocks):
    """Drops a leading and/or trailing block from `blocks` (a list of
    (start, end) tuples, in time order) if it's separated from its
    neighbor by a gap bigger than Config.VOCAL_WINDOW_MAX_EDGE_GAP --
    repeats inward from each end until a within-threshold block is
    reached (or only one block remains). Only ever trims from the two
    EDGES, never the interior -- a large gap in the middle of a song is
    a normal instrumental break (verse/bridge silence), not a detection
    artifact, and is left alone; only an edge block that the rest of the
    song's activity doesn't lead into is treated as suspect.

    Ported from the YouTube branch after detect_vocal_window()'s coarse
    pass confirmed, on a real song, that it can turn up an isolated block
    at the tail of the song separated from all real singing by an
    unusually large gap -- see Config.VOCAL_WINDOW_MAX_EDGE_GAP's
    docstring for the real numbers this was calibrated on. Applied
    symmetrically to the leading edge too, as a precaution, even though
    no real case of a leading false-positive has been observed yet.

    Returns (trimmed_blocks, leading_count, trailing_count) -- the
    counts are purely for the caller's own logging.
    """
    trimmed = list(blocks)
    trailing_count = 0
    while len(trimmed) > 1 and (trimmed[-1][0] - trimmed[-2][1]) > Config.VOCAL_WINDOW_MAX_EDGE_GAP:
        trimmed.pop()
        trailing_count += 1
    leading_count = 0
    while len(trimmed) > 1 and (trimmed[1][0] - trimmed[0][1]) > Config.VOCAL_WINDOW_MAX_EDGE_GAP:
        trimmed.pop(0)
        leading_count += 1
    return trimmed, leading_count, trailing_count


def detect_vocal_window(raw_audio_path, pad_before=2.0, pad_after=2.0):
    """Runs a fast, cheap Whisper pass on the RAW (pre-vocal-separation)
    audio purely to find a rough "where is there likely vocal activity"
    window -- NOT for word-level transcription accuracy, that's what
    forced-alignment against the user's own lyrics text is for.

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

    Before computing the window, also runs _trim_isolated_edge_blocks()
    on the detected blocks: an edge block reached only via a gap bigger
    than Config.VOCAL_WINDOW_MAX_EDGE_GAP from the rest of the song's
    activity is excluded, since real singing doesn't lead into it -- see
    that function's own docstring for the full calibration details.

    Returns (onset, offset) in seconds -- the first and last SURVIVING
    (post-trim) activity block's timestamps, padded outward by
    pad_before/pad_after (clamped to [0, audio duration]) so the
    forced-aligner still gets a little headroom rather than a razor-exact
    cut. Returns None if no vocal activity was detected at all, or if
    anything about this detection step fails -- callers should treat
    None as "skip windowing entirely, run forced-alignment against the
    full audio as before," never as license to guess a window.
    """
    try:
        # temperature=0.0 forces single-pass, deterministic decoding for
        # this coarse pass -- Whisper's own default temperature-fallback
        # (randomly-sampled retries when its confidence checks fail) was
        # observed, directly in testing, to make this detection non-
        # deterministic run-to-run on identical audio. This only affects
        # this detection call -- run_whisper()'s default behavior (used
        # everywhere else, including the real transcription passes) is
        # unchanged.
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

        blocks, trimmed_leading, trimmed_trailing = _trim_isolated_edge_blocks(blocks)
        if trimmed_leading or trimmed_trailing:
            which = []
            if trimmed_leading:
                which.append(f"{trimmed_leading} leading")
            if trimmed_trailing:
                which.append(f"{trimmed_trailing} trailing")
            print(f"  Excluded {' and '.join(which)} isolated block(s) from the vocal-activity "
                  f"window (each reached only via a gap > {Config.VOCAL_WINDOW_MAX_EDGE_GAP:.0f}s "
                  f"from the rest of the song's detected activity -- likely a confidently-"
                  f"mistranscribed instrumental passage, e.g. a sax solo, rather than real singing).")

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



#: How many consecutive lines must share the exact same (normalized) text
#: before they're treated as a Whisper hallucination run rather than a
#: coincidence or a real repeated lyric. Deliberately conservative: a real
#: repeated chorus line in an actual song is either broken up by at least
#: one different line, or lands close enough together to already have been
#: merged into a single block by Config.MERGE_SEGMENT_GAP -- so a bare run
#: of this many IDENTICAL consecutive lines, each its own separate block,
#: is a strong signal, not a normal songwriting pattern. Calibrated on the
#: YouTube branch against a real no-lyrics-file run ("Careless Whisper"):
#: 4 consecutive lines all transcribed as the identical 3 words, landing
#: exactly in the song's ~20s instrumental (saxophone) outro -- a stretch
#: the existing cap_anomalous_word_durations()/flag_slow_lines() warnings
#: had already independently flagged as suspicious on the same run.
MIN_HALLUCINATION_REPEAT_RUN = 3

#: A single very-short trailing line (this many words or fewer) landing
#: immediately after a dropped repeat-run is treated as the tail of the
#: same hallucination burst (e.g. a lone "Ah") rather than a coincidence,
#: provided it starts close enough to the run's end (see
#: _drop_hallucinated_repeats()'s MERGE_SEGMENT_GAP-based gap check).
_HALLUCINATION_TAIL_MAX_WORDS = 2


def _normalize_line_text(text):
    """Lowercases and strips punctuation/whitespace so two transcriptions
    of "the same" phrase (differing only in case or a stray comma) still
    compare as equal -- used only by the repeat-hallucination detector
    below, never for anything that touches on-screen text."""
    return re.sub(r"[^\w\s]", "", text).strip().lower()


def _drop_hallucinated_repeats(lines):
    """Two-pass hallucination-repeat removal. Only ever called on Whisper-
    transcribed text (the no-lyrics-file path) -- when a lyrics file is
    used, on-screen text comes from the user's own lyrics, not Whisper, so
    this kind of fabricated-text hallucination can't occur there; nothing
    calls this function on that path. Ported from the YouTube branch.

    Pass 1 (detection): finds every run of MIN_HALLUCINATION_REPEAT_RUN+
    CONSECUTIVE lines whose normalized text is identical -- see that
    constant's docstring for why this specific pattern is treated as a
    Whisper hallucination fingerprint. Each such run's text becomes a
    "confirmed hallucination phrase" for this song.

    Pass 2 (removal): drops every line anywhere in the song whose
    normalized text matches a confirmed phrase -- not just the original
    tight run. Confirmed necessary on the YouTube branch's real run: the
    same fabricated phrase also showed up twice more in isolation (each
    with big gaps on both sides, so neither formed its own run of 3+) and
    both survived a single-pass version of this function untouched,
    leaving stray fake captions -- and the countdown/gap indicator
    re-triggering around them -- in a stretch that was actually just
    instrumental. Once a phrase is proven fabricated once in a song,
    there's no reason to trust a lone repeat of the exact same phrase
    elsewhere in the same song; a real lyric essentially never recurs as
    a byte-identical fragment in complete isolation like that.

    A single very short (<= _HALLUCINATION_TAIL_MAX_WORDS-word) line
    landing immediately after any dropped stretch is swallowed too, on
    the assumption it's the tail of the same burst (e.g. a lone short
    word right after a run of fabricated lines). Every drop is printed as
    its own WARNING with the exact timestamps and text involved, so this
    is never a silent change; if it ever fires on what was actually a
    real repeated lyric, that's visible immediately from the console
    output (and this function can be disabled per-run by commenting out
    its call site)."""
    if not lines:
        return lines
    n = len(lines)

    # Pass 1: detect confirmed hallucination phrases via tight repeat runs.
    confirmed_phrases = set()
    originating_starts = set()
    i = 0
    while i < n:
        norm_i = _normalize_line_text(lines[i]["text"])
        j = i
        while j + 1 < n and norm_i and _normalize_line_text(lines[j + 1]["text"]) == norm_i:
            j += 1
        if norm_i and (j - i + 1) >= MIN_HALLUCINATION_REPEAT_RUN:
            confirmed_phrases.add(norm_i)
            originating_starts.add(i)
        i = j + 1

    if not confirmed_phrases:
        return lines

    # Pass 2: drop every stretch (anywhere in the song) whose normalized
    # text matches a confirmed phrase, plus a close short trailing line.
    kept = []
    i = 0
    while i < n:
        norm_i = _normalize_line_text(lines[i]["text"])
        if norm_i in confirmed_phrases:
            is_original = i in originating_starts
            j = i
            while j + 1 < n and _normalize_line_text(lines[j + 1]["text"]) == norm_i:
                j += 1
            drop_end = j
            if drop_end + 1 < n:
                tail = lines[drop_end + 1]
                gap = tail["start"] - lines[drop_end]["end"]
                if len(tail["words"]) <= _HALLUCINATION_TAIL_MAX_WORDS and gap < Config.MERGE_SEGMENT_GAP * 3:
                    drop_end += 1
            dropped = lines[i:drop_end + 1]
            reason = (
                f"{j - i + 1} consecutive line(s) transcribed as the identical text {lines[i]['text']!r}, "
                f"a classic ASR hallucination pattern on long instrumental passages"
                if is_original else
                f"text {lines[i]['text']!r} was already confirmed as a hallucination pattern elsewhere "
                f"in this song, so this separate, isolated occurrence is being treated the same way"
            )
            print(f"  WARNING: dropped {len(dropped)} line(s) [{dropped[0]['start']:7.2f} - "
                  f"{dropped[-1]['end']:7.2f}] as likely Whisper hallucination -- {reason}. Spot-check "
                  f"this stretch of the rendered video -- if this was actually a real repeated lyric, "
                  f"it will be missing and needs to be added back by hand for this song.")
            i = drop_end + 1
            continue
        kept.append(lines[i])
        i += 1
    return kept


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


def _font_supports_char(font, ch):
    """True if `font` (a PIL ImageFont.FreeTypeFont) has a real glyph for
    `ch` -- not just the font's own "missing glyph" placeholder (a tofu
    box), which PIL/FreeType silently draws for any codepoint the font
    doesn't map, with no error. There's no direct "does this font have a
    glyph for X" call in PIL, so this renders `ch` and compares it
    against the render of a Private Use Area codepoint (U+E100) that's
    virtually guaranteed to be unmapped in any real-world font -- which
    reliably forces that SAME font's own placeholder glyph. An identical
    (or blank) bitmap means `ch` is ALSO just falling back to the
    placeholder, i.e. not actually supported. Used only to filter
    Config.GAP_INDICATOR_SYMBOLS down to characters that will actually
    render correctly. Ported from the YouTube branch.
    """
    from PIL import Image, ImageDraw
    def render_mask(c):
        img = Image.new("L", (120, 120), 0)
        d = ImageDraw.Draw(img)
        d.text((10, 10), c, font=font, fill=255)
        return img.tobytes()
    try:
        placeholder = render_mask("")
        candidate = render_mask(ch)
        if candidate == placeholder:
            return False
        if not any(candidate):
            return False
        return True
    except Exception:
        return False


def _resolve_gap_indicator_symbols(font):
    """Filters Config.GAP_INDICATOR_SYMBOLS down to whichever characters
    the ACTUALLY-loaded bold font can really render (see
    _font_supports_char()), preserving priority order. Falls back to
    Config.GAP_INDICATOR_FALLBACK_SYMBOL ("." by default) if none of the
    candidates render. Prints which symbols will be used (or that it
    fell back), once, so this automatic decision stays visible in the
    console like the pipeline's other auto-fallback behavior."""
    supported = [c for c in Config.GAP_INDICATOR_SYMBOLS if _font_supports_char(font, c)]
    if supported:
        print(f"  Gap-indicator symbols: using {' '.join(supported)!r} "
              f"(font supports {len(supported)}/{len(Config.GAP_INDICATOR_SYMBOLS)} candidates).")
        return "".join(supported)
    print(f"  Gap-indicator symbols: none of {Config.GAP_INDICATOR_SYMBOLS!r} render in this font; "
          f"falling back to {Config.GAP_INDICATOR_FALLBACK_SYMBOL!r}.")
    return Config.GAP_INDICATOR_FALLBACK_SYMBOL


def _build_gap_indicator_line(gap_start, gap_end, symbols=None):
    """Builds one synthetic "line" -- cycling music-note symbols followed
    by a numeric countdown -- to fill a long gap between two real lyric
    lines (or before the very first line). Returned as an ordinary line
    dict (start/end/words/text/is_adlib=False), just marked with
    is_gap_indicator=True, so render_video() scrolls it through the same
    code path as any real line, but draws it with the dedicated
    draw_gap_indicator_line() (a continuously scrolling color-mask
    highlight, not per-word color flips and not an overlay bar) -- see
    that function's docstring for why. Ported from the YouTube branch.

    `symbols` should be the ALREADY-FONT-FILTERED string from
    _resolve_gap_indicator_symbols() -- defaults to the raw, unfiltered
    Config.GAP_INDICATOR_SYMBOLS only for standalone/test use; real
    callers (insert_gap_indicator_lines()) always pass the resolved set.

    Leaves Config.SCROLL_TRANSITION_SECONDS of genuinely empty time at
    BOTH ends (content starts that long after gap_start, ends that long
    before gap_end) so the normal scroll-transition animation has the
    same room to run smoothly into and out of this line that it has
    between any two real lines. Without that buffer, this line would sit
    flush against its neighbor with a zero-length transition window and
    pop instead of scroll -- scroll_center_index()'s transition math
    only produces a nonzero window when there's real dead time between
    lines to run it in.

    Returns None if, after reserving those buffers and the countdown,
    there's no usable time left (shouldn't happen given
    Config.GAP_INDICATOR_MIN_GAP's margin, but this is a synthetic/
    decorative line, not real transcript content, so it's safe to just
    skip rather than force degenerate near-zero-length words).
    """
    content_start = gap_start + Config.SCROLL_TRANSITION_SECONDS
    content_end = gap_end - Config.SCROLL_TRANSITION_SECONDS
    if content_end <= content_start:
        return None

    countdown_secs = int(Config.GAP_INDICATOR_COUNTDOWN_SECONDS)
    countdown_start = max(content_start, content_end - countdown_secs)

    words = []
    chars = symbols if symbols else Config.GAP_INDICATOR_SYMBOLS
    t = content_start
    i = 0
    while t < countdown_start - 1e-6:
        seg_end = min(t + Config.GAP_INDICATOR_SYMBOL_INTERVAL, countdown_start)
        words.append({"text": chars[i % len(chars)], "start": round(t, 3), "end": round(seg_end, 3),
                       "is_countdown": False})
        t = seg_end
        i += 1

    for k in range(countdown_secs, 0, -1):
        w_start = max(countdown_start, content_end - k)
        w_end = min(content_end - k + 1, content_end)
        if w_end <= w_start:
            continue
        # is_countdown=True -- draw_gap_indicator_line() highlights these
        # words in Config.GAP_LINE_COUNTDOWN_ACTIVE_COLOR (bright yellow)
        # instead of GAP_LINE_ACTIVE_COLOR (blue), so the countdown reads
        # as visually distinct from the music-note lead-in.
        words.append({"text": str(k), "start": round(w_start, 3), "end": round(w_end, 3),
                       "is_countdown": True})

    if not words:
        return None

    return {
        "start": words[0]["start"], "end": words[-1]["end"], "words": words,
        "text": " ".join(w["text"] for w in words), "is_adlib": False,
        "is_gap_indicator": True,
    }


def insert_gap_indicator_lines(lines):
    """Returns a NEW list with synthetic gap-indicator lines (see
    _build_gap_indicator_line()) spliced in wherever a real gap is long
    enough (Config.GAP_INDICATOR_MIN_GAP) -- between two consecutive real
    lines, and before the very first line if there's enough lead-in time.
    Ported from the YouTube branch.

    Call this ONLY right before render_video(), on the final, fully-
    processed `lines` list -- after split_long_lines(),
    cap_anomalous_word_durations(), voice classification, and any voice-
    override file have already run, and after line_timing_preview.txt has
    already been written from the real (unmodified) list. These synthetic
    entries are purely a rendering decoration and must never reach the
    preview file, voice classification, or anything else that treats
    `lines` as the real transcript/alignment.
    """
    if not Config.SHOW_GAP_INDICATOR_LINE or not lines:
        return lines

    # Resolve which candidate symbols this run's font can actually render
    # ONCE per call, using the same font/size render_video() will draw
    # gap-indicator lines with (Config.FONT_CANDIDATES_BOLD /
    # Config.LYRIC_FONT_SIZE) -- so the text baked into each synthetic
    # line's "words" here matches what actually shows up on screen, and
    # box-character placeholders never get generated in the first place.
    gap_font = load_font(Config.FONT_CANDIDATES_BOLD, Config.LYRIC_FONT_SIZE)
    resolved_symbols = _resolve_gap_indicator_symbols(gap_font)

    result = []
    if lines[0]["start"] >= Config.GAP_INDICATOR_MIN_GAP:
        indicator = _build_gap_indicator_line(0.0, lines[0]["start"], symbols=resolved_symbols)
        if indicator:
            result.append(indicator)

    for i, line in enumerate(lines):
        result.append(line)
        if i + 1 < len(lines):
            gap = lines[i + 1]["start"] - line["end"]
            if gap >= Config.GAP_INDICATOR_MIN_GAP:
                indicator = _build_gap_indicator_line(line["end"], lines[i + 1]["start"], symbols=resolved_symbols)
                if indicator:
                    result.append(indicator)
    return result


def render_video(lines, vocal_blocks, duration, instrumental_path, out_dir, ticker_text,
                  output_filename="karaoke_final.mp4"):
    from PIL import Image, ImageDraw

    lyric_font = load_font(Config.FONT_CANDIDATES_BOLD, Config.LYRIC_FONT_SIZE)
    ticker_font = load_font(Config.FONT_CANDIDATES_REGULAR, Config.TICKER_FONT_SIZE)

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

    # row vertical centers within the panel layer, for N_VIS visible rows
    # plus 1 extra (off the bottom) that slides into view during transitions
    top_margin = (PANEL_H - N_VIS * SLOT) / 2

    def row_center(i):
        return top_margin + i * SLOT + SLOT / 2

    TICKER_Y0, TICKER_Y1 = Config.OUT_H - 90, Config.OUT_H - 43
    dummy = Image.new("RGB", (10, 10))
    ticker_bbox = ImageDraw.Draw(dummy).textbbox((0, 0), ticker_text, font=ticker_font)
    ticker_text_w = ticker_bbox[2] - ticker_bbox[0]

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
        """Returns (upcoming_color, active_color) for a REAL lyric line
        (never called for the synthetic gap-indicator line -- that has
        its own dedicated draw_gap_indicator_line(), see below). Ad-lib/
        chorus filler lines (Config.ADLIB_COLOR) take priority over
        everything else -- they're never voice- or solo-colored, so they
        read as filler at a glance regardless of ENABLE_VOICE_COLORING.
        Otherwise: for duets, both states are colored by voice (Male=
        blue, Female=dark pink) so the singer knows who's up before the
        line starts; for solo songs, upcoming stays the neutral dim
        gray."""
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

    def draw_gap_indicator_line(panel_image, y_center, line, t, font, x_center=None):
        """Draws the synthetic gap-indicator line as a CONTINUOUSLY
        scrolling color highlight -- ported from the YouTube branch, where
        this was confirmed against standalone demo clips the user
        reviewed and approved before it was ported in (two earlier
        attempts were rejected there: discrete per-word/per-character
        color flips read as choppy steps rather than a smooth scroll, and
        a semi-transparent overlay bar drawn on top of static text ended
        up rendering fully opaque in practice and hid the numbers
        underneath it).

        Technique: render the line's full text TWICE, fully opaque, once
        entirely in Config.GAP_LINE_UPCOMING_COLOR and once entirely in
        Config.GAP_LINE_ACTIVE_COLOR, onto two same-size transparent
        layers. Compute a sweep x-position for time t that moves
        CONTINUOUSLY (interpolating smoothly within whichever word --
        symbol or countdown digit -- is current, not jumping in one step
        per word), build a hard left/right mask at that x position, and
        Image.composite() the two text layers through it: active-colored
        text shows through left of the sweep, upcoming-colored text
        shows through right of it. Because the mask's split point can
        land in the middle of a single glyph's pixels and moves a
        fraction of a pixel every frame, this reads as one continuous
        sweep rather than a stepped flip -- and because BOTH layers are
        fully opaque text (never a translucent fill on top of anything),
        nothing can look washed-out or obscure the text the way the
        overlay-bar version did.

        Deliberately a separate function from draw_line(), not a branch
        inside it, so real lyric lines' rendering is completely
        unaffected. Takes the panel Image directly (not an ImageDraw)
        since it needs to alpha_composite two intermediate layers onto
        it, unlike draw_line()'s single draw.text() calls.
        """
        if x_center is None:
            x_center = PANEL_CX
        words = line["words"]
        txt = " ".join(w["text"] for w in words)
        measure_draw = ImageDraw.Draw(panel_image, "RGBA")
        bbox = measure_draw.textbbox((0, 0), txt, font=font)
        total_w = bbox[2] - bbox[0]
        x0 = x_center - total_w / 2
        y = y_center - (bbox[3] - bbox[1]) / 2

        # Measure each word's x position/width (same advance logic as
        # draw_line()) -- these boundaries only drive the CONTINUOUS
        # interpolation below, they're never used as discrete steps.
        word_x, word_w = [], []
        x = x0
        for w in words:
            word_txt = w["text"] + " "
            wbbox = measure_draw.textbbox((0, 0), word_txt, font=font)
            w_width = wbbox[2] - wbbox[0]
            word_x.append(x)
            word_w.append(w_width)
            x += w_width

        # Continuous sweep position -- interpolates smoothly within
        # whichever word is current at time t.
        sweep_x = x0
        if t >= words[-1]["end"]:
            sweep_x = word_x[-1] + word_w[-1]
        elif t > words[0]["start"]:
            for i, w in enumerate(words):
                if w["start"] <= t <= w["end"]:
                    local = (t - w["start"]) / max(w["end"] - w["start"], 1e-6)
                    sweep_x = word_x[i] + word_w[i] * local
                    break
                if t < w["start"]:
                    sweep_x = word_x[i]
                    break

        size = panel_image.size
        upcoming_layer = Image.new("RGBA", size, (0, 0, 0, 0))
        active_layer = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(upcoming_layer).text((x0, y), txt, font=font, fill=Config.GAP_LINE_UPCOMING_COLOR)
        # Active layer drawn WORD-BY-WORD (not as one string like the
        # upcoming layer) so the countdown digits ("4 3 2 1") get their
        # own highlight color (Config.GAP_LINE_COUNTDOWN_ACTIVE_COLOR,
        # bright yellow) distinct from the music-note symbols
        # (Config.GAP_LINE_ACTIVE_COLOR, blue) -- per the user's request.
        # The single continuous sweep mask below still reveals this layer
        # at whatever x-position time t has reached, so the color simply
        # changes over to yellow once the sweep crosses into the
        # countdown words, with no change to the sweep mechanism itself.
        active_draw = ImageDraw.Draw(active_layer)
        for i, w in enumerate(words):
            color = Config.GAP_LINE_COUNTDOWN_ACTIVE_COLOR if w.get("is_countdown") else Config.GAP_LINE_ACTIVE_COLOR
            active_draw.text((word_x[i], y), w["text"] + " ", font=font, fill=color)

        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle([0, 0, max(0, int(round(sweep_x))), size[1]], fill=255)

        combined = Image.composite(active_layer, upcoming_layer, mask)
        panel_image.alpha_composite(combined)

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

        px0 = PANEL_CX - PANEL_W // 2
        py0 = PANEL_CY - PANEL_H // 2
        # Drawn on its OWN transparent layer and alpha-composited onto
        # `frame`, rather than filled directly via ImageDraw on `frame`
        # itself -- confirmed directly (filling a semi-transparent color
        # over a solid test background left the raw fill color untouched,
        # not a blend) that Pillow's ImageDraw, even in "RGBA" mode, does
        # NOT alpha-blend a shape fill against an image's existing pixels;
        # it overwrites them outright. Since `frame` is later flattened to
        # RGB for the video (frame.convert("RGB") at the end of this
        # function drops the alpha channel rather than applying it), that
        # meant PANEL_BG_COLOR's alpha was never actually visible in the
        # rendered video no matter what it was set to -- the panel always
        # rendered fully opaque. Ported from the YouTube branch, where
        # this was the real root cause of the "opaque black box" the user
        # reported; alpha_composite() is the same technique already used
        # correctly for the lyric-panel content layer just below and for
        # the gap-indicator's own two-color layers (see
        # draw_gap_indicator_line()).
        panel_bg_layer = Image.new("RGBA", (PANEL_W, PANEL_H), (0, 0, 0, 0))
        ImageDraw.Draw(panel_bg_layer, "RGBA").rounded_rectangle(
            [0, 0, PANEL_W, PANEL_H], radius=24,
            fill=Config.PANEL_BG_COLOR, outline=Config.PANEL_OUTLINE_COLOR, width=2,
        )
        frame.alpha_composite(panel_bg_layer, (px0, py0))

        if lines:
            ci = scroll_center_index(t)
            base = int(np.floor(ci))
            frac = ci - base
            panel_layer = Image.new("RGBA", (PANEL_W, PANEL_H), (0, 0, 0, 0))
            panel_draw = ImageDraw.Draw(panel_layer, "RGBA")
            # N_VIS visible rows + 1 extra row sliding in from the bottom
            # during a transition, so no blank row ever appears mid-scroll.
            # A long gap between two real lines (or before the first one)
            # is represented by a synthetic gap-indicator line already
            # spliced into `lines` itself (see insert_gap_indicator_lines()
            # / Config.SHOW_GAP_INDICATOR_LINE) -- it scrolls in and out
            # at this same row position using the same transition math as
            # everything else (no separate overlay or badge to keep in
            # sync), dispatched below to draw_gap_indicator_line() for its
            # own continuous-scrolling-highlight visual style (takes the
            # panel Image itself, not panel_draw, since it composites
            # layers rather than issuing single draw.text() calls).
            for i in range(N_VIS + 1):
                li = base + i
                if 0 <= li < len(lines):
                    y = row_center(i) - frac * SLOT
                    if lines[li].get("is_gap_indicator"):
                        draw_gap_indicator_line(panel_layer, y, lines[li], t, lyric_font, x_center=PANEL_W // 2)
                    else:
                        draw_line(panel_draw, y, lines[li], t, lyric_font, x_center=PANEL_W // 2)
            frame.alpha_composite(panel_layer, (px0, py0))
            draw = ImageDraw.Draw(frame, "RGBA")

        draw_legend(draw)

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
    # Printed unconditionally, right at startup -- so it's impossible to
    # miss whether the 2026-08-25 numeral-spelling fix is actually active
    # this run, rather than only discovering it's not (silently) deep in
    # a lyrics file's vocab pre-flight failure.
    if _NUM2WORDS_AVAILABLE:
        print("  num2words: available (bare numerals in lyrics will be spelled out for alignment)")
    else:
        print("  num2words: NOT INSTALLED in this Python environment -- bare numerals ('8', '85', "
              "etc.) in lyrics files will NOT be spelled out and will likely fail the vocab "
              "pre-flight check. Run 'pip install num2words' in this same environment to fix.")

    video_path = prompt_path("\nEnter path to the input video file (.mp4): ")

    has_lyrics = input("\nDo you have a lyrics .txt file? (y/n): ").strip().lower().startswith("y")
    # Lyrics are loaded and vocab-pre-checked here, right at setup time
    # (added 2026-08-25) -- previously this only happened deep inside
    # forced_align_lyrics(), after Demucs separation and a coarse Whisper
    # pass had already run. Checking now costs a few seconds (text
    # cleaning + one vocabulary lookup, no audio/model work) and surfaces
    # a lyrics-text problem immediately, before you walk away from a run
    # that's going to fail anyway. This does NOT abort -- forced_align_
    # lyrics() still runs its own copy of this same check later and, if
    # the problem wasn't fixed, falls back to Whisper-based timing exactly
    # as before.
    lyric_lines_text = None
    if has_lyrics:
        lyrics_path = prompt_path("Enter path to the lyrics .txt file (one line per lyric line): ")
        lyric_lines_text = read_lyrics_txt(lyrics_path)
        print(f"  Loaded {len(lyric_lines_text)} lyric line(s) from file.")
        print("  Pre-checking lyrics text against the aligner's vocabulary before starting "
              "audio processing...")
        try:
            prepare_alignment_text(lyric_lines_text, Config.FORCED_ALIGN_LANGUAGE)
            print("  Lyrics text pre-check passed.")
        except Exception as e:
            print(f"  WARNING: lyrics text pre-check failed ({e}). Continuing anyway -- "
                  f"forced alignment will very likely fail again once audio processing reaches "
                  f"it, and this song will fall back to the less accurate Whisper-based timing "
                  f"method. Fix the lyrics file now to avoid that, or let this run finish and "
                  f"re-run once it's fixed.")
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
    if lyric_lines_text is not None:
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
            whisper_result = run_whisper(vocals_path, "cpu", temperature=Config.WHISPER_TEMPERATURE)
            vocal_blocks = segments_as_blocks(whisper_result)
            if vocal_blocks:
                allocation_blocks = merge_blocks_to_target_count(vocal_blocks, len(lyric_lines_text))
                lines = allocate_lines_to_blocks(lyric_lines_text, allocation_blocks)
            else:
                duration = get_audio_duration(instrumental_path)
                lines = allocate_lines_to_blocks(lyric_lines_text, [(0.0, duration)])
    else:
        print("  No lyrics file -- using Whisper's transcribed text directly.")
        whisper_result = run_whisper(vocals_path, "cpu", temperature=Config.WHISPER_TEMPERATURE)
        vocal_blocks = segments_as_blocks(whisper_result)
        lines = build_lines_from_whisper_text(whisper_result)
        lines = _drop_hallucinated_repeats(lines)

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
    # insert_gap_indicator_lines() is a RENDERING-ONLY decoration -- built
    # from the final, fully-processed `lines` (after split_long_lines(),
    # word-duration capping, voice classification/overrides, and after
    # line_timing_preview.txt above was already written from the real,
    # unmodified list) -- so it never affects the preview file or anything
    # that treats `lines` as the real transcript/alignment.
    render_lines = insert_gap_indicator_lines(lines)
    final_out = render_video(render_lines, vocal_blocks, duration, instrumental_path, out_dir, ticker_text,
                              output_filename=video_filename)

    print("\n[6/6] Generating thumbnail...")
    thumb_out = generate_thumbnail(song_title, film_name, artist_name, out_dir, output_filename=thumb_filename)

    print("\n" + "=" * 70)
    print(f"  DONE! Video saved to:     {final_out.resolve()}")
    print(f"        Thumbnail saved to: {thumb_out.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()

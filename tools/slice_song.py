#!/usr/bin/env python3
"""
slice_song.py — Slice a flat audio file into intro / loop / outro sections.

Requires: ffmpeg (https://ffmpeg.org) — must be on your PATH.

Usage examples:
  # Specify timestamps directly:
  python slice_song.py prairie_wind.mp3 --intro-end 8.4 --loop-end 134.2

  # Choose output format and directory:
  python slice_song.py song.mp3 --intro-end 8.4 --loop-end 134.2 \\
      --format ogg --out-dir ../audio/bgm/prairie_wind

  # Batch-process from a JSON config file:
  python slice_song.py --config songs.json

JSON config format (songs.json):
  {
    "songs": [
      {
        "input": "raw/prairie_wind.mp3",
        "name":  "prairie_wind",
        "intro_end": 8.4,
        "loop_end":  134.2,
        "out_dir":   "../audio/bgm/prairie_wind",
        "format":    "ogg"
      }
    ]
  }

Output files (all in out_dir):
  <name>_intro.<ext>   — from 0 to intro_end
  <name>_loop.<ext>    — from intro_end to loop_end
  <name>_outro.<ext>   — from loop_end to end of file
  <name>_meta.json     — metadata with timestamps for bgm.js registration

Notes:
  • If intro_end is omitted (or 0), no intro file is produced.
  • If loop_end is omitted (or equals file duration), no outro file is produced.
  • For gapless loops, end the loop section at a zero-crossing.
    Audacity's "Find Zero Crossings" (Z key) is helpful.
  • OGG Vorbis is recommended for web: good quality, small size, patent-free.
    Use --format mp3 if you need wider Safari compatibility (< iOS 11).
"""

import argparse
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def check_ffmpeg():
    if shutil.which('ffmpeg') is None:
        print("ERROR: ffmpeg not found on PATH.")
        print("Install it from https://ffmpeg.org/download.html")
        sys.exit(1)


def get_duration(input_path: str) -> float:
    """Return the duration of an audio file in seconds using ffprobe."""
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            input_path
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    return float(result.stdout.strip())


def ffmpeg_slice(input_path: str, start: float, end: float | None,
                 output_path: str, fmt: str):
    """
    Extract [start, end) from input_path and write to output_path.
    end=None means 'until end of file'.
    Uses -c:a copy when possible for speed, otherwise re-encodes.
    """
    cmd = ['ffmpeg', '-y', '-i', input_path, '-ss', str(start)]

    if end is not None:
        cmd += ['-to', str(end)]

    # Format-specific encoding options
    if fmt == 'ogg':
        cmd += ['-c:a', 'libvorbis', '-q:a', '5']
    elif fmt == 'mp3':
        cmd += ['-c:a', 'libmp3lame', '-q:a', '2']
    elif fmt == 'opus':
        cmd += ['-c:a', 'libopus', '-b:a', '128k']
    elif fmt in ('wav', 'flac'):
        cmd += ['-c:a', fmt]
    else:
        # Generic: let ffmpeg infer from extension
        pass

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {output_path}:\n{result.stderr[-800:]}"
        )


def seconds_to_hms(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    if h:
        return f"{h}:{m:02d}:{sec:06.3f}"
    return f"{m}:{sec:06.3f}"


# ── Core slice logic ──────────────────────────────────────────────────────────

def slice_song(
    input_path: str,
    name: str,
    intro_end: float,   # seconds; 0 = no intro
    loop_end: float,    # seconds; 0 = no outro
    out_dir: str,
    fmt: str = 'ogg',
    verbose: bool = True,
) -> dict:
    """
    Slice input_path into intro/loop/outro segments.
    Returns a dict suitable for bgm.js registration.
    """
    check_ffmpeg()

    input_path = str(Path(input_path).resolve())
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    duration = get_duration(input_path)

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Input  : {input_path}")
        print(f"  Name   : {name}")
        print(f"  Duration: {seconds_to_hms(duration)} ({duration:.3f}s)")
        print(f"  Format : {fmt}")
        print(f"  Output : {out_dir}")
        print(f"{'─'*55}")

    produced = {}

    # ── Intro ─────────────────────────────────────────────────────────────────
    if intro_end > 0:
        out = str(out_dir_path / f"{name}_intro.{fmt}")
        if verbose:
            print(f"  [1/3] intro  0.000 → {intro_end:.3f}s  …  {out}")
        ffmpeg_slice(input_path, 0, intro_end, out, fmt)
        produced['intro'] = out

    # ── Loop ──────────────────────────────────────────────────────────────────
    loop_start = intro_end if intro_end > 0 else 0
    loop_stop  = loop_end  if loop_end  > 0 else duration

    out = str(out_dir_path / f"{name}_loop.{fmt}")
    if verbose:
        print(f"  [2/3] loop   {loop_start:.3f} → {loop_stop:.3f}s  …  {out}")
    ffmpeg_slice(input_path, loop_start, loop_stop, out, fmt)
    produced['loop'] = out

    # ── Outro ─────────────────────────────────────────────────────────────────
    if loop_end > 0 and loop_end < duration - 0.5:
        out = str(out_dir_path / f"{name}_outro.{fmt}")
        if verbose:
            print(f"  [3/3] outro  {loop_end:.3f} → {duration:.3f}s  …  {out}")
        ffmpeg_slice(input_path, loop_end, None, out, fmt)
        produced['outro'] = out
    elif verbose:
        print("  [3/3] outro  (skipped — loop_end not set or too close to end)")

    # ── Meta JSON ─────────────────────────────────────────────────────────────
    meta = {
        'name': name,
        'source': input_path,
        'format': fmt,
        'duration': round(duration, 3),
        'intro_end': intro_end if intro_end > 0 else None,
        'loop_end':  loop_end  if loop_end  > 0 else None,
        'files': {k: str(Path(v).name) for k, v in produced.items()},
        # Ready-to-paste bgm.js registration snippet
        'bgm_js_snippet': (
            f"bgm.register('{name}', {{\n"
            + (f"  intro: '{produced.get('intro', '')}',\n" if 'intro' in produced else '')
            + f"  loop:  '{produced['loop']}',\n"
            + (f"  outro: '{produced.get('outro', '')}',\n" if 'outro' in produced else '')
            + "});"
        )
    }
    meta_path = out_dir_path / f"{name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"\n  Meta   : {meta_path}")
        print(f"\n  ✓ Done!  Paste into bgm.js:")
        print(f"  {meta['bgm_js_snippet']}")

    return produced


# ── Batch processing ──────────────────────────────────────────────────────────

def process_config(config_path: str, verbose: bool = True):
    config = json.loads(Path(config_path).read_text())
    songs = config.get('songs', [])
    print(f"Processing {len(songs)} song(s) from {config_path}")
    for entry in songs:
        slice_song(
            input_path = entry['input'],
            name       = entry['name'],
            intro_end  = entry.get('intro_end', 0),
            loop_end   = entry.get('loop_end', 0),
            out_dir    = entry.get('out_dir', entry['name']),
            fmt        = entry.get('format', 'ogg'),
            verbose    = verbose,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Slice an audio file into intro/loop/outro sections for bgm.js.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Single-file mode
    parser.add_argument('input', nargs='?', help='Input audio file (MP3, OGG, WAV, FLAC…)')
    parser.add_argument('--name',      help='Base name for output files (default: input filename stem)')
    parser.add_argument('--intro-end', type=float, default=0,
                        help='Timestamp (seconds) where intro ends and loop begins. 0 = no intro.')
    parser.add_argument('--loop-end',  type=float, default=0,
                        help='Timestamp (seconds) where loop ends and outro begins. 0 = no outro.')
    parser.add_argument('--format', default='ogg',
                        choices=['ogg', 'mp3', 'opus', 'wav', 'flac'],
                        help='Output audio format (default: ogg)')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: ./<name>)')

    # Batch mode
    parser.add_argument('--config', help='Path to JSON batch config file.')

    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output.')

    args = parser.parse_args()

    verbose = not args.quiet

    if args.config:
        process_config(args.config, verbose)
        return

    if not args.input:
        parser.print_help()
        sys.exit(0)

    name    = args.name or Path(args.input).stem
    out_dir = args.out_dir or name

    slice_song(
        input_path = args.input,
        name       = name,
        intro_end  = args.intro_end,
        loop_end   = args.loop_end,
        out_dir    = out_dir,
        fmt        = args.format,
        verbose    = verbose,
    )


if __name__ == '__main__':
    main()

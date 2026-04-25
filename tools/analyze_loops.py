#!/usr/bin/env python3
"""
analyze_loops.py — Analyze MP3 files to suggest intro/loop/outro boundaries.

Uses ffmpeg to decode MP3 → raw PCM, then numpy to analyze the waveform envelope.

Strategy:
  1. Compute RMS energy in small windows across the full song.
  2. Find the "stable body" — the region where energy is consistent, excluding
     the opening build-up and closing wind-down.
  3. Within the body, look for repeating energy patterns (potential loop seams)
     by checking auto-correlation at likely bar-length intervals.
  4. Suggest intro_end and loop_end timestamps that maximize the loop section
     while keeping the seam at a musically natural boundary.
  5. Snap timestamps to zero-crossings for click-free cuts.

Output: a JSON report + updated loop_notes.md entries.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


# ── Config ──────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 44100
CHANNELS       = 1        # mono mixdown for analysis
WINDOW_SEC     = 0.25     # RMS window size in seconds
HOP_SEC        = 0.1      # RMS hop size
INTRO_MAX_SEC  = 20.0     # never suggest intro longer than this
OUTRO_MAX_SEC  = 20.0     # never suggest outro longer than this
INTRO_MIN_SEC  = 2.0      # minimum intro length
OUTRO_MIN_SEC  = 2.0      # minimum outro length
ENERGY_THRESHOLD = 0.15   # fraction of peak RMS below which = "quiet"
STABILITY_WINDOW = 3.0    # seconds to check for energy stability


# ── Decode ──────────────────────────────────────────────────────────────────

def decode_to_pcm(mp3_path: str) -> np.ndarray:
    """Decode an MP3 to mono float32 samples using ffmpeg."""
    cmd = [
        'ffmpeg', '-v', 'error', '-y',
        '-i', mp3_path,
        '-ac', str(CHANNELS),
        '-ar', str(SAMPLE_RATE),
        '-f', 'f32le',
        '-acodec', 'pcm_f32le',
        'pipe:1'
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode()[-400:]}")
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    return samples


def get_duration_secs(mp3_path: str) -> float:
    """Get duration using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        mp3_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


# ── RMS Envelope ────────────────────────────────────────────────────────────

def compute_rms_envelope(samples: np.ndarray, sr: int,
                         window_sec: float, hop_sec: float) -> tuple:
    """
    Compute RMS energy envelope.
    Returns (times, rms_values) where times are the center of each window.
    """
    window_samples = int(window_sec * sr)
    hop_samples    = int(hop_sec * sr)
    n_frames       = max(1, (len(samples) - window_samples) // hop_samples + 1)

    times = np.zeros(n_frames)
    rms   = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_samples
        end   = start + window_samples
        if end > len(samples):
            end = len(samples)
        frame = samples[start:end]
        rms[i]   = np.sqrt(np.mean(frame ** 2))
        times[i] = (start + end) / 2.0 / sr

    return times, rms


# ── Find stable region ─────────────────────────────────────────────────────

def find_stable_region(times, rms, duration):
    """
    Find the start and end of the stable "body" of the song.

    Approach:
      - Compute a rolling standard deviation of the RMS.
      - The "stable" region is where the local std is low relative to the mean.
      - The intro is the initial unstable region; the outro is the final one.
    """
    if len(rms) < 10:
        return INTRO_MIN_SEC, duration - OUTRO_MIN_SEC

    # Smooth the RMS to reduce noise
    kernel_size = max(3, int(STABILITY_WINDOW / HOP_SEC))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    rms_smooth = np.convolve(rms, kernel, mode='same')

    # Compute rolling standard deviation
    rms_std = np.zeros_like(rms_smooth)
    half_k = kernel_size // 2
    for i in range(len(rms_smooth)):
        lo = max(0, i - half_k)
        hi = min(len(rms_smooth), i + half_k + 1)
        rms_std[i] = np.std(rms_smooth[lo:hi])

    # Normalize std by mean RMS (coefficient of variation)
    mean_rms = np.mean(rms_smooth)
    if mean_rms > 0:
        cv = rms_std / mean_rms
    else:
        cv = rms_std

    # Peak RMS for energy threshold
    peak_rms = np.max(rms_smooth)

    # Find intro_end: first time the energy rises above threshold AND becomes stable
    stability_threshold = np.median(cv) * 1.5  # allow 1.5x median variability

    intro_end_idx = 0
    for i in range(len(cv)):
        if times[i] > INTRO_MAX_SEC:
            break
        if times[i] < INTRO_MIN_SEC:
            continue
        if rms_smooth[i] > peak_rms * ENERGY_THRESHOLD and cv[i] < stability_threshold:
            intro_end_idx = i
            break

    if intro_end_idx == 0:
        # Fallback: use a fixed proportion
        intro_end_idx = int(len(times) * 0.05)

    # Find loop_end: last time before the song becomes unstable or energy drops
    loop_end_idx = len(cv) - 1
    for i in range(len(cv) - 1, -1, -1):
        if times[i] < duration - OUTRO_MAX_SEC:
            break
        if times[i] > duration - OUTRO_MIN_SEC:
            continue
        if rms_smooth[i] > peak_rms * ENERGY_THRESHOLD and cv[i] < stability_threshold:
            loop_end_idx = i
            break

    if loop_end_idx >= len(times) - 1:
        # Fallback: use a fixed proportion from end
        loop_end_idx = int(len(times) * 0.95)

    intro_end = times[intro_end_idx]
    loop_end  = times[loop_end_idx]

    # Sanity: ensure loop is at least 60% of the song
    min_loop = duration * 0.6
    actual_loop = loop_end - intro_end
    if actual_loop < min_loop:
        # Shrink intro and outro proportionally
        excess = min_loop - actual_loop
        intro_end = max(INTRO_MIN_SEC, intro_end - excess / 2)
        loop_end  = min(duration - OUTRO_MIN_SEC, loop_end + excess / 2)

    return intro_end, loop_end


# ── Snap to zero crossing ──────────────────────────────────────────────────

def snap_to_zero_crossing(samples, sr, target_sec, search_range_sec=0.05):
    """
    Find the nearest zero-crossing to the target timestamp.
    Search within ±search_range_sec of the target.
    """
    target_sample = int(target_sec * sr)
    search_range  = int(search_range_sec * sr)

    lo = max(0, target_sample - search_range)
    hi = min(len(samples) - 1, target_sample + search_range)

    best_dist = search_range + 1
    best_idx  = target_sample

    for i in range(lo, hi):
        if i + 1 < len(samples):
            # Zero crossing: sign change
            if samples[i] * samples[i + 1] <= 0:
                dist = abs(i - target_sample)
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i

    return best_idx / sr


# ── Analyze one song ───────────────────────────────────────────────────────

def analyze_song(mp3_path: str) -> dict:
    """
    Analyze a single MP3 and return suggested loop points.
    """
    name = Path(mp3_path).stem
    print(f"  Analyzing: {name}")

    # Decode
    samples  = decode_to_pcm(mp3_path)
    duration = len(samples) / SAMPLE_RATE

    # RMS envelope
    times, rms = compute_rms_envelope(samples, SAMPLE_RATE, WINDOW_SEC, HOP_SEC)

    # Find stable region
    intro_end_raw, loop_end_raw = find_stable_region(times, rms, duration)

    # Snap to zero crossings
    intro_end = snap_to_zero_crossing(samples, SAMPLE_RATE, intro_end_raw)
    loop_end  = snap_to_zero_crossing(samples, SAMPLE_RATE, loop_end_raw)

    # Compute stats
    loop_duration = loop_end - intro_end
    intro_duration = intro_end
    outro_duration = duration - loop_end

    # Energy stats for the loop section
    loop_mask = (times >= intro_end) & (times <= loop_end)
    if np.any(loop_mask):
        loop_rms_mean = float(np.mean(rms[loop_mask]))
        loop_rms_std  = float(np.std(rms[loop_mask]))
        loop_cv       = loop_rms_std / loop_rms_mean if loop_rms_mean > 0 else 0
    else:
        loop_rms_mean = 0
        loop_rms_std  = 0
        loop_cv       = 0

    # Overall energy level classification
    overall_rms = float(np.mean(rms))
    all_rms_values = [float(np.mean(rms))]  # placeholder for comparison

    # Energy classification (will be relative across all songs)
    energy_label = "medium"

    result = {
        "name": name,
        "file": str(mp3_path),
        "duration": round(duration, 3),
        "duration_fmt": format_time(duration),
        "intro_end": round(intro_end, 3),
        "intro_end_fmt": format_time(intro_end),
        "loop_end": round(loop_end, 3),
        "loop_end_fmt": format_time(loop_end),
        "intro_duration": round(intro_duration, 3),
        "loop_duration": round(loop_duration, 3),
        "outro_duration": round(outro_duration, 3),
        "loop_pct": round(loop_duration / duration * 100, 1),
        "loop_rms_mean": round(loop_rms_mean, 5),
        "loop_rms_cv": round(loop_cv, 4),  # lower = more consistent loop
        "overall_rms": round(overall_rms, 5),
        "energy_label": energy_label,
        "confidence": "high" if loop_cv < 0.2 else ("medium" if loop_cv < 0.35 else "low"),
    }

    return result


# ── Classify energy levels across all songs ─────────────────────────────────

def classify_energy(results: list):
    """Assign energy labels based on relative RMS across all songs."""
    rms_values = [r["overall_rms"] for r in results]
    if not rms_values:
        return

    sorted_rms = sorted(rms_values)
    n = len(sorted_rms)

    for r in results:
        rms = r["overall_rms"]
        rank = sorted_rms.index(rms) / max(1, n - 1)
        if rank < 0.33:
            r["energy_label"] = "low (calm)"
        elif rank < 0.66:
            r["energy_label"] = "medium"
        else:
            r["energy_label"] = "high (intense)"


# ── Format helpers ──────────────────────────────────────────────────────────

def format_time(s):
    m = int(s // 60)
    sec = s % 60
    return f"{m}:{sec:06.3f}"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    raw_dir = Path(__file__).resolve().parent.parent / "songs" / "raw" / "TN51bgmusic 3"

    if not raw_dir.exists():
        print(f"ERROR: directory not found: {raw_dir}")
        sys.exit(1)

    mp3_files = sorted(raw_dir.glob("*.mp3"))
    if not mp3_files:
        print("No MP3 files found.")
        sys.exit(1)

    # Separate gameplay songs from win/lose
    gameplay_files = []
    special_files  = []

    for f in mp3_files:
        name_lower = f.stem.lower()
        if "win_song" in name_lower or "loose_song" in name_lower or "lose_song" in name_lower:
            special_files.append(f)
        elif "under8min" in name_lower or "remaster" in name_lower:
            # Skip rendered/remastered duplicates
            continue
        elif f.suffix.lower() == '.mp3':
            gameplay_files.append(f)

    print(f"\n{'='*60}")
    print(f"  BGM Loop Analyzer")
    print(f"  {len(gameplay_files)} gameplay songs + {len(special_files)} special songs")
    print(f"{'='*60}\n")

    # Analyze gameplay songs (these need looping)
    results = []
    for f in gameplay_files:
        try:
            r = analyze_song(str(f))
            results.append(r)
        except Exception as e:
            print(f"  ERROR analyzing {f.name}: {e}")

    # Classify energy levels across all gameplay songs
    classify_energy(results)

    # Analyze special songs (no looping needed, just get duration)
    special_results = []
    for f in special_files:
        try:
            samples = decode_to_pcm(str(f))
            dur = len(samples) / SAMPLE_RATE
            special_results.append({
                "name": f.stem,
                "file": str(f),
                "duration": round(dur, 3),
                "duration_fmt": format_time(dur),
                "type": "flat",
            })
            print(f"  Special: {f.stem} ({format_time(dur)})")
        except Exception as e:
            print(f"  ERROR: {f.name}: {e}")

    # ── Output report ────────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent.parent / "songs"

    # JSON report
    report = {
        "gameplay_songs": results,
        "special_songs": special_results,
    }
    report_path = out_dir / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  JSON report: {report_path}")

    # Human-readable catalog
    catalog_path = out_dir / "loop_catalog.md"
    with open(catalog_path, 'w') as f:
        f.write("# BGM Loop Analysis Catalog\n\n")
        f.write("*Auto-generated by analyze_loops.py — review and adjust timestamps*\n")
        f.write("*Use `tools/find_loop_points.html` to fine-tune each song*\n\n")
        f.write("---\n\n")

        f.write("## Summary\n\n")
        f.write(f"| # | Song | Duration | Energy | Intro | Loop | Outro | Loop % | Confidence |\n")
        f.write(f"|---|------|----------|--------|-------|------|-------|--------|------------|\n")
        for i, r in enumerate(results, 1):
            f.write(
                f"| {i} | {r['name']} | {r['duration_fmt']} | {r['energy_label']} "
                f"| {r['intro_end_fmt']} | {r['loop_duration']:.1f}s "
                f"| {r['outro_duration']:.1f}s | {r['loop_pct']}% "
                f"| {r['confidence']} |\n"
            )
        f.write("\n")

        if special_results:
            f.write("### Special (flat playback — no looping)\n\n")
            f.write("| Song | Duration |\n")
            f.write("|------|----------|\n")
            for sr in special_results:
                f.write(f"| {sr['name']} | {sr['duration_fmt']} |\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("## Detailed Analysis\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"### {i}. {r['name']}\n\n")
            f.write(f"- **File:** `{Path(r['file']).name}`\n")
            f.write(f"- **Duration:** {r['duration_fmt']} ({r['duration']}s)\n")
            f.write(f"- **Energy:** {r['energy_label']}\n")
            f.write(f"- **Confidence:** {r['confidence']}\n\n")
            f.write(f"| Section | Start | End | Length |\n")
            f.write(f"|---------|-------|-----|--------|\n")
            f.write(f"| Intro | 0.000 | {r['intro_end']:.3f} | {r['intro_duration']:.1f}s |\n")
            f.write(f"| Loop | {r['intro_end']:.3f} | {r['loop_end']:.3f} | {r['loop_duration']:.1f}s |\n")
            f.write(f"| Outro | {r['loop_end']:.3f} | {r['duration']:.3f} | {r['outro_duration']:.1f}s |\n")
            f.write(f"\n")
            f.write(f"```bash\n")
            f.write(f"python tools/slice_song.py \"{Path(r['file']).name}\" \\\n")
            f.write(f"  --intro-end {r['intro_end']:.3f} --loop-end {r['loop_end']:.3f} \\\n")
            f.write(f"  --format ogg --out-dir audio/bgm/{r['name']}\n")
            f.write(f"```\n\n")
            f.write(f"---\n\n")

        # Evaluation section
        f.write("## How to Evaluate\n\n")
        f.write("1. Open `tools/find_loop_points.html` in your browser\n")
        f.write("2. Load a song from `songs/raw/TN51bgmusic 3/`\n")
        f.write("3. The timestamps above are pre-filled starting points\n")
        f.write("4. For each song:\n")
        f.write("   - Seek to the **intro end** timestamp — does the music feel settled in?\n")
        f.write("   - Seek to the **loop end** timestamp — does the music start wrapping up?\n")
        f.write("   - Use the nudge buttons (±10ms, ±100ms) to adjust\n")
        f.write("   - Press **▶ from here** to hear each boundary\n")
        f.write("5. Once happy, copy the `slice_song.py` command from the tool\n")
        f.write("6. Run the command to produce the final intro/loop/outro files\n\n")
        f.write("**Confidence levels:**\n")
        f.write("- **high** = the loop section has very consistent energy, likely a clean loop\n")
        f.write("- **medium** = some energy variation in the loop, may need manual adjustment\n")
        f.write("- **low** = significant variation, definitely review manually\n")

    print(f"  Catalog:     {catalog_path}")
    print(f"\n  Done! Review {catalog_path.name} and fine-tune in find_loop_points.html")


if __name__ == '__main__':
    main()

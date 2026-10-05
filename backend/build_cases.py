"""Run the unchanged Scopewatch engine on the real clips and keep the results.

This is the only place the OpenCV measurement is executed ahead of time. The app
serves what this writes; it never re-measures a bundled case on request, and it
never invents a number the engine did not produce. One folder per case:

    cache/cases/<slug>/record.json    the RunRecord the engine produced
    cache/cases/<slug>/loop.pkl       the engine's agent loop (for the checkpoint routes)
    cache/cases/<slug>/evidence/*.jpg the overlay frames the engine saved
    cache/cases/<slug>/clip.mp4       the clip, transcoded for the browser

Run:  python build_cases.py [slug ...]      (clips directory: SCOPEWATCH_CLIPS)
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

from servicekit import JobContext
from visioncore import recording, RunRecord

from scopewatch import service

HERE = Path(__file__).resolve().parent
CASES = HERE / "cache" / "cases"
CLIPS = Path(os.environ.get(
    "SCOPEWATCH_CLIPS", Path.home() / "dev/Hackathons/opencv26/media/real/scopewatch"))

# slug -> (source file, display title, one-line what-it-is). Credits live in cases.py.
SOURCES = {
    "boer-cholecystectomy": "boer-gallbladder-torsion-cholecystectomy.mp4",
    "wses-bleeding-ulcer": "wses-laparoscopic-bleeding-ulcer-repair.mp4",
    "barroso-inguinal-suture": "barroso-laparoscopic-inguinal-ring-suture.mp4",
    "kaplan-pancreatic-dissection": "kaplan-tlpd-technique-3.mp4",
    "gupta-open-surgery": "gupta-gallbladder-torsion-cholecystectomy.mp4",
}


def build(slug: str) -> None:
    src = CLIPS / SOURCES[slug]
    out = CASES / slug
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    record = RunRecord(product="scopewatch", input={"filename": src.name, "job_id": slug})

    def emit(ev):  # progress is not needed offline
        pass

    ctx = JobContext(job_id=slug, input_path=src, filename=src.name, params={},
                     record=record, _emit=emit, evidence_dir=out / "evidence")
    t0 = time.time()
    with recording(record):
        service.analyze(ctx)
    (out / "record.json").write_text(json.dumps(record.to_dict(), default=str))
    loop = service.REGISTRY.get(slug)
    (out / "loop.pkl").write_bytes(pickle.dumps(loop))
    clip = out / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-an",
                    "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(clip)], check=True)
    print(f"{slug}: {time.time() - t0:.0f}s, refused={record.refused}", flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or SOURCES):
        build(s)

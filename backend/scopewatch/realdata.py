"""Real laparoscopic footage: which clips, which frames, and the hand-drawn blood labels.

None of the footage lives in this repository. The clips are openly licensed videos
from Wikimedia Commons and Zenodo (the manifest below carries the source page, the
author and the licence of each), and they are read from a directory you point at.
What *is* in the repository is derived from them: polygon labels for blood on a fixed
sample of frames, and the numbers the evaluation computes.

There is no ground-truth blood volume for any of these clips and there never will be
from these sources. Nobody weighed the swabs. So the labels answer a narrower question
that real footage *can* answer: on this frame, which pixels are blood? That is enough
to measure segmentation precision and recall, which is what the millilitre figure was
silently standing on.

The frame sample is fixed before anything is labelled and it is not chosen by looking
at the segmenter: four frames per clip at 12.5%, 37.5%, 62.5% and 87.5% of its length.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SAMPLE_FRACTIONS: tuple[float, ...] = (0.125, 0.375, 0.625, 0.875)


@dataclass(frozen=True)
class Clip:
    """One real clip, with enough provenance to find and credit it."""

    name: str
    domain: str  # "laparoscopic" or "open"
    split: str  # "dev": thresholds may be chosen on it; "test": they may not
    licence: str
    credit: str
    source: str

    @property
    def filename(self) -> str:
        return f"{self.name}.mp4"


_KAPLAN = (
    "Kaplan M, World J Surg Oncol 2012, doi:10.1186/1477-7819-10-142",
    "CC BY 2.0",
)
_COMMONS = "https://commons.wikimedia.org/wiki/File:"
_ANPOL = ("Anpol42, Wikimedia Commons (own work, 2012)", "CC BY-SA 3.0")

# The split is by clip, never by frame: frames from one clip share a scope, a light
# source and a patient, and a frame-level split would leak all three. Dev and test
# each get a bleeding clip, a clean clip, a Kaplan clip and a TEP clip, and the
# assignment was made from the provenance notes before any label existed.
CLIPS: tuple[Clip, ...] = (
    Clip("wses-laparoscopic-bleeding-ulcer-repair", "laparoscopic", "dev", "CC BY 4.0",
         "Di Saverio S et al., World J Emerg Surg 2014, doi:10.1186/1749-7922-9-45",
         _COMMONS + "Diagnosis-and-treatment-of-perforated-or-bleeding-peptic-ulcers-"
         "2013-WSES-position-paper-1749-7922-9-45-S1.ogv"),
    Clip("barroso-laparoscopic-inguinal-ring-suture", "laparoscopic", "dev", "CC BY 4.0",
         "Barroso C et al., Front Pediatr 2017, doi:10.3389/fped.2017.00207",
         _COMMONS + "Learning-Curves-for-Laparoscopic-Repair-of-Inguinal-Hernia-and-"
         "Communicating-Hydrocele-in-Children-video_1.ogv"),
    Clip("kaplan-tlpd-technique-3", "laparoscopic", "dev", _KAPLAN[1], _KAPLAN[0],
         _COMMONS + "A-case-report-of-an-ampullary-tumor-presenting-with-spontaneous-"
         "perforation-of-an-aberrant-bile-1477-7819-10-142-S3.ogv"),
    Clip("kaplan-tlpd-technique-6", "laparoscopic", "dev", _KAPLAN[1], _KAPLAN[0],
         _COMMONS + "A-case-report-of-an-ampullary-tumor-presenting-with-spontaneous-"
         "perforation-of-an-aberrant-bile-1477-7819-10-142-S6.ogv"),
    Clip("anpol42-tep-hernia-2", "laparoscopic", "dev", _ANPOL[1], _ANPOL[0],
         _COMMONS + "TEP_Operation_of_Groin_Hernia_Video_2.ogv"),
    Clip("anpol42-tep-hernia-4", "laparoscopic", "dev", _ANPOL[1], _ANPOL[0],
         _COMMONS + "TEP_Operation_of_Groin_Hernia_Video_4.ogv"),
    Clip("anpol42-tep-recurrent-hernia", "laparoscopic", "dev", _ANPOL[1], _ANPOL[0],
         _COMMONS + "TEP_Operation_Video_view_of_Reccurence_Hernia.ogv"),
    Clip("gupta-gallbladder-torsion-cholecystectomy", "open", "dev", "CC BY 2.0",
         "Gupta V et al., Cases J 2009, doi:10.1186/1757-1626-2-193",
         _COMMONS + "Torsion-of-gall-bladder-a-rare-entity-a-case-report-and-review-"
         "article-1757-1626-2-193-S1.ogv"),
    Clip("boer-gallbladder-torsion-cholecystectomy", "laparoscopic", "test", "CC BY 2.0",
         "Boer J, Boerma D, de Vries Reilingh T, J Med Case Rep 2011, "
         "doi:10.1186/1752-1947-5-588",
         _COMMONS + "A-gallbladder-torsion-presenting-as-acute-cholecystitis-in-an-"
         "elderly-woman-A-case-report-1752-1947-5-588-S1.ogv"),
    Clip("kavalakat-omental-infarction", "laparoscopic", "test", "CC BY 2.0",
         "Kavalakat A, Varghese C, Cases J 2008, doi:10.1186/1757-1626-1-164",
         _COMMONS + "Laparoscopic-management-of-an-uncommon-cause-for-right-lower-"
         "quadrant-pain-A-case-report-1757-1626-1-164-S1.ogv"),
    Clip("kaplan-tlpd-diagnostic-laparoscopy", "laparoscopic", "test", _KAPLAN[1], _KAPLAN[0],
         _COMMONS + "A-case-report-of-an-ampullary-tumor-presenting-with-spontaneous-"
         "perforation-of-an-aberrant-bile-1477-7819-10-142-S1.ogv"),
    Clip("kaplan-tlpd-technique-5", "laparoscopic", "test", _KAPLAN[1], _KAPLAN[0],
         _COMMONS + "A-case-report-of-an-ampullary-tumor-presenting-with-spontaneous-"
         "perforation-of-an-aberrant-bile-1477-7819-10-142-S5.ogv"),
    Clip("anpol42-tep-hernia-1", "laparoscopic", "test", _ANPOL[1], _ANPOL[0],
         _COMMONS + "TEP_Operation_of_Groin_Hernia_Video_1..ogv"),
    Clip("anpol42-tep-hernia-3", "laparoscopic", "test", _ANPOL[1], _ANPOL[0],
         _COMMONS + "TEP_Operation_of_Groin_Hernia_Video_3.ogv"),
    Clip("anpol42-tep-indirect-hernia", "laparoscopic", "test", _ANPOL[1], _ANPOL[0],
         _COMMONS + "Indirect_groin_hernia._Intraoperative_view_by_TEP.ogv"),
    Clip("admlsir-real-surgical-smoke-D-V09", "laparoscopic", "test", "CC BY 4.0",
         "Guo N et al., ADM_LSIR, Zenodo 2026, doi:10.5281/zenodo.20470138",
         "https://zenodo.org/records/20470138"),
)

CLIP_BY_NAME = {c.name: c for c in CLIPS}


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def sample_indices(n_frames: int, fractions: tuple[float, ...] = SAMPLE_FRACTIONS) -> list[int]:
    return [min(n_frames - 1, round(f * n_frames)) for f in fractions]


def read_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Decode sequentially and keep the requested frames. Seeking in H.264 is not exact."""
    wanted = set(indices)
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    try:
        i = -1
        while wanted - out.keys():
            ok, image = cap.read()
            if not ok:
                break
            i += 1
            if i in wanted:
                out[i] = image
    finally:
        cap.release()
    return out


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def load_labels(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rasterise(entry: dict[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(blood, ignore) masks at `shape` from a label entry's normalised polygons.

    Polygons are stored in 0..1 coordinates so a label drawn on one resolution scores
    a pipeline that analysed another. `ignore` marks pixels the labeller could not
    call either way (a blurred edge, dark maroon tissue that might be clot); they
    count as neither a hit nor a miss.
    """
    h, w = shape
    blood = np.zeros((h, w), np.uint8)
    ignore = np.zeros((h, w), np.uint8)
    for key, target in (("blood", blood), ("ignore", ignore)):
        for poly in entry.get(key, []):
            pts = np.array([[x * w, y * h] for x, y in poly], np.float32)
            cv2.fillPoly(target, [np.round(pts).astype(np.int32)], 1)
    return blood, ignore


@dataclass
class Tally:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, predicted: np.ndarray, blood: np.ndarray, ignore: np.ndarray) -> None:
        scored = ignore == 0
        p = predicted.astype(bool) & scored
        t = blood.astype(bool) & scored
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.fn += int((~p & t).sum())

    @property
    def precision(self) -> float | None:
        d = self.tp + self.fp
        return None if d == 0 else self.tp / d

    @property
    def recall(self) -> float | None:
        d = self.tp + self.fn
        return None if d == 0 else self.tp / d

    def to_dict(self) -> dict[str, Any]:
        def r(v: float | None) -> float | None:
            return None if v is None else round(v, 3)

        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": r(self.precision), "recall": r(self.recall)}

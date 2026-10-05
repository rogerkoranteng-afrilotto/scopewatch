# Scopewatch

OpenCV 5 measures how much of a laparoscopic field is
blood-covered, refuses frames it cannot read, and withholds numbers it cannot stand behind.
**NVIDIA Nemotron 3 Super, on Nebius Token Factory, reads those measurements** and does the
reasoning a threshold cannot: it decides whether a safety checkpoint belongs before the
irreversible step (with cited, machine-checked evidence), explains each refusal in plain words,
and writes the surgeon's summary. The model never sees the video and measures nothing.

Live: https://y3sbvbcpui.eu-west-3.awsapprunner.com

## NVIDIA Nemotron and Nebius Token Factory

- **Nemotron:** `nvidia/nemotron-3-super-120b-a12b` reads the OpenCV measurements. It decides whether a safety checkpoint belongs before the irreversible step (with cited, machine-checked evidence), explains each refusal, and writes the surgeon's summary.
- **Token Factory:** Every model call runs on Nebius Token Factory, which is serverless and OpenAI-compatible, so one base URL and the standard OpenAI client reached every model with no GPU and no dedicated endpoint to provision. That is what let this be built and deployed quickly. Cached reasoning is served by default; only the live button spends tokens.
- **Other models:** none. No other model is used; measurement is classical OpenCV code.

Decision support and a retrospective measurement instrument. Not a medical device, never run on
clinical video, takes no clinical action.

Run: `python -m venv .venv && .venv/bin/pip install -r requirements.txt && cd backend && ../.venv/bin/uvicorn main:app --port 8080`
(`NEBIUS_API_KEY` only for the live button.) Rebuild caches: `backend/build_cases.py`, `backend/build_reasoning.py`.

Clips: openly
licensed (CC BY) case-report videos, credited in the app.

Measurement quality (hand-drawn masks, real frames): held-out precision 63.6% / recall 7.7%; dev 82.2% / 40.1%; pooled 78.2% / 23.2%. Precision-weighted by design.

Docker: `docker build -t scopewatch . && docker run -p 8080:8080 -e NEBIUS_API_KEY=... scopewatch`

Stack: Python, OpenCV, FastAPI, plain HTML/CSS/JS frontend, NVIDIA Nemotron on Nebius Token Factory.

Licensed under the Apache License 2.0.

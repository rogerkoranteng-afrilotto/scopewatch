/* Scopewatch on Nebius — the operating field instrument, driven by the real service.
 *
 * No framework, no build step. The HTTP surface comes from servicekit's shared
 * api.js; everything below turns one RunRecord into the screens in
 * docs/mockups/operating-field.html.
 */
import { api, ApiError } from '/shell/js/api.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  jobId: null, record: null, agent: null, evaluation: null, cases: [], slug: null,
  reasoning: null, reasoningSource: null, liveAvailable: false, uploaded: false,
  videoUrl: null, events: [], evidence: [], geom: null, closeStream: null, polling: null,
  cursorMs: 0, decodable: true,
};

const PHASE_NAMES = {
  preparation: 'Preparation', exposure: 'Exposure', dissection: 'Dissection',
  critical_approach: 'Critical approach', division: 'Division', extraction: 'Extraction',
};
// Each refusal code in the words a person uses, and the thing that would fix it.
const REFUSALS = {
  OUT_OF_FOCUS: ['no usable edge energy in the frame', 'Refocus the scope and record again.'],
  LENS_FOGGED: ['haze or smoke across the lens', 'Clean the lens, or submit a clip where the cavity is clear.'],
  OCCLUDED: ['too much of the field covered', 'Pull back until the field, not the instrument, fills the frame.'],
  EXPOSURE_CLIPPED: ['the frame is clipped', 'Lower the light source; the sensor is saturating.'],
  NO_SCALE_REFERENCE: ['no instrument shaft in view, so no scale',
    'Bring an instrument shaft into view for a few seconds, or set millimetres per pixel directly.'],
  SCALE_INCONSISTENT: ['the shaft scale wandered too much across the case',
    'The blood-covered share still stands. A volume needs a scale that holds steady; set millimetres per pixel directly if you know it.'],
  OUT_OF_DOMAIN: ['not a laparoscopic view of the abdomen',
    'Scopewatch reads laparoscopic video from inside the abdomen. Open surgery, drapes and gloved hands are refused.'],
  NO_USABLE_FRAMES: ['no frame in the clip was good enough to measure',
    'Submit a clip with the scope inside the cavity and the light on.'],
  DECODE_FAILED: ['the file could not be decoded as video', 'Submit an MP4, AVI or MOV the decoder can open.'],
};
const refusalWord = (code) => REFUSALS[code]?.[0] || 'refused by a quality gate';
const refusalNext = (code) => REFUSALS[code]?.[1] || 'Submit a clip where the cavity is lit, in focus and not obscured.';

/* ────────────────────────── formatting ────────────────────────── */
const pad = (n) => String(n).padStart(2, '0');

function clock(ms) {
  if (ms == null || !isFinite(ms)) return 'not known';
  const t = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  return h ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}
function num(v, d = 2) {
  if (v == null || !isFinite(v)) return 'not measured';
  return Number(v).toFixed(d);
}
function ms(v) { return v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`; }
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function sentence(s) {
  const t = String(s ?? '').replace(/_/g, ' ');
  return t.charAt(0).toUpperCase() + t.slice(1);
}
function localTime(epochSeconds) {
  if (!epochSeconds) return '';
  return new Date(epochSeconds * 1000).toLocaleTimeString([], { hour12: false });
}

/* ────────────────────────── boot ────────────────────────── */
async function boot() {
  try {
    const [cfg, ver] = await Promise.all([api.config(), api.version()]);
    $('#st-opencv').textContent = `${ver.opencv_version || cfg.opencv_version || 'unknown'}`;
    document.title = `${cfg.product?.title || 'Scopewatch'} — operating field instrument`;
  } catch (err) { showError(err); }

  try {
    const res = await fetch('/api/cases');
    const body = await res.json();
    state.cases = body.cases || [];
    state.liveAvailable = Boolean(body.live_available);
    renderCases();
  } catch (err) { showError(err); }

  try {
    const res = await fetch('/api/evaluation');
    if (res.ok) state.evaluation = await res.json();
  } catch { /* the page works without it */ }

  $('#file-input').addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) runFile(file);
  });
  $('#r-run').addEventListener('click', runNemotron);

  $('#cp-actor').addEventListener('input', syncCheckpointButtons);
  $('#cp-reason').addEventListener('change', syncCheckpointButtons);
  $('#cp-confirm').addEventListener('click', () => decide('confirm'));
  $('#cp-dismiss').addEventListener('click', () => decide('dismiss'));

  const video = $('#video');
  video.addEventListener('timeupdate', () => { state.cursorMs = video.currentTime * 1000; onPlayhead(); });
  video.addEventListener('seeked', () => { state.cursorMs = video.currentTime * 1000; onPlayhead(); });
  // A <video> with nothing painted reads as a broken panel, so park the playhead
  // on the frame the case is actually about as soon as there is one to park on.
  video.addEventListener('loadeddata', () => {
    checkCodec();
    if (video.currentTime > 0.05) return;
    const onset = state.record?.metrics?.onset;
    seek(onset?.detected ? onset.timestamp_ms : 100);
  });
  video.addEventListener('error', () => setDecodable(false));
  video.addEventListener('canplay', checkCodec);

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { drawTrace(); onPlayhead(); }, 140);
  });

  watchSections();
  renderEmpty();
  if (state.cases.length) loadCase(new URLSearchParams(location.search).get('case') || state.cases[0].slug);
}

/* ────────────────────────── running a job ────────────────────────── */
async function runFile(file) {
  busy(true);
  hideError();
  state.slug = null; state.uploaded = true;
  state.reasoning = null; state.reasoningSource = null;
  renderCases();
  if (state.closeStream) { state.closeStream(); state.closeStream = null; }
  clearInterval(state.polling);

  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = URL.createObjectURL(file);
  state.cursorMs = 0;
  state.decodable = true;
  const video = $('#video');
  video.hidden = false;
  video.src = state.videoUrl;
  $('#field-img').hidden = true;
  $('#vid-nocodec').hidden = true;
  $('#vid-empty').hidden = true;
  setTimeout(checkCodec, 1500);
  $('#tb-file').textContent = file.name;
  $('#tb-dot').className = 'dot warn';

  $('#prog').hidden = false;
  $('#prog-notes').innerHTML = '';
  progress(0, 'Uploading the clip.');

  try {
    const { job_id: jobId } = await api.submit(file, {});
    state.jobId = jobId;
    state.closeStream = api.events(jobId, {
      progress: (d) => progress(d.percent ?? 0, d.message || ''),
      note: (d) => addNote(d.message || ''),
      status: (d) => { if (d.status === 'failed' && d.error) showError(new ApiError(d.error, 500)); },
      end: () => finish(jobId),
      error: () => { /* the poller below is the safety net */ },
    });
    state.polling = setInterval(async () => {
      try {
        const job = await api.job(jobId);
        if (job.status === 'done' || job.status === 'failed') finish(jobId);
      } catch { /* keep polling */ }
    }, 2500);
  } catch (err) { showError(err); busy(false); $('#prog').hidden = true; }
}

async function finish(jobId) {
  clearInterval(state.polling);
  if (state.closeStream) { state.closeStream(); state.closeStream = null; }
  try {
    const job = await api.job(jobId);
    if (job.status === 'failed') {
      showError(new ApiError(job.error || { message: 'the analysis failed' }, 500));
      $('#prog').hidden = true; busy(false); return;
    }
    state.record = job.result;
    state.agent = await fetchAgent(jobId);
    state.procedure = $('#r-proc').value.trim();
    progress(100, 'Done.');
    setTimeout(() => { $('#prog').hidden = true; }, 700);
    renderAll();
  } catch (err) { showError(err); }
  busy(false);
}

function busy(on) {
  const l = $('#file-label');
  l.textContent = on ? 'OpenCV is measuring…' : 'Measure a clip of your own';
  l.classList.toggle('busy', on);
  $('#file-input').disabled = on;
}
async function fetchAgent(jobId) {
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/checkpoints`);
    if (res.ok) return await res.json();
  } catch { /* fall through */ }
  return state.record?.metrics?.agent || null;
}
function progress(pct, message) {
  $('#prog-fill').style.width = `${Math.max(0, Math.min(100, pct))}%`;
  $('#prog-pct').textContent = `${Math.round(pct)}%`;
  if (message) $('#prog-msg').textContent = sentence(message);
}
function addNote(message) {
  if (!message) return;
  const li = document.createElement('li');
  li.textContent = sentence(message);
  $('#prog-notes').prepend(li);
}
function showError(err) {
  const box = $('#errbox');
  const code = err instanceof ApiError ? err.code : (err?.name || 'ERROR');
  box.innerHTML = `<b>${esc(code)}</b>${esc(err?.message || 'something went wrong')}`;
  const offered = err?.details?.reasons_offered;
  if (Array.isArray(offered) && offered.length) {
    box.innerHTML += `<p style="margin-top:6px;color:var(--fg-muted)">Reasons the server will accept: ${esc(offered.join('; '))}.</p>`;
  }
  box.hidden = false;
}
function hideError() { $('#errbox').hidden = true; }

/* ────────────────────────── rendering ────────────────────────── */
function renderEmpty() {
  $('#kpis').innerHTML = ['Blood-covered field, peak', 'Blood on the field, volume', 'Bleeding onset',
    'Frames measurable', 'Safety checkpoint']
    .map((k) => `<div class="kpi"><div class="k">${esc(k)}</div><div class="v faint">not measured</div>
      <div class="n">no case has been analysed yet</div></div>`).join('');
  $('#cr-body').innerHTML = `<p class="pad faint">No checkpoint has been raised.</p>`;
  $('#ev-body').innerHTML = `<tr><td colspan="3" class="faint">The log fills once a case has run.</td></tr>`;
  $('#in-body').innerHTML = '';
  renderReasoning();
  drawTrace();
}

function renderAll() {
  const rec = state.record;
  if (!rec) return;
  const cannot = isUnmeasurable(rec);

  renderTopbar(rec, cannot);
  renderHead(rec, cannot);
  renderCredit();
  renderCannot(rec, cannot);
  state.events = buildEvents(rec);
  state.evidence = pickEvidence(rec);
  renderKpis(rec);
  renderCheckpoint();
  renderEvents();
  renderEvidence();
  renderInstruments(rec);
  renderRail(rec);
  renderPhases(rec);
  renderReasoning();
  drawTrace();
  checkCodec();
  if (state.cursorMs < 50) {
    const onset = rec.metrics?.onset;
    seek(onset?.detected ? onset.timestamp_ms : 100);
  }
  onPlayhead();
  $('#foot-run').textContent = `Run ${rec.run_id}, OpenCV ${rec.env?.opencv_version || '—'}.`;
}

function isUnmeasurable(rec) {
  const q = rec.metrics?.quality;
  return Boolean(rec.refused) || (q && q.frames > 0 && q.usable_fraction < 0.5);
}

function renderTopbar(rec, cannot) {
  const v = rec.metrics?.video || {};
  $('#tb-file').textContent = caseMeta()?.title || rec.input?.filename || 'case';
  $('#tb-duration').textContent = clock(v.duration_ms);
  $('#tb-video').textContent = v.width
    ? `${v.width}×${v.height}, ${num(v.fps, 0)} fps, ${v.frame_count || 0} frames`
    : 'resolution not read';
  const open = state.agent?.open_checkpoint;
  $('#tb-dot').className = `dot ${cannot ? 'warn' : open ? 'hot' : ''}`.trim();
  const chip = $('#tb-state');
  if (cannot) { chip.className = 'chip warn'; chip.textContent = '▲ Measurement suspended'; }
  else if (open) { chip.className = 'chip hot'; chip.textContent = '▲ Checkpoint held'; }
  else { chip.className = 'chip ok'; chip.textContent = '✓ Case measured'; }
}

function renderHead(rec, cannot) {
  const meta = caseMeta();
  $('#page-title').textContent = cannot ? 'Cannot be measured' : (meta?.title || 'Live field');
  $('#page-lede').textContent = '';
}

function renderCredit() {
  const m = caseMeta();
  $('#credit').innerHTML = m ? `${esc(m.credit)} · ${esc(m.licence)}` : '';
}

function renderCannot(rec, cannot) {
  const box = $('#cannot');
  box.hidden = !cannot;
  if (!cannot) return;
  const q = rec.metrics?.quality || {};
  const refusal = rec.refusals?.[0];
  const code = refusal?.code || 'NO_USABLE_FRAMES';
  $('#cannot-why').textContent = `${q.usable || 0} of ${q.frames || 0} frames passed the gates (${num((q.usable_fraction || 0) * 100, 1)}%). ${refusal ? sentence(refusalWord(code)) + '.' : ''}`;
  const counts = q.rejected_by || {};
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  $('#cannot-codes').innerHTML = rows.length
    ? `<table><thead><tr><th style="width:200px">Refusal</th><th>Why the frame was refused</th><th class="num" style="width:110px">Frames</th></tr></thead><tbody>${
      rows.map(([c, n]) => `<tr><td><span class="chip warn">${esc(c)}</span></td>
        <td class="dim">${esc(refusalWord(c))}</td>
        <td class="num">${n}</td></tr>`).join('')}</tbody></table>`
    : '<p class="faint">No frame carried a refusal code.</p>';
}

function realSegmentationLine() {
  const seg = state.evaluation?.real?.segmentation?.after?.test;
  if (!seg || seg.precision == null) return '';
  return ` On hand-labelled frames from held-out real clips: precision ${num(seg.precision * 100, 0)}%, recall ${num(seg.recall * 100, 0)}%.`;
}

function rawPeak(rec) {
  const s = rec.metrics?.series;
  if (!s) return null;
  let best = null;
  s.raw.forEach((v, i) => { if (s.measurable[i] && (!best || v > best.v)) best = { v, t: s.times_ms[i] }; });
  return best;
}

function renderKpis(rec) {
  const rows = rec.results || [];
  const q = rec.metrics?.quality || {};
  $('#kpis').innerHTML = rows.map((r) => {
    const k = `<div class="k">${esc(r.label)}</div>`;
    if (r.label === 'Safety checkpoint') return checkpointKpi();
    if (r.label.startsWith('Blood-covered')) {
      const pk = rawPeak(rec);
      return `<div class="kpi">${k}<div class="v">${esc(num(r.value, 1))}<u>%</u></div>
        <div class="n">${pk ? `single-frame peak, ${esc(clock(pk.t))}` : 'peak'}</div></div>`;
    }
    if (r.status === 'CANNOT_MEASURE') {
      return `<div class="kpi">${k}<div class="v faint warnfg">Cannot measure</div><div class="n">scale not steady</div></div>`;
    }
    if (r.label === 'Bleeding onset') {
      if (r.measured && r.value != null) {
        return `<div class="kpi hot">${k}<div class="v">${esc(clock(r.value * 1000))}${r.plus_minus ? `<u>± ${num(r.plus_minus, 1)} s</u>` : ''}</div><div class="n">detected</div></div>`;
      }
      return `<div class="kpi">${k}<div class="v faint">Not detected</div><div class="n">${rec.metrics?.onset?.candidates ? 'rises blocked by gates' : 'no rise'}</div></div>`;
    }
    if (r.label === 'Frames measurable') {
      return `<div class="kpi ${r.value >= 50 ? 'good' : ''}">${k}<div class="v">${esc(num(r.value, 1))}<u>%</u></div><div class="n">${q.usable ?? 0} of ${q.frames ?? 0} frames</div></div>`;
    }
    return `<div class="kpi">${k}<div class="v">${esc(r.value ?? '—')}</div></div>`;
  }).join('');
}

function checkpointKpi() {
  const head = `<div class="k">Safety checkpoint</div>`;
  const cp = state.reasoning?.checkpoint;
  if (!cp) {
    return `<div class="kpi">${head}<div class="v faint">Not decided</div>
      <div class="n">${state.record ? 'not run yet' : ''}</div></div>`;
  }
  const last = (state.agent?.checkpoints || []).slice(-1)[0];
  if (cp.decision === 'raise') {
    const answered = last && !last.open;
    const note = answered ? decidedLine(last) : 'Nemotron · awaiting a person';
    return `<div class="kpi ${answered ? 'good' : 'hot'}">${head}<div class="v">${answered ? 'Answered' : 'Raised'}<u>${esc(clock((cp.at_s || 0) * 1000))}</u></div>
      <div class="n">${esc(note)}</div></div>`;
  }
  if (cp.decision === 'cannot_assess') {
    return `<div class="kpi">${head}<div class="v faint warnfg">Cannot assess</div><div class="n">Nemotron</div></div>`;
  }
  return `<div class="kpi nv">${head}<div class="v">Not raised</div><div class="n">Nemotron</div></div>`;
}

function openCheckpoint() {
  const id = state.agent?.open_checkpoint;
  if (!id) return null;
  return (state.agent.checkpoints || []).find((c) => c.checkpoint_id === id) || null;
}
function decidedLine(cp) {
  if (!cp || !cp.decided_by) return 'no checkpoint has been answered';
  const word = cp.state === 'confirmed' ? 'confirmed' : 'dismissed with a reason';
  return `${word} by ${cp.decided_by} at ${localTime(cp.decided_at)}`;
}

/* ────────────────────── checkpoint card ────────────────────── */
function renderCheckpoint() {
  const card = $('#checkpoint-card'), resolved = $('#checkpoint-resolved');
  const cps = state.agent?.checkpoints || [];
  const open = openCheckpoint();
  const last = cps.length ? cps[cps.length - 1] : null;

  card.hidden = !open;
  resolved.hidden = !(last && !open);

  if (open) {
    $('#cp-hold').innerHTML =
      `<svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true"><path d="M9 1.6l7.4 13H1.6z" stroke="#fff" stroke-width="1.8" stroke-linejoin="round"/><path d="M9 6.6v3.4" stroke="#fff" stroke-width="1.9" stroke-linecap="round"/><circle cx="9" cy="12.2" r="1" fill="#fff"/></svg>
       ${open.checkpoint_id.startsWith('nm') ? 'Nemotron raised a checkpoint at' : 'Held at'} ${esc(clock(open.at_ms))} — frame ${open.frame_index} pinned as evidence
       <span style="margin-left:auto">Phase ${esc(PHASE_NAMES[open.phase] || open.phase)}</span>`;
    $('#cp-question').textContent = open.question;
    
    const sel = $('#cp-reason');
    if (sel.options.length <= 1) {
      (open.reasons_offered || []).forEach((r) => {
        const o = document.createElement('option');
        o.value = r; o.textContent = r; sel.appendChild(o);
      });
    }
    $('#cp-live').textContent = `A safety checkpoint is held at ${clock(open.at_ms)} and is waiting on a named person.`;
    syncCheckpointButtons();
  }
  if (last && !open) {
    const confirmed = last.state === 'confirmed';
    $('#cpr-chip').className = `chip ${confirmed ? 'ok' : ''}`;
    $('#cpr-chip').textContent = confirmed ? '✓ Confirmed' : '● Reason recorded';
    $('#cpr-line').textContent = confirmed
      ? `${last.decided_by} confirmed the critical view of safety was established.`
      : `${last.decided_by} proceeded without confirming, and gave a reason.`;
    $('#cpr-reason').textContent = [last.reason, last.note].filter(Boolean).join(' — ');
    $('#cpr-meta').textContent = `${clock(last.at_ms)} · frame ${last.frame_index}`;
    $('#cp-live').textContent = `Checkpoint ${last.state} by ${last.decided_by}.`;
  }

  const sub = $('#cr-sub');
  sub.textContent = cps.length ? `${cps.length} raised · agent ${state.agent?.state || 'observing'}` : 'none raised';
  $('#cr-body').innerHTML = cps.length
    ? `<table><thead><tr><th style="width:86px">Time</th><th>Decision</th><th style="width:110px">By</th></tr></thead><tbody>${
      cps.map((c) => {
        const chip = c.state === 'held' ? '<span class="chip hot">Awaiting</span>'
          : c.state === 'confirmed' ? '<span class="chip ok">Confirmed</span>'
            : '<span class="chip">Reason given</span>';
        const why = c.state === 'dismissed' && c.reason ? esc(c.reason) : 'Safety view before the irreversible step';
        return `<tr class="seek" data-ms="${c.at_ms}"><td class="mono dim">${esc(clock(c.at_ms))}</td>
          <td>${chip} ${why}</td><td class="dim">${esc(c.decided_by || 'nobody yet')}</td></tr>`;
      }).join('')}</tbody></table>`
    : `<p class="pad faint">${state.reasoning?.checkpoint
      ? (state.reasoning.checkpoint.decision === 'cannot_assess'
        ? 'None raised.'
        : 'None raised.')
      : 'None raised.'}</p>`;
  wireSeek($('#cr-body'));
}

function syncCheckpointButtons() {
  const actor = $('#cp-actor').value.trim();
  const reason = $('#cp-reason').value;
  $('#cp-confirm').disabled = !actor;
  $('#cp-dismiss').disabled = !(actor && reason);
}

async function decide(kind) {
  const cp = openCheckpoint();
  if (!cp || !state.jobId) return;
  const body = {
    actor: $('#cp-actor').value.trim(),
    note: $('#cp-note').value.trim(),
  };
  if (kind === 'dismiss') body.reason = $('#cp-reason').value;
  $('#cp-confirm').disabled = $('#cp-dismiss').disabled = true;
  hideError();
  try {
    const url = `/api/jobs/${encodeURIComponent(state.jobId)}/checkpoints/${encodeURIComponent(cp.checkpoint_id)}/${kind}`;
    const res = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const payload = await res.json().catch(() => null);
    if (!res.ok) throw new ApiError(payload?.error || { message: res.statusText }, res.status);
    state.agent = payload.agent;
    state.events = buildEvents(state.record);
    renderCheckpoint();
    renderKpis(state.record);
    renderEvents();
    renderRail(state.record);
    $('#checkpoint-resolved').scrollIntoView({ block: 'nearest' });
  } catch (err) {
    showError(err);
    syncCheckpointButtons();
  }
}

/* ────────────────────────── the event log ────────────────────────── */
const ACTION_TEXT = {
  rescan_window: (d) => `Re-read ${clock(d.start_ms)} to ${clock(d.end_ms)} at full frame rate`,
  record_onset: () => 'Onset recorded from the coarse pass',
  request_clean_lens: (d) => `Clean lens requested — ${d.consecutive_fogged_frames} fogged frames suppressed onset detection`,
  hold_checkpoint: () => 'Held for a person',
  resume: () => 'Resumed observing; the phase left the critical approach',
};
const ACTION_VALUE = {
  rescan_window: (d, a) => a.result?.frames_read != null ? `${a.result.frames_read} frames` : `stride ${d.stride}`,
  record_onset: (d) => d.field_fraction != null ? `${num(d.field_fraction * 100, 1)}% of field` : 'recorded',
  request_clean_lens: () => 'suppressed',
  hold_checkpoint: () => 'held',
  resume: () => 'observing',
};

function buildEvents(rec) {
  const out = [];
  const agent = state.agent || rec.metrics?.agent || {};
  const onset = rec.metrics?.onset;

  if (onset?.detected) {
    out.push({
      ms: onset.timestamp_ms, cls: 'hotfg',
      text: `Bleeding onset — rate crossed ${num(onset.threshold_per_min, 1)} ${onset.unit || ''}`,
      value: `${num(onset.rate_per_min, 1)} ${onset.unit || ''}`,
    });
  }
  (agent.actions || []).filter((a) => a.kind !== 'hold_checkpoint').forEach((a) => {
    const text = (ACTION_TEXT[a.kind] || (() => sentence(a.kind)))(a.detail || {}, a);
    const value = (ACTION_VALUE[a.kind] || (() => (a.performed ? 'performed' : 'planned')))(a.detail || {}, a);
    out.push({ ms: a.at_ms, text, value, cls: a.performed ? '' : 'dim', uri: a.evidence_uri });
  });
  (agent.transitions || []).forEach((t) => {
    const human = t.actor && t.actor !== 'system';
    out.push({
      ms: t.at_ms,
      text: t.to === 'held' ? 'Checkpoint held' : t.to === 'confirmed' ? 'Checkpoint confirmed' : t.to === 'dismissed' ? `Dismissed${t.reason ? `: ${t.reason}` : ''}` : sentence(t.to),
      value: human ? (t.actor === 'nemotron' ? 'Nemotron' : t.actor) : 'system',
      cls: t.to === 'held' ? 'hotfg' : (t.to === 'confirmed' ? 'okfg' : 'dim'),
      uri: t.evidence_uri,
    });
  });

  // Runs where the gates refused to measure. The suspension is itself a measurement.
  const refs = state.reasoning?.refusal_events;
  const plain = new Map((state.reasoning?.explanations || []).map((x) => [x.id, x]));
  if (refs?.length) {
    refs.forEach((r) => {
      const ex = plain.get(r.id);
      out.push({
        ms: r.start_s * 1000, cls: 'warnfg',
        text: `Measurement suspended for ${(r.end_s - r.start_s + 0.2).toFixed(1)} s — ${refusalWord(r.code)}`,
        value: `${r.frames} frame${r.frames === 1 ? '' : 's'}`,
        plain: ex ? firstSentences(ex.plain, 1) : null,
      });
    });
    return out.sort((x, y) => y.ms - x.ms);
  }
  const s = rec.metrics?.series;
  if (s?.measurable?.length) {
    const codesAt = (a, b) => {
      const set = new Set();
      (rec.evidence || []).forEach((e) => {
        const code = e.metrics?.refusal;
        if (code && e.timestamp_ms >= a - 1 && e.timestamp_ms <= b + 1) set.add(code);
      });
      return Array.from(set);
    };
    let i = 0;
    while (i < s.measurable.length) {
      if (s.measurable[i]) { i += 1; continue; }
      let j = i;
      while (j + 1 < s.measurable.length && !s.measurable[j + 1]) j += 1;
      const a = s.times_ms[i], b = s.times_ms[j];
      const codes = codesAt(a, b);
      out.push({
        ms: a, cls: 'warnfg',
        text: `Measurement suspended for ${((b - a) / 1000 + 0.1).toFixed(1)} s — ${codes.length ? codes.map(refusalWord).join(', ') : 'the frame failed a quality gate'}`,
        value: `${j - i + 1} frames`,
      });
      i = j + 1;
    }
  }
  return out.sort((x, y) => y.ms - x.ms);
}

function renderEvents() {
  const rows = state.events;
  $('#ev-sub').textContent = `${rows.length} event${rows.length === 1 ? '' : 's'}`;
  $('#ev-body').innerHTML = rows.length
    ? rows.map((e) => `<tr class="seek" data-ms="${e.ms}" tabindex="0">
        <td class="mono dim">${esc(clock(e.ms))}</td>
        <td>${esc(e.text)}${e.plain ? `<span class="plain"><em>Nemotron</em>${esc(e.plain)}</span>` : ''}</td>
        <td class="num ${e.cls === 'dim' ? 'dim' : esc(e.cls)}">${esc(e.value)}</td></tr>`).join('')
    : '<tr><td colspan="3" class="faint">Nothing happened that was worth a line in the log.</td></tr>';
  wireSeek($('#ev-body'));
}

function wireSeek(root) {
  $$('.seek', root).forEach((tr) => {
    tr.tabIndex = 0;
    tr.addEventListener('click', () => seek(Number(tr.dataset.ms)));
    tr.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seek(Number(tr.dataset.ms)); }
    });
  });
}

function seek(msValue) {
  if (!isFinite(msValue)) return;
  const video = $('#video');
  state.cursorMs = Math.max(0, msValue);
  if (video.src && state.decodable) {
    try { video.currentTime = state.cursorMs / 1000; } catch { /* not seekable yet */ }
    video.pause();
  }
  onPlayhead();
}

/* Not every laparoscopic clip is in a codec a browser can decode — the bundled
 * sample is MPEG-4 Part 2, which Chromium will not touch. Rather than leave a
 * black rectangle where the field should be, the panel falls back to the frames
 * the measurement was actually made on and says so. */
function checkCodec() {
  const video = $('#video');
  if (!video.src) return;
  const broken = Boolean(video.error) || (video.readyState >= 2 && video.videoWidth === 0);
  setDecodable(!broken);
}

function setDecodable(ok) {
  if (state.decodable === ok) return;
  state.decodable = ok;
  const video = $('#video');
  video.hidden = !ok;
  $('#field-img').hidden = ok;
  const note = $('#vid-nocodec');
  note.hidden = ok;
  $('#live-field').classList.toggle('fallback', !ok);
  if (!ok) {
    note.innerHTML = '<b>This browser cannot decode the clip.</b> Showing the overlay frames the measurement was made on. Click an event, a phase or an evidence frame to move through the case.';
    $('#vid-empty').hidden = true;
  }
  onPlayhead();
}

/* ────────────────────────── evidence ────────────────────────── */
function pickEvidence(rec) {
  const all = rec.evidence || [];
  if (all.length <= 10) return all;
  // Every measured frame earns its place. Refused frames earn two each at most:
  // ten pictures of the same complaint is a wall, not evidence.
  const picked = [];
  const perCode = new Map();
  all.forEach((e) => {
    const code = e.metrics?.refusal;
    if (!code) { picked.push(e); return; }
    const n = perCode.get(code) || 0;
    if (n < 3) { perCode.set(code, n + 1); picked.push(e); }
  });
  // A clip that refused everything still deserves a strip worth looking at.
  const seen = new Set(picked.map((e) => e.uri));
  const step = Math.max(1, Math.floor(all.length / 6));
  for (let i = 0; i < all.length && picked.length < 6; i += step) {
    if (!seen.has(all[i].uri)) { picked.push(all[i]); seen.add(all[i].uri); }
  }
  return picked.slice(0, 12).sort((a, b) => (a.timestamp_ms || 0) - (b.timestamp_ms || 0));
}

function renderEvidence() {
  const items = state.evidence;
  const total = state.record?.evidence?.length || 0;
  $('#evd-sub').textContent = total ? `${items.length} of ${total}` : '';
  $('#evd-strip').innerHTML = items.map((e) => {
    const refused = e.metrics?.refusal;
    return `<button type="button" class="ev ${refused ? 'flag' : ''}" data-ms="${e.timestamp_ms || 0}" data-uri="${esc(e.uri)}">
      <span class="shot"><img src="${esc(e.uri)}" alt="Overlay frame at ${esc(clock(e.timestamp_ms))}" loading="lazy"></span>
      <span class="t">${esc(clock(e.timestamp_ms))} · frame ${e.frame_index ?? '—'}</span>
      <span class="c">${esc(e.caption || e.label)}</span></button>`;
  }).join('') || '<p class="cell faint">No evidence frames were saved for this run.</p>';
  $$('#evd-strip .ev').forEach((b) => b.addEventListener('click', () => {
    seek(Number(b.dataset.ms));
    showOverlay(state.record.evidence.find((e) => e.uri === b.dataset.uri));
  }));
}

function showOverlay(ev) {
  const img = $('#ov-img');
  if (!ev) { img.hidden = true; return; }
  img.src = ev.uri;
  img.alt = `Overlay frame at ${clock(ev.timestamp_ms)}: ${ev.caption || ev.label}`;
  img.hidden = false;
  $('#ov-cap').textContent = ev.caption || ev.label;
  $('#ov-tm').textContent = `${clock(ev.timestamp_ms)} · frame ${ev.frame_index ?? '—'}${ev.metrics?.refusal ? ` · refused ${ev.metrics.refusal}` : ''}`;
}

/* ────────────────────── instruments and cost ────────────────────── */
function renderInstruments(rec) {
  const scale = rec.metrics?.scale || {};
  const q = rec.metrics?.quality || {};
  const stages = rec.timings?.stages || [];
  const totalMs = rec.timings?.total_ms || 0;
  const counts = (rec.evidence || []).map((e) => e.metrics?.instruments?.count).filter((n) => n != null);
  const maxCount = counts.length ? Math.max(...counts) : 0;

  const gate = scale.gate || {};
  $('#in-sub').textContent = gate.passed
    ? `scale from ${String(scale.source).replace(/_/g, ' ')}, gate passed`
    : `volume withheld: ${String(gate.reason_code || 'no scale').replace(/_/g, ' ').toLowerCase()}`;

  $('#in-body').innerHTML = `
    <div class="cell"><h3>Scale</h3>
      <div class="kv"><span>Millimetres per pixel</span><span>${scale.mm_per_px != null ? `${num(scale.mm_per_px, 4)} ± ${num(scale.mm_per_px_sigma, 4)}` : 'not recovered'}</span></div>
      <div class="kv"><span>Scale gate</span><span>${esc(sentence(gate.status || 'none'))}${gate.passed ? '' : ' — volume withheld'}</span></div>
      <div class="kv"><span>Frames with a shaft scale</span><span>${gate.frames_with_scale ?? 0} of ${gate.measurable_frames ?? 0}</span></div>
      <div class="kv"><span>Scale variation across the case</span><span>${gate.robust_cv != null ? `${num(gate.robust_cv * 100, 0)}% (limit ${num((gate.max_robust_cv || 0) * 100, 0)}%)` : '—'}</span></div>
      <div class="kv"><span>Implied field width</span><span>${scale.implied_field_width_mm != null ? `${num(scale.implied_field_width_mm, 0)} mm` : '—'}</span></div>
      <div class="kv"><span>Assumed shaft</span><span>${num(scale.assumed_shaft_mm, 1)} mm</span></div>
      <div class="kv"><span>Instruments in view, peak</span><span>${maxCount}</span></div>
    </div>
    <div class="cell"><h3>Frame quality</h3>
      <div class="kv"><span>Frames read</span><span>${q.frames ?? 0}</span></div>
      <div class="kv"><span>Measurable</span><span>${q.usable ?? 0} — ${num((q.usable_fraction || 0) * 100, 1)}%</span></div>
      ${Object.entries(q.rejected_by || {}).map(([c, n]) =>
    `<div class="kv"><span>${esc(c)}</span><span>${n}</span></div>`).join('') ||
    '<div class="kv"><span>Refusals</span><span>none</span></div>'}
    </div>
    <div class="cell"><h3>Stage timings</h3>
      ${stages.map((s) => `<div class="kv"><span>${esc(s.name)} · ${s.calls} calls</span><span>${ms(s.ms)} — ${num(s.ms_per_call, 1)} ms each</span></div>`).join('')}
      <div class="kv"><span>Total</span><span>${ms(totalMs)}</span></div>
    </div>
    <div class="cell"><h3>Accuracy of the blood share</h3>
      ${validationCell()}
    </div>`;
}

function validationCell() {
  const seg = state.evaluation?.real?.segmentation?.after;
  if (!seg?.test) return '<p class="faint">Validation figures are not loaded.</p>';
  const pc = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);
  return `
    <div class="valcmp">
      <div><b>${pc(seg.test.precision)} · ${pc(seg.test.recall)}</b><span>precision · recall, held-out clips</span></div>
      <div><b>${pc(seg.dev.precision)} · ${pc(seg.dev.recall)}</b><span>dev clips</span></div>
      <div><b>${pc(seg.all_laparoscopic.precision)} · ${pc(seg.all_laparoscopic.recall)}</b><span>pooled, 15 laparoscopic clips</span></div>
    </div>
    <p class="faint" style="font-size:13.5px;margin-top:8px">Precision-weighted by design.</p>`;
}

function renderRail(rec) {
  const q = rec.metrics?.quality || {};
  const stages = rec.timings?.stages || [];
  const slow = stages.slice().sort((a, b) => b.ms - a.ms)[0];
  $('#st-frames').textContent = `${q.usable ?? 0} of ${q.frames ?? 0} measurable`;
  const pct = Math.round((q.usable_fraction || 0) * 100);
  $('#st-meter').className = `meter ${pct < 50 ? 'warn' : ''}`;
  $('#st-meter').firstElementChild.style.width = `${pct}%`;
  $('#st-stage').textContent = slow ? `${slow.name} ${num(slow.ms_per_call, 1)} ms/call` : 'none';
  $('#st-total').textContent = ms(rec.timings?.total_ms || 0);
  $('#st-opencv').textContent = rec.env?.opencv_version || $('#st-opencv').textContent;

  const counts = (rec.evidence || []).map((e) => e.metrics?.instruments?.count).filter((n) => n != null);
  $('#ct-live').textContent = q.usable ?? 0;
  $('#ct-trace').textContent = rec.metrics?.series?.times_ms?.length || 0;
  $('#ct-events').textContent = state.events.length;
  $('#ct-checkpoints').textContent = (state.agent?.checkpoints || []).length;
  $('#ct-instruments').textContent = counts.length ? Math.max(...counts) : 0;
  const d = state.reasoning?.checkpoint?.decision;
  $('#ct-reasoning').textContent = !d ? '—' : d === 'raise' ? 'raise' : d === 'cannot_assess' ? 'n/a' : 'clear';
  $('#ct-evidence').textContent = (rec.evidence || []).length;
}

function renderPhases(rec) {
  const p = rec.metrics?.phases;
  const host = $('#phases');
  const spans = p?.spans || [];
  if (!spans.length) { host.innerHTML = ''; return; }
  const open = openCheckpoint();
  host.setAttribute('aria-label', 'Operative phases, experimental: validated on scripted synthetic sequences only');
  host.innerHTML = spans.map((s) => {
    const now = open && open.at_ms >= s.start_ms && open.at_ms <= s.end_ms;
    return `<button type="button" class="pz seek ${now ? 'now' : 'done'}" role="listitem" data-ms="${s.start_ms}"
      style="flex:${Math.max(1, s.duration_ms)} 1 0" title="${esc(s.frames)} frames">
      <b>${esc(PHASE_NAMES[s.phase] || sentence(s.phase))}</b>${esc(clock(s.start_ms))} to ${esc(clock(s.end_ms))}${
      now ? '<i>▲ held here</i>' : ''}</button>`;
  }).join('');
  wireSeek(host);
}

/* ────────────────────────── the field trace ────────────────────────── */
function drawTrace() {
  const host = $('#trace-host');
  const rec = state.record;
  const s = rec?.metrics?.series;
  const W = Math.max(320, host.clientWidth || 720);
  const tight = W < 520;
  const H = 110, L = tight ? 48 : 58, R = 14, TOP = 8, BOT = 18;
  if (!s || !s.times_ms?.length) {
    host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="The field trace is empty until a case has been analysed.">
      <rect width="${W}" height="${H}" fill="#FFFFFF"/>
      <text x="${L}" y="52" font-family="Archivo" font-size="14" fill="#5F6E69">The field trace draws here across the whole case, once one has run.</text></svg>`;
    state.geom = null;
    return;
  }

  const t0 = s.times_ms[0], t1 = s.times_ms[s.times_ms.length - 1] || t0 + 1;
  const span = Math.max(1, t1 - t0);
  const peak = Math.max(...s.raw.filter((_, k) => s.measurable[k]), ...s.smoothed, 0);
  // Nothing measurable: a zeroed axis would be a lie. A measured 0% is not that.
  const flat = !s.measurable.some(Boolean);
  const vmax = Math.max(0.5, peak * 1.18);
  const x0 = L, x1 = W - R, yT = TOP, yB = H - BOT;
  const X = (t) => x0 + ((t - t0) / span) * (x1 - x0);
  const Y = (v) => yB - (Math.max(0, v) / vmax) * (yB - yT);
  state.geom = { x0, x1, t0, t1, W };

  const parts = [];
  parts.push(`<rect width="${W}" height="${H}" fill="#FFFFFF"/>`);

  // ruled paper
  const ticks = [0, 0.5, 1];
  const rules = ticks.map((f) => `M${x0} ${Y(vmax * f).toFixed(1)}H${x1}`).join('');
  const nx = tight ? 3 : 6;
  const verticals = Array.from({ length: nx + 1 }, (_, i) =>
    `M${(x0 + ((x1 - x0) * i) / nx).toFixed(1)} ${yT}V${yB}`).join('');
  parts.push(`<g stroke="#E1E8E5" stroke-width="1" fill="none"><path d="${rules}"/><path d="${verticals}"/></g>`);
  if (!flat) {
    parts.push(`<g font-family="Archivo" font-size="14" fill="#5F6E69">${
      ticks.map((f) => `<text x="4" y="${(Y(vmax * f) + 4.5).toFixed(1)}">${num(vmax * f, 1)}%</text>`).join('')}</g>`);
  }

  // gaps, drawn before the line so the line sits on top
  let i = 0;
  while (i < s.measurable.length) {
    if (s.measurable[i]) { i += 1; continue; }
    let j = i;
    while (j + 1 < s.measurable.length && !s.measurable[j + 1]) j += 1;
    const gx0 = X(s.times_ms[i]), gx1 = Math.max(X(s.times_ms[j]), X(s.times_ms[i]) + 2);
    parts.push(`<rect x="${gx0.toFixed(1)}" y="${yT}" width="${(gx1 - gx0).toFixed(1)}" height="${yB - yT}" fill="#B7791F" opacity=".14"/>`);
    parts.push(`<path d="M${gx0.toFixed(1)} ${yT}V${yB}M${gx1.toFixed(1)} ${yT}V${yB}" stroke="#B7791F" stroke-width="1" opacity=".55"/>`);
    if (gx1 - gx0 > 74) {
      parts.push(`<rect x="${(gx0 + 3).toFixed(1)}" y="${yB - 21}" width="68" height="18" rx="2" fill="#F3E3B8"/>`);
      parts.push(`<text x="${(gx0 + 8).toFixed(1)}" y="${yB - 7}" font-family="Archivo" font-size="14" font-weight="700" fill="#6B4700">no data</text>`);
    }
    i = j + 1;
  }

  // the line, broken where the field could not be measured
  const segments = [];
  let seg = [];
  for (let k = 0; k < s.smoothed.length; k += 1) {
    if (s.measurable[k]) seg.push(`${X(s.times_ms[k]).toFixed(1)} ${Y(s.smoothed[k]).toFixed(1)}`);
    else if (seg.length) { segments.push(seg); seg = []; }
  }
  if (seg.length) segments.push(seg);
  const d = segments.filter((g) => g.length > 1).map((g) => `M${g.join('L')}`).join(' ');
  const area = segments.filter((g) => g.length > 1).map((g) => {
    const first = g[0].split(' ')[0], last = g[g.length - 1].split(' ')[0];
    return `M${first} ${yB}L${g.join('L')}L${last} ${yB}Z`;
  }).join(' ');
  if (area) parts.push(`<path d="${area}" fill="#C2410C" opacity=".10"/>`);
  if (d) {
    const len = Math.round((x1 - x0) * 1.6);
    parts.push(`<path class="line" style="--len:${len}" d="${d}" fill="none" stroke="#C2410C" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>`);
  }

  // per-frame values, thin, so the single-frame peak in the KPI band is visible here
  const rawSegs = []; let rs = [];
  for (let k = 0; k < s.raw.length; k += 1) {
    if (s.measurable[k]) rs.push(`${X(s.times_ms[k]).toFixed(1)} ${Y(s.raw[k]).toFixed(1)}`);
    else if (rs.length) { rawSegs.push(rs); rs = []; }
  }
  if (rs.length) rawSegs.push(rs);
  const rd = rawSegs.filter((g) => g.length > 1).map((g) => `M${g.join('L')}`).join(' ');
  if (rd) parts.push(`<path d="${rd}" fill="none" stroke="#C2410C" stroke-width="1" opacity=".35"/>`);
  const pkp = rawPeak(rec);
  if (pkp && pkp.v > 0) {
    const px = X(pkp.t), py = Y(pkp.v);
    parts.push(`<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="3.6" fill="#fff" stroke="#C2410C" stroke-width="1.8"/>`);
    const pl = `peak ${num(pkp.v, 1)}%`, pw = 14 + pl.length * 7;
    const lx2 = px + pw + 8 > x1 ? px - pw - 8 : px + 8;
    parts.push(`<text x="${lx2.toFixed(1)}" y="${(py + 4).toFixed(1)}" font-family="Archivo" font-size="14" font-weight="700" fill="#C2410C">${pl}</text>`);
  }

  // the onset: a vertical amber rule, its uncertainty, and a time label
  const onset = rec.metrics?.onset;
  if (onset?.detected && onset.timestamp_ms != null) {
    const ox = X(onset.timestamp_ms);
    const u = (onset.uncertainty_ms || 0) / span * (x1 - x0);
    if (u > 1) parts.push(`<rect x="${(ox - u).toFixed(1)}" y="${yT}" width="${(u * 2).toFixed(1)}" height="${yB - yT}" fill="#C2410C" opacity=".07"/>`);
    parts.push(`<path d="M${ox.toFixed(1)} ${yT}V${yB}" stroke="#C2410C" stroke-width="1.6" stroke-dasharray="4 3"/>`);
    const idx = s.times_ms.findIndex((t) => t >= onset.timestamp_ms);
    if (idx >= 0) parts.push(`<circle cx="${ox.toFixed(1)}" cy="${Y(s.smoothed[idx]).toFixed(1)}" r="4" fill="#C2410C"/>`);
    const label = `onset ${clock(onset.timestamp_ms)} ± ${num(onset.uncertainty_ms / 1000, 0)} s`;
    const w = 20 + label.length * 7.6;
    const lx = Math.max(x0 + 2, ox + w + 8 > x1 ? ox - w - 6 : ox + 6);
    // The caption owns the top-left corner, so a label that would land on it
    // drops to the second row rather than printing through it.
    const ly = lx < x0 + 158 ? yT + 21 : yT;
    parts.push(`<rect x="${lx.toFixed(1)}" y="${ly}" width="${w.toFixed(1)}" height="19" rx="2" fill="#C2410C"/>`);
    parts.push(`<text x="${(lx + 8).toFixed(1)}" y="${ly + 14}" font-family="Archivo" font-size="14" font-weight="700" fill="#FFFFFF">${esc(label)}</text>`);
  }

  parts.push(`<text x="${x0 + 6}" y="${yT + 14}" font-family="Archivo" font-size="14" fill="#475651">Blood-covered field, % of view · smoothed (thin line: per frame)</text>`);
  if (flat) {
    parts.push(`<text x="${((x0 + x1) / 2).toFixed(1)}" y="${((yT + yB) / 2 + 5).toFixed(1)}" text-anchor="middle"
      font-family="Archivo" font-size="14" font-weight="600" fill="#8F5F00">No frame in this clip could be measured, so the trace carries no line</text>`);
  }
  const labels = Array.from({ length: nx + 1 }, (_, k) => t0 + (span * k) / nx);
  parts.push(`<g font-family="Archivo" font-size="14" fill="#5F6E69">${
    labels.map((t, k) => {
      const x = X(t);
      const anchor = k === 0 ? 'start' : k === nx ? 'end' : 'middle';
      return `<text x="${x.toFixed(1)}" y="${H - 4}" text-anchor="${anchor}">${clock(t)}</text>`;
    }).join('')}</g>`);

  const summary = `The field trace: share of the visible field covered in blood from ${clock(t0)} to ${clock(t1)}, peak ${num(peak, 1)} per cent${
    onset?.detected ? `, onset marked at ${clock(onset.timestamp_ms)}` : ', no onset detected'}.`;
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(summary)}">${parts.join('')}</svg>`;
}

function onPlayhead() {
  const video = $('#video');
  const cursor = $('#trace-cursor');
  if (!state.geom || !video.src) { cursor.style.display = 'none'; return; }
  const t = state.cursorMs;
  const { x0, x1, t0, t1, W } = state.geom;
  const host = $('#trace-host');
  const scale = (host.clientWidth || W) / W;
  const x = (x0 + ((Math.min(Math.max(t, t0), t1) - t0) / Math.max(1, t1 - t0)) * (x1 - x0)) * scale;
  cursor.style.display = 'block';
  cursor.style.left = `${x.toFixed(1)}px`;

  const bar = $('#vidbar');
  const dur = state.record?.metrics?.video?.duration_ms;
  const open = state.agent?.open_checkpoint;
  bar.hidden = false;
  bar.innerHTML = `<span class="rec ${isUnmeasurableNow(t) ? 'warn' : ''}"></span>${
    isUnmeasurableNow(t) ? 'Not measurable at this frame' : 'Measured at this frame'}
    <span style="margin-left:auto">${esc(clock(t))} / ${esc(clock(dur))}${open ? ' · step held' : ''}</span>`;

  const nearest = nearestEvidence(t);
  if (nearest) {
    if (!$('#ov-img').src.endsWith(nearest.uri)) showOverlay(nearest);
    if (!state.decodable) {
      const big = $('#field-img');
      if (!big.src.endsWith(nearest.uri)) {
        big.src = nearest.uri;
        big.alt = `The field at ${clock(nearest.timestamp_ms)}: ${nearest.caption || nearest.label}`;
      }
    }
  }
}

function isUnmeasurableNow(t) {
  const s = state.record?.metrics?.series;
  if (!s?.times_ms?.length) return false;
  let best = 0, bestD = Infinity;
  s.times_ms.forEach((tm, k) => { const d = Math.abs(tm - t); if (d < bestD) { bestD = d; best = k; } });
  return !s.measurable[best];
}

function nearestEvidence(t) {
  const all = state.record?.evidence || [];
  if (!all.length) return null;
  let best = null, bestD = Infinity;
  all.forEach((e) => {
    const d = Math.abs((e.timestamp_ms ?? 0) - t);
    if (d < bestD) { bestD = d; best = e; }
  });
  return best;
}

/* ────────────────────── nav highlighting ────────────────────── */
function watchSections() {
  const map = {
    'live-field': 'live', 'field-trace': 'trace', reasoning: 'reasoning', events: 'events',
    checkpoints: 'checkpoints', instruments: 'instruments', evidence: 'evidence',
  };
  const links = $$('.nav a');
  const setActive = (key) => links.forEach((a) => a.classList.toggle('on', a.dataset.nav === key));
  setActive('live');
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((e) => e.isIntersecting)
      .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (visible) setActive(map[visible.target.id]);
  }, { rootMargin: '-80px 0px -55% 0px', threshold: 0.01 });
  Object.keys(map).forEach((id) => { const el = document.getElementById(id); if (el) observer.observe(el); });

  $('.nav a[data-nav="checkpoints"]').addEventListener('click', (e) => {
    const card = $('#checkpoint-card').hidden ? ($('#checkpoint-resolved').hidden ? null : $('#checkpoint-resolved')) : $('#checkpoint-card');
    if (card) { e.preventDefault(); card.scrollIntoView({ block: 'center' }); }
  });
}


/* ────────────────────────── the cases ────────────────────────── */
function caseMeta() { return state.cases.find((c) => c.slug === state.slug) || null; }

function decPill(d) {
  if (d === 'raise') return '<span class="dec raise">Raise</span>';
  if (d === 'do_not_raise') return '<span class="dec no">Not raised</span>';
  if (d === 'cannot_assess') return '<span class="dec cannot">Cannot assess</span>';
  return '';
}

function renderCases() {
  $('#cases').innerHTML = state.cases.map((c) => {
    const cur = c.slug === state.slug;
    const d = cur && state.reasoning ? state.reasoning.checkpoint?.decision : c.decision;
    return `<button type="button" class="case" role="listitem" data-slug="${esc(c.slug)}" ${cur ? 'aria-current="true"' : ''}>
      <b>${esc(c.title)}</b><span class="meta">${esc(clock(c.duration_ms))} ${decPill(d)}</span></button>`;
  }).join('');
  $$('#cases .case').forEach((b) => b.addEventListener('click', () => loadCase(b.dataset.slug)));
}

async function loadCase(slug) {
  hideError();
  if (state.closeStream) { state.closeStream(); state.closeStream = null; }
  clearInterval(state.polling);
  $('#prog').hidden = true;
  state.slug = slug; state.uploaded = false; state.jobId = slug;
  state.cursorMs = 0; state.decodable = true;
  renderCases();
  try {
    const [job, rs] = await Promise.all([
      api.job(slug),
      fetch(`/api/jobs/${encodeURIComponent(slug)}/reasoning`).then((r) => r.json()),
    ]);
    state.record = job.result;
    state.agent = await fetchAgent(slug);
    state.reasoning = rs.reasoning; state.reasoningSource = rs.source;
    state.liveAvailable = Boolean(rs.live_available);
    state.procedure = rs.procedure || '';
    if (state.videoUrl) { URL.revokeObjectURL(state.videoUrl); state.videoUrl = null; }
    const video = $('#video');
    video.hidden = false; $('#field-img').hidden = true; $('#vid-nocodec').hidden = true; $('#vid-empty').hidden = true;
    video.src = `/api/cases/${encodeURIComponent(slug)}/clip.mp4`;
    setTimeout(checkCodec, 1500);
    history.replaceState(null, '', `?case=${encodeURIComponent(slug)}`);
    renderCases();
    renderAll();
  } catch (err) { showError(err); }
}

/* ────────────────────────── the Nemotron layer ────────────────────────── */
const METRIC = {
  blood_pct: ['Blood-covered field', (v) => `${v}%`],
  rate_pct_per_min: ['Rate of change', (v) => `${v} %/min`],
  phase: ['Phase track', (v) => String(v)],
  refusal: ['Refusal', (v) => String(v)],
  instruments: ['Instruments in view', (v) => String(v)],
};
const WITHHELD = { CLIP: 'The whole clip', ONSET: 'Bleeding onset', VOLUME: 'Millilitres' };

function figLabel(f) {
  const map = { peak_blood_pct: 'peak', median_blood_pct: 'median', blood_pct: 'at ' + clock((f.t_s || 0) * 1000),
    refused_frames: 'frames refused', usable_frames: 'frames usable' };
  const unit = f.metric.endsWith('pct') ? '%' : '';
  return [map[f.metric] || f.metric, `${f.value}${unit}`];
}

function firstSentences(t, n) {
  const parts = String(t || '').trim().split(/(?<=[.!?])\s+(?=[A-Z])/);
  return parts.slice(0, n).join(' ');
}

function renderReasoning() {
  const r = state.reasoning, body = $('#r-body'), btn = $('#r-run');
  const have = Boolean(state.record);
  $('#r-proc-row').hidden = !(have && state.uploaded);
  $('#ct-reasoning').title = '';
  btn.disabled = !have || !state.liveAvailable;
  btn.classList.remove('busy');
  btn.textContent = !r ? 'Run Nemotron' : 'Run it live';
  const src = $('#r-src');
  if (!have) { src.textContent = ''; body.innerHTML = ''; return; }
  if (!state.liveAvailable) {
    src.textContent = 'Live run unavailable';
  } else if (!r) {
    src.textContent = '';
  } else {
    const c = r.calls?.checkpoint || {};
    const live = state.reasoningSource === 'live';
    src.textContent = `${live ? 'Live' : 'Cached'} · ${r.made_at || ''} · effort ${c.reasoning_effort || 'high'}`;
  }
  if (!r) {
    body.innerHTML = `<div class="rempty"><b>Not read yet.</b></div>`;
    return;
  }

  const cp = r.checkpoint || {};
  const label = cp.decision === 'raise' ? 'Raise a checkpoint' : cp.decision === 'cannot_assess' ? 'Cannot assess' : 'Do not raise';
  const cls = cp.decision === 'raise' ? 'raise' : cp.decision === 'cannot_assess' ? 'cannot' : 'no';
  const at = cp.decision === 'raise' && cp.at_s != null
    ? `<span class="vat">at <button type="button" data-ms="${cp.at_s * 1000}">${esc(clock(cp.at_s * 1000))}</button> into the clip</span>` : '';
  const ev = (cp.evidence || []);
  const okN = ev.filter((e) => e.verified).length;
  const rows = ev.map((e) => {
    const [name, fmt] = METRIC[e.metric] || [e.metric, (v) => String(v)];
    const check = e.verified
      ? '<span class="ck ok">✓ matches OpenCV</span>'
      : `<span class="ck bad">✕ OpenCV has ${esc(e.measured ?? 'nothing')}</span>`;
    return `<div class="row"><button type="button" class="t" data-ms="${(e.t_s || 0) * 1000}">${esc(clock((e.t_s || 0) * 1000))}</button>
      <span class="metric">${esc(name)}</span>
      <span class="what"><b>${esc(fmt(e.value))}</b><span>${esc(e.reading || '')}</span></span>${check}</div>`;
  }).join('');
  const withheld = (r.explanations || []).filter((x) => WITHHELD[x.id]);
  const summary = r.summary || {};
  const figs = (summary.figures || []).map((f) => {
    const [a, b] = figLabel(f);
    return `<span class="fig"><b>${esc(b)}</b> ${esc(a)} <span class="ck ${f.verified ? 'ok' : 'bad'}">${f.verified ? '✓' : '✕'}</span></span>`;
  }).join('');

  body.innerHTML = `
    <section aria-label="Checkpoint decision">
      <h3>Checkpoint decision</h3>
      <div class="verdict"><span class="vpill ${cls}"><i></i>${label}</span>${at}</div>
      <p class="headline">${esc((cp.headline || '').replace(/^(cannot_assess|do_not_raise|raise)\s*[:\-]\s*/i, ''))}</p>
      <p class="why">${esc(firstSentences(cp.reasoning, 2))}</p>
      ${ev.length ? `<div class="cites" role="table" aria-label="Figures the decision cites">
        <div class="row head" role="row"><span>Time</span><span class="metric">Measured</span><span>Reading</span><span>Check</span></div>${rows}</div>
        <p class="fine"><b>Grounding</b> ${okN} of ${ev.length} cited figures match OpenCV</p>` : ''}
      ${(cp.limits || cp.would_change_if || state.procedure) ? `<details class="more"><summary>Limits and context</summary>
        ${cp.limits ? `<p class="fine"><b>Limits</b> ${esc(cp.limits)}</p>` : ''}
        ${cp.would_change_if ? `<p class="fine"><b>Would change if</b> ${esc(cp.would_change_if)}</p>` : ''}
        ${state.procedure ? `<p class="fine"><b>Procedure</b> ${esc(state.procedure)}</p>` : ''}</details>` : ''}
    </section>
    <section aria-label="Case summary">
      <h3>Summary</h3>
      <p class="summary">${esc(firstSentences(summary.summary, 3) || '—')}</p>
      ${figs ? `<div class="figs" aria-label="Figures quoted">${figs}</div>` : ''}
      ${withheld.length ? `<div class="withheld"><h3>Withheld</h3>${withheld.map((x) => `
        <div class="wh"><b>${esc(WITHHELD[x.id])}</b><p>${esc(firstSentences(x.plain, 1))}</p></div>`).join('')}</div>` : ''}
    </section>`;
  $$('#r-body [data-ms]').forEach((b) => b.addEventListener('click', () => {
    seek(Number(b.dataset.ms));
    $('#live-field').scrollIntoView({ block: 'center', behavior: 'smooth' });
  }));
}

async function runNemotron() {
  if (!state.jobId || !state.record) return;
  const btn = $('#r-run');
  btn.disabled = true; btn.classList.add('busy');
  btn.textContent = 'Nemotron is reading…';
  $('#r-src').textContent = '15 to 40 s';
  hideError();
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(state.jobId)}/reason`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ procedure: state.uploaded ? $('#r-proc').value.trim() : undefined }),
    });
    const payload = await res.json().catch(() => null);
    if (!res.ok) throw new ApiError(payload?.error || { message: res.statusText }, res.status);
    state.reasoning = payload.reasoning; state.reasoningSource = 'live';
    state.agent = payload.agent;
    state.procedure = payload.procedure || state.procedure;
    state.events = buildEvents(state.record);
    renderCases(); renderHead(state.record, isUnmeasurable(state.record));
    renderKpis(state.record); renderCheckpoint(); renderEvents(); renderRail(state.record);
    renderPhases(state.record); renderReasoning();
  } catch (err) { showError(err); renderReasoning(); }
}

boot();

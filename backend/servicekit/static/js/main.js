// Wires the shell together: config, upload, SSE progress, result and evidence.
// A product may import from './api.js' and './render.js' and extend this, or
// replace the whole file by shipping its own /assets/js/product.js.

import { api, ApiError } from './api.js';
import { initTheme } from './theme.js';
import { collectParams, renderEvidence, renderNotice, renderParams, renderResult } from './render.js';

const $ = (id) => document.getElementById(id);

const ui = {
  form: $('upload-form'),
  dropzone: $('dropzone'),
  fileInput: $('file-input'),
  fileLabel: $('dropzone-file'),
  hint: $('dropzone-hint'),
  params: $('params'),
  submit: $('submit'),
  progressPanel: $('progress-panel'),
  progressBar: $('progress-bar'),
  progressMessage: $('progress-message'),
  status: $('job-status'),
  log: $('log'),
  result: $('result'),
  resultTiming: $('result-timing'),
  evidence: $('evidence'),
  evidenceCount: $('evidence-count'),
  chipOpenCV: $('chip-opencv'),
  chipSha: $('chip-sha'),
};

let selectedFile = null;
let closeStream = null;

initTheme($('theme-toggle'));

function appendLog(line) {
  ui.log.textContent += `${line}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
}

function setProgress(percent, message) {
  ui.progressBar.style.width = `${percent}%`;
  ui.progressBar.setAttribute('aria-valuenow', String(Math.round(percent)));
  if (message) ui.progressMessage.textContent = message;
}

function setFile(file) {
  selectedFile = file;
  ui.fileLabel.textContent = file
    ? `${file.name} · ${(file.size / (1024 * 1024)).toFixed(1)} MB`
    : 'Drop a file, or click to choose';
  ui.submit.disabled = !file;
}

// --- drag and drop --------------------------------------------------------

ui.dropzone.addEventListener('click', () => ui.fileInput.click());
ui.dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); ui.fileInput.click(); }
});
ui.fileInput.addEventListener('change', () => setFile(ui.fileInput.files[0] ?? null));

for (const name of ['dragenter', 'dragover']) {
  ui.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    ui.dropzone.dataset.dragging = 'true';
  });
}
for (const name of ['dragleave', 'drop']) {
  ui.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    ui.dropzone.dataset.dragging = 'false';
  });
}
ui.dropzone.addEventListener('drop', (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) setFile(file);
});

// --- submit ---------------------------------------------------------------

ui.form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!selectedFile) return;

  closeStream?.();
  ui.submit.disabled = true;
  ui.progressPanel.hidden = false;
  ui.log.textContent = '';
  ui.result.replaceChildren();
  ui.resultTiming.hidden = true;
  setProgress(0, 'uploading');
  ui.status.textContent = 'queued';

  try {
    const { job_id: jobId } = await api.submit(selectedFile, collectParams(ui.params));
    appendLog(`job ${jobId} accepted`);
    follow(jobId);
  } catch (error) {
    failed(error);
  }
});

function follow(jobId) {
  closeStream = api.events(jobId, {
    progress: (event) => {
      setProgress(event.percent ?? 0, event.message ?? '');
      if (event.message) appendLog(`${String(Math.round(event.percent ?? 0)).padStart(3)}%  ${event.message}`);
    },
    note: (event) => appendLog(`      ${event.message}`),
    status: (event) => {
      ui.status.textContent = event.status;
      if (event.status === 'done' || event.status === 'failed') finish(jobId, event);
    },
    error: (error) => failed(error),
  });
}

async function finish(jobId, event) {
  setProgress(100, event.status === 'done' ? 'complete' : 'failed');
  ui.submit.disabled = !selectedFile;
  let job;
  try {
    job = await api.job(jobId);
  } catch (error) {
    failed(error);
    return;
  }
  renderResult(ui.result, job);
  renderEvidence(ui.evidence, ui.evidenceCount, job.result);
  const total = job.result?.timings?.total_ms;
  if (total !== undefined) {
    ui.resultTiming.hidden = false;
    ui.resultTiming.textContent = `${total.toFixed(0)} ms`;
  }
}

function failed(error) {
  ui.submit.disabled = !selectedFile;
  ui.status.textContent = 'failed';
  const notice = error instanceof ApiError
    ? renderNotice('error', error.code, error.message, error.details)
    : renderNotice('error', 'NETWORK', error.message ?? String(error));
  ui.result.replaceChildren(notice);
  appendLog(`error: ${error.message ?? error}`);
}

// --- boot -----------------------------------------------------------------

(async function boot() {
  try {
    const [config, version] = await Promise.all([api.config(), api.version()]);
    document.title = config.product.title;
    for (const node of document.querySelectorAll('[data-bind="product.title"]')) {
      node.textContent = config.product.title;
    }
    for (const node of document.querySelectorAll('[data-bind="product.tagline"]')) {
      node.textContent = config.product.tagline ?? '';
    }
    if (config.accent) document.documentElement.style.setProperty('--accent', config.accent);
    ui.hint.textContent = `Accepts ${config.accepts.join(', ')} up to ` +
      `${Math.round(config.max_upload_bytes / (1024 * 1024))} MB`;
    ui.fileInput.accept = config.accepts.join(',');
    renderParams(ui.params, config.params);
    ui.chipOpenCV.textContent = `OpenCV ${version.opencv_version}`;
    ui.chipSha.textContent = version.git_sha;
    ui.chipSha.title = `${version.python_version} · ${version.machine} · ` +
      `HAL ${version.opencv_build?.custom_hal ?? 'unknown'}`;
  } catch (error) {
    ui.hint.textContent = 'Could not load service configuration.';
    console.error(error);
  }
})();

export { api, ui };

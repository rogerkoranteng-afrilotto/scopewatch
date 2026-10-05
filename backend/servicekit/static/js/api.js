// Thin wrapper over the servicekit HTTP API. No framework, no build step.

export class ApiError extends Error {
  constructor(payload, status) {
    super(payload?.message || 'request failed');
    this.code = payload?.code || 'INTERNAL';
    this.requestId = payload?.request_id || '';
    this.details = payload?.details || {};
    this.status = status;
  }
}

async function unwrap(response) {
  const text = await response.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!response.ok) {
    throw new ApiError(body?.error ?? { message: text || response.statusText }, response.status);
  }
  return body;
}

export const api = {
  async config() {
    return unwrap(await fetch('/api/config'));
  },

  async version() {
    return unwrap(await fetch('/version'));
  },

  async submit(file, params = {}) {
    const body = new FormData();
    body.append('file', file);
    body.append('params', JSON.stringify(params));
    return unwrap(await fetch('/api/jobs', { method: 'POST', body }));
  },

  async job(jobId) {
    return unwrap(await fetch(`/api/jobs/${encodeURIComponent(jobId)}`));
  },

  /**
   * Subscribe to a job's Server-Sent Events. Returns a close() function.
   * `handlers` may define progress, note, status and end.
   */
  events(jobId, handlers = {}) {
    const source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
    const wire = (name) => source.addEventListener(name, (event) => {
      let data = {};
      try { data = JSON.parse(event.data); } catch { /* heartbeat */ }
      handlers[name]?.(data);
      if (name === 'end') source.close();
    });
    ['progress', 'note', 'status', 'end'].forEach(wire);
    source.onerror = () => {
      // EventSource retries on its own; only surface a hard close.
      if (source.readyState === EventSource.CLOSED) handlers.error?.(new Error('stream closed'));
    };
    return () => source.close();
  },
};

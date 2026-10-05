// Rendering the RunRecord. Every product gets these three panels for free.

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

function formatValue(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/\.?0+$/, '');
  }
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (Array.isArray(value)) return value.map(formatValue).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function humanise(key) {
  return key.replace(/_/g, ' ').replace(/\b([a-z])/g, (m) => m.toUpperCase());
}

export function renderNotice(kind, code, message, details) {
  const box = el('div', `notice notice--${kind}`);
  if (code) box.append(el('div', 'notice__code', code));
  box.append(el('div', 'notice__message', message));
  if (details && Object.keys(details).length) {
    box.append(el('p', 'notice__details',
      Object.entries(details).map(([k, v]) => `${humanise(k)}: ${formatValue(v)}`).join(' · ')));
  }
  return box;
}

export function renderResult(container, job) {
  container.replaceChildren();

  if (job.error) {
    container.append(renderNotice('error', job.error.code, job.error.message, job.error.details));
    return;
  }

  const record = job.result;
  if (!record) {
    container.append(el('p', 'empty', 'Nothing analysed yet.'));
    return;
  }

  // Refusals come first: a declined measurement is the headline, not a footnote.
  for (const refusal of record.refusals ?? []) {
    container.append(renderNotice('refusal', refusal.code, refusal.message, refusal.details));
  }
  for (const warning of record.warnings ?? []) {
    container.append(renderNotice('warn', '', warning));
  }

  const metrics = Object.entries(record.metrics ?? {});
  if (metrics.length) {
    const grid = el('div', 'metrics');
    for (const [key, value] of metrics) {
      const card = el('div', 'metric');
      card.append(el('div', 'metric__label', humanise(key)));
      card.append(el('div', 'metric__value', formatValue(value)));
      grid.append(card);
    }
    container.append(grid);
  }

  const stages = record.timings?.stages ?? [];
  if (stages.length) {
    const table = el('table', 'timings');
    const thead = el('thead');
    const head = el('tr');
    head.append(el('th', '', 'Stage'), el('th', '', 'Calls'), el('th', '', 'ms'));
    thead.append(head);
    table.append(thead);
    const body = el('tbody');
    for (const stage of stages) {
      const row = el('tr');
      row.append(el('td', '', stage.name), el('td', '', String(stage.calls)),
                 el('td', '', stage.ms.toFixed(1)));
      body.append(row);
    }
    const total = el('tr');
    total.append(el('td', '', 'total'), el('td', '', ''),
                 el('td', '', (record.timings.total_ms ?? 0).toFixed(1)));
    body.append(total);
    table.append(body);
    container.append(table);
  }

  if (!metrics.length && !(record.refusals ?? []).length) {
    container.append(el('p', 'empty', 'The analysis produced no metrics.'));
  }

  const raw = el('details', 'raw');
  raw.append(el('summary', '', 'Raw run record (JSON)'));
  raw.append(el('pre', 'mono', JSON.stringify(record, null, 2)));
  container.append(raw);
}

export function renderEvidence(container, countChip, record) {
  container.replaceChildren();
  const items = record?.evidence ?? [];
  if (!items.length) {
    container.append(el('p', 'empty', 'The frames behind each result appear here.'));
    if (countChip) countChip.hidden = true;
    return;
  }
  if (countChip) {
    countChip.hidden = false;
    countChip.textContent = `${items.length} item${items.length === 1 ? '' : 's'}`;
  }
  const grid = el('div', 'evidence');
  for (const item of items) {
    const card = el('figure', 'evidence__item');
    card.style.margin = '0';
    if (item.uri) {
      const img = el('img');
      img.src = item.uri;
      img.alt = item.caption || item.label;
      img.loading = 'lazy';
      card.append(img);
    }
    const caption = el('figcaption', 'evidence__caption');
    caption.append(el('div', 'evidence__label', item.label));
    const meta = [];
    if (item.frame_index !== null && item.frame_index !== undefined) meta.push(`frame ${item.frame_index}`);
    if (item.timestamp_ms !== null && item.timestamp_ms !== undefined) {
      meta.push(`${(item.timestamp_ms / 1000).toFixed(2)}s`);
    }
    for (const [k, v] of Object.entries(item.metrics ?? {})) meta.push(`${k} ${formatValue(v)}`);
    if (meta.length) caption.append(el('div', 'evidence__meta', meta.join(' · ')));
    if (item.caption) caption.append(el('div', 'evidence__meta', item.caption));
    card.append(caption);
    grid.append(card);
  }
  container.append(grid);
}

export function renderParams(container, schema) {
  container.replaceChildren();
  for (const param of schema ?? []) {
    const field = el('div', 'field');
    const id = `param-${param.name}`;
    const label = el('label', 'field__label', param.label ?? humanise(param.name));
    label.htmlFor = id;
    field.append(label);
    let input;
    if (param.type === 'select') {
      input = el('select');
      for (const option of param.options ?? []) {
        const node = el('option', '', option.label ?? String(option.value ?? option));
        node.value = option.value ?? option;
        input.append(node);
      }
    } else {
      input = el('input');
      input.type = param.type === 'number' ? 'number' : 'text';
      if (param.step !== undefined) input.step = param.step;
      if (param.min !== undefined) input.min = param.min;
      if (param.max !== undefined) input.max = param.max;
    }
    input.id = id;
    input.name = param.name;
    input.dataset.paramType = param.type ?? 'text';
    if (param.default !== undefined) input.value = param.default;
    if (param.help) input.title = param.help;
    field.append(input);
    container.append(field);
  }
}

export function collectParams(container) {
  const params = {};
  for (const input of container.querySelectorAll('[data-param-type]')) {
    const raw = input.value;
    if (raw === '') continue;
    params[input.name] = input.dataset.paramType === 'number' ? Number(raw) : raw;
  }
  return params;
}

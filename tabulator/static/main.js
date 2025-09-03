(() => {
  function onReady(fn) {
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
      setTimeout(fn, 0);
    } else {
      document.addEventListener('DOMContentLoaded', fn);
    }
  }

  function setupCollapsibles(root = document) {
    root.addEventListener('click', (e) => 
    {
      const btn = e.target.closest('.card-toggle');
      if (!btn) return;
      const card = btn.closest('.card');
      if (!card) return;
      const expanded = btn.getAttribute('aria-expanded') === 'true';
      btn.setAttribute('aria-expanded', expanded ? 'false' : 'true');
      card.classList.toggle('collapsed', expanded);
    });
  }

  let plotSeq = 0;

  function createPlotCard() {
    plotSeq += 1;
    const id = `plot-${plotSeq}`;
    const bodyId = `${id}-body`;
    const wrapper = document.createElement('section');
    wrapper.className = 'card plot-card';
    wrapper.dataset.seq = String(plotSeq);
    wrapper.innerHTML = `
      <div class="card-header">
        <button class="card-toggle" aria-expanded="true" aria-controls="${bodyId}">
          <span class="chevron">▾</span>
          <span class="card-title">Plot ${plotSeq}</span>
        </button>
        <div class="card-actions">
          <button type="button" class="secondary remove-plot">Remove</button>
        </div>
      </div>
      <div class="card-body" id="${bodyId}">
        <div class="form-row">
          <label for="${id}-type">Plot Type</label>
          <select id="${id}-type" class="plot-type">
            <option value="bar">bar</option>
            <option value="scatter">scatter</option>
            <option value="heatmap">heatmap</option>
          </select>
        </div>
        <div class="plot-config" id="${id}-config"></div>
        <div class="plot-preview" id="${id}-preview"></div>
      </div>
    `;
    return wrapper;
  }

  function setupPlotsUI() {
    const addBtn = document.getElementById('add-plot-btn');
    const container = document.getElementById('plots-container');
    if (!addBtn || !container) return;

    addBtn.addEventListener('click', () => {
      const card = createPlotCard();
      container.appendChild(card);
      initializePlotCard(card);
    });

    container.addEventListener('click', (e) => {
      const btn = e.target.closest('.remove-plot');
      if (!btn) return;
      const card = btn.closest('.plot-card');
      if (card) card.remove();
    });
  }

  onReady(() => {
    setupCollapsibles(document);
    setupPlotsUI();
  });

  // -------- Plot helpers --------
  async function fetchJSON(url) {
    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    return await res.json();
  }

  async function initializePlotCard(card) {
    const typeSel = card.querySelector('.plot-type');
    const configEl = card.querySelector('.plot-config');
    const previewEl = card.querySelector('.plot-preview');

    async function renderForType() {
      const t = typeSel.value;
      if (t === 'bar') {
        await buildBarConfig(card, configEl, previewEl);
      } else {
        configEl.innerHTML = '';
        previewEl.innerHTML = '';
      }
    }

    typeSel.addEventListener('change', renderForType);
    await renderForType();
  }

  async function buildBarConfig(card, configEl, previewEl) {
    // Get columns and split numeric/all for selections
    let cols;
    try {
      const meta = await fetchJSON('/api/columns');
      cols = meta.columns || [];
    } catch (e) {
      configEl.innerHTML = '<div style="color: var(--muted);">No dataset loaded.</div>';
      previewEl.innerHTML = '';
      return;
    }
    const numeric = cols.filter(c => c.is_numeric);
    const allCols = cols;

    // Build selects
    const id = card.dataset.seq;
    const valueId = `plot-${id}-value`;
    const groupId = `plot-${id}-group`;
    const errId = `plot-${id}-err`;
    const logId = `plot-${id}-log`;
    const yminId = `plot-${id}-ymin`;
    const ymaxId = `plot-${id}-ymax`;
    configEl.innerHTML = `
      <div class="form-row">
        <label for="${valueId}">Display variable</label>
        <select id="${valueId}"></select>
        <label for="${groupId}">Group by</label>
        <select id="${groupId}"></select>
        <label for="${errId}">Error bars</label>
        <select id="${errId}">
          <option value="none">None</option>
          <option value="sd">SD</option>
          <option value="sem">SEM</option>
        </select>
        <label for="${logId}">Log scale</label>
        <input id="${logId}" type="checkbox" />
      </div>
      <div class="form-row" style="margin-top:8px;">
        <label for="${yminId}">Y min</label>
        <input type="number" id="${yminId}" placeholder="auto" step="any" />
        <label for="${ymaxId}">Y max</label>
        <input type="number" id="${ymaxId}" placeholder="auto" step="any" />
        <button type="button" class="secondary" id="plot-${id}-autoscale">Autoscale</button>
      </div>
    `;
    const valueSel = document.getElementById(valueId);
    const groupSel = document.getElementById(groupId);
    const errSel = document.getElementById(errId);
    const logChk = document.getElementById(logId);
    const yminInp = document.getElementById(yminId);
    const ymaxInp = document.getElementById(ymaxId);
    const autoBtn = document.getElementById(`plot-${id}-autoscale`);

    // Populate options
    for (const c of allCols) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = c.name;
      valueSel.appendChild(opt);
    }
    for (const c of allCols) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = c.name;
      groupSel.appendChild(opt);
    }

    // Choose defaults: first numeric for value, first non-value for group if possible
    if (numeric.length) valueSel.value = numeric[0].name;
    if (allCols.length) {
      groupSel.value = allCols.find(c => c.name !== valueSel.value)?.name || allCols[0].name;
    }

    async function refreshPlot() {
      const value = valueSel.value;
      const group = groupSel.value;
      const err = errSel.value; // none|sd|sem
      const log = !!logChk.checked;
      const yMin = parseFloat(yminInp.value);
      const yMax = parseFloat(ymaxInp.value);
      try {
        const data = await fetchJSON(`/api/plot/bar?value=${encodeURIComponent(value)}&group=${encodeURIComponent(group)}`);
        const opts = { err, log };
        if (!Number.isNaN(yMin)) opts.yMin = yMin;
        if (!Number.isNaN(yMax)) opts.yMax = yMax;
        renderBarPlot(previewEl, data, opts);
      } catch (e) {
        previewEl.innerHTML = `<div style=\"color: var(--muted);\">${String(e)}</div>`;
      }
    }

    valueSel.addEventListener('change', refreshPlot);
    groupSel.addEventListener('change', refreshPlot);
    errSel.addEventListener('change', refreshPlot);
    logChk.addEventListener('change', refreshPlot);
    yminInp.addEventListener('change', refreshPlot);
    ymaxInp.addEventListener('change', refreshPlot);
    autoBtn.addEventListener('click', () => { yminInp.value = ''; ymaxInp.value = ''; refreshPlot(); });
    await refreshPlot();
  }

  function renderBarPlot(container, data, opts = { err: 'none', log: false }) {
    const groups = Array.isArray(data.groups) ? data.groups : [];
    const unit = data.unit || '';
    const label = data.value || '';
    const groupName = data.group || '';

    const xlabels = groups.map(g => String(g.name));
    const xpos = groups.map((_, i) => i);

    const rawMeans = groups.map(g => (typeof g.mean === 'number' && isFinite(g.mean)) ? g.mean : null);
    const means = rawMeans.map(m => (opts.log && !(m > 0) ? null : m));
    const errArray = groups.map((g, i) => {
      let eb = computeError(g.values || [], opts.err);
      const m = rawMeans[i];
      if (opts.log) {
        if (!(m > 0) || !(typeof eb === 'number' && isFinite(eb))) return 0;
        // Clamp so that mean - error stays positive for log display
        eb = Math.min(eb, m - 1e-12);
        if (!(eb > 0)) return 0;
      }
      return (typeof eb === 'number' && isFinite(eb)) ? eb : 0;
    });

    const barTrace = {
      type: 'bar',
      x: xpos,
      y: means,
      marker: { color: '#60a5fa' },
      error_y: {
        type: 'data',
        array: errArray,
        visible: opts.err !== 'none',
        color: '#111827',
        thickness: 1.5,
        width: 4,
      },
      hovertemplate: '%{y}<extra>%{x}</extra>',
    };

    const pointX = [];
    const pointY = [];
    const jitter = 0.25;
    groups.forEach((g, i) => {
      const vals = (g.values || []).filter(v => typeof v === 'number' && isFinite(v));
      vals.forEach(v => {
        if (opts.log && !(v > 0)) return;
        pointX.push(i + (Math.random() - 0.5) * 2 * jitter);
        pointY.push(v);
      });
    });
    const pointsTrace = {
      type: 'scatter',
      mode: 'markers',
      x: pointX,
      y: pointY,
      marker: { color: '#111827', opacity: 0.5, size: 5 },
      hovertemplate: '%{y}<extra></extra>',
      showlegend: false,
    };

    // Compute data extents for optional manual limits
    const allValues = groups.flatMap(g => (g.values || [])).filter(v => typeof v === 'number' && isFinite(v));
    const numericMeans = rawMeans.filter(v => typeof v === 'number' && isFinite(v));
    const linMin = (allValues.concat(numericMeans).length ? Math.min(...allValues, ...numericMeans) : 0);
    const linMax = (allValues.concat(numericMeans).length ? Math.max(...allValues, ...numericMeans) : 1);
    const posValues = allValues.filter(v => v > 0).concat(numericMeans.filter(v => v > 0));

    const yaxis = {
      title: unit ? `${label} (${unit})` : label,
      type: opts.log ? 'log' : 'linear',
      autorange: true,
      gridcolor: '#e5e7eb',
      zeroline: !opts.log,
      zerolinecolor: '#9ca3af',
    };
    if (!opts.log) {
      // Linear scale: allow partial manual limits (min-only or max-only)
      const hasMin = typeof opts.yMin === 'number' && isFinite(opts.yMin);
      const hasMax = typeof opts.yMax === 'number' && isFinite(opts.yMax);
      if (hasMin || hasMax) {
        const yMin = hasMin ? opts.yMin : linMin;
        const yMax = hasMax ? opts.yMax : linMax;
        if (yMax > yMin) {
          yaxis.autorange = false;
          yaxis.range = [yMin, yMax];
        }
      }
    } else {
      // Log scale: choose a nonzero default Y-min (decade below/at data min)
      let yMinP;
      let yMaxP;
      const hasMin = typeof opts.yMin === 'number' && isFinite(opts.yMin) && opts.yMin > 0;
      const hasMax = typeof opts.yMax === 'number' && isFinite(opts.yMax) && opts.yMax > 0;
      if (hasMin) yMinP = opts.yMin;
      if (hasMax) yMaxP = opts.yMax;

      if (posValues.length) {
        const dataMin = Math.min(...posValues);
        const dataMax = Math.max(...posValues);
        // Default to decade-rounded bounds when not provided
        if (!hasMin) yMinP = Math.pow(10, Math.floor(Math.log10(dataMin)));
        if (!hasMax) yMaxP = Math.pow(10, Math.ceil(Math.log10(dataMax)));
      }

      if (yMinP && yMaxP && yMaxP > yMinP) {
        yaxis.autorange = false;
        yaxis.range = [Math.log10(yMinP), Math.log10(yMaxP)];
        yaxis.tickmode = 'array';
        yaxis.tickvals = decadeTicks(yMinP, yMaxP);
        yaxis.ticktext = yaxis.tickvals.map(v => formatTickLog(v));
      } else {
        yaxis.autorange = true;
        if (posValues.length) {
          const dataMin = Math.min(...posValues);
          const dataMax = Math.max(...posValues);
          yaxis.tickmode = 'array';
          yaxis.tickvals = decadeTicks(dataMin, dataMax);
          yaxis.ticktext = yaxis.tickvals.map(v => formatTickLog(v));
        } else {
          yaxis.dtick = 1;
        }
      }
    }

    const layout = {
      height: 360,
      margin: { l: 70, r: 20, t: 10, b: 60 },
      xaxis: {
        title: groupName,
        tickmode: 'array',
        tickvals: xpos,
        ticktext: xlabels,
        gridcolor: '#f3f4f6',
      },
      yaxis,
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      showlegend: false,
    };

    const config = { responsive: true, displayModeBar: false };

    container.innerHTML = '';
    const div = document.createElement('div');
    container.appendChild(div);
    Plotly.newPlot(div, [barTrace, pointsTrace], layout, config);
  }

  // Helpers: nice ticks for linear scale
  function niceTicks(min, max, count = 5) {
    if (!isFinite(min) || !isFinite(max)) { min = 0; max = 1; }
    if (min === max) { max = min + 1; }
    const range = niceNum(max - min, false);
    const step = niceNum(range / (count - 1), true);
    const niceMin = Math.floor(min / step) * step;
    const niceMax = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = niceMin; v <= niceMax + 1e-9; v += step) ticks.push(v);
    return { min: niceMin, max: niceMax, step, ticks };
  }
  function niceNum(range, round) {
    const exponent = Math.floor(Math.log10(range));
    const fraction = range / Math.pow(10, exponent);
    let niceFraction;
    if (round) {
      if (fraction < 1.5) niceFraction = 1;
      else if (fraction < 3) niceFraction = 2;
      else if (fraction < 7) niceFraction = 5;
      else niceFraction = 10;
    } else {
      if (fraction <= 1) niceFraction = 1;
      else if (fraction <= 2) niceFraction = 2;
      else if (fraction <= 5) niceFraction = 5;
      else niceFraction = 10;
    }
    return niceFraction * Math.pow(10, exponent);
  }
  function formatTickLinear(v) {
    const av = Math.abs(v);
    if (av >= 1000) return v.toFixed(0);
    if (av >= 1) return v.toFixed(2);
    if (av >= 0.01) return v.toFixed(3);
    return v.toExponential(1);
  }
  function formatTickLog(v) {
    return v.toExponential(0).replace('e+','e');
  }
  function decadeTicks(minPos, maxPos) {
    const pmin = Math.floor(Math.log10(minPos));
    const pmax = Math.ceil(Math.log10(maxPos));
    const vals = [];
    for (let p = pmin; p <= pmax; p++) vals.push(Math.pow(10, p));
    return vals;
  }
  function computeError(values, mode) {
    if (mode === 'none') return null;
    const xs = values.filter(v => typeof v === 'number' && isFinite(v));
    const n = xs.length;
    if (n === 0) return null;
    const mean = xs.reduce((a,b)=>a+b,0) / n;
    const variance = n > 1 ? xs.reduce((a,b)=>a+(b-mean)*(b-mean),0) / (n - 1) : 0;
    const sd = Math.sqrt(variance);
    if (mode === 'sd') return sd;
    if (mode === 'sem') return n > 0 ? sd / Math.sqrt(n) : null;
    return null;
  }
})();

(() => {
  function onReady(fn) {
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
      setTimeout(fn, 0);
    } else {
      document.addEventListener('DOMContentLoaded', fn);
    }
  }

  const datasetId = document.body?.dataset?.datasetId || '';

  function withDatasetId(url) {
    if (!datasetId) return url;
    try {
      const u = new URL(url, window.location.origin);
      if (!u.searchParams.has('dataset_id')) {
        u.searchParams.set('dataset_id', datasetId);
      }
      return u.pathname + u.search;
    } catch {
      return url;
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

  function setupUploadCard() {
    const card = document.getElementById('upload-card');
    if (!card) return;
    const buttons = card.querySelectorAll('.mode-switch-btn');
    if (!buttons.length) return;
    const panels = card.querySelectorAll('.mode-panel');
    const tableSelect = card.querySelector('#db-table-select');
    const userQuerySelect = card.querySelector('#db-user-query');
    const projectQuerySelect = card.querySelector('#db-project-query');
    const statusEl = card.querySelector('#db-load-status');
    const dbBtn = card.querySelector('#db-load-btn');
    let tablesFetched = false;
    let fetchingTables = false;
    let queriesFetched = false;
    let fetchingQueries = false;

    function setStatus(msg) {
      if (statusEl) statusEl.textContent = msg || '';
    }

    function setSelectMessage(text) {
      if (!tableSelect) return;
      tableSelect.innerHTML = '';
      const option = document.createElement('option');
      option.textContent = text;
      option.value = '';
      tableSelect.appendChild(option);
    }

    async function ensureTables(force = false) {
      if (!tableSelect) return;
      if (fetchingTables) return;
      if (tablesFetched && !force) return;
      fetchingTables = true;
      tableSelect.disabled = true;
      setSelectMessage(force ? 'Refreshing tables…' : 'Loading tables…');
      try {
        const res = await fetch('/api/db/tables', { headers: { 'Accept': 'application/json' }, credentials: 'same-origin' });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error((data && data.detail) || data.error || `Request failed (${res.status})`);
        }
        const tables = Array.isArray(data.tables) ? data.tables : [];
        if (!tables.length) {
          setSelectMessage('No tables found for this schema.');
          setStatus('No tables are available in the configured schema.');
          tablesFetched = false;
          return;
        }
        const fragment = document.createDocumentFragment();
        tables.forEach((entry) => {
          const name = (entry && entry.name) || entry;
          if (!name) return;
          const option = document.createElement('option');
          option.value = name;
          option.textContent = name;
          fragment.appendChild(option);
        });
        tableSelect.innerHTML = '';
        tableSelect.appendChild(fragment);
        tableSelect.disabled = false;
        tablesFetched = true;
        setStatus('');
      } catch (err) {
        setSelectMessage('Unable to load tables.');
        setStatus(`Failed to load tables: ${err.message}`);
        tablesFetched = false;
      } finally {
        fetchingTables = false;
      }
    }

    function setQuerySelectMessage(select, text) {
      if (!select) return;
      select.innerHTML = '';
      const option = document.createElement('option');
      option.value = '';
      option.textContent = text;
      select.appendChild(option);
    }

    async function ensureQueries(force = false) {
      if ((!userQuerySelect && !projectQuerySelect) || fetchingQueries) return;
      if (queriesFetched && !force) return;
      fetchingQueries = true;
      if (userQuerySelect) {
        userQuerySelect.disabled = true;
        setQuerySelectMessage(userQuerySelect, 'Loading user queries…');
      }
      if (projectQuerySelect) {
        projectQuerySelect.disabled = true;
        setQuerySelectMessage(projectQuerySelect, 'Loading project queries…');
      }
      try {
        const res = await fetch('/api/db/queries', { headers: { 'Accept': 'application/json' }, credentials: 'same-origin' });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error((data && data.detail) || data.error || `Request failed (${res.status})`);
        }
        const users = Array.isArray(data.user_queries) ? data.user_queries : [];
        const projects = Array.isArray(data.project_queries) ? data.project_queries : [];
        if (userQuerySelect) {
          const frag = document.createDocumentFragment();
          const optAll = document.createElement('option');
          optAll.value = '';
          optAll.textContent = 'All users';
          frag.appendChild(optAll);
          users.forEach((entry) => {
            const option = document.createElement('option');
            option.value = entry.query_name || '';
            option.textContent = entry.label || entry.query_name || '';
            frag.appendChild(option);
          });
          userQuerySelect.innerHTML = '';
          userQuerySelect.appendChild(frag);
          userQuerySelect.disabled = users.length === 0;
        }
        if (projectQuerySelect) {
          const frag = document.createDocumentFragment();
          const optAll = document.createElement('option');
          optAll.value = '';
          optAll.textContent = 'All projects';
          frag.appendChild(optAll);
          projects.forEach((entry) => {
            const option = document.createElement('option');
            option.value = entry.query_name || '';
            option.textContent = entry.label || entry.query_name || '';
            frag.appendChild(option);
          });
          projectQuerySelect.innerHTML = '';
          projectQuerySelect.appendChild(frag);
          projectQuerySelect.disabled = projects.length === 0;
        }
        queriesFetched = true;
      } catch (err) {
        if (userQuerySelect) {
          setQuerySelectMessage(userQuerySelect, 'User queries unavailable');
          userQuerySelect.disabled = true;
        }
        if (projectQuerySelect) {
          setQuerySelectMessage(projectQuerySelect, 'Project queries unavailable');
          projectQuerySelect.disabled = true;
        }
        setStatus(`Failed to load queries: ${err.message}`);
        queriesFetched = false;
      } finally {
        fetchingQueries = false;
      }
    }

    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const targetId = btn.dataset.target;
        if (!targetId || btn.classList.contains('active')) return;
        buttons.forEach((other) => {
          const isActive = other === btn;
          other.classList.toggle('active', isActive);
          other.setAttribute('aria-selected', isActive ? 'true' : 'false');
        });
        panels.forEach((panel) => {
          panel.hidden = panel.id !== targetId;
        });
        if (targetId === 'upload-mode-db') {
          void ensureTables(false);
          void ensureQueries(false);
        }
      });
    });
    if (tableSelect) {
      tableSelect.addEventListener('change', () => setStatus(''));
    }
    function resetOtherQuery(changed) {
      if (changed === userQuerySelect && projectQuerySelect) {
        projectQuerySelect.value = '';
      } else if (changed === projectQuerySelect && userQuerySelect) {
        userQuerySelect.value = '';
      }
      setStatus('');
    }
    if (userQuerySelect) {
      userQuerySelect.addEventListener('change', () => resetOtherQuery(userQuerySelect));
    }
    if (projectQuerySelect) {
      projectQuerySelect.addEventListener('change', () => resetOtherQuery(projectQuerySelect));
    }
    if (dbBtn) {
      dbBtn.addEventListener('click', async () => {
        await ensureTables(false);
        await ensureQueries(false);
        if (!tableSelect || tableSelect.disabled) {
          setStatus('Table list is still loading.');
          return;
        }
        const tableName = tableSelect.value;
        if (!tableName) {
          setStatus('Select a table to load.');
          return;
        }
        const prevText = dbBtn.textContent;
        dbBtn.disabled = true;
        dbBtn.textContent = 'Loading…';
        setStatus('');
        try {
          const queryName = (userQuerySelect && userQuerySelect.value) || (projectQuerySelect && projectQuerySelect.value) || '';
          const body = { table: tableName };
          if (queryName) body.query_name = queryName;
          const res = await fetch('/api/db/load', {
            method: 'POST',
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json',
            },
            credentials: 'same-origin',
            body: JSON.stringify(body),
          });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) {
            throw new Error((data && data.detail) || data.error || `Request failed (${res.status})`);
          }
          const target = data.redirect || window.location.pathname + window.location.search + window.location.hash;
          const url = new URL(target, window.location.origin);
          url.searchParams.set('_dj_refresh', Date.now().toString());
          setStatus(`${data.message || 'Table loaded.'} Showing data…`);
          setTimeout(() => { window.location.assign(url.toString()); }, 50);
        } catch (err) {
          setStatus(`Load failed: ${err.message}`);
        } finally {
          dbBtn.disabled = false;
          dbBtn.textContent = prevText;
        }
      });
    }
    const activeBtn = card.querySelector('.mode-switch-btn.active');
    if (activeBtn && activeBtn.dataset.target === 'upload-mode-db') {
      void ensureTables(false);
      void ensureQueries(false);
    }
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
          <span class="field">
            <label for="${id}-type">Plot Type</label>
            <select id="${id}-type" class="plot-type">
              <option value="bar">bar</option>
              <option value="scatter">scatter</option>
              <option value="line_by_row">Line plot by row</option>
              <option value="heatmap">heatmap</option>
            </select>
          </span>
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
    setupUploadCard();
    const clearBtn = document.getElementById('clear-data-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', async () => {
        clearBtn.disabled = true;
        clearBtn.textContent = 'Clearing…';
        try {
          const res = await fetch('/api/clear', { method: 'POST', headers: { 'Accept': 'application/json' }, credentials: 'same-origin' });
          if (!res.ok) throw new Error('Failed to clear');
        } catch (err) {
          console.error(err);
        } finally {
          const dest = window.location.pathname + window.location.search;
          window.location.assign(dest);
        }
      });
    }
    setupPlotsUI();
    setupPCA();
    setupDR();
    setupClassifier();
  });

  // -------- Plot helpers --------
  async function fetchJSON(url) {
    const target = withDatasetId(url);
    const res = await fetch(target, { headers: { 'Accept': 'application/json' }, credentials: 'same-origin' });
    if (!res.ok) {
      let detail = '';
      try {
        const body = await res.json();
        if (body && body.error) detail = ` (${body.error})`;
      } catch {}
      throw new Error(`Request failed: ${res.status}${detail}`);
    }
    return await res.json();
  }

  // Generic Plotly rendering with a small export toolbar (PNG/JSON)
  function renderPlot(container, traces, layout, config = {}, nameHint = 'plot') {
    container.innerHTML = '';
    const plotDiv = document.createElement('div');
    container.appendChild(plotDiv);
    const cfg = { responsive: true, displayModeBar: false, ...config };
    Plotly.newPlot(plotDiv, traces, layout, cfg);
    plotDiv.__plotlyCfg = cfg;

    // Toolbar overlay
    const bar = document.createElement('div');
    bar.className = 'plot-toolbar';
    const btnPng = document.createElement('button'); btnPng.className = 'secondary'; btnPng.textContent = 'PNG'; btnPng.title = 'Download PNG';
    const btnJson = document.createElement('button'); btnJson.className = 'secondary'; btnJson.textContent = 'JSON'; btnJson.title = 'Download Plotly JSON';
    bar.appendChild(btnPng); bar.appendChild(btnJson);
    container.appendChild(bar);

    function filename(ext) {
      const ts = new Date().toISOString().replace(/[:T]/g, '-').split('.')[0];
      return `${nameHint}-${ts}.${ext}`;
    }
    btnPng.addEventListener('click', async () => {
      try {
        const url = await Plotly.toImage(plotDiv, { format: 'png', width: plotDiv.clientWidth || 800, height: (layout && layout.height) || 360, scale: 2 });
        const a = document.createElement('a'); a.href = url; a.download = filename('png'); a.click();
      } catch (e) { console.error('PNG export failed', e); }
    });
    btnJson.addEventListener('click', () => {
      try {
        const fig = { data: plotDiv.data || [], layout: plotDiv.layout || {}, config: plotDiv.__plotlyCfg || {} };
        const blob = new Blob([JSON.stringify(fig, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = filename('plotly.json'); a.click();
        setTimeout(() => URL.revokeObjectURL(url), 500);
      } catch (e) { console.error('JSON export failed', e); }
    });
    // No WYSIWYG edit toggle — simplified toolbar
  }

  function displayCategoryLabel(value) {
    return value == null || value === '' ? '(blank)' : String(value);
  }

  function formatCountLabel(baseText, rowCount, animalCount) {
    const animalText = animalCount == null || animalCount === '' ? '?' : animalCount;
    return `${baseText} (n=${rowCount} rows, ${animalText} animals)`;
  }

  function stripCountLabel(text) {
    const value = String(text ?? '');
    return value
      .replace(/\s*\(n=\s*[^)]*rows,\s*[^)]*animals\)\s*$/, '')
      .replace(/\s*<br>\s*n=\s*.*$/i, '')
      .trim();
  }

  function formatPValueThreshold(pValue) {
    const p = Number(pValue);
    if (!Number.isFinite(p) || p < 0) return 'p=?';
    if (p === 0) return 'p<1E-300';
    if (p >= 1) return 'p=1';
    let exponent = Math.ceil(Math.log10(p));
    let threshold = Math.pow(10, exponent);
    if (threshold <= p) {
      exponent += 1;
      threshold *= 10;
    }
    return `p<1E${exponent}`;
  }

  async function initializePlotCard(card) {
    const typeSel = card.querySelector('.plot-type');
    const configEl = card.querySelector('.plot-config');
    const previewEl = card.querySelector('.plot-preview');

    async function renderForType() {
      const t = typeSel.value;
      if (t === 'bar') {
        await buildBarConfig(card, configEl, previewEl);
      } else if (t === 'scatter') {
        await buildScatterConfig(card, configEl, previewEl);
      } else if (t === 'line_by_row') {
        await buildLineByRowConfig(card, configEl, previewEl);
      } else {
        configEl.innerHTML = '';
        previewEl.innerHTML = '';
      }
    }

    typeSel.addEventListener('change', renderForType);
    await renderForType();
  }

  // -------- PCA UI --------
  function setupPCA() {
    const btn = document.getElementById('run-pca-btn');
    const preview = document.getElementById('pca-preview');
    const controls = document.getElementById('pca-controls');
    const pcSelect = document.getElementById('pca-loadings-pc');
    const info = document.getElementById('pca-info');
    const scatterControls = document.getElementById('pca-scatter-controls');
    const pcaDimSel = document.getElementById('pca-dim');
    const pcaPcX = document.getElementById('pca-pc-x');
    const pcaPcY = document.getElementById('pca-pc-y');
    const pcaPcZField = document.getElementById('pca-pc-z-field');
    const pcaPcZ = document.getElementById('pca-pc-z');
    const pcaColorSel = document.getElementById('pca-color');
    const scatterDiv = document.getElementById('pca-scatter');
    if (!btn || !preview) return;

    let pcaData = null;

    btn.addEventListener('click', async () => {
      btn.disabled = true;
      const prevLabel = btn.textContent;
      btn.textContent = 'Running…';
      info.textContent = '';
      try {
        const data = await fetchJSON('/api/pca');
        pcaData = data;
        renderPCAElbow(preview, data);
        setupLoadingsControls(controls, pcSelect, data);
        renderPCALoadings(document.getElementById('pca-loadings'), data, 0);
        await setupPCAScatter(scatterControls, data, { pcaDimSel, pcaPcX, pcaPcY, pcaPcZField, pcaPcZ, pcaColorSel, scatterDiv });
        if (info && data) {
          const ns = data.n_samples ?? '?';
          const nf = data.n_features ?? '?';
          info.textContent = `Computed on ${ns} rows × ${nf} numeric columns.`;
        }
      } catch (e) {
        preview.innerHTML = `<div style=\"color: var(--muted);\">${String(e)}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = prevLabel;
      }
    });
  }

  function renderPCAElbow(container, data) {
    const evr = Array.isArray(data.explained_variance_ratio) ? data.explained_variance_ratio : [];
    const cum = Array.isArray(data.cumulative_ratio) ? data.cumulative_ratio : [];
    const n = evr.length;
    const x = Array.from({ length: n }, (_, i) => i + 1);
    const xlabels = x.map(i => `PC ${i}`);

    const bar = {
      type: 'bar',
      x,
      y: evr,
      marker: { color: '#60a5fa' },
      name: 'Explained variance',
      hovertemplate: 'PC %{x}<br>%{y:.3f} of variance<extra></extra>'
    };
    const line = {
      type: 'scatter',
      mode: 'lines+markers',
      x,
      y: cum.length === n ? cum : undefined,
      marker: { color: '#111827', size: 6 },
      line: { color: '#111827' },
      name: 'Cumulative',
      hovertemplate: '≤ PC %{x}<br>%{y:.3f} cumulative<extra></extra>'
    };

    const width = container.clientWidth || 800;
    const small = width < 560;
    const height = small ? 280 : 360;
    const layout = {
      height,
      margin: { l: 70, r: 20, t: 10, b: 60 },
      xaxis: { title: 'Component', tickmode: 'array', tickvals: x, ticktext: xlabels, automargin: true, gridcolor: '#f3f4f6' },
      yaxis: { title: 'Fraction of variance explained', rangemode: 'tozero', automargin: true, gridcolor: '#f3f4f6' },
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      showlegend: true,
      legend: { orientation: 'h', y: -0.2 }
    };
    const traces = [bar];
    if (line.y) traces.push(line);
    renderPlot(container, traces, layout, { displayModeBar: false }, 'pca-elbow');
  }

  function setupLoadingsControls(controls, pcSelect, data) {
    if (!controls || !pcSelect) return;
    const k = typeof data.components === 'number' ? data.components : (Array.isArray(data.explained_variance_ratio) ? data.explained_variance_ratio.length : 0);
    pcSelect.innerHTML = '';
    for (let i = 0; i < k; i++) {
      const opt = document.createElement('option');
      opt.value = String(i);
      opt.textContent = `PC ${i + 1}`;
      pcSelect.appendChild(opt);
    }
    controls.style.display = k > 0 ? '' : 'none';
    pcSelect.value = '0';
    pcSelect.onchange = () => {
      renderPCALoadings(document.getElementById('pca-loadings'), data, parseInt(pcSelect.value, 10) || 0);
    };
  }

  function renderPCALoadings(container, data, pcIndex = 0) {
    if (!container) return;
    const loadings = Array.isArray(data.loadings) ? data.loadings : [];
    const cols = Array.isArray(data.columns) ? data.columns : [];
    if (!loadings.length || !cols.length) {
      container.innerHTML = '<div style="color: var(--muted);">No loadings to display.</div>';
      return;
    }
    const k = loadings.length;
    const p = cols.length;
    const idx = Math.max(0, Math.min(pcIndex, k - 1));
    const v = loadings[idx]; // array length p
    // Keep variables in the same order for each PC (no sorting)
    const y = cols.map(name => String(name));
    const x = cols.map((_, i) => Number(v[i]));
    const colors = x.map(val => (val >= 0 ? '#34d399' : '#ef4444'));

    const bar = {
      type: 'bar',
      orientation: 'h',
      x,
      y,
      marker: { color: colors },
      hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
    };
    const width = container.clientWidth || 800;
    const small = width < 560;
    const height = Math.max(220, Math.min(600, y.length * (small ? 18 : 22)));
    const layout = {
      height,
      margin: { l: 160, r: 20, t: 10, b: 40 },
      xaxis: { title: 'Loading coefficient', zeroline: true, zerolinecolor: '#9ca3af', gridcolor: '#f3f4f6' },
      yaxis: { automargin: true },
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      showlegend: false,
    };
    renderPlot(container, [bar], layout, { displayModeBar: false }, 'pca-loadings');
  }

  async function setupPCAScatter(ctrlEl, pcaMeta, refs) {
    if (!ctrlEl) return;
    const { pcaDimSel, pcaPcX, pcaPcY, pcaPcZField, pcaPcZ, pcaColorSel, scatterDiv } = refs;
    const k = typeof pcaMeta.components === 'number' ? pcaMeta.components : (Array.isArray(pcaMeta.explained_variance_ratio) ? pcaMeta.explained_variance_ratio.length : 0);
    // Populate PC selects
    function fillPCSelect(sel) {
      sel.innerHTML = '';
      for (let i = 1; i <= k; i++) {
        const opt = document.createElement('option');
        opt.value = String(i);
        const frac = (Array.isArray(pcaMeta.explained_variance_ratio) && pcaMeta.explained_variance_ratio[i-1] != null)
          ? ` (${(pcaMeta.explained_variance_ratio[i-1] * 100).toFixed(1)}%)` : '';
        opt.textContent = `PC ${i}${frac}`;
        sel.appendChild(opt);
      }
    }
    fillPCSelect(pcaPcX);
    fillPCSelect(pcaPcY);
    fillPCSelect(pcaPcZ);
    if (k >= 2) { pcaPcX.value = '1'; pcaPcY.value = '2'; }
    if (k >= 3) { pcaPcZ.value = '3'; }

    // Populate color-by
    pcaColorSel.innerHTML = '<option value="">None</option>';
    try {
      const meta = await fetchJSON('/api/columns');
      const cols = (meta.columns || []).filter(c => c.is_simple);
      for (const c of cols) {
        const opt = document.createElement('option');
        opt.value = opt.textContent = c.name;
        pcaColorSel.appendChild(opt);
      }
    } catch {}

    // Toggle 2D/3D Z control and availability
    const dim3Opt = Array.from(pcaDimSel.options).find(o => o.value === '3d');
    if (k < 3 && dim3Opt) {
      dim3Opt.disabled = true;
      pcaDimSel.value = '2d';
    } else if (dim3Opt) {
      dim3Opt.disabled = false;
    }
    // Toggle 2D/3D Z control
    function syncDim() {
      const is3d = pcaDimSel.value === '3d';
      pcaPcZField.style.display = is3d ? '' : 'none';
    }
    pcaDimSel.addEventListener('change', () => { syncDim(); refresh(); });
    pcaPcX.addEventListener('change', refresh);
    pcaPcY.addEventListener('change', refresh);
    pcaPcZ.addEventListener('change', refresh);
    pcaColorSel.addEventListener('change', refresh);

    ctrlEl.style.display = '';
    syncDim();
    await refresh();

    async function refresh() {
      const pcs = [parseInt(pcaPcX.value, 10), parseInt(pcaPcY.value, 10)];
      if (pcaDimSel.value === '3d') pcs.push(parseInt(pcaPcZ.value, 10));
      const pcsValid = pcs.every(v => Number.isFinite(v) && v >= 1);
      if (!pcsValid) return;
      const params = new URLSearchParams();
      params.set('pcs', pcs.join(','));
      const color = pcaColorSel.value || '';
      if (color) params.set('color', color);
      try {
        const data = await fetchJSON(`/api/pca/scores?${params.toString()}`);
        renderPCAScatter(scatterDiv, data);
      } catch (e) {
        scatterDiv.innerHTML = `<div style=\"color: var(--muted);\">${String(e)}</div>`;
      }
    }
  }

  function renderPCAScatter(container, data) {
    const xs = Array.isArray(data.x) ? data.x : [];
    const ys = Array.isArray(data.y) ? data.y : [];
    const zs = Array.isArray(data.z) ? data.z : null;
    const ids = Array.isArray(data.id) ? data.id : [];
    const colors = Array.isArray(data.color_values) ? data.color_values : null;
    const colorCol = data.color_column || '';
    const pcLabels = Array.isArray(data.pcs) ? data.pcs : [1,2];
    const evr = Array.isArray(data.explained_variance_ratio) ? data.explained_variance_ratio : [];
    function axisTitle(axis, idx1based) {
      const frac = (evr[idx1based-1] != null) ? ` ${(evr[idx1based-1]*100).toFixed(1)}%` : '';
      return `PC${idx1based}${frac}`;
    }

    const width = container.clientWidth || 800;
    const small = width < 560;
    const height = small ? 320 : 400;

    // Build traces depending on dimension and coloring
    const palette = ['#60a5fa','#34d399','#f59e0b','#ef4444','#a78bfa','#10b981','#f472b6','#22d3ee','#fb7185','#93c5fd','#fbbf24','#d946ef','#14b8a6','#fca5a5','#4ade80'];

    let traces = [];
    const is3d = Array.isArray(zs);
    if (!colors) {
      const text = ids.map((id, i) => `${id || ''}${id ? ', ' : ''}(${xs[i]}, ${ys[i]}${is3d ? ', ' + zs[i] : ''})`);
      const base = {
        name: 'points', text,
        hovertemplate: '%{text}<extra></extra>'
      };
      traces.push(is3d ? {
        type: 'scatter3d', mode: 'markers', x: xs, y: ys, z: zs, marker: { color: '#111827', size: 3, opacity: 0.9 }, ...base
      } : {
        type: 'scattergl', mode: 'markers', x: xs, y: ys, marker: { color: '#111827', size: 5, opacity: 0.9 }, ...base
      });
    } else if (colors.every(v => typeof v === 'number')) {
      const text = ids.map((id, i) => `${id || ''}${id ? ', ' : ''}${colorCol}: ${colors[i]}\n(${xs[i]}, ${ys[i]}${is3d ? ', ' + zs[i] : ''})`);
      traces.push(is3d ? {
        type: 'scatter3d', mode: 'markers', x: xs, y: ys, z: zs,
        marker: { size: 3, opacity: 0.9, color: colors, colorscale: 'Viridis', colorbar: { title: colorCol } },
        text, hovertemplate: '%{text}<extra></extra>'
      } : {
        type: 'scattergl', mode: 'markers', x: xs, y: ys,
        marker: { size: 6, opacity: 0.9, color: colors, colorscale: 'Viridis', colorbar: { title: colorCol } },
        text, hovertemplate: '%{text}<extra></extra>'
      });
    } else {
      const groups = new Map();
      for (let i = 0; i < xs.length; i++) {
        const key = colors[i] == null ? 'NA' : String(colors[i]);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(i);
      }
      Array.from(groups.entries()).forEach(([name, idxs], j) => {
        const x = idxs.map(i => xs[i]);
        const y = idxs.map(i => ys[i]);
        const z = is3d ? idxs.map(i => zs[i]) : null;
        const text = idxs.map(i => {
          const id = ids[i] || '';
          return `${id}${id ? ', ' : ''}${colorCol}: ${name}\n(${xs[i]}, ${ys[i]}${is3d ? ', ' + zs[i] : ''})`;
        });
        traces.push(is3d ? {
          type: 'scatter3d', mode: 'markers', name: String(name), legendgroup: String(name), x, y, z,
          marker: { color: palette[j % palette.length], size: 3, opacity: 0.9 }, text,
          hovertemplate: '%{text}<extra></extra>'
        } : {
          type: 'scattergl', mode: 'markers', name: String(name), legendgroup: String(name), x, y,
          marker: { color: palette[j % palette.length], size: 6, opacity: 0.9 }, text,
          hovertemplate: '%{text}<extra></extra>'
        });
      });
    }

    const layout = is3d ? {
      height,
      margin: { l: 0, r: 0, t: 10, b: 0 },
      scene: {
        xaxis: { title: axisTitle('x', pcLabels[0]) },
        yaxis: { title: axisTitle('y', pcLabels[1]) },
        zaxis: { title: axisTitle('z', pcLabels[2]) },
      },
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      showlegend: !!colors && !colors.every(v => typeof v === 'number'),
    } : {
      height,
      margin: { l: 60, r: 20, t: 10, b: 50 },
      xaxis: { title: axisTitle('x', pcLabels[0]), automargin: true, zeroline: false, gridcolor: '#f3f4f6' },
      yaxis: { title: axisTitle('y', pcLabels[1]), automargin: true, zeroline: false, gridcolor: '#f3f4f6' },
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      showlegend: !!colors && !colors.every(v => typeof v === 'number'),
    };

    renderPlot(container, traces, layout, { displayModeBar: false }, 'pca-scores');
  }

  // -------- Dimensionality Reduction UI --------
  async function setupDR() {
    const btn = document.getElementById('run-dr-btn');
    const preview = document.getElementById('dr-preview');
    const info = document.getElementById('dr-info');
    const methodSel = document.getElementById('dr-method');
    const modeSel = document.getElementById('dr-mode');
    const npcsInp = document.getElementById('dr-npcs');
    const colorSel = document.getElementById('dr-color');
    const clusterSel = document.getElementById('dr-cluster');
    const kmeansParams = document.getElementById('dr-kmeans-params');
    const kmeansK = document.getElementById('dr-kmeans-k');
    const dbscanParams = document.getElementById('dr-dbscan-params');
    const dbscanEps = document.getElementById('dr-dbscan-eps');
    const dbscanMin = document.getElementById('dr-dbscan-min');
    const agglomParams = document.getElementById('dr-agglom-params');
    const agglomK = document.getElementById('dr-agglom-k');
    const agglomLink = document.getElementById('dr-agglom-linkage');
    const hdbscanParams = document.getElementById('dr-hdbscan-params');
    const hdbscanMinSize = document.getElementById('dr-hdbscan-minsize');
    const hdbscanMinSamples = document.getElementById('dr-hdbscan-minsamples');
    if (!btn || !preview) return;

    // Populate color-by options
    try {
      const meta = await fetchJSON('/api/columns');
      const cols = (meta.columns || []).filter(c => c.is_simple);
      for (const c of cols) {
        const opt = document.createElement('option');
        opt.value = opt.textContent = c.name;
        colorSel.appendChild(opt);
      }
    } catch (e) {
      // ignore if no dataset
    }

    function syncNpcsVisibility() {
      const show = modeSel.value === 'pcs';
      npcsInp.style.display = show ? '' : 'none';
      const label = document.querySelector('label[for="dr-npcs"]');
      if (label) label.style.display = show ? '' : 'none';
    }
    modeSel.addEventListener('change', syncNpcsVisibility);
    syncNpcsVisibility();

    function syncClusterParams() {
      const m = clusterSel.value;
      kmeansParams.style.display = (m === 'kmeans') ? '' : 'none';
      dbscanParams.style.display = (m === 'dbscan') ? '' : 'none';
      agglomParams.style.display = (m === 'agglomerative') ? '' : 'none';
      hdbscanParams.style.display = (m === 'hdbscan') ? '' : 'none';
    }
    clusterSel.addEventListener('change', syncClusterParams);
    syncClusterParams();

    btn.addEventListener('click', async () => {
      btn.disabled = true;
      const prev = btn.textContent;
      btn.textContent = 'Running…';
      info.textContent = '';
      try {
        const params = new URLSearchParams();
        params.set('method', methodSel.value);
        params.set('mode', modeSel.value);
        if (modeSel.value === 'pcs') {
          const k = parseInt(npcsInp.value, 10);
          if (Number.isFinite(k) && k > 0) params.set('n_pcs', String(k));
        }
        const color = colorSel.value || '';
        if (color) params.set('color', color);
        // clustering
        const cl = clusterSel.value;
        params.set('cluster', cl);
        if (cl === 'kmeans') {
          const k = parseInt(kmeansK.value, 10);
          if (Number.isFinite(k) && k >= 2) params.set('kmeans_k', String(k));
        } else if (cl === 'dbscan') {
          const eps = parseFloat(dbscanEps.value);
          const minPts = parseInt(dbscanMin.value, 10);
          if (Number.isFinite(eps) && eps > 0) params.set('dbscan_eps', String(eps));
          if (Number.isFinite(minPts) && minPts >= 1) params.set('dbscan_min_samples', String(minPts));
        } else if (cl === 'agglomerative') {
          const k = parseInt(agglomK.value, 10);
          if (Number.isFinite(k) && k >= 2) params.set('agglom_k', String(k));
          params.set('agglom_linkage', agglomLink.value);
        } else if (cl === 'hdbscan') {
          const mcs = parseInt(hdbscanMinSize.value, 10);
          const ms = parseInt(hdbscanMinSamples.value, 10);
          if (Number.isFinite(mcs) && mcs >= 2) params.set('hdbscan_min_cluster_size', String(mcs));
          if (Number.isFinite(ms) && ms >= 1) params.set('hdbscan_min_samples', String(ms));
        }
        const data = await fetchJSON(`/api/dr?${params.toString()}`);
        // If clustering was run, append color option and render with clusters colored by default
        let dataForRender = data;
        if (Array.isArray(data.cluster_labels) && typeof data.cluster_algorithm === 'string') {
          const optName = `cluster_ids_${data.cluster_algorithm}`;
          // Append to dropdown if missing
          let exists = false;
          for (const opt of colorSel.options) { if (opt.value === optName) { exists = true; break; } }
          if (!exists) {
            const opt = document.createElement('option');
            opt.value = opt.textContent = optName;
            colorSel.appendChild(opt);
          }
          // Select the new option and color by clusters
          colorSel.value = optName;
          const clusterColors = data.cluster_labels.map(v => (v == null ? null : String(v)));
          dataForRender = { ...data, color_values: clusterColors, color_column: optName };
        }
        renderDimred(preview, dataForRender);
        const meth = data.method || methodSel.value;
        const mode = data.preprocess && data.preprocess.mode ? data.preprocess.mode : modeSel.value;
        const k = data.preprocess && data.preprocess.n_pcs ? data.preprocess.n_pcs : undefined;
        const used = (mode === 'pcs' && k) ? `first ${k} PCs` : 'all numeric features';
        const n = data.n_points ?? (data.x ? data.x.length : '?');
        info.textContent = `Method: ${meth.toUpperCase()} · ${used} · Points: ${n}` + (data.color_column ? ` · Colored by: ${data.color_column}` : '');
        const sil = (typeof data.silhouette === 'number' && isFinite(data.silhouette)) ? data.silhouette : null;
        if (sil !== null) {
          const excl = (typeof data.n_noise === 'number' && data.n_noise > 0) ? ' (excl. noise)' : '';
          info.textContent += ` · Silhouette: ${sil.toFixed(3)}${excl}`;
        }
      } catch (e) {
        let msg = String(e);
        if (msg.includes('(missing_dep_sklearn)')) {
          msg += `\nInstall scikit-learn to use t-SNE.`;
        } else if (msg.includes('(missing_dep_umap)')) {
          msg += `\nInstall umap-learn to use UMAP.`;
        } else if (msg.includes('(missing_dep_hdbscan)')) {
          msg += `\nInstall hdbscan to use HDBSCAN clustering.`;
        }
        preview.innerHTML = `<div style=\"color: var(--muted); white-space: pre-wrap;\">${msg}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = prev;
      }
    });
  }

  function renderDimred(container, data) {
    const xs = Array.isArray(data.x) ? data.x : [];
    const ys = Array.isArray(data.y) ? data.y : [];
    const ids = Array.isArray(data.id) ? data.id : [];
    const colors = Array.isArray(data.color_values) ? data.color_values : null;
    const colorCol = data.color_column || '';
    const points = [];

    const width = container.clientWidth || 800;
    const small = width < 560;
    const height = small ? 320 : 400;

    const palette = ['#60a5fa','#34d399','#f59e0b','#ef4444','#a78bfa','#10b981','#f472b6','#22d3ee','#fb7185','#93c5fd','#fbbf24','#d946ef','#14b8a6','#fca5a5','#4ade80'];

    if (!colors) {
      // No color — single trace
      const text = ids.map((id, i) => `${id || ''}${id ? ', ' : ''}(${xs[i]}, ${ys[i]})`);
      points.push({
        type: 'scattergl', mode: 'markers', name: 'points', x: xs, y: ys,
        marker: { color: '#111827', size: 5, opacity: 0.9 }, text,
        hovertemplate: '%{text}<extra></extra>'
      });
    } else if (colors.every(v => typeof v === 'number')) {
      // Numeric color scale
      const text = ids.map((id, i) => `${id || ''}${id ? ', ' : ''}${colorCol}: ${colors[i]}\n(${xs[i]}, ${ys[i]})`);
      points.push({
        type: 'scattergl', mode: 'markers', name: 'points', x: xs, y: ys,
        marker: { size: 6, opacity: 0.9, color: colors, colorscale: 'Viridis', colorbar: { title: colorCol } },
        text, hovertemplate: '%{text}<extra></extra>'
      });
    } else {
      // Categorical — bucket into groups
      const groups = new Map();
      for (let i = 0; i < xs.length; i++) {
        const key = colors[i] == null ? 'NA' : String(colors[i]);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(i);
      }
      Array.from(groups.entries()).forEach(([name, idxs], j) => {
        const x = idxs.map(i => xs[i]);
        const y = idxs.map(i => ys[i]);
        const text = idxs.map(i => {
          const id = ids[i] || '';
          return `${id}${id ? ', ' : ''}${colorCol}: ${name}\n(${xs[i]}, ${ys[i]})`;
        });
        points.push({
          type: 'scattergl', mode: 'markers', name: String(name), legendgroup: String(name),
          x, y, text, marker: { color: palette[j % palette.length], size: 6, opacity: 0.9 },
          hovertemplate: '%{text}<extra></extra>'
        });
      });
    }

    const layout = {
      height,
      margin: { l: 60, r: 20, t: 10, b: 50 },
      xaxis: { title: 'Dim 1', automargin: true, zeroline: false, gridcolor: '#f3f4f6' },
      yaxis: { title: 'Dim 2', automargin: true, zeroline: false, gridcolor: '#f3f4f6' },
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      showlegend: !!colors && !colors.every(v => typeof v === 'number'),
    };
    renderPlot(container, points, layout, { displayModeBar: false }, 'dimred');
  }

  // -------- Classifier UI --------
  async function setupClassifier() {
    const btn = document.getElementById('clf-train-btn');
    const labelSel = document.getElementById('clf-label-col');
    const typeSel = document.getElementById('clf-type');
    const testFracInp = document.getElementById('clf-test-frac');
    const learnDiv = document.getElementById('clf-learning');
    const metricsDiv = document.getElementById('clf-metrics');
    const confDiv = document.getElementById('clf-confusion');
    const itersInp = document.getElementById('clf-iters');
    const earlyChk = document.getElementById('clf-early');
    const patienceInp = document.getElementById('clf-patience');
    if (!btn || !labelSel) return;

    // Populate label column choices (all columns)
    try {
      const meta = await fetchJSON('/api/columns');
      const cols = (meta.columns || []).filter(c => c.is_simple);
      labelSel.innerHTML = '';
      for (const c of cols) {
        const opt = document.createElement('option');
        opt.value = opt.textContent = c.name;
        labelSel.appendChild(opt);
      }
    } catch {}

    btn.addEventListener('click', async () => {
      btn.disabled = true;
      const prev = btn.textContent;
      btn.textContent = 'Training…';
      metricsDiv.textContent = '';
      learnDiv.innerHTML = '';
      confDiv.innerHTML = '';
      try {
        const body = {
          label: labelSel.value,
          clf: typeSel.value,
          test_frac: parseFloat(testFracInp.value),
          max_iters: Math.max(1, parseInt(itersInp?.value || '50', 10)),
          patience: Math.max(1, parseInt(patienceInp?.value || '5', 10)),
          early_stop: !!(earlyChk && earlyChk.checked),
        };
        const res = await fetch(withDatasetId('/api/classify/train'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify(body)
        });
        if (!res.ok) {
          let detail = '';
          try { const j = await res.json(); if (j && j.error) detail = ` (${j.error})`; } catch {}
          throw new Error(`Request failed: ${res.status}${detail}`);
        }
        const data = await res.json();
        renderLearningCurve(learnDiv, data.history, data.classes);
        const acc = (typeof data.test_accuracy === 'number' && isFinite(data.test_accuracy)) ? (data.test_accuracy * 100).toFixed(1) : '?';
        const bestIt = data.best_iteration || '?';
        const ran = Array.isArray(data?.history?.iter) ? data.history.iter.length : null;
        const early = !!data.stopped_early;
        const runNote = early ? `Stopped early at ${ran} iters` : (ran ? `Ran ${ran} iters` : '');
        metricsDiv.textContent = `Classifier: ${data.classifier} · Best iter: ${bestIt} · Test accuracy: ${acc}%` + (runNote ? ` · ${runNote}` : '');
        if (Array.isArray(data.confusion_matrix) && Array.isArray(data.classes)) {
          renderConfusion(confDiv, data.confusion_matrix, data.classes);
        }
      } catch (e) {
        let msg = String(e);
        learnDiv.innerHTML = `<div style=\"color: var(--muted); white-space: pre-wrap;\">${msg}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = prev;
      }
    });
  }

  function renderLearningCurve(container, history, classes) {
    const it = Array.isArray(history?.iter) ? history.iter : [];
    const tr = Array.isArray(history?.train_error) ? history.train_error : [];
    const vl = Array.isArray(history?.val_error) ? history.val_error : [];
    const traces = [
      { type: 'scatter', mode: 'lines+markers', name: 'Train error', x: it, y: tr, line: { color: '#60a5fa' }, marker: { size: 6 } },
      { type: 'scatter', mode: 'lines+markers', name: 'Validation error', x: it, y: vl, line: { color: '#ef4444' }, marker: { size: 6 } }
    ];
    // Add random-chance baseline error = 1 - 1/num_classes
    const nC = Array.isArray(classes) ? classes.length : null;
    if (typeof nC === 'number' && nC > 1) {
      const baseErr = 1 - 1 / nC;
      const xline = it.length >= 2 ? [it[0], it[it.length - 1]] : (it.length === 1 ? [it[0], it[0] + 1] : [0, 1]);
      const yline = [baseErr, baseErr];
      traces.push({ type: 'scatter', mode: 'lines', name: 'Random chance', x: xline, y: yline, line: { color: '#9ca3af', dash: 'dot' } });
    }
    const layout = {
      height: 280,
      margin: { l: 60, r: 20, t: 10, b: 40 },
      xaxis: { title: 'Iteration', gridcolor: '#f3f4f6' },
      yaxis: { title: 'Error (1 - accuracy)', rangemode: 'tozero', gridcolor: '#f3f4f6' },
      plot_bgcolor: '#ffffff', paper_bgcolor: '#ffffff', showlegend: true
    };
    renderPlot(container, traces, layout, { displayModeBar: false }, 'classifier-learning');
  }

  function renderConfusion(container, matrix, classes) {
    const z = matrix;
    const x = classes; const y = classes;
    const trace = { type: 'heatmap', z, x, y, colorscale: 'Blues', colorbar: { title: 'Count' } };
    const layout = {
      height: 320,
      margin: { l: 80, r: 20, t: 10, b: 80 },
      xaxis: { title: 'Predicted', automargin: true },
      yaxis: { title: 'True', automargin: true },
      plot_bgcolor: '#ffffff', paper_bgcolor: '#ffffff'
    };
    renderPlot(container, [trace], layout, { displayModeBar: false }, 'confusion');
  }

  async function buildScatterConfig(card, configEl, previewEl) {
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
    const simple = cols.filter(c => c.is_simple);

    const id = card.dataset.seq;
    const xId = `plot-${id}-x`;
    const yId = `plot-${id}-y`;
    const groupId = `plot-${id}-group`;
    const errxId = `plot-${id}-errx`;
    const erryId = `plot-${id}-erry`;
    const logxId = `plot-${id}-logx`;
    const logyId = `plot-${id}-logy`;
    const xminId = `plot-${id}-xmin`;
    const xmaxId = `plot-${id}-xmax`;
    const yminId = `plot-${id}-ymin`;
    const ymaxId = `plot-${id}-ymax`;

    configEl.innerHTML = `
      <div class="form-row">
        <span class="field">
          <label for="${xId}">X variable</label>
          <select id="${xId}"></select>
        </span>
        <span class="field">
          <label for="${yId}">Y variable</label>
          <select id="${yId}"></select>
        </span>
        <span class="field">
          <label for="${groupId}">Group by</label>
          <select id="${groupId}">
            <option value="">None</option>
          </select>
        </span>
      </div>
      <div class="form-row" style="margin-top:8px;">
        <span class="field">
          <label for="${errxId}">X error</label>
          <select id="${errxId}">
            <option value="none">None</option>
            <option value="sd">SD</option>
            <option value="sem">SEM</option>
          </select>
        </span>
        <span class="field">
          <label for="${erryId}">Y error</label>
          <select id="${erryId}">
            <option value="none">None</option>
            <option value="sd">SD</option>
            <option value="sem">SEM</option>
          </select>
        </span>
        <label class="toggle"><input id="${logxId}" type="checkbox" /> Log X</label>
        <label class="toggle"><input id="${logyId}" type="checkbox" /> Log Y</label>
      </div>
      <div class="form-row" style="margin-top:8px;">
        <span class="field">
          <label for="${xminId}">X min</label>
          <input type="number" id="${xminId}" placeholder="auto" step="any" />
        </span>
        <span class="field">
          <label for="${xmaxId}">X max</label>
          <input type="number" id="${xmaxId}" placeholder="auto" step="any" />
        </span>
        <span class="field">
          <label for="${yminId}">Y min</label>
          <input type="number" id="${yminId}" placeholder="auto" step="any" />
        </span>
        <span class="field">
          <label for="${ymaxId}">Y max</label>
          <input type="number" id="${ymaxId}" placeholder="auto" step="any" />
        </span>
        <button type="button" class="secondary" id="plot-${id}-scatter-autoscale">Autoscale</button>
      </div>
    `;

    const xSel = document.getElementById(xId);
    const ySel = document.getElementById(yId);
    const groupSel = document.getElementById(groupId);
    const errxSel = document.getElementById(errxId);
    const errySel = document.getElementById(erryId);
    const logxChk = document.getElementById(logxId);
    const logyChk = document.getElementById(logyId);
    const xminInp = document.getElementById(xminId);
    const xmaxInp = document.getElementById(xmaxId);
    const yminInp = document.getElementById(yminId);
    const ymaxInp = document.getElementById(ymaxId);
    const autoBtn = document.getElementById(`plot-${id}-scatter-autoscale`);

    for (const c of numeric) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = c.name;
      xSel.appendChild(opt);
    }
    for (const c of numeric) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = c.name;
      ySel.appendChild(opt);
    }
    for (const c of simple) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = c.name;
      groupSel.appendChild(opt);
    }

    if (numeric.length) xSel.value = numeric[0].name;
    if (numeric.length > 1) ySel.value = numeric[1].name;
    // Prefer 'cell_type' as default group if available
    const hasCellType = simple.some(c => c.name === 'cell_type');
    if (hasCellType) {
      groupSel.value = 'cell_type';
    }

    async function refreshPlot() {
      const x = xSel.value;
      const y = ySel.value;
      const group = groupSel.value || '';
      const errx = errxSel.value; // none|sd|sem
      const erry = errySel.value; // none|sd|sem
      const logx = !!logxChk.checked;
      const logy = !!logyChk.checked;
      try {
        const q = new URLSearchParams({ x, y });
        if (group) q.set('group', group);
        const data = await fetchJSON(`/api/plot/scatter?${q.toString()}`);
        const opts = { errx, erry, logx, logy };
        const xMin = parseFloat(xminInp.value);
        const xMax = parseFloat(xmaxInp.value);
        const yMin = parseFloat(yminInp.value);
        const yMax = parseFloat(ymaxInp.value);
        if (!Number.isNaN(xMin)) opts.xMin = xMin;
        if (!Number.isNaN(xMax)) opts.xMax = xMax;
        if (!Number.isNaN(yMin)) opts.yMin = yMin;
        if (!Number.isNaN(yMax)) opts.yMax = yMax;
        renderScatterPlot(previewEl, data, opts);
      } catch (e) {
        previewEl.innerHTML = `<div style=\"color: var(--muted);\">${String(e)}</div>`;
      }
    }

    xSel.addEventListener('change', refreshPlot);
    ySel.addEventListener('change', refreshPlot);
    groupSel.addEventListener('change', refreshPlot);
    errxSel.addEventListener('change', refreshPlot);
    errySel.addEventListener('change', refreshPlot);
    logxChk.addEventListener('change', refreshPlot);
    logyChk.addEventListener('change', refreshPlot);
    xminInp.addEventListener('change', refreshPlot);
    xmaxInp.addEventListener('change', refreshPlot);
    yminInp.addEventListener('change', refreshPlot);
    ymaxInp.addEventListener('change', refreshPlot);
    autoBtn.addEventListener('click', () => { xminInp.value = ''; xmaxInp.value=''; yminInp.value=''; ymaxInp.value=''; refreshPlot(); });
    await refreshPlot();
  }

  function renderScatterPlot(container, data, opts = { errx: 'none', erry: 'none', logx: false, logy: false }) {
    const groups = Array.isArray(data.groups) ? data.groups : [];
    const xLabel = data.x || '';
    const yLabel = data.y || '';
    const xUnit = data.x_unit || '';
    const yUnit = data.y_unit || '';
    const titleX = xUnit ? `${xLabel} (${xUnit})` : xLabel;
    const titleY = yUnit ? `${yLabel} (${yUnit})` : yLabel;

    // Simple palette matching points and means per group
    const palette = ['#60a5fa','#34d399','#f59e0b','#ef4444','#a78bfa','#10b981','#f472b6','#22d3ee','#fb7185','#93c5fd'];

    const traces = [];
    groups.forEach((g, i) => {
      const color = palette[i % palette.length];
      const pts = Array.isArray(g.points) ? g.points : [];
      const x = [], y = [], text = [];
      pts.forEach(p => {
        const xv = p.x, yv = p.y;
        if (!(typeof xv === 'number' && isFinite(xv))) return;
        if (!(typeof yv === 'number' && isFinite(yv))) return;
        if (opts.logx && !(xv > 0)) return;
        if (opts.logy && !(yv > 0)) return;
        x.push(xv);
        y.push(yv);
        const id = (p.id ?? '').toString();
        text.push(id ? `${id}, (${xv}, ${yv})` : `(${xv}, ${yv})`);
      });
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: String(g.name ?? 'Group'),
        legendgroup: `g${i}`,
        x, y, text,
        marker: { color, opacity: 0.8, size: 6 },
        hovertemplate: '%{text}<extra></extra>',
      });

      const m = g.mean || {};
      const mx = (typeof m.x === 'number' && isFinite(m.x)) ? m.x : null;
      const my = (typeof m.y === 'number' && isFinite(m.y)) ? m.y : null;
      if (mx !== null && my !== null) {
        let errx = 0, erry = 0, vx = true, vy = true;
        if (opts.errx === 'sd') { errx = (typeof m.errx_sd === 'number' && isFinite(m.errx_sd)) ? m.errx_sd : 0; }
        else if (opts.errx === 'sem') { errx = (typeof m.errx_sem === 'number' && isFinite(m.errx_sem)) ? m.errx_sem : 0; }
        else { vx = false; }
        if (opts.erry === 'sd') { erry = (typeof m.erry_sd === 'number' && isFinite(m.erry_sd)) ? m.erry_sd : 0; }
        else if (opts.erry === 'sem') { erry = (typeof m.erry_sem === 'number' && isFinite(m.erry_sem)) ? m.erry_sem : 0; }
        else { vy = false; }
        // For log axes, ensure positive mean and clamp errors so lower bound remains positive
        if (opts.logx) {
          if (!(mx > 0)) { vx = false; vy = vy && (opts.logy ? my > 0 : true); /* skip mean if invalid later */ }
          if (vx && errx > 0) { errx = Math.min(errx, mx - 1e-12); if (!(errx > 0)) vx = false; }
        }
        if (opts.logy) {
          if (!(my > 0)) { vy = false; vx = vx && (opts.logx ? mx > 0 : true); }
          if (vy && erry > 0) { erry = Math.min(erry, my - 1e-12); if (!(erry > 0)) vy = false; }
        }
        // Skip mean marker entirely if any required log axis has non-positive mean
        if ((opts.logx && !(mx > 0)) || (opts.logy && !(my > 0))) {
          // do not add mean trace
        } else {
          traces.push({
            type: 'scatter',
            mode: 'markers',
            name: `${String(g.name ?? 'Group')} mean`,
            legendgroup: `g${i}`,
            x: [mx], y: [my],
            marker: { color, symbol: 'x', size: 10, line: { color: '#111827', width: 1 } },
            error_x: { type: 'data', array: [errx], visible: vx, color: '#111827', thickness: 1 },
            error_y: { type: 'data', array: [erry], visible: vy, color: '#111827', thickness: 1 },
            hovertemplate: `${String(g.name ?? 'Group')} mean: (${mx}, ${my})<extra></extra>`,
          });
        }
      }
    });

    const width = container.clientWidth || 800;
    const small = width < 560;
    const height = small ? 280 : 360;

    // Data extents for optional manual bounds
    const allX = [];
    const allY = [];
    groups.forEach(g => {
      const pts = Array.isArray(g.points) ? g.points : [];
      pts.forEach(p => {
        if (isFinite(p.x) && (!opts.logx || p.x > 0)) allX.push(p.x);
        if (isFinite(p.y) && (!opts.logy || p.y > 0)) allY.push(p.y);
      });
      const m = g.mean || {};
      if (isFinite(m.x) && (!opts.logx || m.x > 0)) allX.push(m.x);
      if (isFinite(m.y) && (!opts.logy || m.y > 0)) allY.push(m.y);
    });
    const xLinMin = allX.length ? Math.min(...allX) : 0;
    const xLinMax = allX.length ? Math.max(...allX) : 1;
    const yLinMin = allY.length ? Math.min(...allY) : 0;
    const yLinMax = allY.length ? Math.max(...allY) : 1;

    const hasXMin = typeof opts.xMin === 'number' && isFinite(opts.xMin);
    const hasXMax = typeof opts.xMax === 'number' && isFinite(opts.xMax);
    const hasYMin = typeof opts.yMin === 'number' && isFinite(opts.yMin);
    const hasYMax = typeof opts.yMax === 'number' && isFinite(opts.yMax);

    const xaxis = { title: titleX, automargin: true, gridcolor: '#f3f4f6', type: opts.logx ? 'log' : 'linear', zeroline: !opts.logx, zerolinecolor: '#9ca3af' };
    if (hasXMin || hasXMax) {
      const minX = hasXMin ? opts.xMin : xLinMin;
      const maxX = hasXMax ? opts.xMax : xLinMax;
      if (opts.logx) {
        if ((hasXMin ? minX > 0 : true) && (hasXMax ? maxX > 0 : true) && maxX > minX) {
          xaxis.autorange = false;
          xaxis.range = [Math.log10(minX > 0 ? minX : xLinMin || 1), Math.log10(maxX > 0 ? maxX : xLinMax || 10)];
        }
      } else if (maxX > minX) {
        xaxis.autorange = false; xaxis.range = [minX, maxX];
      }
    }
    const yaxis = { title: titleY, automargin: true, gridcolor: '#e5e7eb', type: opts.logy ? 'log' : 'linear', zeroline: !opts.logy, zerolinecolor: '#9ca3af' };
    if (hasYMin || hasYMax) {
      const minY = hasYMin ? opts.yMin : yLinMin;
      const maxY = hasYMax ? opts.yMax : yLinMax;
      if (opts.logy) {
        if ((hasYMin ? minY > 0 : true) && (hasYMax ? maxY > 0 : true) && maxY > minY) {
          yaxis.autorange = false;
          yaxis.range = [Math.log10(minY > 0 ? minY : yLinMin || 1), Math.log10(maxY > 0 ? maxY : yLinMax || 10)];
        }
      } else if (maxY > minY) {
        yaxis.autorange = false; yaxis.range = [minY, maxY];
      }
    }

    const layout = {
      height,
      margin: { l: 70, r: 20, t: 10, b: 60 },
      xaxis,
      yaxis,
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      showlegend: true,
    };

    renderPlot(container, traces, layout, { displayModeBar: false }, 'scatter');
  }

  async function buildLineByRowConfig(card, configEl, previewEl) {
    let cols;
    try {
      const meta = await fetchJSON('/api/columns');
      cols = meta.columns || [];
    } catch (e) {
      configEl.innerHTML = '<div style="color: var(--muted);">No dataset loaded.</div>';
      previewEl.innerHTML = '';
      return;
    }
    const vectorCols = cols.filter(c => c.is_vector);
    if (!vectorCols.length) {
      configEl.innerHTML = '<div style="color: var(--muted);">No 1D numeric vector columns are available.</div>';
      previewEl.innerHTML = '';
      return;
    }

    const id = card.dataset.seq;
    const xId = `plot-${id}-line-x`;
    const yId = `plot-${id}-line-y`;
    const colorId = `plot-${id}-line-color`;
    const xLabelId = `plot-${id}-line-x-label`;
    const yLabelId = `plot-${id}-line-y-label`;
    const legendTitleId = `plot-${id}-line-legend-title`;
    const xMinId = `plot-${id}-line-xmin`;
    const xMaxId = `plot-${id}-line-xmax`;
    const yMinId = `plot-${id}-line-ymin`;
    const yMaxId = `plot-${id}-line-ymax`;
    const avgMetricId = `plot-${id}-line-avg-metric`;
    const avgStyleId = `plot-${id}-line-avg-style`;
    const invertYId = `plot-${id}-line-invert-y`;
    const presetNameId = `plot-${id}-line-preset-name`;
    const presetSelectId = `plot-${id}-line-preset-select`;
    const presetSaveId = `plot-${id}-line-preset-save`;
    const presetLoadId = `plot-${id}-line-preset-load`;
    const presetDeleteId = `plot-${id}-line-preset-delete`;
    const addFilterId = `plot-${id}-line-add-filter`;
    const filtersId = `plot-${id}-line-filters`;
    const colorEditorId = `plot-${id}-line-colors`;
    const simpleCols = cols.filter(c => c.is_simple);
    configEl.innerHTML = `
      <div class="form-row">
        <span class="field">
          <label for="${xId}">X variable</label>
          <select id="${xId}"></select>
        </span>
        <span class="field">
          <label for="${yId}">Y variable</label>
          <select id="${yId}"></select>
        </span>
        <span class="field">
          <label for="${colorId}">Color by</label>
          <select id="${colorId}">
            <option value="">None</option>
          </select>
        </span>
      </div>
      <div class="form-row" style="margin-top:8px;">
        <span class="field">
          <label for="${xLabelId}">X axis label</label>
          <input id="${xLabelId}" type="text" />
        </span>
        <span class="field">
          <label for="${yLabelId}">Y axis label</label>
          <input id="${yLabelId}" type="text" />
        </span>
        <span class="field">
          <label for="${legendTitleId}">Legend text</label>
          <input id="${legendTitleId}" type="text" />
        </span>
      </div>
      <div class="form-row" style="margin-top:8px;">
        <span class="field">
          <label for="${xMinId}">X min</label>
          <input id="${xMinId}" type="number" step="any" placeholder="auto" />
        </span>
        <span class="field">
          <label for="${xMaxId}">X max</label>
          <input id="${xMaxId}" type="number" step="any" placeholder="auto" />
        </span>
        <span class="field">
          <label for="${yMinId}">Y min</label>
          <input id="${yMinId}" type="number" step="any" placeholder="auto" />
        </span>
        <span class="field">
          <label for="${yMaxId}">Y max</label>
          <input id="${yMaxId}" type="number" step="any" placeholder="auto" />
        </span>
      </div>
      <div class="form-row" style="margin-top:8px;">
        <span class="field">
          <label for="${avgMetricId}">Average spread</label>
          <select id="${avgMetricId}">
            <option value="sem">SEM</option>
            <option value="sd">SD</option>
          </select>
        </span>
        <span class="field">
          <label for="${avgStyleId}">Error display</label>
          <select id="${avgStyleId}">
            <option value="shaded">Shaded region</option>
            <option value="bars">Bars</option>
          </select>
        </span>
        <label class="toggle"><input id="${invertYId}" type="checkbox" /> Invert Y values</label>
      </div>
      <div class="form-row" style="margin-top:8px; align-items:flex-end;">
        <span class="field">
          <label for="${presetNameId}">Preset name</label>
          <input id="${presetNameId}" type="text" placeholder="Save current settings" />
        </span>
        <button type="button" class="secondary" id="${presetSaveId}">Save Preferences</button>
        <span class="field">
          <label for="${presetSelectId}">Saved presets</label>
          <select id="${presetSelectId}">
            <option value="">Choose preset</option>
          </select>
        </span>
        <button type="button" class="secondary" id="${presetLoadId}">Load</button>
        <button type="button" class="secondary" id="${presetDeleteId}">Delete</button>
      </div>
      <div class="form-row" style="margin-top:8px; align-items:flex-end;">
        <div id="${filtersId}" style="display:flex; flex-direction:column; gap:8px; flex:1 1 auto;"></div>
        <button type="button" class="secondary" id="${addFilterId}">Add Filter</button>
      </div>
      <div class="form-row" id="${colorEditorId}" style="margin-top:8px; gap:12px; flex-wrap:wrap;"></div>
    `;

    const xSel = document.getElementById(xId);
    const ySel = document.getElementById(yId);
    const colorSel = document.getElementById(colorId);
    const xLabelInp = document.getElementById(xLabelId);
    const yLabelInp = document.getElementById(yLabelId);
    const legendTitleInp = document.getElementById(legendTitleId);
    const xMinInp = document.getElementById(xMinId);
    const xMaxInp = document.getElementById(xMaxId);
    const yMinInp = document.getElementById(yMinId);
    const yMaxInp = document.getElementById(yMaxId);
    const avgMetricSel = document.getElementById(avgMetricId);
    const avgStyleSel = document.getElementById(avgStyleId);
    const invertYChk = document.getElementById(invertYId);
    const presetNameInp = document.getElementById(presetNameId);
    const presetSel = document.getElementById(presetSelectId);
    const presetSaveBtn = document.getElementById(presetSaveId);
    const presetLoadBtn = document.getElementById(presetLoadId);
    const presetDeleteBtn = document.getElementById(presetDeleteId);
    const filtersEl = document.getElementById(filtersId);
    const addFilterBtn = document.getElementById(addFilterId);
    const colorEditorEl = document.getElementById(colorEditorId);
    for (const c of vectorCols) {
      const optX = document.createElement('option');
      optX.value = optX.textContent = c.name;
      xSel.appendChild(optX);
      const optY = document.createElement('option');
      optY.value = optY.textContent = c.name;
      ySel.appendChild(optY);
    }
    for (const c of simpleCols) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = c.name;
      colorSel.appendChild(opt);
    }

    const preferredX = vectorCols.find(c => c.name === 'spot_sizes');
    const preferredY = vectorCols.find(c => c.name === 'spikes_stim_mean');
    xSel.value = preferredX ? preferredX.name : vectorCols[0].name;
    ySel.value = preferredY ? preferredY.name : (vectorCols.find(c => c.name !== xSel.value)?.name || vectorCols[0].name);
    if (cols.some(c => c.name === 'cell_type' && c.is_simple)) colorSel.value = 'cell_type';

    let currentData = null;
    let currentColors = {};
    let currentLegendLabels = {};
    let currentLegendAutoLabels = {};
    let filterSeq = 0;
    let lastAutoLegend = '';

    async function fetchColumnValues(column) {
      const data = await fetchJSON(`/api/column_values?column=${encodeURIComponent(column)}`);
      return Array.isArray(data.values) ? data.values : [];
    }

    async function fetchPresetList() {
      const data = await fetchJSON('/api/plot_prefs/line_by_row');
      return Array.isArray(data.presets) ? data.presets : [];
    }

    async function fetchPreset(name) {
      return await fetchJSON(`/api/plot_prefs/line_by_row/${encodeURIComponent(name)}`);
    }

    async function savePreset(name, preferences) {
      const target = withDatasetId('/api/plot_prefs/line_by_row');
      const res = await fetch(target, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        credentials: 'same-origin',
        body: JSON.stringify({ name, preferences }),
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      return await res.json();
    }

    async function deletePreset(name) {
      const target = withDatasetId(`/api/plot_prefs/line_by_row/${encodeURIComponent(name)}`);
      const res = await fetch(target, {
        method: 'DELETE',
        headers: { 'Accept': 'application/json' },
        credentials: 'same-origin',
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      return await res.json();
    }

    async function refreshPresetOptions(selectedName = '') {
      const presets = (await fetchPresetList()).sort((a, b) => String(a.name || '').localeCompare(String(b.name || '')));
      presetSel.innerHTML = '<option value="">Choose preset</option>';
      for (const preset of presets) {
        if (!preset || !preset.name) continue;
        const opt = document.createElement('option');
        opt.value = preset.name;
        opt.textContent = preset.name;
        presetSel.appendChild(opt);
      }
      if (selectedName && presets.some(p => p.name === selectedName)) {
        presetSel.value = selectedName;
      }
    }

    async function addFilterRow(defaultMode = 'include', defaultColumn = '', defaultValue = '') {
      filterSeq += 1;
      const row = document.createElement('div');
      row.className = 'form-row';
      row.dataset.filterSeq = String(filterSeq);
      row.innerHTML = `
        <span class="field">
          <label>Filter</label>
          <select class="line-filter-op">
            <option value="include">Select by</option>
            <option value="exclude">Unselect by</option>
          </select>
        </span>
        <span class="field">
          <label>Variable</label>
          <select class="line-filter-col">
            <option value="">Choose variable</option>
          </select>
        </span>
        <span class="field">
          <label>Equals</label>
          <select class="line-filter-val" disabled>
            <option value="">Choose value</option>
          </select>
        </span>
        <button type="button" class="secondary line-filter-remove">Remove</button>
      `;
      filtersEl.appendChild(row);
      const opSel = row.querySelector('.line-filter-op');
      const colSel = row.querySelector('.line-filter-col');
      const valSel = row.querySelector('.line-filter-val');
      const removeBtn = row.querySelector('.line-filter-remove');
      for (const c of simpleCols) {
        const opt = document.createElement('option');
        opt.value = opt.textContent = c.name;
        colSel.appendChild(opt);
      }
      async function refreshValues(selectedValue = '') {
        const col = colSel.value;
        valSel.innerHTML = '<option value="">Choose value</option>';
        valSel.disabled = !col;
        if (!col) return;
        const values = await fetchColumnValues(col);
        for (const item of values) {
          const opt = document.createElement('option');
          opt.value = item.value;
          opt.textContent = item.label;
          valSel.appendChild(opt);
        }
        if (selectedValue || selectedValue === '') valSel.value = selectedValue;
      }
      colSel.addEventListener('change', async () => {
        await refreshValues('');
        await refreshPlot(true);
      });
      opSel.addEventListener('change', () => refreshPlot(true));
      valSel.addEventListener('change', () => refreshPlot(true));
      removeBtn.addEventListener('click', () => {
        row.remove();
        refreshPlot(true);
      });
      opSel.value = defaultMode === 'exclude' ? 'exclude' : 'include';
      if (defaultColumn) {
        colSel.value = defaultColumn;
        await refreshValues(defaultValue);
      }
    }

    async function setFilters(filters) {
      filtersEl.innerHTML = '';
      filterSeq = 0;
      const safeFilters = Array.isArray(filters) ? filters.filter(f => f && typeof f.column === 'string') : [];
      if (!safeFilters.length) {
        return;
      }
      for (const filter of safeFilters) {
        await addFilterRow(filter.mode || filter.op || 'include', filter.column || '', filter.value ?? '');
      }
    }

    function collectFilters() {
      const filters = [];
      for (const row of filtersEl.querySelectorAll('[data-filter-seq]')) {
        const mode = row.querySelector('.line-filter-op')?.value || 'include';
        const col = row.querySelector('.line-filter-col')?.value || '';
        const val = row.querySelector('.line-filter-val')?.value ?? '';
        if (col && val !== '') filters.push({ mode, column: col, value: val });
      }
      return filters;
    }

    function parseMaybeNumber(value) {
      const n = parseFloat(value);
      return Number.isFinite(n) ? n : null;
    }

    function getRenderOptions() {
      const opts = {
        colorMap: currentColors,
        legendLabelMap: currentLegendLabels,
        xLabel: (xLabelInp.value || '').trim() || (currentData?.x || xSel.value || ''),
        yLabel: (yLabelInp.value || '').trim() || (currentData?.y || ySel.value || ''),
        legendTitle: (legendTitleInp.value || '').trim() || (currentData?.color_column || ''),
        avgMetric: avgMetricSel.value || 'sem',
        avgStyle: avgStyleSel.value || 'shaded',
        invertY: !!invertYChk.checked,
      };
      const xMin = parseMaybeNumber(xMinInp.value);
      const xMax = parseMaybeNumber(xMaxInp.value);
      const yMin = parseMaybeNumber(yMinInp.value);
      const yMax = parseMaybeNumber(yMaxInp.value);
      if (xMin !== null) opts.xMin = xMin;
      if (xMax !== null) opts.xMax = xMax;
      if (yMin !== null) opts.yMin = yMin;
      if (yMax !== null) opts.yMax = yMax;
      return opts;
    }

    function getCurrentPreferences() {
      const savedLegendLabels = {};
      for (const [key, value] of Object.entries(currentLegendLabels)) {
        savedLegendLabels[key] = stripCountLabel(value);
      }
      const savedColors = { ...currentColors };
      for (const inp of colorEditorEl.querySelectorAll('input[type="color"][data-color-key]')) {
        const key = inp.dataset.colorKey || '';
        if (key) savedColors[key] = inp.value;
      }
      return {
        x: xSel.value || '',
        y: ySel.value || '',
        color: colorSel.value || '',
        xLabel: xLabelInp.value || '',
        yLabel: yLabelInp.value || '',
        legendTitle: legendTitleInp.value || '',
        xMin: xMinInp.value || '',
        xMax: xMaxInp.value || '',
        yMin: yMinInp.value || '',
        yMax: yMaxInp.value || '',
        avgMetric: avgMetricSel.value || 'sem',
        avgStyle: avgStyleSel.value || 'shaded',
        invertY: !!invertYChk.checked,
        filters: collectFilters(),
        colorMap: savedColors,
        legendLabelMap: savedLegendLabels,
      };
    }

    function applyTextInputValue(input, value) {
      input.value = value || '';
      if (value) input.dataset.userEdited = '1';
      else delete input.dataset.userEdited;
    }

    async function applyPreferences(prefs) {
      if (!prefs || typeof prefs !== 'object') return;
      if (prefs.x && vectorCols.some(c => c.name === prefs.x)) xSel.value = prefs.x;
      if (prefs.y && vectorCols.some(c => c.name === prefs.y)) ySel.value = prefs.y;
      if (typeof prefs.color === 'string' && (!prefs.color || simpleCols.some(c => c.name === prefs.color))) {
        colorSel.value = prefs.color;
      }
      applyTextInputValue(xLabelInp, prefs.xLabel || '');
      applyTextInputValue(yLabelInp, prefs.yLabel || '');
      applyTextInputValue(legendTitleInp, prefs.legendTitle || '');
      xMinInp.value = prefs.xMin || '';
      xMaxInp.value = prefs.xMax || '';
      yMinInp.value = prefs.yMin || '';
      yMaxInp.value = prefs.yMax || '';
      avgMetricSel.value = prefs.avgMetric === 'sd' ? 'sd' : 'sem';
      avgStyleSel.value = prefs.avgStyle === 'bars' ? 'bars' : 'shaded';
      invertYChk.checked = !!prefs.invertY;
      currentColors = { ...(prefs.colorMap || {}) };
      currentLegendLabels = { ...(prefs.legendLabelMap || {}) };
      currentLegendAutoLabels = {};
      await setFilters(prefs.filters || []);
      await refreshPlot(false);
    }

    function rerenderCurrent() {
      if (!currentData) return;
      renderLineByRowPlot(previewEl, currentData, getRenderOptions());
    }

    function refreshColorEditor(data) {
      colorEditorEl.innerHTML = '';
      const colorColumn = data && data.color_column ? data.color_column : '';
      const traces = Array.isArray(data?.traces) ? data.traces : [];
      const categories = [];
      if (colorColumn) {
        for (const row of traces) {
          const key = row.color_value == null ? 'NA' : String(row.color_value);
          if (!categories.includes(key)) categories.push(key);
        }
      } else {
        categories.push('All lines');
      }
      const palette = ['#60a5fa','#34d399','#f59e0b','#ef4444','#a78bfa','#10b981','#f472b6','#22d3ee','#fb7185','#93c5fd','#fbbf24','#d946ef'];
      categories.forEach((key, i) => {
        const rowsForKey = traces.filter(row => (row.color_value == null ? 'NA' : String(row.color_value)) === key);
        const animalCount = new Set(
          rowsForKey
            .map(row => row.animal_value == null ? null : String(row.animal_value))
            .filter(v => v != null && v !== '')
        ).size;
        const autoLegendLabel = formatCountLabel(displayCategoryLabel(key), rowsForKey.length, animalCount > 0 ? animalCount : '?');
        if (!currentColors[key]) currentColors[key] = palette[i % palette.length];
        if (!Object.prototype.hasOwnProperty.call(currentLegendLabels, key) || currentLegendLabels[key] === currentLegendAutoLabels[key]) {
          currentLegendLabels[key] = autoLegendLabel;
        } else if (stripCountLabel(currentLegendLabels[key]) === currentLegendLabels[key]) {
          currentLegendLabels[key] = formatCountLabel(currentLegendLabels[key], rowsForKey.length, animalCount > 0 ? animalCount : '?');
        }
        currentLegendAutoLabels[key] = autoLegendLabel;
        const wrap = document.createElement('label');
        wrap.className = 'field';
        wrap.style.minWidth = '180px';
        wrap.innerHTML = `<span style="display:block; margin-bottom:4px;">${displayCategoryLabel(key)}</span>`;
        const textInp = document.createElement('input');
        textInp.type = 'text';
        textInp.value = currentLegendLabels[key];
        textInp.style.display = 'block';
        textInp.style.marginBottom = '6px';
        textInp.addEventListener('input', () => {
          currentLegendLabels[key] = textInp.value;
          rerenderCurrent();
        });
        const inp = document.createElement('input');
        inp.type = 'color';
        inp.value = currentColors[key];
        inp.dataset.colorKey = key;
        const handleColorChange = () => {
          currentColors[key] = inp.value;
          rerenderCurrent();
        };
        inp.addEventListener('input', handleColorChange);
        inp.addEventListener('change', handleColorChange);
        wrap.appendChild(textInp);
        wrap.appendChild(inp);
        colorEditorEl.appendChild(wrap);
      });
    }

    async function refreshPlot(resetColors = false) {
      const x = xSel.value;
      const y = ySel.value;
      const color = colorSel.value;
      try {
        const q = new URLSearchParams({ x, y });
        if (color) q.set('color', color);
        for (const filter of collectFilters()) {
          q.append('filter_op', filter.mode);
          q.append('filter_col', filter.column);
          q.append('filter_val', filter.value);
        }
        const data = await fetchJSON(`/api/plot/line_by_row?${q.toString()}`);
        const prevAutoLegend = lastAutoLegend;
        currentData = data;
        if (resetColors) {
          currentColors = {};
          currentLegendLabels = {};
          currentLegendAutoLabels = {};
        }
        if (!xLabelInp.dataset.userEdited || !xLabelInp.value) xLabelInp.value = data.x || '';
        if (!yLabelInp.dataset.userEdited || !yLabelInp.value) yLabelInp.value = data.y || '';
        lastAutoLegend = data.color_column || '';
        if (!legendTitleInp.dataset.userEdited || !legendTitleInp.value || legendTitleInp.value === prevAutoLegend) {
          legendTitleInp.value = lastAutoLegend;
        }
        refreshColorEditor(data);
        rerenderCurrent();
      } catch (e) {
        previewEl.innerHTML = `<div style=\"color: var(--muted);\">${String(e)}</div>`;
        colorEditorEl.innerHTML = '';
      }
    }

    [xLabelInp, yLabelInp, legendTitleInp].forEach((inp) => {
      inp.addEventListener('input', () => {
        inp.dataset.userEdited = '1';
        rerenderCurrent();
      });
    });
    [xMinInp, xMaxInp, yMinInp, yMaxInp, avgMetricSel, avgStyleSel, invertYChk].forEach((el) => {
      el.addEventListener('input', rerenderCurrent);
      el.addEventListener('change', rerenderCurrent);
    });
    xSel.addEventListener('change', () => refreshPlot(true));
    ySel.addEventListener('change', () => refreshPlot(true));
    colorSel.addEventListener('change', () => refreshPlot(true));
    addFilterBtn.addEventListener('click', async () => { await addFilterRow('include', ''); });
    presetSaveBtn.addEventListener('click', async () => {
      const name = (presetNameInp.value || '').trim();
      if (!name) return;
      await savePreset(name, getCurrentPreferences());
      await refreshPresetOptions(name);
    });
    presetLoadBtn.addEventListener('click', async () => {
      const name = presetSel.value;
      if (!name) return;
      const preset = await fetchPreset(name);
      presetNameInp.value = preset.name || name;
      await applyPreferences(preset.preferences || {});
    });
    presetDeleteBtn.addEventListener('click', async () => {
      const name = presetSel.value;
      if (!name) return;
      await deletePreset(name);
      if ((presetNameInp.value || '').trim() === name) presetNameInp.value = '';
      await refreshPresetOptions('');
    });
    await refreshPresetOptions('');
    await addFilterRow('include', 'cell_type');
    await refreshPlot(true);
  }

  function renderLineByRowPlot(container, data, opts = {}) {
    const rows = Array.isArray(data.traces) ? data.traces : [];
    const xLabel = opts.xLabel || data.x || '';
    const yLabel = opts.yLabel || data.y || '';
    const invertY = !!opts.invertY;
    const xUnit = data.x_unit || '';
    const yUnit = data.y_unit || '';
    const colorColumn = data.color_column || '';
    const legendTitle = opts.legendTitle || colorColumn || '';
    const colorMap = opts.colorMap || {};
    const legendLabelMap = opts.legendLabelMap || {};
    const titleX = xUnit ? `${xLabel} (${xUnit})` : xLabel;
    const titleY = yUnit ? `${yLabel} (${yUnit})` : yLabel;
    const palette = ['#60a5fa','#34d399','#f59e0b','#ef4444','#a78bfa','#10b981','#f472b6','#22d3ee','#fb7185','#93c5fd','#fbbf24','#d946ef'];

    function clamp(v, lo, hi) {
      return Math.min(hi, Math.max(lo, v));
    }

    function hexToRgb(hex) {
      if (!hex || typeof hex !== 'string') return null;
      const clean = hex.replace('#', '').trim();
      if (clean.length !== 6) return null;
      const num = parseInt(clean, 16);
      if (!Number.isFinite(num)) return null;
      return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
    }

    function rgbToHex(r, g, b) {
      const toHex = (v) => clamp(Math.round(v), 0, 255).toString(16).padStart(2, '0');
      return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
    }

    function lightenColor(hex, amount = 0.5) {
      const rgb = hexToRgb(hex);
      if (!rgb) return hex;
      return rgbToHex(
        rgb.r + (255 - rgb.r) * amount,
        rgb.g + (255 - rgb.g) * amount,
        rgb.b + (255 - rgb.b) * amount
      );
    }

    function computeAverage(rowsForGroup, metric = 'sem') {
      if (!rowsForGroup.length) return null;
      function prepareRow(row) {
        const xs = Array.isArray(row.x) ? row.x : [];
        const ys = Array.isArray(row.y) ? row.y : [];
        const pairs = [];
        for (let j = 0; j < Math.min(xs.length, ys.length); j += 1) {
          const x = xs[j];
          const y = ys[j];
          if (typeof x === 'number' && isFinite(x) && typeof y === 'number' && isFinite(y)) {
            pairs.push({ x, y });
          }
        }
        pairs.sort((a, b) => a.x - b.x);
        if (!pairs.length) return { x: [], y: [] };
        const deduped = [];
        for (const pair of pairs) {
          const prev = deduped[deduped.length - 1];
          if (prev && Math.abs(prev.x - pair.x) < 1e-9) {
            prev.values.push(pair.y);
          } else {
            deduped.push({ x: pair.x, values: [pair.y] });
          }
        }
        return {
          x: deduped.map(p => p.x),
          y: deduped.map(p => p.values.reduce((acc, v) => acc + v, 0) / p.values.length),
        };
      }

      function interpolateRow(row, xTarget) {
        const xs = row.x;
        const ys = row.y;
        if (!xs.length) return null;
        if (xTarget < xs[0] || xTarget > xs[xs.length - 1]) return null;
        for (let j = 0; j < xs.length; j += 1) {
          if (Math.abs(xs[j] - xTarget) < 1e-9) return ys[j];
        }
        for (let j = 0; j < xs.length - 1; j += 1) {
          const x0 = xs[j];
          const x1 = xs[j + 1];
          if (xTarget < x0 || xTarget > x1) continue;
          if (Math.abs(x1 - x0) < 1e-12) return ys[j];
          const t = (xTarget - x0) / (x1 - x0);
          return ys[j] + t * (ys[j + 1] - ys[j]);
        }
        return null;
      }

      const preparedRows = rowsForGroup
        .map(prepareRow)
        .filter(row => row.x.length > 0 && row.y.length > 0);
      if (!preparedRows.length) return null;
      const unionXMap = new Map();
      for (const row of preparedRows) {
        for (const x of row.x) {
          unionXMap.set(Number(x).toPrecision(12), x);
        }
      }
      const unionXs = Array.from(unionXMap.values()).sort((a, b) => a - b);
      if (!unionXs.length) return null;
      const xs = [];
      const mean = [];
      const err = [];
      const counts = [];
      for (const xTarget of unionXs) {
        const vals = [];
        for (const row of preparedRows) {
          const yInterp = interpolateRow(row, xTarget);
          if (typeof yInterp === 'number' && isFinite(yInterp)) vals.push(yInterp);
        }
        if (!vals.length) continue;
        const nHere = vals.length;
        const mu = vals.reduce((acc, v) => acc + v, 0) / nHere;
        let spread = 0;
        if (nHere >= 2) {
          const variance = vals.reduce((acc, v) => acc + ((v - mu) ** 2), 0) / (nHere - 1);
          const sd = Math.sqrt(Math.max(variance, 0));
          spread = metric === 'sd' ? sd : sd / Math.sqrt(nHere);
        }
        xs.push(xTarget);
        mean.push(mu);
        err.push(spread);
        counts.push(nHere);
      }
      return { x: xs, mean, err, n: rowsForGroup.length, counts };
    }

    const grouped = new Map();
    const colorIndex = new Map();
    rows.forEach((row, i) => {
      const x = Array.isArray(row.x) ? row.x : [];
      const yBase = Array.isArray(row.y) ? row.y : [];
      const y = invertY ? yBase.map(v => (typeof v === 'number' && isFinite(v) ? -v : v)) : yBase;
      const label = row.id || `row ${i + 1}`;
      const colorKey = colorColumn ? (row.color_value == null ? 'NA' : String(row.color_value)) : 'All lines';
      const animalValue = row.animal_value == null ? null : String(row.animal_value);
      if (!colorIndex.has(colorKey)) colorIndex.set(colorKey, colorIndex.size);
      if (!grouped.has(colorKey)) grouped.set(colorKey, []);
      grouped.get(colorKey).push({ id: label, x, y, animalValue });
    });

    const traces = [];
    const overlayShading = [];
    const overlayMeans = [];
    for (const [colorKey, groupRows] of grouped.entries()) {
      const baseColor = colorMap[colorKey] || palette[colorIndex.get(colorKey) % palette.length];
      const rowColor = lightenColor(baseColor, 0.55);
      const baseLegendLabel = displayCategoryLabel(colorKey);
      const animalCount = new Set(
        groupRows
          .map(row => row.animalValue)
          .filter(v => v != null && v !== '')
      ).size;
      const animalText = animalCount > 0 ? animalCount : '?';
      const autoLegendLabel = `${baseLegendLabel} (n=${groupRows.length} rows, ${animalText} animals)`;
      const legendDisplayLabel = Object.prototype.hasOwnProperty.call(legendLabelMap, colorKey)
        ? legendLabelMap[colorKey]
        : autoLegendLabel;
      groupRows.forEach((row) => {
        const text = row.x.map((xv, j) => `${row.id}<br>${xLabel}: ${xv}<br>${yLabel}: ${row.y[j]}`);
        traces.push({
          type: 'scatter',
          mode: 'lines',
          name: legendDisplayLabel,
          legendgroup: colorKey,
          showlegend: false,
          x: row.x,
          y: row.y,
          text,
          line: { color: rowColor, width: 1 },
          opacity: grouped.size > 1 || rows.length > 25 ? 0.5 : 0.7,
          hovertemplate: '%{text}<extra></extra>',
        });
      });
      const avg = computeAverage(groupRows, opts.avgMetric || 'sem');
      if (avg && avg.n >= 3) {
        if (opts.avgStyle === 'shaded') {
          const upper = avg.mean.map((v, i) => v + avg.err[i]);
          const lower = avg.mean.map((v, i) => v - avg.err[i]);
          overlayShading.push({
            type: 'scatter',
            mode: 'lines',
            x: avg.x.concat(avg.x.slice().reverse()),
            y: upper.concat(lower.slice().reverse()),
            fill: 'toself',
            fillcolor: lightenColor(baseColor, 0.75),
            line: { color: 'rgba(0,0,0,0)', width: 0 },
            hoverinfo: 'skip',
            showlegend: false,
            legendgroup: colorKey,
          });
        }
        overlayMeans.push({
          type: 'scatter',
          mode: 'lines',
          name: legendDisplayLabel,
          legendgroup: colorKey,
          showlegend: !!colorColumn,
          x: avg.x,
          y: avg.mean,
          line: { color: baseColor, width: 3 },
          error_y: {
            type: 'data',
            array: avg.err,
            visible: opts.avgStyle === 'bars',
            color: baseColor,
            thickness: 1.5,
            width: 3,
          },
          hovertemplate: `${legendDisplayLabel}<br>${xLabel}: %{x}<br>${yLabel}: %{y}<extra></extra>`,
        });
      } else if (colorColumn) {
        overlayMeans.push({
          type: 'scatter',
          mode: 'lines',
          name: legendDisplayLabel,
          legendgroup: colorKey,
          showlegend: true,
          visible: 'legendonly',
          x: [0],
          y: [0],
          line: { color: baseColor, width: 3 },
          hoverinfo: 'skip',
        });
      }
    }
    traces.push(...overlayShading);
    traces.push(...overlayMeans);

    const width = container.clientWidth || 800;
    const small = width < 560;
    const height = small ? 300 : 380;
    const xaxis = { title: titleX, automargin: true, gridcolor: '#f3f4f6' };
    const yaxis = { title: titleY, automargin: true, gridcolor: '#e5e7eb' };
    const allX = rows.flatMap(row => Array.isArray(row.x) ? row.x : []).filter(v => typeof v === 'number' && isFinite(v));
    const allY = rows.flatMap(row => Array.isArray(row.y) ? row.y : []).filter(v => typeof v === 'number' && isFinite(v));
    if (typeof opts.xMin === 'number' || typeof opts.xMax === 'number') {
      const min = typeof opts.xMin === 'number' ? opts.xMin : (allX.length ? Math.min(...allX) : null);
      const max = typeof opts.xMax === 'number' ? opts.xMax : (allX.length ? Math.max(...allX) : null);
      if (min !== null && max !== null && max > min) xaxis.range = [min, max];
    }
    if (typeof opts.yMin === 'number' || typeof opts.yMax === 'number') {
      const min = typeof opts.yMin === 'number' ? opts.yMin : (allY.length ? Math.min(...allY) : null);
      const max = typeof opts.yMax === 'number' ? opts.yMax : (allY.length ? Math.max(...allY) : null);
      if (min !== null && max !== null && max > min) yaxis.range = [min, max];
    }
    const layout = {
      height,
      margin: { l: 70, r: 20, t: 10, b: 60 },
      xaxis,
      yaxis,
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      showlegend: !!colorColumn,
      legend: colorColumn ? { title: { text: legendTitle || colorColumn } } : undefined,
    };

    renderPlot(container, traces, layout, { displayModeBar: false }, 'line-by-row');
  }

  async function buildBarConfig(card, configEl, previewEl) {
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
    const vectorCols = cols.filter(c => c.is_vector || c.is_reducible_vector);
    const simple = cols.filter(c => c.is_simple);
    const barValueOptions = [
      ...numeric.map(c => ({ value: c.name, label: c.name })),
      ...vectorCols.flatMap(c => ([
        { value: `${c.name}::mean`, label: `${c.name} (mean)` },
        { value: `${c.name}::median`, label: `${c.name} (median)` },
      ])),
    ];
    const validBarValueSet = new Set(barValueOptions.map(opt => opt.value));

    const id = card.dataset.seq;
    const valueId = `plot-${id}-value`;
    const groupId = `plot-${id}-group`;
    const xLabelId = `plot-${id}-bar-x-label`;
    const yLabelId = `plot-${id}-bar-y-label`;
    const xTickAngleId = `plot-${id}-bar-x-tick-angle`;
    const errId = `plot-${id}-err`;
    const orderId = `plot-${id}-order`;
    const logId = `plot-${id}-log`;
    const yminId = `plot-${id}-ymin`;
    const ymaxId = `plot-${id}-ymax`;
    const presetNameId = `plot-${id}-bar-preset-name`;
    const presetSelectId = `plot-${id}-bar-preset-select`;
    const presetSaveId = `plot-${id}-bar-preset-save`;
    const presetLoadId = `plot-${id}-bar-preset-load`;
    const presetDeleteId = `plot-${id}-bar-preset-delete`;
    const addFilterId = `plot-${id}-bar-add-filter`;
    const filtersId = `plot-${id}-bar-filters`;
    const statsControlsId = `plot-${id}-bar-stats`;
    const testMethodId = `plot-${id}-bar-test-method`;
    const runTestId = `plot-${id}-bar-run-test`;
    const showSigId = `plot-${id}-bar-show-sig`;
    const testResultId = `plot-${id}-bar-test-result`;
    const colorEditorId = `plot-${id}-bar-colors`;
    const orderEditorId = `plot-${id}-bar-order-editor`;
    configEl.innerHTML = `
      <div class="form-row">
        <span class="field">
          <label for="${valueId}">Display variable</label>
          <select id="${valueId}"></select>
        </span>
        <span class="field">
          <label for="${groupId}">Group by</label>
          <select id="${groupId}"></select>
        </span>
      </div>
      <div class="form-row" style="margin-top:8px;">
        <span class="field">
          <label for="${xLabelId}">X axis label</label>
          <input id="${xLabelId}" type="text" />
        </span>
        <span class="field">
          <label for="${yLabelId}">Y axis label</label>
          <input id="${yLabelId}" type="text" />
        </span>
        <span class="field">
          <label for="${xTickAngleId}">Label angle</label>
          <select id="${xTickAngleId}">
            <option value="0">0°</option>
            <option value="45">45°</option>
            <option value="90">90°</option>
          </select>
        </span>
        <span class="field">
          <label for="${errId}">Error bars</label>
          <select id="${errId}">
            <option value="none">None</option>
            <option value="sd">SD</option>
            <option value="sem">SEM</option>
          </select>
        </span>
        <span class="field">
          <label for="${orderId}">Order by</label>
          <select id="${orderId}">
            <option value="label_asc">Label (A→Z)</option>
            <option value="label_desc">Label (Z→A)</option>
            <option value="value_asc">Value/mean (low→high)</option>
            <option value="value_desc">Value/mean (high→low)</option>
            <option value="custom">Custom</option>
          </select>
        </span>
        <label class="toggle"><input id="${logId}" type="checkbox" /> Log scale</label>
      </div>
      <div class="form-row" style="margin-top:8px;">
        <span class="field">
          <label for="${yminId}">Y min</label>
          <input type="number" id="${yminId}" placeholder="auto" step="any" />
        </span>
        <span class="field">
          <label for="${ymaxId}">Y max</label>
          <input type="number" id="${ymaxId}" placeholder="auto" step="any" />
        </span>
        <button type="button" class="secondary" id="plot-${id}-autoscale">Autoscale</button>
      </div>
      <div class="form-row" style="margin-top:8px; align-items:flex-end;">
        <span class="field">
          <label for="${presetNameId}">Preset name</label>
          <input id="${presetNameId}" type="text" placeholder="Save current settings" />
        </span>
        <button type="button" class="secondary" id="${presetSaveId}">Save Preferences</button>
        <span class="field">
          <label for="${presetSelectId}">Saved presets</label>
          <select id="${presetSelectId}">
            <option value="">Choose preset</option>
          </select>
        </span>
        <button type="button" class="secondary" id="${presetLoadId}">Load</button>
        <button type="button" class="secondary" id="${presetDeleteId}">Delete</button>
      </div>
      <div class="form-row" style="margin-top:8px; align-items:flex-end;">
        <div id="${filtersId}" style="display:flex; flex-direction:column; gap:8px; flex:1 1 auto;"></div>
        <button type="button" class="secondary" id="${addFilterId}">Add Filter</button>
      </div>
      <div class="form-row" id="${statsControlsId}" style="margin-top:8px; align-items:flex-end; display:none;">
        <span class="field">
          <label for="${testMethodId}">Two-group test</label>
          <select id="${testMethodId}">
            <option value="ttest">2-tailed t-test</option>
            <option value="mannwhitney">Mann-Whitney U</option>
          </select>
        </span>
        <button type="button" class="secondary" id="${runTestId}">Run Test</button>
        <label class="toggle"><input id="${showSigId}" type="checkbox" /> Show significance line</label>
        <span id="${testResultId}" style="color: var(--muted);"></span>
      </div>
      <div class="form-row" id="${colorEditorId}" style="margin-top:8px; gap:12px; flex-wrap:wrap;"></div>
      <div class="form-row" id="${orderEditorId}" style="margin-top:8px; gap:8px; flex-wrap:wrap;"></div>
    `;
    const valueSel = document.getElementById(valueId);
    const groupSel = document.getElementById(groupId);
    const xLabelInp = document.getElementById(xLabelId);
    const yLabelInp = document.getElementById(yLabelId);
    const xTickAngleSel = document.getElementById(xTickAngleId);
    const errSel = document.getElementById(errId);
    const orderSel = document.getElementById(orderId);
    const logChk = document.getElementById(logId);
    const yminInp = document.getElementById(yminId);
    const ymaxInp = document.getElementById(ymaxId);
    const presetNameInp = document.getElementById(presetNameId);
    const presetSel = document.getElementById(presetSelectId);
    const presetSaveBtn = document.getElementById(presetSaveId);
    const presetLoadBtn = document.getElementById(presetLoadId);
    const presetDeleteBtn = document.getElementById(presetDeleteId);
    const filtersEl = document.getElementById(filtersId);
    const addFilterBtn = document.getElementById(addFilterId);
    const statsControlsEl = document.getElementById(statsControlsId);
    const testMethodSel = document.getElementById(testMethodId);
    const runTestBtn = document.getElementById(runTestId);
    const showSigChk = document.getElementById(showSigId);
    const testResultEl = document.getElementById(testResultId);
    const colorEditorEl = document.getElementById(colorEditorId);
    const orderEditorEl = document.getElementById(orderEditorId);
    const autoBtn = document.getElementById(`plot-${id}-autoscale`);

    for (const c of barValueOptions) {
      const opt = document.createElement('option');
      opt.value = c.value;
      opt.textContent = c.label;
      valueSel.appendChild(opt);
    }
    for (const c of simple) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = c.name;
      groupSel.appendChild(opt);
    }

    if (barValueOptions.length) valueSel.value = barValueOptions[0].value;
    if (simple.length) {
      const hasCellType = simple.some(c => c.name === 'cell_type');
      if (hasCellType) {
        groupSel.value = 'cell_type';
      } else {
        const selectedBaseValue = String(valueSel.value || '').replace(/::(mean|median)$/, '');
        groupSel.value = simple.find(c => c.name !== selectedBaseValue)?.name || simple[0].name;
      }
    }

    let currentData = null;
    let currentColors = {};
    let currentLabelMap = {};
    let currentLabelAutoMap = {};
    let currentCustomOrder = [];
    let currentTestResult = null;
    let filterSeq = 0;
    let lastAutoXLabel = '';
    let lastAutoYLabel = '';

    async function fetchColumnValues(column) {
      const data = await fetchJSON(`/api/column_values?column=${encodeURIComponent(column)}`);
      return Array.isArray(data.values) ? data.values : [];
    }

    async function fetchPresetList() {
      const data = await fetchJSON('/api/plot_prefs/bar');
      return Array.isArray(data.presets) ? data.presets : [];
    }

    async function fetchPreset(name) {
      return await fetchJSON(`/api/plot_prefs/bar/${encodeURIComponent(name)}`);
    }

    async function savePreset(name, preferences) {
      const target = withDatasetId('/api/plot_prefs/bar');
      const res = await fetch(target, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        credentials: 'same-origin',
        body: JSON.stringify({ name, preferences }),
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      return await res.json();
    }

    async function deletePreset(name) {
      const target = withDatasetId(`/api/plot_prefs/bar/${encodeURIComponent(name)}`);
      const res = await fetch(target, {
        method: 'DELETE',
        headers: { 'Accept': 'application/json' },
        credentials: 'same-origin',
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      return await res.json();
    }

    async function refreshPresetOptions(selectedName = '') {
      const presets = (await fetchPresetList()).sort((a, b) => String(a.name || '').localeCompare(String(b.name || '')));
      presetSel.innerHTML = '<option value="">Choose preset</option>';
      for (const preset of presets) {
        if (!preset || !preset.name) continue;
        const opt = document.createElement('option');
        opt.value = preset.name;
        opt.textContent = preset.name;
        presetSel.appendChild(opt);
      }
      if (selectedName && presets.some(p => p.name === selectedName)) {
        presetSel.value = selectedName;
      }
    }

    async function addFilterRow(defaultMode = 'include', defaultColumn = '', defaultValue = '') {
      filterSeq += 1;
      const row = document.createElement('div');
      row.className = 'form-row';
      row.dataset.filterSeq = String(filterSeq);
      row.innerHTML = `
        <span class="field">
          <label>Filter</label>
          <select class="bar-filter-op">
            <option value="include">Select by</option>
            <option value="exclude">Unselect by</option>
          </select>
        </span>
        <span class="field">
          <label>Variable</label>
          <select class="bar-filter-col">
            <option value="">Choose variable</option>
          </select>
        </span>
        <span class="field">
          <label>Equals</label>
          <select class="bar-filter-val" disabled>
            <option value="">Choose value</option>
          </select>
        </span>
        <button type="button" class="secondary bar-filter-remove">Remove</button>
      `;
      filtersEl.appendChild(row);
      const opSel = row.querySelector('.bar-filter-op');
      const colSel = row.querySelector('.bar-filter-col');
      const valSel = row.querySelector('.bar-filter-val');
      const removeBtn = row.querySelector('.bar-filter-remove');
      for (const c of simple) {
        const opt = document.createElement('option');
        opt.value = opt.textContent = c.name;
        colSel.appendChild(opt);
      }
      async function refreshValues(selectedValue = '') {
        const col = colSel.value;
        valSel.innerHTML = '<option value="">Choose value</option>';
        valSel.disabled = !col;
        if (!col) return;
        const values = await fetchColumnValues(col);
        for (const item of values) {
          const opt = document.createElement('option');
          opt.value = item.value;
          opt.textContent = item.label;
          valSel.appendChild(opt);
        }
        if (selectedValue || selectedValue === '') valSel.value = selectedValue;
      }
      colSel.addEventListener('change', async () => {
        await refreshValues('');
        await refreshPlot();
      });
      opSel.addEventListener('change', () => refreshPlot());
      valSel.addEventListener('change', () => refreshPlot());
      removeBtn.addEventListener('click', () => {
        row.remove();
        refreshPlot();
      });
      opSel.value = defaultMode === 'exclude' ? 'exclude' : 'include';
      if (defaultColumn) {
        colSel.value = defaultColumn;
        await refreshValues(defaultValue);
      }
    }

    async function setFilters(filters) {
      filtersEl.innerHTML = '';
      filterSeq = 0;
      const safeFilters = Array.isArray(filters) ? filters.filter(f => f && typeof f.column === 'string') : [];
      if (!safeFilters.length) return;
      for (const filter of safeFilters) {
        await addFilterRow(filter.mode || filter.op || 'include', filter.column || '', filter.value ?? '');
      }
    }

    function collectFilters() {
      const filters = [];
      for (const row of filtersEl.querySelectorAll('[data-filter-seq]')) {
        const mode = row.querySelector('.bar-filter-op')?.value || 'include';
        const col = row.querySelector('.bar-filter-col')?.value || '';
        const val = row.querySelector('.bar-filter-val')?.value ?? '';
        if (col && val !== '') filters.push({ mode, column: col, value: val });
      }
      return filters;
    }

    function parseMaybeNumber(value) {
      const n = parseFloat(value);
      return Number.isFinite(n) ? n : null;
    }

    function getRenderOptions() {
      const value = valueSel.value;
      return {
        err: errSel.value || 'none',
        order: orderSel.value || 'label_asc',
        log: !!logChk.checked,
        xLabel: (xLabelInp.value || '').trim() || (currentData?.group || groupSel.value || ''),
        yLabel: (yLabelInp.value || '').trim() || (currentData?.value || value || ''),
        xTickAngle: xTickAngleSel.value === '90' ? 90 : (xTickAngleSel.value === '45' ? 45 : 0),
        colorMap: currentColors,
        labelMap: currentLabelMap,
        customOrder: currentCustomOrder.slice(),
        testResult: showSigChk.checked ? currentTestResult : null,
        showSignificanceLine: !!showSigChk.checked,
        yMin: parseMaybeNumber(yminInp.value),
        yMax: parseMaybeNumber(ymaxInp.value),
      };
    }

    function getCurrentPreferences() {
      const savedLabelMap = {};
      for (const [key, value] of Object.entries(currentLabelMap)) {
        savedLabelMap[key] = stripCountLabel(value);
      }
      const savedColors = { ...currentColors };
      for (const inp of colorEditorEl.querySelectorAll('input[type="color"][data-color-key]')) {
        const key = inp.dataset.colorKey || '';
        if (key) savedColors[key] = inp.value;
      }
      return {
        value: valueSel.value || '',
        group: groupSel.value || '',
        xLabel: xLabelInp.value || '',
        yLabel: yLabelInp.value || '',
        xTickAngle: xTickAngleSel.value === '90' ? '90' : (xTickAngleSel.value === '45' ? '45' : '0'),
        err: errSel.value || 'none',
        order: orderSel.value || 'label_asc',
        log: !!logChk.checked,
        yMin: yminInp.value || '',
        yMax: ymaxInp.value || '',
        filters: collectFilters(),
        colorMap: savedColors,
        labelMap: savedLabelMap,
        customOrder: currentCustomOrder.slice(),
        barTestMethod: testMethodSel.value || 'ttest',
        showSignificanceLine: !!showSigChk.checked,
      };
    }

    function applyTextInputValue(input, value) {
      input.value = value || '';
      if (value) input.dataset.userEdited = '1';
      else delete input.dataset.userEdited;
    }

    async function applyPreferences(prefs) {
      if (!prefs || typeof prefs !== 'object') return;
      if (prefs.value && validBarValueSet.has(prefs.value)) valueSel.value = prefs.value;
      if (prefs.group && simple.some(c => c.name === prefs.group)) groupSel.value = prefs.group;
      applyTextInputValue(xLabelInp, prefs.xLabel || '');
      applyTextInputValue(yLabelInp, prefs.yLabel || '');
      xTickAngleSel.value = String(prefs.xTickAngle || '0') === '90'
        ? '90'
        : (String(prefs.xTickAngle || '0') === '45' ? '45' : '0');
      errSel.value = prefs.err === 'sd' || prefs.err === 'sem' ? prefs.err : 'none';
      orderSel.value = prefs.order || 'label_asc';
      logChk.checked = !!prefs.log;
      yminInp.value = prefs.yMin || '';
      ymaxInp.value = prefs.yMax || '';
      currentColors = { ...(prefs.colorMap || {}) };
      currentLabelMap = { ...(prefs.labelMap || {}) };
      currentLabelAutoMap = {};
      currentCustomOrder = Array.isArray(prefs.customOrder) ? prefs.customOrder.map(v => String(v)) : [];
      testMethodSel.value = prefs.barTestMethod === 'mannwhitney' ? 'mannwhitney' : 'ttest';
      showSigChk.checked = !!prefs.showSignificanceLine;
      currentTestResult = null;
      await setFilters(prefs.filters || []);
      await refreshPlot();
    }

    function rerenderCurrent() {
      if (!currentData) return;
      renderBarPlot(previewEl, currentData, getRenderOptions());
    }

    function normalizeCustomOrder(categories) {
      const incoming = Array.isArray(categories) ? categories.map(v => String(v)) : [];
      const seen = new Set();
      const ordered = [];
      for (const key of currentCustomOrder) {
        if (incoming.includes(key) && !seen.has(key)) {
          ordered.push(key);
          seen.add(key);
        }
      }
      for (const key of incoming) {
        if (!seen.has(key)) {
          ordered.push(key);
          seen.add(key);
        }
      }
      currentCustomOrder = ordered;
      return ordered;
    }

    function refreshColorEditor(data) {
      colorEditorEl.innerHTML = '';
      const groups = Array.isArray(data?.groups) ? data.groups : [];
      const categories = groups.map(g => String(g.name ?? ''));
      const palette = ['#60a5fa','#34d399','#f59e0b','#ef4444','#a78bfa','#10b981','#f472b6','#22d3ee','#fb7185','#93c5fd','#fbbf24','#d946ef'];
      categories.forEach((key, i) => {
        const group = groups.find(g => String(g.name ?? '') === key) || {};
        const rowCount = Number.isFinite(group?.count) ? group.count : 0;
        const animalCount = Number.isFinite(group?.animal_count) ? group.animal_count : '?';
        const autoLabel = formatCountLabel(displayCategoryLabel(key), rowCount, animalCount);
        if (!currentColors[key]) currentColors[key] = palette[i % palette.length];
        if (!Object.prototype.hasOwnProperty.call(currentLabelMap, key) || currentLabelMap[key] === currentLabelAutoMap[key]) {
          currentLabelMap[key] = autoLabel;
        } else if (stripCountLabel(currentLabelMap[key]) === currentLabelMap[key]) {
          currentLabelMap[key] = formatCountLabel(currentLabelMap[key], rowCount, animalCount);
        }
        currentLabelAutoMap[key] = autoLabel;
        const wrap = document.createElement('label');
        wrap.className = 'field';
        wrap.style.minWidth = '180px';
        wrap.innerHTML = `<span style="display:block; margin-bottom:4px;">${displayCategoryLabel(key)}</span>`;
        const textInp = document.createElement('input');
        textInp.type = 'text';
        textInp.value = currentLabelMap[key];
        textInp.style.display = 'block';
        textInp.style.marginBottom = '6px';
        textInp.addEventListener('input', () => {
          currentLabelMap[key] = textInp.value;
          rerenderCurrent();
        });
        const inp = document.createElement('input');
        inp.type = 'color';
        inp.value = currentColors[key];
        inp.dataset.colorKey = key;
        const handleColorChange = () => {
          currentColors[key] = inp.value;
          rerenderCurrent();
        };
        inp.addEventListener('input', handleColorChange);
        inp.addEventListener('change', handleColorChange);
        wrap.appendChild(textInp);
        wrap.appendChild(inp);
        colorEditorEl.appendChild(wrap);
      });
    }

    function refreshOrderEditor(data) {
      orderEditorEl.innerHTML = '';
      const groups = Array.isArray(data?.groups) ? data.groups : [];
      const categories = normalizeCustomOrder(groups.map(g => String(g.name ?? '')));
      if (!categories.length) return;
      categories.forEach((key, index) => {
        const wrap = document.createElement('div');
        wrap.className = 'field';
        wrap.style.minWidth = '220px';
        wrap.style.display = 'flex';
        wrap.style.alignItems = 'center';
        wrap.style.gap = '6px';
        const label = document.createElement('span');
        label.style.flex = '1 1 auto';
        label.textContent = currentLabelMap[key] || displayCategoryLabel(key);
        const upBtn = document.createElement('button');
        upBtn.type = 'button';
        upBtn.className = 'secondary';
        upBtn.textContent = 'Up';
        upBtn.disabled = index === 0;
        upBtn.addEventListener('click', () => {
          if (index === 0) return;
          const next = currentCustomOrder.slice();
          [next[index - 1], next[index]] = [next[index], next[index - 1]];
          currentCustomOrder = next;
          orderSel.value = 'custom';
          refreshOrderEditor(currentData);
          rerenderCurrent();
        });
        const downBtn = document.createElement('button');
        downBtn.type = 'button';
        downBtn.className = 'secondary';
        downBtn.textContent = 'Down';
        downBtn.disabled = index === categories.length - 1;
        downBtn.addEventListener('click', () => {
          if (index === categories.length - 1) return;
          const next = currentCustomOrder.slice();
          [next[index], next[index + 1]] = [next[index + 1], next[index]];
          currentCustomOrder = next;
          orderSel.value = 'custom';
          refreshOrderEditor(currentData);
          rerenderCurrent();
        });
        wrap.appendChild(label);
        wrap.appendChild(upBtn);
        wrap.appendChild(downBtn);
        orderEditorEl.appendChild(wrap);
      });
    }

    async function runCurrentTest() {
      if (!currentData || !Array.isArray(currentData.groups) || currentData.groups.length !== 2) {
        currentTestResult = null;
        testResultEl.textContent = '';
        rerenderCurrent();
        return;
      }
      const q = new URLSearchParams({
        value: valueSel.value,
        group: groupSel.value,
        method: testMethodSel.value || 'ttest',
      });
      for (const filter of collectFilters()) {
        q.append('filter_op', filter.mode);
        q.append('filter_col', filter.column);
        q.append('filter_val', filter.value);
      }
      try {
        const result = await fetchJSON(`/api/plot/bar_test?${q.toString()}`);
        currentTestResult = result;
        const pText = formatPValueThreshold(result?.p_value);
        testResultEl.textContent = `${result.summary}: ${pText}`;
        rerenderCurrent();
      } catch (e) {
        currentTestResult = null;
        testResultEl.textContent = String(e);
        rerenderCurrent();
      }
    }

    function refreshStatsControls(data) {
      const groups = Array.isArray(data?.groups) ? data.groups : [];
      const visible = groups.length === 2;
      statsControlsEl.style.display = visible ? '' : 'none';
      runTestBtn.disabled = !visible;
      showSigChk.disabled = !visible;
      if (!visible) {
        currentTestResult = null;
        testResultEl.textContent = '';
      }
    }

    async function refreshPlot() {
      const value = valueSel.value;
      const group = groupSel.value;
      try {
        const q = new URLSearchParams({ value, group });
        for (const filter of collectFilters()) {
          q.append('filter_op', filter.mode);
          q.append('filter_col', filter.column);
          q.append('filter_val', filter.value);
        }
        const data = await fetchJSON(`/api/plot/bar?${q.toString()}`);
        const prevAutoXLabel = lastAutoXLabel;
        const prevAutoYLabel = lastAutoYLabel;
        currentData = data;
        lastAutoXLabel = data.group || '';
        lastAutoYLabel = data.value || '';
        if (!xLabelInp.dataset.userEdited || !xLabelInp.value || xLabelInp.value === prevAutoXLabel) {
          xLabelInp.value = lastAutoXLabel;
        }
        if (!yLabelInp.dataset.userEdited || !yLabelInp.value || yLabelInp.value === prevAutoYLabel) {
          yLabelInp.value = lastAutoYLabel;
        }
        refreshStatsControls(data);
        refreshColorEditor(data);
        refreshOrderEditor(data);
        if (Array.isArray(data.groups) && data.groups.length === 2 && (showSigChk.checked || currentTestResult)) {
          await runCurrentTest();
          return;
        }
        rerenderCurrent();
      } catch (e) {
        previewEl.innerHTML = `<div style=\"color: var(--muted);\">${String(e)}</div>`;
        statsControlsEl.style.display = 'none';
        testResultEl.textContent = '';
        colorEditorEl.innerHTML = '';
        orderEditorEl.innerHTML = '';
      }
    }

    [xLabelInp, yLabelInp].forEach((inp) => {
      inp.addEventListener('input', () => {
        inp.dataset.userEdited = '1';
        rerenderCurrent();
      });
    });
    [xTickAngleSel, errSel, orderSel, logChk, yminInp, ymaxInp].forEach((el) => {
      el.addEventListener('input', rerenderCurrent);
      el.addEventListener('change', rerenderCurrent);
    });
    valueSel.addEventListener('change', refreshPlot);
    groupSel.addEventListener('change', refreshPlot);
    addFilterBtn.addEventListener('click', async () => { await addFilterRow('include', ''); });
    testMethodSel.addEventListener('change', () => {
      currentTestResult = null;
      testResultEl.textContent = '';
      if (showSigChk.checked && currentData && Array.isArray(currentData.groups) && currentData.groups.length === 2) {
        void runCurrentTest();
      } else {
        rerenderCurrent();
      }
    });
    runTestBtn.addEventListener('click', () => { void runCurrentTest(); });
    showSigChk.addEventListener('change', () => {
      if (showSigChk.checked && !currentTestResult && currentData && Array.isArray(currentData.groups) && currentData.groups.length === 2) {
        void runCurrentTest();
      } else {
        rerenderCurrent();
      }
    });
    autoBtn.addEventListener('click', () => {
      yminInp.value = '';
      ymaxInp.value = '';
      rerenderCurrent();
    });
    presetSaveBtn.addEventListener('click', async () => {
      const name = (presetNameInp.value || '').trim();
      if (!name) return;
      await savePreset(name, getCurrentPreferences());
      await refreshPresetOptions(name);
    });
    presetLoadBtn.addEventListener('click', async () => {
      const name = presetSel.value;
      if (!name) return;
      const preset = await fetchPreset(name);
      presetNameInp.value = preset.name || name;
      await applyPreferences(preset.preferences || {});
    });
    presetDeleteBtn.addEventListener('click', async () => {
      const name = presetSel.value;
      if (!name) return;
      await deletePreset(name);
      if ((presetNameInp.value || '').trim() === name) presetNameInp.value = '';
      await refreshPresetOptions('');
    });
    await refreshPresetOptions('');
    await addFilterRow('include', 'cell_type');
    await refreshPlot();
  }

  function renderBarPlot(container, data, opts = { err: 'none', log: false, order: 'label_asc' }) {
    const groups = Array.isArray(data.groups) ? data.groups : [];
    const unit = data.unit || '';
    const valueName = data.value || '';
    const groupName = data.group || '';
    const groupUnit = data.group_unit || '';
    const yBaseLabel = opts.yLabel || valueName || '';
    const xBaseLabel = opts.xLabel || groupName || '';
    const labelMap = opts.labelMap || {};
    const colorMap = opts.colorMap || {};
    const customOrder = Array.isArray(opts.customOrder) ? opts.customOrder.map(v => String(v)) : [];
    const testResult = opts.testResult || null;
    const showSignificanceLine = !!opts.showSignificanceLine;
    const palette = ['#60a5fa','#34d399','#f59e0b','#ef4444','#a78bfa','#10b981','#f472b6','#22d3ee','#fb7185','#93c5fd','#fbbf24','#d946ef'];

    function clamp(v, lo, hi) {
      return Math.min(hi, Math.max(lo, v));
    }

    function hexToRgb(hex) {
      if (!hex || typeof hex !== 'string') return null;
      const clean = hex.replace('#', '').trim();
      if (clean.length !== 6) return null;
      const num = parseInt(clean, 16);
      if (!Number.isFinite(num)) return null;
      return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
    }

    function rgbToHex(r, g, b) {
      const toHex = (v) => clamp(Math.round(v), 0, 255).toString(16).padStart(2, '0');
      return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
    }

    function lightenColor(hex, amount = 0.4) {
      const rgb = hexToRgb(hex);
      if (!rgb) return hex;
      return rgbToHex(
        rgb.r + (255 - rgb.r) * amount,
        rgb.g + (255 - rgb.g) * amount,
        rgb.b + (255 - rgb.b) * amount
      );
    }

    const n = groups.length;
    const indices = Array.from({ length: n }, (_, i) => i);
    const rawMeans = groups.map(g => (typeof g.mean === 'number' && isFinite(g.mean)) ? g.mean : null);
    const names = groups.map(g => String(g.name ?? ''));
    const counts = groups.map(g => Number.isFinite(g?.count) ? g.count : 0);
    const animalCounts = groups.map(g => Number.isFinite(g?.animal_count) ? g.animal_count : null);
    const autoLabels = names.map((name, i) => {
      const animalText = animalCounts[i] == null ? '?' : animalCounts[i];
      return `${displayCategoryLabel(name)} (n=${counts[i]} rows, ${animalText} animals)`;
    });
    const displayLabels = names.map((name, i) => {
      if (Object.prototype.hasOwnProperty.call(labelMap, name)) return labelMap[name];
      const animalText = animalCounts[i] == null ? '?' : animalCounts[i];
      return autoLabels[i];
    });
    const groupColors = names.map((name, i) => colorMap[name] || palette[i % palette.length]);
    const orderMode = opts.order || 'label_asc';
    const customPos = new Map(customOrder.map((name, i) => [name, i]));
    const cmp = (a, b) => {
      if (orderMode === 'custom') {
        const pa = customPos.has(names[a]) ? customPos.get(names[a]) : Number.MAX_SAFE_INTEGER;
        const pb = customPos.has(names[b]) ? customPos.get(names[b]) : Number.MAX_SAFE_INTEGER;
        if (pa !== pb) return pa - pb;
        return a - b;
      }
      if (orderMode === 'value_asc' || orderMode === 'value_desc') {
        const va = rawMeans[a];
        const vb = rawMeans[b];
        const na = !(typeof va === 'number' && isFinite(va));
        const nb = !(typeof vb === 'number' && isFinite(vb));
        if (na && nb) return 0;
        if (na) return 1; // push NaNs/nulls to end
        if (nb) return -1;
        return orderMode === 'value_asc' ? (va - vb) : (vb - va);
      } else {
        // label_asc / label_desc; treat missing as empty string
        const sa = names[a] ?? '';
        const sb = names[b] ?? '';
        return orderMode === 'label_desc' ? String(sb).localeCompare(String(sa)) : String(sa).localeCompare(String(sb));
      }
    };
    indices.sort(cmp);
    const posByOrig = new Map(indices.map((origIdx, pos) => [origIdx, pos]));
    const xlabels = indices.map(i => displayLabels[i]);
    const xpos = indices.map((_, i) => i);
    const barColors = indices.map(i => groupColors[i]);

    const meansOrderedRaw = indices.map(i => rawMeans[i]);
    const means = meansOrderedRaw.map(m => (opts.log && !(m > 0) ? null : m));
    const errArray = indices.map((i) => {
      const g = groups[i];
      let eb = computeError((g && g.values) || [], opts.err);
      const m = groups[i] && (typeof groups[i].mean === 'number' && isFinite(groups[i].mean)) ? groups[i].mean : null;
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
      marker: { color: barColors, opacity: 0.45 },
      customdata: indices.map(i => xlabels[posByOrig.get(i) ?? i]),
      error_y: {
        type: 'data',
        array: errArray,
        visible: opts.err !== 'none',
        color: '#111827',
        thickness: 1.5,
        width: 4,
      },
      hovertemplate: `${yBaseLabel}: %{y}<extra>%{customdata}</extra>`,
    };

    const pointX = [];
    const pointY = [];
    const pointText = [];
    const pointColor = [];
    const jitter = 0.25;
    groups.forEach((g, i) => {
      const pts = Array.isArray(g.points) && g.points.length
        ? g.points
        : (g.values || []).map(v => ({ id: '', value: v }));
      pts.forEach(p => {
        const v = p.value;
        if (!(typeof v === 'number' && isFinite(v))) return;
        if (opts.log && !(v > 0)) return;
        const newPos = posByOrig.get(i) ?? i;
        pointX.push(newPos + (Math.random() - 0.5) * 2 * jitter);
        pointY.push(v);
        const id = (p.id ?? '').toString();
        const displayName = xlabels[newPos] || displayLabels[i] || names[i];
        pointText.push(id ? `${displayName}<br>${id}: ${v}` : `${displayName}<br>${v}`);
        pointColor.push(lightenColor(groupColors[i], 0.15));
      });
    });
    const pointsTrace = {
      type: 'scatter',
      mode: 'markers',
      x: pointX,
      y: pointY,
      text: pointText,
      marker: { color: pointColor, opacity: 0.7, size: 5 },
      hovertemplate: '%{text}<extra></extra>',
      showlegend: false,
    };

    const allValues = groups.flatMap(g => (g.values || [])).filter(v => typeof v === 'number' && isFinite(v));
    const numericMeans = rawMeans.filter(v => typeof v === 'number' && isFinite(v));
    const linMin = (allValues.concat(numericMeans).length ? Math.min(...allValues, ...numericMeans) : 0);
    const linMax = (allValues.concat(numericMeans).length ? Math.max(...allValues, ...numericMeans) : 1);
    const posValues = allValues.filter(v => v > 0).concat(numericMeans.filter(v => v > 0));

    const yTitle = unit ? `${yBaseLabel}<br>(${unit})` : yBaseLabel;
    const yaxis = {
      title: { text: yTitle, standoff: 8 },
      type: opts.log ? 'log' : 'linear',
      autorange: true,
      gridcolor: '#e5e7eb',
      zeroline: !opts.log,
      zerolinecolor: '#9ca3af',
    };
    if (!opts.log) {
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
      let yMinP;
      let yMaxP;
      const hasMin = typeof opts.yMin === 'number' && isFinite(opts.yMin) && opts.yMin > 0;
      const hasMax = typeof opts.yMax === 'number' && isFinite(opts.yMax) && opts.yMax > 0;
      if (hasMin) yMinP = opts.yMin;
      if (hasMax) yMaxP = opts.yMax;

      if (posValues.length) {
        const dataMin = Math.min(...posValues);
        const dataMax = Math.max(...posValues);
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

    const width = container.clientWidth || 800;
    const small = width < 560;
    const requestedAngle = opts.xTickAngle === 90 ? 90 : (opts.xTickAngle === 45 ? 45 : 0);
    const tickAngle = requestedAngle === 0 ? 0 : -requestedAngle;
    const bottomMargin = requestedAngle === 90 ? 160 : (requestedAngle === 45 ? 110 : 60);
    const yTitleHasBreak = yTitle.indexOf('<br>') !== -1;
    const height = (small ? 280 : 360) + (yTitleHasBreak ? 40 : 0);
    const leftMargin = (yTitle.length > 24) ? 90 : 70;

    const layout = {
      height,
      margin: { l: leftMargin, r: 20, t: showSignificanceLine ? 52 : 10, b: bottomMargin },
      xaxis: {
        title: groupUnit ? `${xBaseLabel} (${groupUnit})` : xBaseLabel,
        tickmode: 'array',
        tickvals: xpos,
        ticktext: xlabels,
        tickangle: tickAngle,
        automargin: true,
        gridcolor: '#f3f4f6',
      },
      yaxis: { ...yaxis, automargin: true },
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      showlegend: false,
    };

    if (showSignificanceLine && testResult && groups.length === 2) {
      const pText = formatPValueThreshold(testResult.p_value);
      layout.shapes = [
        { type: 'line', xref: 'x', yref: 'paper', x0: 0, x1: 0, y0: 1.01, y1: 1.045, line: { color: '#111827', width: 1.5 } },
        { type: 'line', xref: 'x', yref: 'paper', x0: 0, x1: 1, y0: 1.045, y1: 1.045, line: { color: '#111827', width: 1.5 } },
        { type: 'line', xref: 'x', yref: 'paper', x0: 1, x1: 1, y0: 1.01, y1: 1.045, line: { color: '#111827', width: 1.5 } },
      ];
      layout.annotations = [
        { x: 0.5, y: 1.06, xref: 'x', yref: 'paper', text: pText, showarrow: false, yanchor: 'bottom', font: { color: '#111827' }, bgcolor: '#ffffff' },
      ];
    }

    renderPlot(container, [barTrace, pointsTrace], layout, { displayModeBar: false }, 'bar');
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

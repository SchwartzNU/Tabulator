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
          <span class="field">
            <label for="${id}-type">Plot Type</label>
            <select id="${id}-type" class="plot-type">
              <option value="bar">bar</option>
              <option value="scatter">scatter</option>
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
    setupPlotsUI();
    setupPCA();
    setupDR();
    setupClassifier();
  });

  // -------- Plot helpers --------
  async function fetchJSON(url) {
    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
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
      const cols = meta.columns || [];
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
      const cols = meta.columns || [];
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
      const cols = meta.columns || [];
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
        const res = await fetch('/api/classify/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
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
    const allCols = cols;

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
    for (const c of allCols) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = c.name;
      groupSel.appendChild(opt);
    }

    if (numeric.length) xSel.value = numeric[0].name;
    if (numeric.length > 1) ySel.value = numeric[1].name;
    // Prefer 'cell_type' as default group if available
    const hasCellType = allCols.some(c => c.name === 'cell_type');
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
    const orderId = `plot-${id}-order`;
    const logId = `plot-${id}-log`;
    const yminId = `plot-${id}-ymin`;
    const ymaxId = `plot-${id}-ymax`;
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
    `;
    const valueSel = document.getElementById(valueId);
    const groupSel = document.getElementById(groupId);
    const errSel = document.getElementById(errId);
    const orderSel = document.getElementById(orderId);
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

    // Choose defaults: first numeric for value, prefer 'cell_type' for group if present, else first non-value column
    if (numeric.length) valueSel.value = numeric[0].name;
    if (allCols.length) {
      const hasCellType = allCols.some(c => c.name === 'cell_type');
      if (hasCellType) {
        groupSel.value = 'cell_type';
      } else {
        groupSel.value = allCols.find(c => c.name !== valueSel.value)?.name || allCols[0].name;
      }
    }

    async function refreshPlot() {
      const value = valueSel.value;
      const group = groupSel.value;
      const err = errSel.value; // none|sd|sem
      const order = orderSel.value; // label_asc|label_desc|value_asc|value_desc
      const log = !!logChk.checked;
      const yMin = parseFloat(yminInp.value);
      const yMax = parseFloat(ymaxInp.value);
      try {
        const data = await fetchJSON(`/api/plot/bar?value=${encodeURIComponent(value)}&group=${encodeURIComponent(group)}`);
        const opts = { err, log, order };
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
    orderSel.addEventListener('change', refreshPlot);
    logChk.addEventListener('change', refreshPlot);
    yminInp.addEventListener('change', refreshPlot);
    ymaxInp.addEventListener('change', refreshPlot);
    autoBtn.addEventListener('click', () => { yminInp.value = ''; ymaxInp.value = ''; refreshPlot(); });
    await refreshPlot();
  }

  function renderBarPlot(container, data, opts = { err: 'none', log: false, order: 'label_asc' }) {
    const groups = Array.isArray(data.groups) ? data.groups : [];
    const unit = data.unit || '';
    const label = data.value || '';
    const groupName = data.group || '';
    const groupUnit = data.group_unit || '';

    // Determine ordering of groups for plotting
    const n = groups.length;
    const indices = Array.from({ length: n }, (_, i) => i);
    const rawMeans = groups.map(g => (typeof g.mean === 'number' && isFinite(g.mean)) ? g.mean : null);
    const names = groups.map(g => String(g.name));
    const orderMode = opts.order || 'label_asc';
    const cmp = (a, b) => {
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
    const xlabels = indices.map(i => names[i]);
    const xpos = indices.map((_, i) => i);

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
    const pointText = [];
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
        pointText.push(id ? `${id}, ${v}` : `${v}`);
      });
    });
    const pointsTrace = {
      type: 'scatter',
      mode: 'markers',
      x: pointX,
      y: pointY,
      text: pointText,
      marker: { color: '#111827', opacity: 0.5, size: 5 },
      hovertemplate: '%{text}<extra></extra>',
      showlegend: false,
    };

    // Compute data extents for optional manual limits
    const allValues = groups.flatMap(g => (g.values || [])).filter(v => typeof v === 'number' && isFinite(v));
    const numericMeans = rawMeans.filter(v => typeof v === 'number' && isFinite(v));
    const linMin = (allValues.concat(numericMeans).length ? Math.min(...allValues, ...numericMeans) : 0);
    const linMax = (allValues.concat(numericMeans).length ? Math.max(...allValues, ...numericMeans) : 1);
    const posValues = allValues.filter(v => v > 0).concat(numericMeans.filter(v => v > 0));

    const yTitle = unit ? `${label}<br>(${unit})` : label;
    const yaxis = {
      title: { text: yTitle, standoff: 8 },
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

    // Responsive layout tweaks based on container width and crowding
    const width = container.clientWidth || 800;
    const crowded = xlabels.length > 10;
    const small = width < 560;
    const tickAngle = crowded ? -45 : 0;
    const bottomMargin = crowded ? 110 : 60;
    const yTitleHasBreak = yTitle.indexOf('<br>') !== -1;
    const height = (small ? 280 : 360) + (yTitleHasBreak ? 40 : 0);
    const leftMargin = (yTitle.length > 24) ? 90 : 70;

    const layout = {
      height,
      margin: { l: leftMargin, r: 20, t: 10, b: bottomMargin },
      xaxis: {
        title: groupUnit ? `${groupName} (${groupUnit})` : groupName,
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

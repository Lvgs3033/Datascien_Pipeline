/* ==========================================================================
   AUTODS front-end controller
   Talks to the Flask JSON API, drives the pipeline stepper, and renders
   results with Chart.js + hand-rolled heatmaps. No external services.
   ========================================================================== */
(() => {
  "use strict";

  const STAGES = ["upload", "clean", "eda", "features", "train", "explain", "predict", "dashboard"];
  const state = {
    datasetLoaded: false,
    columns: [],
    suggestedTarget: null,
    target: null,
    taskType: null,
    charts: {},
    lastEda: null,
    fullDataShown: false,
    cleanFullDataShown: false,
    excludedFeatures: new Set(),
    lastRanking: [],
    excludedModels: new Set(),
  };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // ---------------------------------------------------------------- helpers
  async function api(path, { method = "GET", json, form } = {}) {
    const opts = { method };
    if (json) { opts.headers = { "Content-Type": "application/json" }; opts.body = JSON.stringify(json); }
    if (form) { opts.body = form; }
    let res;
    try {
      res = await fetch(path, opts);
    } catch (networkErr) {
      throw new Error("Could not reach the server. Make sure app.py is still running, then try again.");
    }
    let data;
    try { data = await res.json(); } catch { data = { ok: false, error: "Invalid server response" }; }
    if (!res.ok || data.ok === false) throw new Error(data.error || `Request to ${path} failed`);
    return data;
  }

  function toast(msg, ok = false) {
    const el = $("#toast");
    el.textContent = msg;
    el.classList.toggle("ok", ok);
    el.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => el.classList.remove("show"), 3600);
  }

  function setTopStatus(text) {
    $("#topStatus").innerHTML = `<span class="pulse"></span> ${text}`;
  }

  function fmt(n, digits = 3) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    if (typeof n !== "number") return n;
    return Math.abs(n) >= 1000 ? n.toLocaleString() : n.toFixed(digits).replace(/\.?0+$/, "") || "0";
  }

  function statCard(label, value, cls = "") {
    return `<div class="stat-card"><div class="label">${label}</div><div class="value ${cls}">${value}</div></div>`;
  }

  function downloadFile(url) {
    const a = document.createElement("a");
    a.href = url;
    a.click();
  }

  // ---------------------------------------------------------- stage stepper
  function setStage(name) {
    STAGES.forEach(s => {
      $(`.stage[data-stage="${s}"]`)?.classList.toggle("active", s === name);
      $(`.panel[data-panel="${s}"]`)?.classList.toggle("active", s === name);
    });
  }
  function markStage(name, status) {
    const el = $(`.stage[data-stage="${name}"]`);
    el.classList.remove("running", "done");
    if (status) el.classList.add(status);
  }

  $$(".stage").forEach(el => el.addEventListener("click", () => setStage(el.dataset.stage)));

  // -------------------------------------------------------------- upload
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  dropzone.addEventListener("click", () => fileInput.click());
  ["dragover", "dragleave", "drop"].forEach(evt =>
    dropzone.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.toggle("drag", evt === "dragover");
    })
  );
  dropzone.addEventListener("drop", e => {
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  });
  fileInput.addEventListener("change", e => {
    if (e.target.files[0]) handleUpload(e.target.files[0]);
  });

  $$(".chip[data-sample]").forEach(btn =>
    btn.addEventListener("click", () => loadSample(btn.dataset.sample))
  );

  async function handleUpload(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      toast("Please choose a .csv file.");
      return;
    }
    setTopStatus("ingesting dataset…");
    const form = new FormData();
    form.append("file", file);
    try {
      const data = await api("/api/upload", { method: "POST", form });
      onDatasetLoaded(data, null);
      toast(`Loaded ${file.name}`, true);
    } catch (e) {
      toast(e.message);
      setTopStatus("idle — awaiting dataset");
    }
  }

  async function loadSample(name) {
    setTopStatus(`loading sample "${name}"…`);
    try {
      const data = await api("/api/sample", { method: "POST", json: { name } });
      onDatasetLoaded(data, data.suggested_target);
      toast(`Sample dataset "${name}" loaded`, true);
    } catch (e) {
      toast(e.message);
    }
  }

  function onDatasetLoaded(data, suggestedTarget) {
    state.datasetLoaded = true;
    state.columns = data.columns;
    state.suggestedTarget = suggestedTarget;
    state.fullDataShown = false;
    state.cleanFullDataShown = false;
    state.excludedFeatures = new Set();
    state.excludedModels = new Set();
    markStage("upload", "done");
    ["clean", "eda", "features", "train", "explain", "predict", "dashboard"].forEach(s => markStage(s, null));

    $("#uploadSummary")?.classList.remove("hidden");
    $("#uploadSummary").innerHTML = [
      statCard("Rows", data.shape.rows),
      statCard("Columns", data.shape.cols),
      statCard("Duplicate rows", data.duplicate_rows, data.duplicate_rows > 0 ? "warn" : ""),
      statCard("Columns w/ missing", Object.values(data.missing).filter(v => v > 0).length,
        Object.values(data.missing).some(v => v > 0) ? "warn" : ""),
    ].join("");

    renderPreviewTable(data.columns, data.preview);
    $("#tableToolbar")?.classList.remove("hidden");
    $("#tableCaption").textContent = `Showing first ${data.preview.length} of ${data.shape.rows} rows`;
    $("#fullTableWrap")?.classList.add("hidden");
    $("#toggleFullDataBtn").textContent = "View full dataset";
    $("#cleanTableToolbar")?.classList.add("hidden");
    $("#cleanFullTableWrap")?.classList.add("hidden");
    $("#downloadModelBtn")?.classList.add("hidden");

    populateTargetSelect(data.columns, suggestedTarget);
    $("#topMeta").textContent = `${data.shape.rows} rows × ${data.shape.cols} cols`;
    setTopStatus("dataset ready — proceed to Clean");
  }

  function renderPreviewTable(columns, rows) {
    const wrap = $("#previewTableWrap");
    wrap.classList.remove("hidden");
    wrap.innerHTML = buildTableHtml(columns, rows);
  }

  function buildTableHtml(columns, rows) {
    const thead = `<thead><tr>${columns.map(c => `<th>${c}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${rows.map(r =>
      `<tr>${columns.map(c => `<td>${r[c] ?? ""}</td>`).join("")}</tr>`
    ).join("")}</tbody>`;
    return `<table>${thead}${tbody}</table>`;
  }

  $("#toggleFullDataBtn").addEventListener("click", async () => {
    state.fullDataShown = !state.fullDataShown;
    const btn = $("#toggleFullDataBtn");
    const previewWrap = $("#previewTableWrap");
    const fullWrap = $("#fullTableWrap");
    if (!state.fullDataShown) {
      btn.textContent = "View full dataset";
      previewWrap.classList.remove("hidden");
      fullWrap.classList.add("hidden");
      return;
    }
    btn.textContent = "Loading…";
    try {
      const data = await api("/api/full_data?stage=raw");
      previewWrap.classList.add("hidden");
      fullWrap.classList.remove("hidden");
      fullWrap.innerHTML = buildTableHtml(data.columns, data.rows);
      btn.textContent = "Show preview only";
      $("#tableCaption").textContent = data.truncated
        ? `Showing first ${data.shown_rows} of ${data.total_rows} rows (scroll for more)`
        : `Showing all ${data.total_rows} rows (scroll for more)`;
    } catch (e) {
      toast(e.message);
      state.fullDataShown = false;
      btn.textContent = "View full dataset";
    }
  });

  function populateTargetSelect(columns, suggested) {
    const sel = $("#targetSelect");
    sel.innerHTML = columns.map(c => `<option value="${c}">${c}</option>`).join("");
    sel.value = suggested && columns.includes(suggested) ? suggested : columns[columns.length - 1];
  }

  // -------------------------------------------------------------- clean
  async function runClean() {
    if (!state.datasetLoaded) { toast("Load a dataset first."); return; }
    markStage("clean", "running");
    setTopStatus("cleaning data…");
    try {
      const data = await api("/api/clean", { method: "POST" });
      renderCleanReport(data.report);
      markStage("clean", "done");
      setTopStatus("cleaning complete — proceed to Explore");
      return data;
    } catch (e) {
      toast(e.message);
      markStage("clean", null);
      throw e;
    }
  }

  function renderCleanReport(report) {
    $("#cleanReport")?.classList.remove("hidden");
    $("#cleanReport").innerHTML = report.actions.map(a => `<div class="line">${a}</div>`).join("");
    $("#cleanShapes")?.classList.remove("hidden");
    $("#cleanShapes").innerHTML = [
      statCard("Rows before", report.before_shape[0]),
      statCard("Rows after", report.after_shape[0]),
      statCard("Cols before", report.before_shape[1]),
      statCard("Cols after", report.after_shape[1]),
    ].join("");
    $("#cleanTableToolbar")?.classList.remove("hidden");
    state.cleanFullDataShown = false;
    $("#toggleCleanFullDataBtn").textContent = "View full dataset";
    $("#cleanFullTableWrap")?.classList.add("hidden");
  }

  $("#toggleCleanFullDataBtn").addEventListener("click", async () => {
    state.cleanFullDataShown = !state.cleanFullDataShown;
    const btn = $("#toggleCleanFullDataBtn");
    const wrap = $("#cleanFullTableWrap");
    if (!state.cleanFullDataShown) {
      btn.textContent = "View full dataset";
      wrap.classList.add("hidden");
      return;
    }
    btn.textContent = "Loading…";
    try {
      const data = await api("/api/full_data?stage=clean");
      wrap.classList.remove("hidden");
      wrap.innerHTML = buildTableHtml(data.columns, data.rows);
      btn.textContent = "Hide full dataset";
    } catch (e) {
      toast(e.message);
      state.cleanFullDataShown = false;
      btn.textContent = "View full dataset";
    }
  });

  $("#downloadCleanedBtn").addEventListener("click", () => downloadFile("/api/download/cleaned_csv"));
  $("#downloadModelBtn").addEventListener("click", () => downloadFile("/api/download/model"));

  // -------------------------------------------------------------- eda
  async function runEda() {
    markStage("eda", "running");
    setTopStatus("running exploratory analysis…");
    try {
      const data = await api("/api/eda", { method: "POST" });
      state.lastEda = data.eda;
      renderEda(data.eda);
      markStage("eda", "done");
      setTopStatus("EDA complete — proceed to Select");
      return data;
    } catch (e) {
      toast(e.message);
      markStage("eda", null);
      throw e;
    }
  }

  function renderEda(eda) {
    $("#edaContent")?.classList.remove("hidden");
    renderCorrHeatmap(eda.correlation);

    const histSelect = $("#histSelect");
    const histCols = Object.keys(eda.histograms);
    histSelect.innerHTML = histCols.map(c => `<option value="${c}">${c}</option>`).join("");
    histSelect.onchange = () => renderHistogram(eda.histograms[histSelect.value], histSelect.value);
    if (histCols.length) renderHistogram(eda.histograms[histCols[0]], histCols[0]);
    $("#histPicker")?.classList.toggle("hidden", histCols.length === 0);

    const catWrap = $("#catCharts");
    catWrap.innerHTML = "";
    Object.entries(eda.categorical_counts).forEach(([col, cc], i) => {
      const card = document.createElement("div");
      card.className = "scope-card";
      card.innerHTML = `<div class="scope-card-head"><span>${col}</span></div><canvas id="catChart${i}"></canvas>`;
      catWrap.appendChild(card);
      makeChart(`catChart${i}`, {
        type: "bar",
        data: { labels: cc.labels, datasets: [{ data: cc.counts, backgroundColor: "#2FE0C4" }] },
        options: baseChartOptions(false),
      });
    });

    renderFeatureRanges(eda.stats);
  }

  function renderFeatureRanges(stats) {
    const el = $("#rangeViz");
    const cols = Object.keys(stats);
    if (!cols.length) { $("#rangeCard")?.classList.add("hidden"); return; }
    $("#rangeCard")?.classList.remove("hidden");
    el.innerHTML = cols.map(c => {
      const s = stats[c];
      const span = (s.max - s.min) || 1;
      const meanPct = Math.min(100, Math.max(0, (s.mean - s.min) / span * 100));
      return `
      <div class="range-row">
        <span class="range-name">${c}</span>
        <div>
          <div class="range-track">
            <div class="range-fill" style="left:0%; width:100%;"></div>
            <div class="range-mean-marker" style="left:calc(${meanPct}% - 1.5px)" title="mean ${fmt(s.mean)}"></div>
          </div>
          <div class="range-vals"><span>min ${fmt(s.min)}</span><span>mean ${fmt(s.mean)}</span><span>max ${fmt(s.max)}</span></div>
        </div>
      </div>`;
    }).join("");
  }

  function renderCorrHeatmap(corr) {
    const el = $("#corrHeatmap");
    if (!corr || !corr.columns || corr.columns.length < 2) {
      el.innerHTML = `<div class="panel-sub" style="margin:0;">Not enough numeric columns for a correlation matrix.</div>`;
      return;
    }
    const colorFor = v => {
      const t = (v + 1) / 2;
      const r = Math.round(240 - t * (240 - 47));
      const g = Math.round(89 + t * (224 - 89));
      const b = Math.round(107 + t * (196 - 107));
      return `rgba(${r},${g},${b},${0.25 + Math.abs(v) * 0.55})`;
    };
    renderSimpleGridHeatmap(el, corr.columns, corr.matrix, colorFor, v => v.toFixed(2));
  }

  function renderSimpleGridHeatmap(el, cols, matrix, colorFor, fmtCell = v => v) {
    const n = cols.length;
    const wrap = document.createElement("div");
    wrap.style.display = "grid";
    wrap.style.gridTemplateColumns = `90px repeat(${n}, 46px)`;
    wrap.style.gap = "2px";
    wrap.style.overflowX = "auto";

    wrap.appendChild(document.createElement("div"));
    cols.forEach(c => {
      const d = document.createElement("div");
      d.className = "heatmap-colhead";
      d.textContent = c;
      wrap.appendChild(d);
    });
    cols.forEach((rowName, i) => {
      const label = document.createElement("div");
      label.className = "heatmap-label";
      label.textContent = rowName;
      wrap.appendChild(label);
      cols.forEach((_, j) => {
        const v = matrix[i][j];
        const cell = document.createElement("div");
        cell.className = "heatmap-cell";
        cell.style.background = colorFor(v);
        cell.textContent = fmtCell(v);
        wrap.appendChild(cell);
      });
    });
    el.innerHTML = "";
    el.appendChild(wrap);
  }

  function renderHistogram(hist, label) {
    if (!hist) return;
    const labels = hist.edges.slice(0, -1).map((e, i) => `${fmt(e, 2)}–${fmt(hist.edges[i + 1], 2)}`);
    makeChart("histChart", {
      type: "bar",
      data: { labels, datasets: [{ label, data: hist.counts, backgroundColor: "#F5A623" }] },
      options: baseChartOptions(false),
    });
  }

  // -------------------------------------------------------------- features
  async function runFeatures() {
    const target = $("#targetSelect").value;
    if (!target) { toast("Choose a target column."); return; }
    markStage("features", "running");
    setTopStatus(`analyzing target "${target}"…`);
    try {
      const tData = await api("/api/target", { method: "POST", json: { target } });
      state.target = target;
      state.taskType = tData.task_type;
      renderTargetInfo(tData);
      const fData = await api("/api/features", { method: "POST" });
      state.excludedFeatures = new Set();
      state.lastRanking = fData.ranking;
      renderFeatureRanking(fData.ranking, fData.selected_features);
      await loadModelChecklist(tData.task_type);
      markStage("features", "done");
      setTopStatus("feature ranking complete — proceed to Train");
      return fData;
    } catch (e) {
      toast(e.message);
      markStage("features", null);
      throw e;
    }
  }

  function renderTargetInfo(tData) {
    $("#targetInfo")?.classList.remove("hidden");
    const cards = [statCard("Task type", tData.task_type)];
    if (tData.task_type === "classification") {
      cards.push(statCard("Classes", tData.classes.length));
    } else {
      cards.push(statCard("Mean", fmt(tData.stats.mean)));
      cards.push(statCard("Std dev", fmt(tData.stats.std)));
    }
    $("#targetInfo").innerHTML = cards.join("");

    $("#targetDistWrap")?.classList.remove("hidden");
    if (tData.task_type === "classification") {
      makeChart("targetDistChart", {
        type: "bar",
        data: {
          labels: tData.classes,
          datasets: [{ label: "count", data: tData.class_counts, backgroundColor: "#2FE0C4" }],
        },
        options: baseChartOptions(false),
      });
    } else if (tData.histogram) {
      const labels = tData.histogram.edges.slice(0, -1).map((e, i) => `${fmt(e, 2)}–${fmt(tData.histogram.edges[i + 1], 2)}`);
      makeChart("targetDistChart", {
        type: "bar",
        data: { labels, datasets: [{ label: "count", data: tData.histogram.counts, backgroundColor: "#F5A623" }] },
        options: baseChartOptions(false),
      });
    }
  }

  function escapeAttr(s) { return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;"); }
  function escapeHtml(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  function renderFeatureRanking(ranking, selected) {
    const wrap = $("#featureRanking");
    wrap.classList.remove("hidden");
    $("#rankHint")?.classList.remove("hidden");
    const maxScore = Math.max(...ranking.map(r => r.score), 0.0001);
    wrap.innerHTML = ranking.map((r, i) => {
      const isSelected = selected.includes(r.feature);
      if (!isSelected) state.excludedFeatures.add(r.feature);
      const tier = i < 3 ? i + 1 : 0;
      return `
      <div class="rank-row ${isSelected ? "" : "excluded"}" data-feature="${escapeAttr(r.feature)}" data-tier="${tier}">
        <input type="checkbox" class="rank-check" data-feature="${escapeAttr(r.feature)}" ${isSelected ? "checked" : ""}>
        <span class="rank-idx">${i + 1}</span>
        <span class="rank-name">${escapeHtml(r.feature)}</span>
        <span class="rank-bar-track"><span class="rank-bar-fill" style="width:${Math.max(4, r.score / maxScore * 100)}%"></span></span>
        <span class="rank-score">${r.score.toFixed(3)}</span>
      </div>`;
    }).join("");

    $$(".rank-check", wrap).forEach(cb => cb.addEventListener("change", () => {
      const f = cb.dataset.feature;
      const row = wrap.querySelector(`.rank-row[data-feature="${CSS.escape(f)}"]`);
      if (cb.checked) { state.excludedFeatures.delete(f); row?.classList.remove("excluded"); }
      else { state.excludedFeatures.add(f); row?.classList.add("excluded"); }
    }));
  }

  function currentFeatureSelection() {
    return state.lastRanking.map(r => r.feature).filter(f => !state.excludedFeatures.has(f));
  }

  // -------------------------------------------------------------- model picker
  async function loadModelChecklist(taskType) {
    try {
      const data = await api(`/api/available_models?task_type=${encodeURIComponent(taskType)}`);
      state.excludedModels = new Set();
      const wrap = $("#modelChecklist");
      wrap.innerHTML = data.models.map(name => `
        <label class="model-check-item">
          <input type="checkbox" class="model-check" data-model="${name}" checked>
          <span>${name}</span>
        </label>
      `).join("");
    } catch (e) {
      toast(e.message);
    }
  }
  function currentModelSelection() {
    return $$(".model-check").filter(cb => cb.checked).map(cb => cb.dataset.model);
  }
  $("#selectAllModelsBtn").addEventListener("click", () => $$(".model-check").forEach(cb => cb.checked = true));
  $("#selectNoneModelsBtn").addEventListener("click", () => $$(".model-check").forEach(cb => cb.checked = false));

  // -------------------------------------------------------------- train
  async function runTrain() {
    const features = currentFeatureSelection();
    if (!features.length) { toast("Select at least one feature."); return; }
    const models = currentModelSelection();
    if (!models.length) { toast("Select at least one model to train."); return; }
    markStage("train", "running");
    setTopStatus("training and racing models…");
    try {
      const data = await api("/api/train", { method: "POST", json: { features, models } });
      renderLeaderboard(data.leaderboard);
      renderMetrics(data.metrics);
      $("#downloadModelBtn")?.classList.remove("hidden");
      markStage("train", "done");
      setTopStatus("training complete — proceed to Explain");
      return data;
    } catch (e) {
      toast(e.message);
      markStage("train", null);
      throw e;
    }
  }

  function renderLeaderboard(rows) {
    const wrap = $("#leaderboard");
    wrap.classList.remove("hidden");
    const maxCv = Math.max(...rows.filter(r => !r.error).map(r => r.cv_mean), 0.0001);
    wrap.innerHTML = rows.map((r, i) => {
      if (r.error) {
        return `<div class="lb-row"><span class="lb-rank">${i + 1}</span><span class="lb-name">${escapeHtml(r.model)}</span>
          <span class="lb-bar-track"></span><span class="lb-metric" style="color:var(--rose)">failed</span><span></span></div>`;
      }
      const pct = Math.max(4, r.cv_mean / maxCv * 100);
      return `<div class="lb-row ${i === 0 ? "best" : ""}">
        <span class="lb-rank">${i + 1}</span>
        <span class="lb-name">${escapeHtml(r.model)}${i === 0 ? '<span class="tag">selected</span>' : ""}</span>
        <span class="lb-bar-track"><span class="lb-bar-fill" style="width:${pct}%"></span></span>
        <span class="lb-metric">${r.metric} <b>${fmt(r.cv_mean)}</b></span>
        <span class="lb-metric">±${fmt(r.cv_std)}</span>
      </div>`;
    }).join("");

    const valid = rows.filter(r => !r.error);
    if (valid.length) {
      $("#lbChartWrap")?.classList.remove("hidden");
      makeChart("lbChart", {
        type: "bar",
        data: {
          labels: valid.map(r => r.model),
          datasets: [{ label: valid[0].metric, data: valid.map(r => r.cv_mean), backgroundColor: valid.map((_, i) => i === 0 ? "#2FE0C4" : "#3A4A66") }],
        },
        options: { ...baseChartOptions(false), indexAxis: "y" },
      });
    }
  }

  function renderMetrics(m) {
    const wrap = $("#trainMetrics");
    wrap.classList.remove("hidden");
    if (m.task_type === "classification") {
      wrap.innerHTML = [
        statCard("Accuracy", (m.accuracy * 100).toFixed(1) + "%"),
        statCard("Precision", fmt(m.precision)),
        statCard("Recall", fmt(m.recall)),
        statCard("F1 score", fmt(m.f1)),
      ].join("");
      $("#cmWrap")?.classList.remove("hidden");
      $("#residWrap")?.classList.add("hidden");
      renderConfusionMatrix(m.confusion_matrix);

      if (m.roc_curve) {
        $("#rocWrap")?.classList.remove("hidden");
        const aucEl = $("#aucBadge");
        if (aucEl) aucEl.textContent = `AUC ${fmt(m.roc_curve.auc)}`;
        makeChart("rocChart", {
          type: "line",
          data: {
            labels: m.roc_curve.fpr.map(v => v.toFixed(2)),
            datasets: [
              { label: `ROC (positive: ${m.roc_curve.positive_class})`, data: m.roc_curve.tpr, borderColor: "#2FE0C4", backgroundColor: "rgba(47,224,196,0.15)", fill: true, pointRadius: 0, tension: 0.2 },
              { label: "chance", data: m.roc_curve.fpr, borderColor: "#4E5C77", borderDash: [5, 5], pointRadius: 0 },
            ],
          },
          options: {
            ...baseChartOptions(true),
            scales: {
              x: { title: { display: true, text: "false positive rate", color: "#7C8AA5" }, ticks: { color: "#7C8AA5" }, grid: { color: "#1A2438" } },
              y: { title: { display: true, text: "true positive rate", color: "#7C8AA5" }, ticks: { color: "#7C8AA5" }, grid: { color: "#1A2438" } },
            },
          },
        });
      } else {
        $("#rocWrap")?.classList.add("hidden");
      }

      if (m.class_balance) {
        $("#classBalanceWrap")?.classList.remove("hidden");
        makeChart("classBalanceChart", {
          type: "bar",
          data: {
            labels: m.class_balance.labels,
            datasets: [
              { label: "actual", data: m.class_balance.actual, backgroundColor: "#2FE0C4" },
              { label: "predicted", data: m.class_balance.predicted, backgroundColor: "#F5A623" },
            ],
          },
          options: baseChartOptions(true),
        });
      } else {
        $("#classBalanceWrap")?.classList.add("hidden");
      }
    } else {
      wrap.innerHTML = [
        statCard("R²", fmt(m.r2)),
        statCard("MAE", fmt(m.mae)),
        statCard("RMSE", fmt(m.rmse)),
      ].join("");
      $("#residWrap")?.classList.remove("hidden");
      $("#cmWrap")?.classList.add("hidden");
      $("#rocWrap")?.classList.add("hidden");
      $("#classBalanceWrap")?.classList.add("hidden");
      renderResiduals(m.residual_sample);
    }
  }

  function renderConfusionMatrix(cm) {
    const el = $("#cmHeatmap");
    const max = Math.max(...cm.matrix.flat());
    const colorFor = v => `rgba(47,224,196,${0.12 + (v / (max || 1)) * 0.7})`;
    renderSimpleGridHeatmap(el, cm.labels, cm.matrix, colorFor);
  }

  function renderResiduals(sample) {
    makeChart("residChart", {
      type: "scatter",
      data: {
        datasets: [{
          label: "predicted vs actual",
          data: sample.actual.map((a, i) => ({ x: a, y: sample.predicted[i] })),
          backgroundColor: "#F5A623",
        }],
      },
      options: {
        ...baseChartOptions(true),
        scales: {
          x: { title: { display: true, text: "actual", color: "#7C8AA5" }, ticks: { color: "#7C8AA5" }, grid: { color: "#1A2438" } },
          y: { title: { display: true, text: "predicted", color: "#7C8AA5" }, ticks: { color: "#7C8AA5" }, grid: { color: "#1A2438" } },
        },
      },
    });
  }

  // -------------------------------------------------------------- explain
  async function runExplain() {
    markStage("explain", "running");
    setTopStatus("generating explanation…");
    try {
      const data = await api("/api/explain", { method: "POST" });
      renderExplain(data);
      markStage("explain", "done");
      setTopStatus("explanation ready — proceed to Predict");
      return data;
    } catch (e) {
      toast(e.message);
      markStage("explain", null);
      throw e;
    }
  }

  function renderExplain(data) {
    $("#explainNarrative")?.classList.remove("hidden");
    $("#explainNarrative").innerHTML = data.narrative.map(l => `<p>${mdBold(l)}</p>`).join("");
    if (data.feature_importance && data.feature_importance.length) {
      $("#importanceWrap")?.classList.remove("hidden");
      makeChart("importanceChart", {
        type: "bar",
        data: {
          labels: data.feature_importance.map(f => f.feature),
          datasets: [{ data: data.feature_importance.map(f => f.importance), backgroundColor: "#2FE0C4" }],
        },
        options: { ...baseChartOptions(false), indexAxis: "y" },
      });
    }
  }
  function mdBold(s) { return s.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>"); }

  // -------------------------------------------------------------- predict
  async function loadPredictForm() {
    setTopStatus("loading prediction form…");
    try {
      const data = await api("/api/feature_meta");
      renderPredictForm(data.features);
      setTopStatus("ready to predict");
    } catch (e) {
      toast(e.message);
    }
  }

  function renderPredictForm(fields) {
    const form = $("#predictForm");
    form.classList.remove("hidden");
    form.innerHTML = fields.map(f => {
      if (f.type === "numeric") {
        return `<div class="predict-field">
          <label>${f.feature} <span style="color:var(--text-dim)">(${fmt(f.min)}–${fmt(f.max)})</span></label>
          <input type="number" step="any" name="${f.feature}" value="${fmt(f.mean, 4)}">
        </div>`;
      }
      return `<div class="predict-field">
        <label>${f.feature}</label>
        <select name="${f.feature}">
          ${f.options.map(o => `<option value="${o}" ${o === f.default ? "selected" : ""}>${o}</option>`).join("")}
        </select>
      </div>`;
    }).join("") + `<button type="submit" class="btn btn-run predict-submit">Predict ▸</button>`;

    form.onsubmit = async e => {
      e.preventDefault();
      const payload = {};
      new FormData(form).forEach((v, k) => payload[k] = v);
      try {
        const result = await api("/api/predict", { method: "POST", json: payload });
        renderPredictResult(result);
        markStage("predict", "done");
      } catch (err) {
        toast(err.message);
      }
    };
  }

  function renderPredictResult(result) {
    const el = $("#predictResult");
    el.classList.remove("hidden");
    let html = `<div class="headline">${escapeHtml(result.model)} predicts</div><div class="value">${escapeHtml(result.prediction)}</div>`;
    if (result.probabilities) {
      html += result.probabilities.map((p, i) => `
        <div class="predict-prob-row ${i === 0 ? "top" : ""}">
          <span>${escapeHtml(p.class)}</span>
          <span class="predict-prob-track"><span class="predict-prob-fill" style="width:${p.probability * 100}%"></span></span>
          <span>${(p.probability * 100).toFixed(1)}%</span>
        </div>`).join("");
    }
    el.innerHTML = html;
  }

  // -------------------------------------------------------------- dashboard
  async function runDashboard() {
    markStage("dashboard", "running");
    try {
      const data = await api("/api/dashboard");
      renderDashboard(data);
      markStage("dashboard", "done");
      setTopStatus("pipeline complete");
      return data;
    } catch (e) {
      toast(e.message);
      markStage("dashboard", null);
      throw e;
    }
  }

  function renderDashboard(d) {
    const el = $("#dashboardContent");
    const parts = [];

    parts.push(`<div class="dash-section"><h3>Dataset</h3><div class="card-grid">
      ${statCard("Rows", d.dataset_shape[0])}
      ${statCard("Columns", d.dataset_shape[1])}
      ${statCard("Target", d.target || "—")}
      ${statCard("Task", d.task_type || "—")}
    </div></div>`);

    if (d.leaderboard) {
      parts.push(`<div class="dash-section"><h3>Model leaderboard</h3><div class="leaderboard">
        ${d.leaderboard.filter(r => !r.error).map((r, i) => `
          <div class="lb-row ${i === 0 ? "best" : ""}">
            <span class="lb-rank">${i + 1}</span>
            <span class="lb-name">${r.model}${i === 0 ? '<span class="tag">selected</span>' : ""}</span>
            <span class="lb-metric">${r.metric} <b>${fmt(r.cv_mean)}</b></span>
            <span class="lb-metric">±${fmt(r.cv_std)}</span>
          </div>`).join("")}
      </div></div>`);
    }

    if (d.metrics) {
      const m = d.metrics;
      const cards = m.task_type === "classification"
        ? [statCard("Accuracy", (m.accuracy * 100).toFixed(1) + "%"), statCard("F1", fmt(m.f1)), statCard("Precision", fmt(m.precision)), statCard("Recall", fmt(m.recall))]
        : [statCard("R²", fmt(m.r2)), statCard("MAE", fmt(m.mae)), statCard("RMSE", fmt(m.rmse))];
      parts.push(`<div class="dash-section"><h3>Best model performance</h3><div class="card-grid">${cards.join("")}</div></div>`);
    }

    if (d.explain) {
      parts.push(`<div class="dash-section"><h3>Explanation</h3><div class="narrative">
        ${d.explain.narrative.map(l => `<p>${mdBold(l)}</p>`).join("")}
      </div></div>`);
    }

    if (d.feature_ranking) {
      const top = d.feature_ranking.slice(0, 8);
      const maxScore = Math.max(...top.map(r => r.score), 0.0001);
      parts.push(`<div class="dash-section"><h3>Top features</h3><div class="rank-list">
        ${top.map((r, i) => `
          <div class="rank-row" style="grid-template-columns:22px 140px 1fr 60px;">
            <span class="rank-idx">${i + 1}</span>
            <span class="rank-name">${r.feature}</span>
            <span class="rank-bar-track"><span class="rank-bar-fill" style="width:${Math.max(4, r.score / maxScore * 100)}%"></span></span>
            <span class="rank-score">${r.score.toFixed(3)}</span>
          </div>`).join("")}
      </div></div>`);
    }

    if (d.cleaning_report) {
      parts.push(`<div class="dash-section"><h3>Cleaning log</h3><div class="log-box">
        ${d.cleaning_report.actions.map(a => `<div class="line">${a}</div>`).join("")}
      </div></div>`);
    }

    parts.push(`<div class="dash-section"><h3>Export</h3><div class="toolbar-actions">
      <button class="btn btn-ghost btn-small" onclick="window.location.href='/api/download/cleaned_csv'">⇩ Cleaned dataset (CSV)</button>
      <button class="btn btn-ghost btn-small" onclick="window.location.href='/api/download/model'">⇩ Trained model (.joblib)</button>
    </div></div>`);

    el.innerHTML = parts.join("");
  }

  // -------------------------------------------------------------- chart util
  function baseChartOptions(showLegend) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: showLegend, labels: { color: "#7C8AA5" } } },
      scales: {
        x: { ticks: { color: "#7C8AA5", maxRotation: 40 }, grid: { color: "#1A2438" } },
        y: { ticks: { color: "#7C8AA5" }, grid: { color: "#1A2438" } },
      },
    };
  }
  function makeChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    if (state.charts[canvasId]) state.charts[canvasId].destroy();
    canvas.parentElement.style.height = canvas.parentElement.style.height || "260px";
    canvas.style.height = "230px";
    state.charts[canvasId] = new Chart(canvas.getContext("2d"), config);
  }

  // -------------------------------------------------------------- wiring
  const ACTIONS = {
    clean: runClean, eda: runEda, features: runFeatures, train: runTrain,
    explain: runExplain, predict: loadPredictForm, dashboard: runDashboard,
  };
  $$('.btn-stage').forEach(btn => btn.addEventListener("click", () => ACTIONS[btn.dataset.action]()));

  $("#runAllBtn").addEventListener("click", async () => {
    if (!state.datasetLoaded) { toast("Load a dataset or sample first."); setStage("upload"); return; }
    const btn = $("#runAllBtn");
    btn.disabled = true;
    try {
      setStage("clean"); await runClean();
      setStage("eda"); await runEda();
      setStage("features"); await runFeatures();
      setStage("train"); await runTrain();
      setStage("explain"); await runExplain();
      setStage("predict"); await loadPredictForm();
      setStage("dashboard"); await runDashboard();
      toast("Full pipeline complete", true);
    } catch (e) {
      // individual stage already toasted the specific error
    } finally {
      btn.disabled = false;
    }
  });

  $("#resetBtn").addEventListener("click", async () => {
    try { await api("/api/reset", { method: "POST" }); } catch {}
    location.reload();
  });

  setStage("upload");

  window.addEventListener("error", (e) => {
    console.error("AUTODS error:", e.error || e.message);
    toast("Something went wrong rendering that step -- check the console for details, or reload.");
  });
  window.addEventListener("unhandledrejection", (e) => {
    console.error("AUTODS unhandled rejection:", e.reason);
    toast((e.reason && e.reason.message) || "Something went wrong -- check the console for details.");
  });
})();
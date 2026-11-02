/* Gopniya demo page. Redaction runs in Pyodide in this tab; only redacted text is POSTed. */
(() => {
  const $ = (id) => document.getElementById(id);
  const samples = window.GOPNIYA_SAMPLES;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const PY_FILES = ["__init__.py", "validators.py", "detect.py", "vault.py", "guard.py", "agent.py"];
  const STEP_LABEL = { local_ner: "local NER", detect: "detect", guard: "guard", block: "blocked", llm: "cloud call", verify: "verify", restore: "restore" };
  let py = null, bridge = null, current = samples[0], previewTimer = null, busy = false;

  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function renderTabs() {
    $("tabs").innerHTML = "";
    for (const s of samples) {
      const b = document.createElement("button");
      b.className = "tab"; b.type = "button"; b.textContent = s.label;
      b.setAttribute("aria-pressed", String(s === current));
      b.onclick = () => { current = s; $("note").value = s.text; $("task").value = s.task; renderTabs(); clearCloud(); preview(); };
      $("tabs").appendChild(b);
    }
  }

  function renderMarked(text, spans) {
    const el = $("marked");
    let html = "", cur = 0;
    spans.sort((a, b) => a.start - b.start).forEach((s, i) => {
      if (s.start < cur) return;
      html += esc(text.slice(cur, s.start));
      html += `<span class="pii" data-i="${i}" data-ph="${esc(s.placeholder || "[" + s.type + "]")}" title="${esc(s.type)} · stays in this browser">${esc(text.slice(s.start, s.end))}</span>`;
      cur = s.end;
    });
    html += esc(text.slice(cur));
    el.innerHTML = html;
    const marks = [...el.querySelectorAll(".pii")];
    marks.forEach((m, i) => {
      const finish = () => { m.textContent = m.dataset.ph; m.classList.add("done"); };
      if (reduce) return finish();
      setTimeout(() => m.classList.add("on"), 60 + i * 40);
      setTimeout(finish, 60 + i * 40 + 300);
    });
  }

  function placeholderHTML(s) {
    return esc(s).replace(/\[([A-Z_]+_\d+)\]/g, '<span class="ph">[$1]</span>');
  }
  // tiny markdown for model answers: **bold**, "* " bullets, "#" headings
  function md(html) {
    return html
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/^\s*[*-]\s+/gm, "• ")
      .replace(/^#{1,4}\s*(.+)$/gm, "<strong>$1</strong>");
  }

  function restoredHTML(answer, answerRed) {
    // underline values that came back from the vault: rebuild from the placeholder version
    const map = lastMap;
    if (!map) return esc(answer);
    return esc(answerRed).replace(/\[{0,2}\s*([A-Z_]+)[ _](\d+)\s*\]{0,2}/g, (m, t, n) => {
      const v = map[`[${t}_${n}]`];
      return v ? `<span class="rv">${esc(v)}</span>` : m;
    });
  }

  let lastMap = null;
  function buildMap(text, spans) {
    const m = {};
    for (const s of spans) if (s.placeholder && !(s.placeholder in m)) m[s.placeholder] = text.slice(s.start, s.end);
    return m;
  }

  function renderSteps(steps) {
    const ol = $("steps");
    ol.innerHTML = "";
    steps.forEach((s, i) => {
      const li = document.createElement("li");
      li.style.setProperty("--i", i);
      if (s.name === "llm") li.className = "cross";
      if (s.name === "block" || /error|not in the note/.test(s.detail)) li.className = "bad";
      li.innerHTML = `<b>${esc(STEP_LABEL[s.name] || s.name)}</b>${esc(s.detail)} <span class="ms">${s.ms ? s.ms.toFixed(1) + " ms" : ""}</span>`;
      ol.appendChild(li);
    });
  }

  function renderCounts(counts) {
    const parts = Object.entries(counts).map(([k, v]) => `${k} ${v}`);
    $("counts").textContent = parts.length ? "distinct values replaced: " + parts.join(" · ") : "no identifiers found";
  }

  function clearCloud() {
    $("answerRed").innerHTML = '<span class="empty">Nothing sent yet.</span>';
    $("restored").innerHTML = '<span class="empty">Run the agent to see the answer with the real values back in place.</span>';
    $("sentStamp").hidden = true;
  }

  async function preview() {
    if (!bridge) return;
    const text = $("note").value;
    const res = JSON.parse(await bridge.preview(text));
    lastMap = buildMap(text, res.spans);
    renderMarked(text, res.spans);
    $("sent").innerHTML = placeholderHTML(res.redacted);
    renderSteps(res.steps);
    renderCounts(res.entity_counts);
  }

  async function run() {
    if (!bridge || busy) return;
    busy = true; $("run").disabled = true; $("run").textContent = "Working…";
    const text = $("note").value;
    try {
      const res = JSON.parse(await bridge.run(text, $("task").value));
      lastMap = buildMap(text, res.spans);
      $("sent").innerHTML = placeholderHTML(res.redacted);
      renderSteps(res.steps);
      renderCounts(res.entity_counts);
      if (res.status === "ok") {
        $("sentStamp").hidden = false;
        $("answerRed").innerHTML = md(placeholderHTML(res.answer_redacted));
        $("restored").innerHTML = md(restoredHTML(res.answer, res.answer_redacted));
      } else if (res.status === "blocked") {
        $("answerRed").innerHTML = '<span class="empty">Blocked on the clinic side: nothing was sent.</span>';
      } else {
        const err = res.steps.filter((s) => s.name === "llm").pop();
        $("answerRed").innerHTML = `<span class="empty">${esc(err ? err.detail : "model call failed")}</span>`;
      }
    } catch (e) {
      $("answerRed").innerHTML = `<span class="empty">${esc(String(e))}</span>`;
    } finally {
      busy = false; $("run").disabled = false; $("run").textContent = "Run agent";
    }
  }

  function toggleEdit() {
    const editing = !$("note").hidden;
    if (editing) {
      $("note").hidden = true; $("marked").hidden = false; $("edit").textContent = "Edit note"; preview();
    } else {
      $("marked").hidden = true; $("note").hidden = false; $("edit").textContent = "Show redaction"; $("note").focus();
    }
  }

  async function boot() {
    $("note").value = current.text;
    $("task").value = current.task;
    renderTabs();
    $("marked").textContent = current.text;
    try {
      const t0 = performance.now();
      py = await loadPyodide();
      py.FS.mkdirTree("/home/pyodide/gopniya");
      await Promise.all(PY_FILES.map(async (f) => {
        const src = await (await fetch("py/gopniya/" + f)).text();
        py.FS.writeFile("/home/pyodide/gopniya/" + f, src);
      }));
      py.FS.writeFile("/home/pyodide/bridge.py", await (await fetch("bridge.py")).text());
      py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
      bridge = py.pyimport("bridge");
      $("runtime").innerHTML = `<b>Python ${py.version}</b> running in this tab · loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s · the note never leaves it unredacted`;
      $("run").disabled = false;
      await preview();
    } catch (e) {
      $("runtime").textContent = "Could not start the local runtime: " + e;
    }
    try {
      const m = await (await fetch("metrics.json")).json();
      renderMetrics(m);
    } catch { $("metrics").innerHTML = '<tr><td class="empty">Measurements unavailable.</td></tr>'; }
  }

  function pct(x) { return x == null ? "—" : (x * 100).toFixed(1) + "%"; }
  function renderMetrics(m) {
    const rows = m.rows.map((r) => `<tr><td>${esc(r.label)}</td><td class="n">${r.notes}</td><td class="n">${r.gold}</td><td class="n">${pct(r.recall)}</td><td class="n">${pct(r.precision)}</td><td class="n">${pct(r.leak_free)}</td></tr>`).join("");
    $("metrics").innerHTML = `<tr><th>Set · detector</th><th>Notes</th><th>Identifiers</th><th>Protection recall</th><th>Precision</th><th>Leak-free notes</th></tr>${rows}`;
  }

  $("run").onclick = run;
  $("edit").onclick = toggleEdit;
  $("note").addEventListener("input", () => { clearTimeout(previewTimer); previewTimer = setTimeout(() => { if (!$("marked").hidden) preview(); }, 250); });
  $("task").addEventListener("change", clearCloud);
  boot();
})();

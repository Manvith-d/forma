// Forma dashboard: the conversation on the left, the agents' live work on the right.
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const AGENTS = {
  Coach: { x: 220, y: 150, sub: "orchestrates" }, Intake: { x: 92, y: 62, sub: "profile" },
  Medical: { x: 348, y: 62, sub: "safety" }, Trainer: { x: 378, y: 218, sub: "training plan" },
  Nutritionist: { x: 220, y: 272, sub: "calories & macros" }, Chef: { x: 62, y: 218, sub: "pantry meals" },
};
const EDGES = [["Coach", "Intake"], ["Coach", "Medical"], ["Coach", "Trainer"], ["Coach", "Nutritionist"], ["Coach", "Chef"],
  ["Intake", "Medical"], ["Medical", "Trainer"], ["Medical", "Nutritionist"], ["Medical", "Chef"], ["Trainer", "Nutritionist"],
  ["Nutritionist", "Chef"]];
const color = a => `var(--${a in AGENTS ? a : "Coach"})`;
const initials = a => a === "Nutritionist" ? "Nu" : a[0];

let session = null, ws = null, busy = false;
const streaming = new Map();

// ---- boot ----------------------------------------------------------------------------------
async function boot(fresh = false) {
  let id = fresh ? null : new URLSearchParams(location.search).get("session") || localStorage.getItem("forma-session");  // ?session= links
  session = id ? await fetch(`/api/sessions/${id}`).then(r => r.ok ? r.json() : null) : null;
  if (!session) session = await fetch("/api/sessions", { method: "POST" }).then(r => r.json());
  localStorage.setItem("forma-session", session.id);
  $("#messages").innerHTML = "";
  $("#timeline").innerHTML = "";
  session.messages.forEach(m => addMessage(m));  // not forEach(addMessage): the index would arrive as "stream"
  (session.activity || []).forEach(ev => onEvent(ev, true));
  showQuick(session.messages.at(-1));
  drawNetwork();
  renderBoard();
  connect();
  health();
  if (document.getElementById(`tab-${location.hash.slice(1)}`)) selectTab(location.hash.slice(1));  // #plan, #meals, ...
}

async function health() {
  const h = await fetch("/api/health").then(r => r.json()).catch(() => null);
  if (!h) return;
  const k = h.knowledge;
  $("#status").innerHTML = [
    `<span class="pill ${h.llm.available ? "on" : "off"}" title="Runs locally through Ollama">${h.llm.available ? `Local model: ${esc(h.llm.model)}` : "Local model offline: rules-only mode"}</span>`,
    `<span class="pill" title="Pipeline run ${esc(k.pipeline_run)}">${k.documents} documents · ${k.labels} FDA labels · ${k.foods.toLocaleString()} foods · ${k.exercises} exercises</span>`,
    k.checks_passed ? `<span class="pill on">Data checks ${k.checks_passed.p}/${k.checks_passed.n}</span>` : "",
  ].join("");
}

function connect() {
  if (ws) ws.close();
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/sessions/${session.id}/ws`);
  ws.onmessage = e => onEvent(JSON.parse(e.data));
  ws.onclose = () => setTimeout(() => { if (ws.readyState === WebSocket.CLOSED) connect(); }, 1500);
}

function send(text) {
  text = text.trim();
  if (!text || busy || ws.readyState !== WebSocket.OPEN) return;
  addMessage({ role: "user", text });
  $("#quick").innerHTML = "";
  setBusy(true, "Coach", "Routing your message");
  ws.send(JSON.stringify({ text }));
}

function setBusy(on, agent = "Coach", text = "") {
  busy = on;
  $("#send").disabled = on;
  $("#working").hidden = !on;
  $("#working").style.setProperty("--agent", color(agent));
  $("#working-text").textContent = on ? `${agent}: ${text}…` : "";
}

// ---- events ----------------------------------------------------------------------------------
function onEvent(ev, replay = false) {  // replay: rebuilding the log after a reload, no animation
  switch (ev.type) {
    case "agent":
      if (!replay) nodeState(ev.agent, ev.status);
      log(ev.agent, ev.status === "working" ? "working" : "done", ev.summary, false, ev.at);
      if (ev.status === "working" && !replay) setBusy(true, ev.agent, ev.summary);
      break;
    case "handoff":
      if (!replay) packet(ev.source, ev.target);
      log(ev.source, "hand-off", `→ <b style="--c:${color(ev.target)}">${esc(ev.target)}</b>: ${esc(ev.text)}`, true, ev.at);
      break;
    case "tool":
      log(ev.agent, "tool", `${esc(ev.tool)}: ${esc(ev.detail)}`, true, ev.at);
      break;
    case "route":
      log("Coach", "route", `sent to <b>${esc(ev.route)}</b>`, true, ev.at);
      break;
    case "message":
      addMessage(ev.message);
      showQuick(ev.message);
      break;
    case "message_start":
      streaming.set(ev.message.id, addMessage({ ...ev.message, text: "" }, true));
      break;
    case "token": {
      const el = streaming.get(ev.id);
      if (el) { el.dataset.raw = (el.dataset.raw || "") + ev.text; el.innerHTML = format(el.dataset.raw); scrollChat(); }
      break;
    }
    case "message_end": {
      const el = streaming.get(ev.message.id);
      if (el) { el.classList.remove("caret"); el.innerHTML = format(ev.message.text); streaming.delete(ev.message.id); }
      showQuick(ev.message);
      break;
    }
    case "state":
      session = ev.session;
      renderBoard();
      setBusy(false);
      document.querySelectorAll("#network .node.done").forEach(n => setTimeout(() => n.classList.remove("done"), 2500));
      break;
    case "error":
      addMessage({ role: "agent", agent: "Coach", text: ev.text });
      setBusy(false);
      break;
  }
}

// ---- chat ------------------------------------------------------------------------------------
function format(text) {
  const lines = esc(text).split("\n");
  let html = "", list = false;
  for (const line of lines) {
    if (/^\s*[-•]\s+/.test(line)) { if (!list) { html += "<ul>"; list = true; } html += `<li>${line.replace(/^\s*[-•]\s+/, "")}</li>`; continue; }
    if (list) { html += "</ul>"; list = false; }
    if (line.trim()) html += `<p>${line}</p>`;
  }
  if (list) html += "</ul>";
  return html.replace(/\[(\d{1,3})\]/g, (_, n) => `<button class="cite" type="button" data-ref="${n}" title="Show source ${n}">${n}</button>`);
}

function addMessage(m, stream = false) {
  const li = document.createElement("li");
  const agent = m.agent || "Coach";
  li.className = `msg ${m.role === "user" ? "user" : "agent"}`;
  li.innerHTML = m.role === "user"
    ? `<div class="bubble">${format(m.text)}</div>`
    : `<span class="avatar" style="background:${color(agent)}">${initials(agent)}</span>
       <div class="bubble"><div class="who" style="color:${color(agent)}">${esc(agent)}</div><div class="body ${stream ? "caret" : ""}">${format(m.text)}</div></div>`;
  $("#messages").append(li);
  scrollChat();
  return $(".body", li);
}

function scrollChat() { const m = $("#messages"); m.scrollTop = m.scrollHeight; }

function showQuick(m) {
  const q = $("#quick");
  q.innerHTML = "";
  for (const text of (m && m.quick) || []) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.onclick = () => send(text);
    q.append(b);
  }
}

// ---- live network ----------------------------------------------------------------------------
function drawNetwork() {
  const svg = $("#network");
  svg.innerHTML = EDGES.map(([a, b]) => `<line class="edge" data-e="${a}-${b}" x1="${AGENTS[a].x}" y1="${AGENTS[a].y}" x2="${AGENTS[b].x}" y2="${AGENTS[b].y}"/>`).join("")
    + Object.entries(AGENTS).map(([name, p]) => `
      <g class="node" data-agent="${name}" style="--c:${color(name)}" transform="translate(${p.x},${p.y})">
        <circle class="ring" r="31"/><circle class="core" r="${name === "Coach" ? 31 : 29}"/>
        <text y="4">${name === "Nutritionist" ? "Nutrition" : name}</text><text class="sub" y="${name === "Coach" ? 46 : 44}">${p.sub}</text>
      </g>`).join("");
}

function nodeState(agent, status) {
  const n = $(`#network .node[data-agent="${agent}"]`);
  if (!n) return;
  n.classList.toggle("working", status === "working");
  n.classList.toggle("done", status === "done");
}

function packet(from, to) {
  const a = AGENTS[from], b = AGENTS[to];
  if (!a || !b) return;
  const svg = $("#network");
  const edge = svg.querySelector(`[data-e="${from}-${to}"], [data-e="${to}-${from}"]`);
  if (edge) { edge.style.setProperty("--from", color(from)); edge.classList.add("hot"); setTimeout(() => edge.classList.remove("hot"), 1400); }
  const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  dot.setAttribute("r", 6);
  dot.setAttribute("class", "packet");
  dot.style.setProperty("--c", color(from));
  svg.append(dot);
  const start = performance.now(), dur = 900;
  (function step(t) {
    const k = Math.min(1, (t - start) / dur), e = k < .5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
    dot.setAttribute("cx", a.x + (b.x - a.x) * e);
    dot.setAttribute("cy", a.y + (b.y - a.y) * e);
    if (k < 1) requestAnimationFrame(step); else dot.remove();
  })(start);
}

function log(agent, kind, html, raw = false, at = null) {
  const li = document.createElement("li");
  li.style.setProperty("--c", color(agent));
  li.innerHTML = `<time>${new Date(at ? at * 1000 : Date.now()).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time>
    <div><span class="kind">${esc(kind)}</span><b>${esc(agent)}</b> ${raw ? html : esc(html)}</div>`;
  const tl = $("#timeline");
  tl.prepend(li);
  while (tl.children.length > 80) tl.lastChild.remove();
}

// ---- dashboard panels --------------------------------------------------------------------------
function renderBoard() {
  renderPlan(); renderNutrition(); renderMeals(); renderSafety(); renderSources();
  const badge = (tab, on) => { const b = $(`.tabs [data-tab="${tab}"]`); b.querySelector(".badge")?.remove(); if (on) b.insertAdjacentHTML("beforeend", `<span class="badge">${on}</span>`); };
  badge("meals", session.meals?.meals.length || 0);
  badge("sources", session.citations.length || 0);
}

const empty = text => `<p class="empty">${text}</p>`;
const refs = ns => (ns || []).map(n => `<button class="cite" type="button" data-ref="${n}">${n}</button>`).join("");

function renderPlan() {
  const p = session.plan, el = $("#tab-plan");
  if (!p) { el.innerHTML = empty("The Trainer builds your plan once your profile and screening are done."); return; }
  const pag = p.pag;
  el.innerHTML = `
    <div class="cards">
      <div class="card"><small>Plan length</small><div class="big">${p.weeks} weeks</div></div>
      <div class="card"><small>Strength days</small><div class="big">${p.strength_days}/week</div></div>
      <div class="card"><small>Aerobic minutes</small><div class="big">${p.aerobic_minutes}/week</div><small>PAG target 150-300 ${refs(p.refs)}</small></div>
      <div class="card"><small>Intensity</small><div class="big" style="font-size:16px">${esc(p.intensity_mode)}</div></div>
    </div>
    <p><span class="tag ${pag.aerobic_ok ? "ok" : "bad"}">aerobic ${pag.aerobic_ok ? "meets" : "below"} PAG</span>
       <span class="tag ${pag.strength_ok ? "ok" : "bad"}">muscle-strengthening ${pag.strength_ok ? "2+ days" : "below 2 days"}</span>
       ${pag.balance ? `<span class="tag ok">balance training added</span>` : ""}</p>
    <h3>Your week</h3>
    <div class="week">${p.template.map((d, i) => `
      <button type="button" class="day ${d.type}" data-day="${i}" aria-pressed="${i === 0}">
        <div class="d">${d.day}</div><div class="n">${esc(d.name)}</div>
        <div class="m">${d.type === "rest" ? (d.cardio ? `${d.cardio.minutes} min ${d.cardio.modality === "walking" ? "walk" : esc(d.cardio.modality)}` : "rest") : `${d.minutes} min · ${d.intensity}`}</div>
      </button>`).join("")}</div>
    <div id="day-detail"></div>
    <h3>Progression</h3>
    <div class="phases">${p.weekly.map(w => `<div class="phase ${w.phase === "Deload" ? "deload" : w.phase === "Foundation" ? "found" : ""}" title="${esc(w.focus)}">W${w.week}<br>${w.phase === "Deload" ? "deload" : w.phase === "Foundation" ? "learn" : "build"}</div>`).join("")}</div>
    <p class="how">Every 4th week is a lighter deload. Hover a week to see its focus.</p>`;
  const show = i => {
    const d = p.template[i];
    el.querySelectorAll(".day").forEach(b => b.setAttribute("aria-pressed", b.dataset.day == i));
    $("#day-detail").innerHTML = `<h3>${esc(d.day)} · ${esc(d.name)}</h3>
      ${d.warmup ? `<p class="how">Warm-up: ${esc(d.warmup)}</p>` : ""}
      ${d.exercises.length ? `<table><tr><th>Exercise</th><th>Sets × reps</th><th>Effort</th></tr>${d.exercises.map(e => `
        <tr><td><b>${esc(e.name)}</b><p class="how">${esc((e.how || []).slice(0, 2).join(" "))}</p></td><td>${e.sets} × ${esc(e.reps)}</td><td>RPE ${esc(e.rpe)}</td></tr>`).join("")}</table>` : ""}
      ${d.cardio ? `<p><b>${esc(d.cardio.modality)}</b>, ${d.cardio.minutes} min at RPE ${esc(d.cardio.rpe)}. <span class="how">${esc(d.cardio.how)}</span></p>` : ""}
      ${!d.exercises.length && !d.cardio ? empty("Rest day.") : ""}`;
  };
  el.querySelectorAll(".day").forEach(b => b.onclick = () => show(+b.dataset.day));
  show(Math.max(0, p.template.findIndex(d => d.day === session.today)));
}

function renderNutrition() {
  const t = session.targets, el = $("#tab-nutrition");
  if (!t) { el.innerHTML = empty("The Nutritionist sets your calories after the training plan exists."); return; }
  const d = t.days[session.today];
  const max = Math.max(...Object.values(t.days).map(x => x.kcal));
  const ring = pct => `<svg viewBox="0 0 120 120" width="130" height="130"><circle cx="60" cy="60" r="50" fill="none" stroke="var(--line)" stroke-width="12"/>
    <circle cx="60" cy="60" r="50" fill="none" stroke="var(--Nutritionist)" stroke-width="12" stroke-linecap="round" stroke-dasharray="${314 * pct} 314" transform="rotate(-90 60 60)"/>
    <text x="60" y="58" text-anchor="middle" font-size="22" font-weight="750" fill="var(--ink)">${d.kcal}</text><text x="60" y="76" text-anchor="middle" font-size="11" fill="var(--muted)">kcal today</text></svg>`;
  const eaten = session.meals?.totals;
  const bar = (label, val, goal, c) => `<div>${label} <b>${val}</b>${goal ? ` / ${goal}` : ""} g</div><div class="bar"><i style="width:${Math.min(100, goal ? 100 * val / goal : 100)}%;background:${c}"></i></div>`;
  el.innerHTML = `
    <div class="ringrow">${ring(eaten ? Math.min(1, eaten.kcal / d.kcal) : 1)}
      <div class="macro">
        ${bar("Protein", eaten ? Math.round(eaten.protein_g) : d.protein_g, eaten ? d.protein_g : null, "var(--Trainer)")}
        ${bar("Carbohydrate", eaten ? Math.round(eaten.carb_g) : d.carb_g, eaten ? d.carb_g : null, "var(--Coach)")}
        ${bar("Fat", eaten ? Math.round(eaten.fat_g) : d.fat_g, eaten ? d.fat_g : null, "var(--Chef)")}
        <small class="how">${eaten ? "Today's meals against the targets." : "Targets for today."} Saturated fat under ${d.sat_fat_max_g} g ${refs(t.refs?.slice(2))}</small>
      </div></div>
    <h3>How today's number was set</h3><p class="explain">${esc(d.explain)}</p>
    <div class="cards">
      <div class="card"><small>Rate of change</small><div class="big">${t.rate_kg_per_week > 0 ? "+" : ""}${t.rate_kg_per_week} kg/wk</div><small>0.5-1% of body weight max ${refs(t.refs?.slice(0, 1))}</small></div>
      <div class="card"><small>Protein</small><div class="big">${t.protein_g_per_kg} g/kg</div><small>of ${t.protein_basis_kg} kg ${refs(t.refs?.slice(1, 2))}</small></div>
      <div class="card"><small>Maintenance (avg)</small><div class="big">${t.maintenance_avg_kcal}</div><small>BMR ${t.bmr} kcal</small></div>
      <div class="card"><small>Floor</small><div class="big">${t.floor_kcal}</div><small>never below this</small></div>
    </div>
    ${t.notes.map(n => `<div class="notice warn">${esc(n)}</div>`).join("")}
    ${t.hydration ? `<div class="notice ok">${esc(t.hydration)}</div>` : ""}
    <h3>Calories by day</h3>
    <div class="daybars">${Object.entries(t.days).map(([k, v]) => `<div class="${k === session.today ? "today" : ""}"><span style="height:${Math.round(110 * v.kcal / max)}px" title="${v.kcal} kcal"></span>${k}<br>${v.kcal}</div>`).join("")}</div>
    <p class="how">Training days get more; the weekly average sets the pace.</p>`;
}

function renderMeals() {
  const m = session.meals, el = $("#tab-meals");
  if (!m) { el.innerHTML = empty("The Chef cooks from your pantry once your targets are set."); return; }
  const day = session.targets.days[session.today];
  el.innerHTML = `
    <div class="cards">
      <div class="card"><small>Today's meals</small><div class="big">${Math.round(m.totals.kcal)} kcal</div><small>${m.fit_pct}% of ${day.kcal}</small></div>
      <div class="card"><small>Protein</small><div class="big">${Math.round(m.totals.protein_g)} g</div><small>target ${day.protein_g} g</small></div>
      <div class="card"><small>Fiber</small><div class="big">${Math.round(m.totals.fiber_g)} g</div></div>
      <div class="card"><small>Sodium</small><div class="big">${Math.round(m.totals.sodium_mg)} mg</div></div>
    </div>
    ${(m.notes || []).map(n => `<div class="notice warn">${esc(n)}</div>`).join("")}
    ${m.excluded_for_safety?.length ? `<div class="notice danger">Left out for safety: ${m.excluded_for_safety.map(esc).join(", ")}</div>` : ""}
    ${m.meals.map(x => `<article class="meal"><header><div><span class="meta">${esc(x.meal)} · ${Math.round(x.total.kcal)} kcal · ${Math.round(x.total.protein_g)} g protein</span>
        <h4>${esc(x.title)}</h4></div><span class="meta">${x.source === "recipe" ? `adapted from <a href="${esc(x.source_url)}" target="_blank" rel="noopener">a recipe</a>` : "composed from your pantry"}${x.written_by === "llm" ? " · written by the local model" : ""}</span></header>
      <div class="ings">${x.ingredients.filter(i => i.grams > 0).map(i => `<span class="ing">${i.grams} g ${esc(i.name)}</span>`).join("")}</div>
      ${x.missing?.length ? `<p class="how">Also needs: ${x.missing.map(esc).join(", ")}</p>` : ""}
      ${(x.timing || []).map(t => `<p class="how timing"><b>Timing</b> ${esc(t)}</p>`).join("")}
      ${x.steps?.length ? `<ol>${x.steps.map(s => `<li>${esc(s)}</li>`).join("")}</ol>` : ""}</article>`).join("")}
    <h3>Your pantry</h3>
    <p>${(session.pantry || []).map(i => `<span class="tag ${m.unmatched.includes(i) ? "bad" : "ok"}">${esc(i)}</span>`).join("") || "No pantry given."}
       ${(m.assumed_staples || []).map(i => `<span class="tag">${esc(i)} (assumed)</span>`).join("")}</p>
    <p class="how">Nutrition is computed from USDA FoodData Central for the exact grams shown. Ask "what can I cook tomorrow?" or send a new pantry list.</p>`;
}

function renderSafety() {
  const c = session.constraints, md = session.medical, el = $("#tab-safety");
  if (!md.screened) { el.innerHTML = empty("The Medical agent screens conditions, medications and symptoms before any plan is written."); return; }
  const clearance = c?.clearance || "none";
  el.innerHTML = `
    ${clearance === "required" ? `<div class="notice danger"><b>See a clinician before starting.</b> Until you're cleared, the plan stays at light activity.</div>`
      : clearance === "vigorous" ? `<div class="notice warn"><b>Check with your clinician before vigorous exercise.</b> The plan uses moderate intensity until then.</div>`
      : `<div class="notice ok">No medical clearance needed for this plan, based on your answers.</div>`}
    ${session.review ? `<div class="notice ${session.review.passed ? "ok" : "danger"}">${session.review.passed ? "Final review passed: every exercise and every food was checked against your constraints." : "Issues: " + session.review.problems.map(esc).join("; ")}</div>` : ""}
    <div class="cards">
      <div class="card"><small>Conditions</small>${md.conditions.length ? md.conditions.map(x => `<span class="tag">${esc(x.replace(/_/g, " "))}</span>`).join("") : "<div>none</div>"}</div>
      <div class="card"><small>Medications</small>${md.medications.length ? md.medications.map(x => `<span class="tag">${esc(x)}</span>`).join("") : "<div>none</div>"}</div>
      <div class="card"><small>Symptoms</small>${md.symptoms.length ? md.symptoms.map(x => `<span class="tag bad">${esc(x)}</span>`).join("") : "<div>none</div>"}</div>
    </div>
    ${md.unrecognized_medications?.length ? `<div class="notice warn">No FDA label data for ${md.unrecognized_medications.map(esc).join(", ")}. Ask your pharmacist whether it affects exercise or diet.</div>` : ""}
    <h3>What changes because of your answers</h3>
    ${c?.notes?.length ? c.notes.map(n => `<div class="rule">${esc(n.message)} ${refs(n.refs)}</div>`).join("") : empty("No rules apply.")}
    ${c?.pre_exercise?.length ? `<h3>Before each session</h3><ul>${c.pre_exercise.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    <p class="how">Forma doesn't diagnose or change medication. Each rule above quotes the FDA label or guideline it comes from.</p>`;
}

function renderSources() {
  const el = $("#tab-sources");
  if (!session.citations.length) { el.innerHTML = empty("Sources appear here as agents cite them."); return; }
  el.innerHTML = session.citations.map((c, i) => `<div class="source" id="src-${i + 1}">
    <div class="t">[${i + 1}] ${esc(c.title)}${c.location ? ` · ${esc(c.location)}` : ""}</div>
    <q>${esc(c.snippet.slice(0, 420))}${c.snippet.length > 420 ? "…" : ""}</q>${c.url ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">Open source</a>` : ""}</div>`).join("");
}

// ---- wiring ----------------------------------------------------------------------------------
document.addEventListener("click", e => {
  const tab = e.target.closest(".tabs button");
  if (tab) selectTab(tab.dataset.tab);
  const cite = e.target.closest(".cite");
  if (cite) {
    selectTab("sources");
    const src = $(`#src-${cite.dataset.ref}`);
    if (src) { src.scrollIntoView({ block: "center", behavior: "smooth" }); src.classList.add("flash"); setTimeout(() => src.classList.remove("flash"), 1600); }
  }
});

function selectTab(name) {
  document.querySelectorAll(".tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach(p => p.hidden = p.id !== `tab-${name}`);
}

$("#composer").addEventListener("submit", e => { e.preventDefault(); const t = $("#input").value; $("#input").value = ""; send(t); });
$("#input").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
$("#input").addEventListener("input", e => { e.target.style.height = "auto"; e.target.style.height = Math.min(160, e.target.scrollHeight) + "px"; });
$("#new").addEventListener("click", () => boot(true));
boot();

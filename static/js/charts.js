// Charts for the SPROUT project page: dot-range rows and a tokens-vs-resolve scatter. Numbers are the paper's.
(function () {
  const $ = id => document.getElementById(id);
  const f1 = v => v.toFixed(1);
  const SER = {
    p1: { name: "Pass@1", color: "#9aa7a0", shape: "hollow" },
    p4: { name: "Pass@4", color: "var(--sky)", shape: "square" },
    shep: { name: "Shepherd", color: "#e07b39", shape: "diamond" },
    sprout: { name: "SPROUT", color: "var(--green)", shape: "leaf" },
    base: { name: "Base (no RL)", color: "#9aa7a0", shape: "hollow" },
    grpo: { name: "GRPO", color: "var(--sky)", shape: "square" },
  };
  const DEEPSWE = [
    { label: "Qwen3.8-27B", p1: 37.2, p4: 71.7, shep: 67.3, sprout: 78.6, tok: { p1: 1921, p4: 8673, shep: 9026, sprout: 4123 }, ent: 71.7 },
    { label: "Kimi K3", p1: 72.3, p4: 92.0, shep: 78.8, sprout: 98.2, tok: { p1: 1154, p4: 4614, shep: 2272, sprout: 1492 } },
    { label: "DeepSeek V4.1 Flash", p1: 57.5, p4: 82.3, shep: 86.7, sprout: 90.3, tok: { p1: 1459, p4: 6109, shep: 3624, sprout: 2538 } },
    { label: "Claude Sonnet 5", p1: 46.9, p4: 78.8, shep: 72.6, sprout: 82.3, tok: { p1: 1872, p4: 7487, shep: 4328, sprout: 3897 } },
    { label: "GPT 5.6 Luna", p1: 42.5, p4: 68.1, shep: 62.8, sprout: 85.8, tok: { p1: 352, p4: 1409, shep: 985, sprout: 652 } },
  ];
  const RL = [
    { label: "SWE-bench Pro v2 Hard", base: 62.7, grpo: 74.5, sprout: 84.3 },
    { label: "DeepSWE", base: 37.2, grpo: 33.6, sprout: 40.7 },
    { label: "Terminal-Bench 2.1", base: 49.4, grpo: 39.3, sprout: 64.0 },
  ];
  const SCALING = [
    { cfg: "1 × 4", tok: 570, rate: 75.2, rounds: 1 }, { cfg: "1 × 8", tok: 745.2, rate: 76.1, rounds: 1 },
    { cfg: "1 × 16", tok: 1100, rate: 77.0, rounds: 1 }, { cfg: "1 × 32", tok: 1850, rate: 77.9, rounds: 1 },
    { cfg: "2 × 1", tok: 465.5, rate: 80.5, rounds: 2 }, { cfg: "2 × 4", tok: 673.6, rate: 85.8, rounds: 2 },
    { cfg: "4 × 2", tok: 566.7, rate: 89.4, rounds: 4 }, { cfg: "8 × 1", tok: 525.7, rate: 91.2, rounds: 8 },
  ];

  function marker(g, s, x, y) {
    const { el } = window.SPROUT_TIP;
    if (s.shape === "hollow") return el("circle", { cx: x, cy: y, r: 6, fill: "#fff", stroke: s.color, "stroke-width": 2 }, g);
    if (s.shape === "square") return el("rect", { x: x - 5.5, y: y - 5.5, width: 11, height: 11, rx: 2, fill: s.color }, g);
    if (s.shape === "diamond") return el("rect", { x: x - 5.5, y: y - 5.5, width: 11, height: 11, rx: 1.5, fill: s.color, transform: `rotate(45 ${x} ${y})` }, g);
    const l = el("g", {}, g);
    el("circle", { cx: x, cy: y, r: 8, fill: s.color }, l);
    el("path", { d: `M${x},${y + 3} c-1,-5 -5,-6 -6,-3 c1,3 4,3 6,3 Z M${x},${y + 3} c1,-5 5,-6 6,-3 c-1,3 -4,3 -6,3 Z`, fill: "#fff", "fill-opacity": .95 }, l);
    return l;
  }

  // Dot-range rows: one row per category, markers for each series on a 0–100 axis, SPROUT value labelled.
  function dotRange(svg, rows, keys, opts) {
    const { el, txt, hover } = window.SPROUT_TIP;
    const W = 760, L = opts.left || 170, Rr = 70, rowH = 48, T = 26, H = T + rows.length * rowH + 10;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const x = v => L + (v / 100) * (W - L - Rr);
    [0, 25, 50, 75, 100].forEach(t => {
      el("line", { x1: x(t), x2: x(t), y1: T - 6, y2: H - 10, stroke: "var(--line)", "stroke-dasharray": "2 4" }, svg);
      txt(svg, x(t), T - 12, t + (t === 100 ? "%" : ""), { "text-anchor": "middle", "font-size": 11 });
    });
    rows.forEach((r, i) => {
      const y = T + i * rowH + rowH / 2;
      const g = el("g", { class: "mark" }, svg);
      txt(g, L - 14, y + 4, r.label, { "text-anchor": "end", "font-size": 13, "font-weight": 600, fill: "var(--ink)" });
      const vals = keys.map(k => r[k]);
      el("line", { x1: x(Math.min(...vals)), x2: x(Math.max(...vals)), y1: y, y2: y, stroke: "var(--line-2)", "stroke-width": 2 }, g);
      // gain arrow from the strongest baseline to SPROUT
      const best = Math.max(...keys.filter(k => k !== "sprout").map(k => r[k]));
      if (r.sprout > best) el("line", { x1: x(best), x2: x(r.sprout), y1: y, y2: y, stroke: "var(--green-2)", "stroke-width": 4, opacity: .55, "stroke-linecap": "round" }, g);
      keys.filter(k => k !== "sprout").forEach(k => marker(g, SER[k], x(r[k]), y));
      marker(g, SER.sprout, x(r.sprout), y);
      txt(g, x(r.sprout) + 14, y + 4.5, f1(r.sprout), { "font-size": 13, "font-weight": 800, fill: "var(--green)" });
      const hit = el("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent" }, svg);
      hover(hit, () => opts.tip(r), () => { svg.classList.add("dim"); g.classList.add("hot"); }, () => { svg.classList.remove("dim"); g.classList.remove("hot"); });
    });
  }

  function legend(node, keys) {
    node.innerHTML = keys.map(k => {
      const s = SER[k], sw = s.shape === "hollow" ? `border:2px solid ${s.color};background:#fff` : `background:${s.color}` + (s.shape === "diamond" ? ";transform:rotate(45deg);border-radius:2px" : s.shape === "square" ? ";border-radius:2px" : "");
      return `<span><i style="${sw}"></i>${s.name}</span>`;
    }).join("");
  }

  // Scaling on a linear token axis: Pass@k, one round with k branches, k rounds with one branch, equal-budget splits.
  // Points are the paper's figure data (official verifier verdicts, GPT 5.6 Luna on DeepSWE).
  const SC = {
    passk: [["Pass@1", .390, 46.0], ["Pass@2", .658, 54.0], ["Pass@4", 1.409, 68.1], ["Pass@8", 2.742, 77.9]],
    single: [["1×1", .437, 68.1], ["1×2", .482, 72.6], ["1×4", .574, 75.2], ["1×8", .745, 76.1], ["1×16", 1.097, 77.0], ["1×32", 1.850, 78.8]],
    rounds: [["1×1", .437, 68.1], ["2×1", .466, 80.5], ["4×1", .510, 85.8], ["8×1", .526, 92.0]],
    budget8: [["8×1", .526, 92.0], ["4×2", .567, 90.3], ["2×4", .674, 86.7], ["1×8", .745, 76.1]],
    budget4: [["4×1", .510, 85.8], ["2×2", .532, 82.3], ["1×4", .574, 75.2]],
    reported: ["4+3 (reported)", .652, 85.8],
  };
  function scaling2d(svg) {
    const { el, txt, hover } = window.SPROUT_TIP;
    const W = 760, H = 430, L = 50, Rr = 26, T = 22, B = 46;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const X0 = .2, X1 = 2.95, Y0 = 40, Y1 = 96;
    const x = v => L + (v - X0) / (X1 - X0) * (W - L - Rr), y = v => T + (1 - (v - Y0) / (Y1 - Y0)) * (H - T - B);
    for (let t = 40; t <= 90; t += 10) { el("line", { x1: L, x2: W - Rr, y1: y(t), y2: y(t), stroke: "var(--line)" }, svg); txt(svg, L - 10, y(t) + 4, t, { "text-anchor": "end", "font-size": 11 }); }
    [.5, 1, 1.5, 2, 2.5].forEach(t => txt(svg, x(t), H - B + 18, t.toFixed(1), { "text-anchor": "middle", "font-size": 11 }));
    txt(svg, (L + W - Rr) / 2, H - 6, "total tokens, billions (baseline rollout included)", { "text-anchor": "middle", "font-size": 11.5, "font-weight": 600, fill: "var(--ink-2)" });
    txt(svg, 12, T + 4, "resolve rate, %", { "font-size": 11, "font-weight": 600, fill: "var(--ink-2)" });
    const GREY = "#8a9790", BLUE = "var(--sky)", GREEN = "var(--green)", ORANGE = "#d98b4f", RED = "#b5473b";
    const line = (pts, stroke, extra) => el("path", { d: "M" + pts.map(p => `${x(p[1])},${y(p[2])}`).join(" L"), fill: "none", stroke, "stroke-linejoin": "round", "stroke-linecap": "round", ...extra }, svg);
    // Pass@8 level
    const p8 = SC.passk[3];
    el("line", { x1: L, x2: x(p8[1]), y1: y(p8[2]), y2: y(p8[2]), stroke: GREY, "stroke-dasharray": "5 5", opacity: .8 }, svg);
    line(SC.passk, GREY, { "stroke-width": 2.2 });
    line(SC.budget4, RED, { "stroke-width": 1.8, "stroke-dasharray": "4 4" });
    line(SC.budget8, ORANGE, { "stroke-width": 1.8, "stroke-dasharray": "5 4" });
    line(SC.single, BLUE, { "stroke-width": 2.4, "stroke-dasharray": "9 5" });
    line(SC.rounds, GREEN, { "stroke-width": 4 });
    const marks = [];
    const point = (p, kind, labelPos) => {
      const px = x(p[1]), py = y(p[2]), g = el("g", { class: "mark" }, svg);
      if (kind === "passk") el("rect", { x: px - 6, y: py - 6, width: 12, height: 12, rx: 2, fill: "#fff", stroke: GREY, "stroke-width": 2.2 }, g);
      else if (kind === "single") el("circle", { cx: px, cy: py, r: 5, fill: BLUE, stroke: "#fff", "stroke-width": 1.5 }, g);
      else if (kind === "rounds") el("circle", { cx: px, cy: py, r: 7, fill: GREEN, stroke: "#fff", "stroke-width": 2 }, g);
      else if (kind === "budget8") el("rect", { x: px - 5, y: py - 5, width: 10, height: 10, rx: 1.5, fill: ORANGE, transform: `rotate(45 ${px} ${py})` }, g);
      else if (kind === "budget4") el("path", { d: `M${px},${py - 6} L${px + 6},${py + 5} L${px - 6},${py + 5} Z`, fill: RED }, g);
      else if (kind === "star") el("path", { d: star(px, py, 11, 5), fill: "var(--ink)", stroke: "#fff", "stroke-width": 1.5 }, g);
      const [dx, dy, anchor] = labelPos;
      const color = kind === "passk" ? GREY : kind === "single" ? BLUE : kind === "rounds" ? GREEN : kind === "budget8" ? ORANGE : kind === "budget4" ? RED : "var(--ink)";
      txt(g, px + dx, py + dy, p[0], { "text-anchor": anchor, "font-size": kind === "rounds" ? 13 : 11.5, "font-weight": kind === "rounds" || kind === "star" ? 800 : 600, fill: color });
      const hit = el("circle", { cx: px, cy: py, r: 14, fill: "transparent" }, svg);
      const what = { passk: "Independent sampling: Pass@1 is the baseline rollout; Pass@2/4/8 are unions of independent attempts with their summed tokens.", single: "One review round with k branches: the branches never see each other's failures.", rounds: "k review rounds with one branch each: every round starts from the previous round's failure and its diagnostics.", budget8: "A budget of eight branches split across rounds.", budget4: "A budget of four branches split across rounds.", star: "The configuration the main results report: two rounds, 4 then 3 branches." }[kind];
      hover(hit, `<div class="tt-h">${p[0]}</div>${p[2].toFixed(1)}% resolved · ${p[1].toFixed(2)}B tokens<br>${what}`, () => { svg.classList.add("dim"); g.classList.add("hot"); }, () => { svg.classList.remove("dim"); g.classList.remove("hot"); });
      marks.push(g);
    };
    function star(cx, cy, r, n) { let d = ""; for (let i = 0; i < 2 * n; i++) { const rr = i % 2 ? r * .45 : r, a = Math.PI / n * i - Math.PI / 2; d += (i ? "L" : "M") + (cx + rr * Math.cos(a)).toFixed(1) + "," + (cy + rr * Math.sin(a)).toFixed(1); } return d + "Z"; }
    point(SC.passk[0], "passk", [14, 4, "start"]); point(SC.passk[1], "passk", [12, 16, "start"]); point(SC.passk[2], "passk", [-10, 22, "middle"]); point(SC.passk[3], "passk", [-12, 22, "end"]);
    point(SC.single[1], "single", [10, 16, "start"]); point(SC.single[2], "single", [10, 16, "start"]); point(SC.single[3], "single", [10, 17, "start"]); point(SC.single[4], "single", [0, 20, "middle"]); point(SC.single[5], "single", [0, 22, "middle"]);
    point(SC.rounds[0], "rounds", [-12, 5, "end"]); point(SC.rounds[1], "rounds", [-12, 5, "end"]); point(SC.rounds[2], "rounds", [-12, 5, "end"]); point(SC.rounds[3], "rounds", [-12, 5, "end"]);
    point(SC.budget8[1], "budget8", [10, -6, "start"]); point(SC.budget8[2], "budget8", [10, -6, "start"]);
    point(SC.budget4[1], "budget4", [10, 4, "start"]);
    point(SC.reported, "star", [12, 22, "start"]);
    txt(svg, x(SC.single[5][1]) + 12, y(SC.single[5][2]) - 22, "78.8% resolved,", { "font-size": 11, fill: GREY });
    txt(svg, x(SC.single[5][1]) + 12, y(SC.single[5][2]) - 9, "a third fewer tokens than Pass@8", { "font-size": 11, fill: GREY });
    txt(svg, x(p8[1]) + 4, y(p8[2]) - 10, "", { "font-size": 11 });
  }

  function bibtex() {
    const btn = document.querySelector(".copy"); if (!btn) return;
    btn.addEventListener("click", async () => {
      const t = document.querySelector(".bib pre").textContent;
      try { await navigator.clipboard.writeText(t); btn.textContent = "Copied"; } catch (e) { btn.textContent = "Select & copy"; }
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const { swatch } = {};
    legend($("legend-deepswe"), ["p1", "p4", "shep", "sprout"]);
    dotRange($("chart-deepswe"), DEEPSWE, ["p1", "p4", "shep", "sprout"], {
      tip: r => `<div class="tt-h">${r.label} · DeepSWE</div>` + ["p1", "p4", "shep", "sprout"].map(k => `<div class="tt-row"><i style="background:${SER[k].color}"></i>${SER[k].name}<span style="margin-left:auto;padding-left:12px">${f1(r[k])}% · ${r.tok[k].toLocaleString()}M tok</span></div>`).join("") +
        (r.ent ? `<div class="tt-row"><i style="background:var(--violet)"></i>Entropy-based<span style="margin-left:auto;padding-left:12px">${f1(r.ent)}%</span></div>` : "") +
        `<div style="margin-top:4px;color:#c9d6cf">SPROUT spends ${Math.round(100 * r.tok.sprout / r.tok.p4)}% of Pass@4's tokens</div>`,
    });
    legend($("legend-rl"), ["base", "grpo", "sprout"]);
    dotRange($("chart-rl"), RL, ["base", "grpo", "sprout"], {
      left: 190,
      tip: r => `<div class="tt-h">${r.label}</div>` + ["base", "grpo", "sprout"].map(k => `<div class="tt-row"><i style="background:${SER[k].color}"></i>${SER[k].name}<span style="margin-left:auto;padding-left:12px">${f1(r[k])}%</span></div>`).join("") +
        `<div style="margin-top:4px;color:#c9d6cf">SPROUT − GRPO: +${(r.sprout - r.grpo).toFixed(1)} points</div>`,
    });
    scaling2d($("chart-scaling"));
    bibtex();
  });
})();

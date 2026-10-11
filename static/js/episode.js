// One round of SPROUT, step by step: a state-driven SVG canvas with a vertical stepper.
// Illustrative scenario, not run data. Standalone (no dependencies).
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const k in attrs || {}) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; };
  const txt = (parent, x, y, s, attrs) => { const t = el("text", Object.assign({ x, y, "font-size": 12, fill: "var(--muted)" }, attrs || {}), parent); t.textContent = s; return t; };

  // ---- tooltip (shared with charts.js through window.SPROUT_TIP) ----
  const tipNode = document.createElement("div"); tipNode.className = "tooltip"; tipNode.setAttribute("role", "status");
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(tipNode));
  const tip = {
    show(evt, html) {
      tipNode.innerHTML = html; tipNode.style.opacity = 1;
      const pad = 14, box = tipNode.getBoundingClientRect();
      let x = evt.clientX + pad, y = evt.clientY - box.height - 8;
      if (x + box.width > window.innerWidth - 8) x = evt.clientX - box.width - pad;
      if (y < 8) y = evt.clientY + pad;
      tipNode.style.left = x + "px"; tipNode.style.top = y + "px";
    },
    hide() { tipNode.style.opacity = 0; },
  };
  const hover = (node, html, onEnter, onLeave) => {
    node.addEventListener("mousemove", e => tip.show(e, typeof html === "function" ? html() : html));
    node.addEventListener("mouseenter", () => onEnter && onEnter());
    node.addEventListener("mouseleave", () => { tip.hide(); onLeave && onLeave(); });
  };
  window.SPROUT_TIP = { el, txt, hover, tip };

  // ---- geometry ----
  const W = 840, H = 500;
  const C = { call: "var(--sky)", ckpt: "var(--green-2)", feedback: "var(--coral)", ink: "var(--ink)", ink2: "var(--ink-2)", muted: "var(--muted)", green: "var(--green)", soil: "var(--soil)", coral: "var(--coral-ink)", ok: "var(--green)", sun: "var(--sun-ink)" };
  const colX = i => 85 + i * 134;        // card centres, six columns
  const CW = 120, CH = 84;
  const ROW1 = { top: 102, stem: 214, lbl: 240, hdr: 92 };
  const ROW2 = { top: 304, stem: 416, lbl: 442, hdr: 294 };
  const POL = { x: 0, y: 8, w: 400, h: 46 };
  const VER = { x: 560, y: 8, w: 280, h: 46 };
  const STEM_X0 = 25, STEM_X1 = 815;
  const BRANCH_X0 = colX(3) - 34;       // where the fork lands on the branch stem
  const TURNS = [
    { tag: "t1", cmd: 'rg -n "def parse"', note: "found src/cli.py:42" },
    { tag: "t2", cmd: "cat -n src/cli.py", note: "return argv[1]" },
    { tag: "t3", cmd: "pytest -q", note: "..F.  1 failed" },
    { tag: "t4", cmd: "sed -i s/[1]/[0]/", note: "a fix: wrong index" },
    { tag: "t5", cmd: "pytest -q", note: "..F.  1 failed" },
    { tag: "t6", cmd: "echo done", note: "submits anyway" },
  ];
  const BRANCH = [
    { tag: "4′", cmd: "pytest --tb=short", note: "wants argv[2]", start: true },
    { tag: "5′", cmd: "sed -i s/[1]/[2]/", note: "the fix" },
    { tag: "6′", cmd: "pytest -q", note: "....  4 passed" },
  ];
  const ROLES = ["actor", "analyst", "reviewer"];

  function initialState() {
    return {
      role: "actor", msg: "Waits for the task", thinking: false,
      verifier: { msg: "Waits for a finished attempt", detail: "", kind: null, busy: false },
      root: TURNS.map(() => ({ shown: false, saved: false, bad: false, glow: false, pairing: false, into: false })),
      branch: BRANCH.map(() => ({ shown: false, saved: false, pairing: false, into: false })),
      rootStem: 0, fork: 0, branchStem: 0, fail: false, pass: false, lane2: false, train: false, training: false, updated: false,
    };
  }

  // ---- scenario ----
  const STEPS = [
    { title: "One attempt, one sandbox", text: "The policy runs mini-swe-agent in its own microVM. Each turn is reasoning followed by shell calls; the harness runs them and returns the output.", run: async () => {} },
    { title: "A checkpoint after every turn", text: "SPROUT takes an incremental environment snapshot and pairs it with the agent context at that boundary. Turn 4 edits the wrong index; the agent does not notice and submits.",
      run: async (S, A) => { for (let i = 0; i < TURNS.length; i++) await A.turn(S, "root", i); S.msg = "Attempt finished after 6 turns"; } },
    { title: "The verifier fails it", text: "The task's own tests run in a fresh sandbox: 1 of 4 fails. A binary reward gives the whole attempt zero, although turns 1–3 were right.",
      run: async (S, A) => {
        S.verifier = { msg: "running the tests…", detail: "", kind: null, busy: true }; A.render(); await A.pause(1100);
        S.verifier = { msg: "1 of 4 tests failed", detail: "FAILED test_cli.py::test_parse_argv", kind: "bad", busy: false }; S.fail = true; A.render(); await A.pause(600);
      } },
    { title: "The same policy, as analyst", text: "Prompted as analyst, the policy reads the trajectory with the verifier's diagnostics: the fix in turn 4 went wrong, and checkpoint 3 is the last state worth keeping. No stronger model is involved.",
      run: async (S, A) => {
        S.role = "analyst"; S.msg = "Reads the attempt and the verifier's feedback"; await A.think(1400);
        S.root[3].bad = true; S.root[2].glow = true; S.msg = "turn 4 went wrong · last good state: checkpoint 3"; A.render(); await A.pause(800);
      } },
    { title: "…and as reviewer", text: "Prompted as reviewer, it picks checkpoint 3, the boundary before the mistake, and writes the branch's first turn in the actor's own voice: probe the failing test, then fix, then verify. That starting turn replaces turn 4.",
      run: async (S, A) => {
        S.role = "reviewer"; S.msg = "Plans the branch from checkpoint 3"; S.lane2 = true; await A.think(1400);
        S.branch[0].shown = true; S.msg = "starting turn: pytest --tb=short, then fix, then verify"; A.render(); await A.pause(800);
      } },
    { title: "Restore and continue", text: "A fresh microVM boots from checkpoint 3's environment snapshot and the agent context is restored up to turn 3. The starting turn runs like any other turn, then the actor continues on its own: same history up to 3, new future.",
      run: async (S, A) => {
        S.role = "actor"; S.msg = "Continues from checkpoint 3"; A.render();
        await A.grow(S, "fork", 1, 800);
        for (let i = 0; i < BRANCH.length; i++) await A.turn(S, "branch", i);
        S.msg = "Branch finished after 3 new turns";
      } },
    { title: "The verifier passes it", text: "4 of 4 tests pass. The branch inherited turns 1–3 and spent three new turns; a restart would have spent all six again. On DeepSWE this is why SPROUT beats four independent attempts with 32–52% of their tokens.",
      run: async (S, A) => {
        S.verifier = { msg: "running the tests…", detail: "", kind: null, busy: true }; A.render(); await A.pause(1100);
        S.verifier = { msg: "4 of 4 tests passed", detail: "....  4 passed", kind: "ok", busy: false }; S.pass = true; A.render(); await A.pause(600);
      } },
    { title: "Train on both", text: "For RL, roots and branches form one GRPO group: the branch is credited for its success, shared prefixes are counted once, and the analysis never enters the learner's input. The policy learns to make the corrective turn from its ordinary context.",
      run: async (S, A) => {
        S.train = true; S.training = true; S.msg = "GRPO step over the group: root reward 0, branch reward 1"; A.render();
        await A.flicker(1800);
        S.training = false; S.updated = true; S.msg = "Updated policy: the corrective turn now comes from its ordinary context"; A.render(); await A.pause(500);
      } },
  ];

  // ---- skeleton ----
  function build(svg) {
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const defs = el("defs", {}, svg);
    const sh = el("filter", { id: "ep-shadow", x: "-10%", y: "-10%", width: "120%", height: "130%" }, defs);
    el("feDropShadow", { dx: 0, dy: 1.5, stdDeviation: 1.6, "flood-color": "#1e2a24", "flood-opacity": .10 }, sh);
    const R = {};

    // policy token with a role dial
    const pg = el("g", {}, svg);
    R.polBox = el("rect", { x: POL.x, y: POL.y, width: POL.w, height: POL.h, rx: 12, class: "box" }, pg);
    const rb = el("g", { transform: "translate(14 17)" }, pg);
    el("rect", { x: 0, y: 6, width: 22, height: 16, rx: 5, fill: "#fff", stroke: C.green, "stroke-width": 1.6 }, rb);
    el("circle", { cx: 7, cy: 14, r: 2, fill: C.green }, rb); el("circle", { cx: 15, cy: 14, r: 2, fill: C.green }, rb);
    el("path", { d: "M11,6 V1", stroke: C.green, "stroke-width": 1.6 }, rb);
    el("path", { d: "M11,1 c-1,-4 -5,-5 -6,-2 c1,3 4,3 6,2 Z M11,1 c1,-4 5,-5 6,-2 c-1,3 -4,3 -6,2 Z", fill: "var(--green-2)", stroke: C.green, "stroke-width": 1 }, rb);
    R.polTitle = txt(pg, 46, 27, "Policy", { "font-size": 14, "font-weight": 800, fill: C.ink });
    R.polSub = txt(pg, 46, 44, "one model, three roles", { "font-size": 10.5, fill: C.muted });
    R.think = el("g", { class: "think" }, pg);
    [0, 1, 2].forEach(i => el("circle", { cx: 192 + i * 9, cy: 31, r: 3 }, R.think));
    el("rect", { x: 220, y: 15, width: 170, height: 32, rx: 10, class: "dial" }, pg);
    R.segs = ROLES.map((r, i) => {
      const g = el("g", { class: "seg " + r }, pg);
      const x = 223 + i * 55;
      el("rect", { x, y: 18, width: 53, height: 26, rx: 8 }, g);
      txt(g, x + 26.5, 35, r, { "text-anchor": "middle", "font-size": 10.5 });
      return g;
    });
    R.msg = txt(svg, 0, 76, "", { "font-size": 12, fill: C.ink2, "font-weight": 600 });
    hover(pg, () => `<div class="tt-h">One policy, three prompts</div>The actor runs the task. The analyst reads a failed attempt with the verifier's feedback. The reviewer picks checkpoints and writes starting turns.<br>Same weights in every role; current role: <b>${R.state.role}</b>.`);

    // policy weights: a small matrix that flickers during the gradient step
    R.weights = el("g", { class: "group hidden" }, svg);
    R.cells = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 10; c++) R.cells.push(el("rect", { x: 420 + c * 11, y: 13 + r * 11, width: 9, height: 9, rx: 2, fill: "var(--leaf)", class: "cell" }, R.weights));
    R.wLabel = txt(R.weights, 474, 58, "policy weights θ", { "text-anchor": "middle", "font-size": 10, "font-weight": 700, fill: C.muted });
    el("path", { d: "M402,31 H416 M544,31 H556", stroke: "var(--line-2)", "stroke-width": 1.5, "stroke-dasharray": "2 3" }, R.weights);

    // verifier
    const vg = el("g", {}, svg);
    R.verBox = el("rect", { x: VER.x, y: VER.y, width: VER.w, height: VER.h, rx: 12, class: "box" }, vg);
    txt(vg, VER.x + 14, 27, "Verifier", { "font-size": 14, "font-weight": 800, fill: C.ink });
    R.verMsg = txt(vg, VER.x + 84, 27, "", { "font-size": 12, "font-weight": 700, fill: C.ink2 });
    R.verDetail = txt(vg, VER.x + 14, 44, "", { "font-size": 10.5, "font-family": "var(--mono)", fill: C.muted });
    R.vbusy = el("g", { class: "think verifier" }, vg);
    [0, 1, 2].forEach(i => el("circle", { cx: VER.x + 20 + i * 9, cy: 42, r: 3 }, R.vbusy));
    hover(vg, `<div class="tt-h">Verifier</div>The benchmark's own grader, run on the finished attempt in a separate sandbox. It returns a binary reward and diagnostics; only the analyst and reviewer prompts ever see the diagnostics.`);

    // rows
    function row(RW, label, kind, x0) {
      const g = el("g", {}, svg);
      const lbl = txt(g, STEM_X0 + 2, RW.hdr, label, { "font-size": 10.5, "font-weight": 800, "letter-spacing": ".12em", fill: C.muted });
      const stemBg = el("path", { d: `M${x0},${RW.stem} H${STEM_X1}`, fill: "none", stroke: "var(--line)", "stroke-width": 4, "stroke-linecap": "round", "stroke-dasharray": "2 7" }, g);
      const stem = el("path", { d: `M${x0},${RW.stem} H${STEM_X1}`, fill: "none", stroke: kind === "root" ? "var(--soil-2)" : "var(--green-2)", "stroke-width": 4, "stroke-linecap": "round", class: "stem" }, g);
      const len = stem.getTotalLength(); stem.style.strokeDasharray = `${len} ${len}`; stem.style.strokeDashoffset = len;
      const vd = el("g", { class: "verdict hidden" }, g);
      el("rect", { x: 718, y: RW.hdr - 17, width: 122, height: 24, rx: 12 }, vd);
      const vdT = txt(vd, 779, RW.hdr - 1, "", { "text-anchor": "middle", "font-size": 11.5, "font-weight": 800 });
      return { g, lbl, stem, stemLen: len, stemBg, vd, vdT, cards: [] };
    }
    R.row1 = row(ROW1, "ATTEMPT · ROOT ROLLOUT", "root", STEM_X0);
    R.row2 = row(ROW2, "BRANCH · FROM CHECKPOINT 3", "branch", BRANCH_X0);
    R.row2.lbl.setAttribute("opacity", 0); R.row2.stemBg.setAttribute("opacity", 0);
    // the fork: leaves C3 along the stem and lands on the branch stem, horizontal tangents at both ends
    R.fork = el("path", { d: `M${colX(2)},${ROW1.stem} C${colX(2) + 70},${ROW1.stem} ${BRANCH_X0 - 70},${ROW2.stem} ${BRANCH_X0},${ROW2.stem}`, fill: "none", stroke: "var(--green-2)", "stroke-width": 4, "stroke-linecap": "round" }, svg);
    R.forkLen = R.fork.getTotalLength(); R.fork.style.strokeDasharray = `${R.forkLen} ${R.forkLen}`; R.fork.style.strokeDashoffset = R.forkLen;

    function card(RW, turn, i, col, which) {
      const cx = colX(col), x = cx - CW / 2, y = RW.top, isB = which === "branch";
      const g = el("g", { class: "cardg hidden" }, svg);
      const box = el("rect", { x, y, width: CW, height: CH, rx: 10, class: "card" + (turn.start ? " start" : "") }, g);
      el("path", { d: `M${x + 12},${y + 1.5} H${x + CW - 12}`, class: "accent " + (turn.start ? "start" : isB ? "branch" : "root"), fill: "none" }, g);
      el("rect", { x: x + 10, y: y + 12, width: 26, height: 17, rx: 8, fill: turn.start ? "var(--green)" : isB ? "var(--leaf-2)" : "var(--sun-2)" }, g);
      txt(g, x + 23, y + 24.5, turn.tag, { "text-anchor": "middle", "font-size": 10.5, "font-weight": 800, fill: turn.start ? "#fff" : isB ? C.green : C.sun });
      if (turn.start) txt(g, x + 42, y + 24.5, "starting turn", { "font-size": 10, "font-weight": 700, fill: C.green });
      txt(g, x + 10, y + 49, turn.cmd, { "font-size": 9.8, "font-family": "var(--mono)", fill: C.ink });
      const note = txt(g, x + 10, y + 67, turn.note, { "font-size": 10, fill: C.muted });
      const pair = el("g", { class: "pair hidden" }, svg);
      el("rect", { x: cx - 50, y: y + CH + 7, width: 48, height: 13, rx: 4, fill: "var(--soil-3)", stroke: "var(--soil-2)" }, pair);
      txt(pair, cx - 26, y + CH + 16.5, "snapshot", { "text-anchor": "middle", "font-size": 8.5, "font-weight": 700, fill: C.soil });
      el("rect", { x: cx + 2, y: y + CH + 7, width: 48, height: 13, rx: 4, fill: "var(--sky-2)", stroke: "var(--sky)" }, pair);
      txt(pair, cx + 26, y + CH + 16.5, "context", { "text-anchor": "middle", "font-size": 8.5, "font-weight": 700, fill: C.ink2 });
      const node = el("g", { class: "node hidden" + (isB ? " branch" : "") }, svg);
      el("circle", { cx, cy: RW.stem, r: 9 }, node);
      el("rect", { x: cx - 4, y: RW.stem - 4, width: 8, height: 8, rx: 1.5, fill: "#fff", "fill-opacity": .9 }, node);
      txt(node, cx, RW.lbl, isB ? `C${col + 1}′` : `C${col + 1}`, { "text-anchor": "middle", "font-size": 11, "font-weight": 800, fill: C.ink2 });
      hover(g, () => {
        const st = (isB ? R.state.branch : R.state.root)[i];
        return `<div class="tt-h">Turn ${turn.tag}</div><code style="background:rgba(255,255,255,.14);color:#fff">$ ${turn.cmd}</code> → ${turn.note}<br>${turn.start ? "Written by the reviewer in the actor's voice; executed through the normal tool path, so the agent context and the environment agree." : "Reasoning, then shell calls, then their observations."}${st.saved ? "<br>Checkpoint saved: environment snapshot + agent context up to this turn." : ""}`;
      });
      hover(node, `<div class="tt-h">Checkpoint</div>Two halves at one turn boundary: the environment snapshot after the turn's tool calls (an incremental layer) and the agent context up to the same observation. Any checkpoint can be restored in a fresh sandbox.`);
      return { g, box, note, pair, node };
    }
    R.row1.cards = TURNS.map((t, i) => card(ROW1, t, i, i, "root"));
    R.row2.cards = BRANCH.map((t, i) => card(ROW2, t, i, i + 3, "branch"));

    R.train = el("g", { class: "group hidden" }, svg);
    el("rect", { x: STEM_X0, y: 468, width: STEM_X1 - STEM_X0, height: 24, rx: 12, fill: "var(--leaf-2)", stroke: "var(--green-2)", "stroke-width": 1.2 }, R.train);
    txt(R.train, (STEM_X0 + STEM_X1) / 2, 484, "GRPO group: root (reward 0) + branch (reward 1) · shared prefix counted once · analysis kept out of the learner's input", { "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: C.green });

    return R;
  }

  // ---- render ----
  function render(R, S) {
    R.state = S;
    R.msg.textContent = S.msg; R.msg.setAttribute("fill", S.train ? C.ok : C.ink2);
    R.polBox.setAttribute("class", "box" + (S.train ? " ok" : ""));
    R.segs.forEach((g, i) => g.setAttribute("class", "seg " + ROLES[i] + (S.role === ROLES[i] ? " on" : "")));
    R.think.setAttribute("class", "think " + S.role + (S.thinking ? " on" : ""));
    R.vbusy.setAttribute("class", "think verifier" + (S.verifier.busy ? " on" : ""));
    R.verMsg.textContent = S.verifier.msg; R.verDetail.textContent = S.verifier.detail;
    R.verMsg.setAttribute("fill", S.verifier.kind === "ok" ? C.ok : S.verifier.kind === "bad" ? C.coral : C.ink2);
    R.verBox.setAttribute("class", "box" + (S.verifier.kind ? " " + S.verifier.kind : ""));
    R.row1.stem.style.strokeDashoffset = R.row1.stemLen * (1 - S.rootStem);
    R.fork.style.strokeDashoffset = R.forkLen * (1 - S.fork);
    R.row2.stem.style.strokeDashoffset = R.row2.stemLen * (1 - S.branchStem);
    R.row2.lbl.setAttribute("opacity", S.lane2 ? 1 : 0); R.row2.stemBg.setAttribute("opacity", S.lane2 ? 1 : 0);
    const verdict = (row, kind, label) => { row.vd.setAttribute("class", "verdict" + (kind ? " " + kind : " hidden")); row.vdT.textContent = label; };
    verdict(R.row1, S.fail ? "fail" : null, "✗ fail · reward 0"); verdict(R.row2, S.pass ? "pass" : null, "✓ pass · reward 1");
    S.root.forEach((c, i) => {
      const r = R.row1.cards[i];
      r.g.classList.toggle("hidden", !c.shown); r.box.setAttribute("class", "card" + (c.bad ? " bad" : ""));
      r.note.setAttribute("fill", c.bad ? C.coral : C.muted); r.note.setAttribute("font-weight", c.bad ? 700 : 400);
      r.pair.setAttribute("class", "pair" + (c.pairing ? "" : " hidden") + (c.into ? " into" : ""));
      r.node.setAttribute("class", "node" + (c.saved ? "" : " hidden") + (c.glow ? " glow" : "") + (c.bad ? " bad" : ""));
    });
    S.branch.forEach((c, i) => {
      const r = R.row2.cards[i];
      r.g.classList.toggle("hidden", !c.shown); r.pair.setAttribute("class", "pair" + (c.pairing ? "" : " hidden") + (c.into ? " into" : ""));
      r.node.setAttribute("class", "node branch" + (c.saved ? "" : " hidden"));
    });
    R.train.classList.toggle("hidden", !S.train);
    R.weights.classList.toggle("hidden", !(S.training || S.updated));
    R.wLabel.textContent = S.updated ? "policy weights θ′" : "policy weights θ";
    if (!S.training) R.cells.forEach((cell, i) => cell.setAttribute("fill", S.updated ? (i % 10 < 6 ? "var(--green)" : "var(--green-2)") : "var(--leaf)"));
    R.polTitle.textContent = S.updated ? "Policy′" : "Policy";
    R.polSub.textContent = S.updated ? "weights updated by the GRPO step" : "one model, three roles";
  }

  // ---- animation ----
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function makeAnim(R, S, ctl, instant) {
    const alive = () => ctl.token === ctl.current && !instant;
    const A = {
      render: () => { if (instant || ctl.token === ctl.current) render(R, S); },
      pause: ms => (alive() ? new Promise(r => setTimeout(r, ms)) : Promise.resolve()),
      // the policy "thinks": three pulsing dots in the role's colour, then the result lands
      async think(ms) { S.thinking = true; A.render(); await A.pause(ms); S.thinking = false; A.render(); },
      // the matrix flickers like a gradient step: random cells light up, then the pattern settles
      flicker(ms) {
        if (!alive()) return Promise.resolve();
        const shades = ["var(--leaf)", "var(--leaf-2)", "var(--green-2)", "var(--green)", "var(--sun)", "#fff"];
        return new Promise(resolve => {
          const t0 = performance.now(); let last = 0;
          const frame = now => {
            if (now - last > 90) { last = now; R.cells.forEach(cell => { if (Math.random() < .45) cell.setAttribute("fill", shades[Math.floor(Math.random() * shades.length)]); }); }
            if (now - t0 < ms && alive()) requestAnimationFrame(frame); else resolve();
          };
          requestAnimationFrame(frame);
        });
      },
      grow(S_, key, to, ms) {
        if (!alive()) { S[key] = to; A.render(); return Promise.resolve(); }
        const from = S[key];
        return new Promise(resolve => {
          const t0 = performance.now();
          const frame = now => {
            const k = Math.min(1, (now - t0) / ms), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
            S[key] = from + (to - from) * e; A.render();
            if (k < 1 && alive()) requestAnimationFrame(frame); else { S[key] = to; A.render(); resolve(); }
          };
          requestAnimationFrame(frame);
        });
      },
      async turn(S_, which, i) {
        const lane = which === "root" ? S.root : S.branch, defs = which === "root" ? TURNS : BRANCH;
        const col = which === "root" ? i : i + 3, cx = colX(col), RW = which === "root" ? ROW1 : ROW2, turn = defs[i];
        const first = !lane[i].shown;
        S.msg = `Turn ${turn.tag}: $ ${turn.cmd}`; A.render();
        if (first) await A.think(600);
        lane[i].shown = true; A.render();
        if (first && !instant) { const b = (which === "root" ? R.row1 : R.row2).cards[i].box; b.classList.remove("flash"); void b.getBBox(); b.classList.add("flash"); }
        await A.pause(260);
        lane[i].pairing = true; A.render(); await A.pause(i < 2 && which === "root" ? 900 : 550);   // the two halves sit under the card long enough to read
        lane[i].into = true; A.render(); await A.pause(340);                                        // then shrink into the checkpoint, which pops as they vanish
        lane[i].pairing = false; lane[i].into = false; lane[i].saved = true;
        if (which === "root") S.rootStem = Math.max(S.rootStem, (cx - STEM_X0) / (STEM_X1 - STEM_X0));
        else S.branchStem = Math.max(S.branchStem, (cx - BRANCH_X0) / (STEM_X1 - BRANCH_X0));
        A.render(); await A.pause(140);
      },
    };
    return A;
  }

  // ---- player ----
  function init() {
    const root = document.getElementById("episode");
    if (!root) return;
    const svg = root.querySelector("svg"), R = build(svg);
    const list = root.querySelector(".steps");
    const btnPlay = root.querySelector("[data-act=play]"), btnPrev = root.querySelector("[data-act=prev]"), btnNext = root.querySelector("[data-act=next]");
    const ctl = { token: 0, current: 0, step: 0, playing: false, run: 0 };
    let S = initialState();
    const items = STEPS.map((s, i) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="n">${i + 1}</span><b>${s.title}</b><p>${s.text}</p>`;
      li.addEventListener("click", () => { stop(); goTo(i); });
      list.appendChild(li); return li;
    });
    function mark(i) {
      items.forEach((li, j) => { li.classList.toggle("on", j === i); li.classList.toggle("done", j < i); });
      btnPrev.disabled = i === 0; btnNext.disabled = i === STEPS.length - 1;
      const sc = root.querySelector(".ep-scroll");
      if (sc && window.innerWidth <= 960) sc.scrollTo({ left: i >= 4 ? sc.scrollWidth : 0, behavior: "smooth" });
    }
    async function goTo(i) {
      ctl.token++; S = initialState();
      root.classList.add("instant");
      const A = makeAnim(R, S, ctl, true);
      for (let k = 0; k <= i; k++) await STEPS[k].run(S, A);
      ctl.step = i; render(R, S); mark(i);
      void svg.getBoundingClientRect(); root.classList.remove("instant");
    }
    async function animateStep(i) {
      ctl.current = ++ctl.token;
      const A = makeAnim(R, S, ctl, reduceMotion);
      mark(i);
      await STEPS[i].run(S, A);
      if (ctl.token !== ctl.current) return false;
      ctl.step = i; render(R, S); return true;
    }
    async function play() {
      if (ctl.playing) return;
      if (ctl.step >= STEPS.length - 1) await goTo(0);
      const run = ++ctl.run;   // a paused loop that wakes up must not resume alongside a new one
      ctl.playing = true; btnPlay.textContent = "❚❚ Pause"; root.classList.add("playing");
      while (ctl.playing && ctl.run === run && ctl.step < STEPS.length - 1) {
        const ok = await animateStep(ctl.step + 1);
        if (!ok || !ctl.playing || ctl.run !== run) break;
        const words = STEPS[ctl.step].text.split(/\s+/).length;
        await new Promise(r => setTimeout(r, Math.min(5200, Math.max(2600, words * 230))));
      }
      if (ctl.run === run) stop();
    }
    function stop() { ctl.playing = false; btnPlay.textContent = ctl.step >= STEPS.length - 1 ? "↻ Replay" : "▶ Play"; root.classList.remove("playing"); }
    btnPlay.addEventListener("click", () => (ctl.playing ? stop() : play()));
    btnNext.addEventListener("click", async () => { stop(); if (ctl.step < STEPS.length - 1) { await goTo(ctl.step); await animateStep(ctl.step + 1); stop(); } });
    btnPrev.addEventListener("click", () => { stop(); goTo(Math.max(0, ctl.step - 1)); });
    goTo(0);
    if ("IntersectionObserver" in window && !reduceMotion) {
      const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); setTimeout(play, 500); } }, { threshold: .4 });
      io.observe(svg);
    }
    window.SPROUT_EPISODE = { goTo, play, stop, steps: STEPS.length, state: () => S };
  }
  document.addEventListener("DOMContentLoaded", init);
})();

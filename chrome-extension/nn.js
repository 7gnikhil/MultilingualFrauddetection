/**
 * nn.js — a live neural-network animation in the style of the TensorFlow
 * Playground: layers of neurons, weighted connections, and signal pulses
 * that travel along the edges and light neurons up as they pass.
 *
 * It is a visualisation of the Fraud Detector pipeline:
 *   Message tokens -> Anonymized text -> Language -> MuRIL embedding
 *   + lexical features -> per-language classifier -> Verdict
 *
 * Usage:
 *   const nn = FraudNN.attach(canvas, { onStage: (i) => {} });
 *   nn.idle();                 // gentle ambient pulses
 *   nn.scan();                 // a full left-to-right activation wave
 *   nn.setVerdict("fraud", .93); // light the output neuron
 */
(function (global) {
  "use strict";

  var LAYERS = [6, 9, 7, 10, 8, 5, 2];
  var LAYER_NAMES = [
    "Message tokens", "Anonymized", "Language", "MuRIL embedding",
    "Lexical + structured", "Classifier head", "Verdict"
  ];
  var NODE_COLORS = ["#38bdf8", "#38bdf8", "#a78bfa", "#22d3ee", "#fbbf24", "#f472b6", "#94a3b8"];

  function attach(canvas, opts) {
    opts = opts || {};
    var ctx = canvas.getContext("2d");
    var W = 0, H = 0, nodes = [], edges = [], pulses = [], raf = null;
    var verdict = null; // {label, probability}
    var stage = -1;
    var reduced = false;
    try {
      reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch (e) { /* ignore */ }

    function layout() {
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var rect = canvas.getBoundingClientRect();
      W = Math.max(320, rect.width); H = Math.max(180, rect.height);
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      nodes = []; edges = [];
      var padX = 46, top = 34, bottom = H - 30;
      LAYERS.forEach(function (count, li) {
        var x = padX + (W - padX * 2) * (LAYERS.length === 1 ? 0.5 : li / (LAYERS.length - 1));
        var layer = [];
        for (var i = 0; i < count; i++) {
          var y = count === 1 ? (top + bottom) / 2
            : top + (bottom - top) * (i / (count - 1));
          layer.push({ x: x, y: y, layer: li, glow: 0, r: li === LAYERS.length - 1 ? 13 : 8 });
        }
        nodes.push(layer);
      });
      for (var l = 0; l < nodes.length - 1; l++) {
        nodes[l].forEach(function (a) {
          nodes[l + 1].forEach(function (b) {
            edges.push({ a: a, b: b, w: Math.random() * 0.9 + 0.1 });
          });
        });
      }
    }

    function spawnPulse(fromLayer, strength) {
      if (fromLayer >= nodes.length - 1) return;
      var srcs = nodes[fromLayer];
      var a = srcs[Math.floor(Math.random() * srcs.length)];
      var dsts = nodes[fromLayer + 1];
      var b = dsts[Math.floor(Math.random() * dsts.length)];
      a.glow = Math.min(1, a.glow + 0.7);
      pulses.push({ a: a, b: b, t: 0, speed: 0.012 + Math.random() * 0.02, strength: strength || 0.8 });
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      // layer labels
      ctx.font = "600 10px system-ui, sans-serif";
      ctx.textAlign = "center";
      nodes.forEach(function (layer, li) {
        var active = stage === li;
        ctx.fillStyle = active ? "#e0f2fe" : "rgba(148,163,184,.75)";
        ctx.fillText(LAYER_NAMES[li].toUpperCase(), layer[0].x, 16);
      });
      // edges
      edges.forEach(function (e) {
        var hot = Math.max(e.a.glow, e.b.glow);
        ctx.strokeStyle = hot > 0.25
          ? "rgba(56,189,248," + (0.10 + hot * 0.35) + ")"
          : "rgba(100,116,139,.16)";
        ctx.lineWidth = 0.6 + e.w * 0.9;
        ctx.beginPath(); ctx.moveTo(e.a.x, e.a.y); ctx.lineTo(e.b.x, e.b.y); ctx.stroke();
      });
      // pulses
      pulses = pulses.filter(function (p) { return p.t <= 1; });
      pulses.forEach(function (p) {
        p.t += p.speed;
        if (p.t >= 1 && p.b) { p.b.glow = Math.min(1, p.b.glow + 0.55 * p.strength); }
        var x = p.a.x + (p.b.x - p.a.x) * p.t;
        var y = p.a.y + (p.b.y - p.a.y) * p.t;
        var grad = ctx.createRadialGradient(x, y, 0, x, y, 9);
        grad.addColorStop(0, "rgba(125,211,252,.95)");
        grad.addColorStop(1, "rgba(125,211,252,0)");
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#e0f2fe";
        ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI * 2); ctx.fill();
      });
      // nodes
      nodes.forEach(function (layer, li) {
        layer.forEach(function (n, ni) {
          n.glow *= 0.94;
          var isOut = li === LAYERS.length - 1;
          var base = NODE_COLORS[li];
          if (isOut && verdict) {
            base = ni === 0 ? "#f87171" : "#4ade80"; // 0 = FRAUD, 1 = SAFE
            n.glow = Math.max(n.glow,
              (ni === 0 && verdict.label !== "safe") || (ni === 1 && verdict.label === "safe")
                ? 0.9 : 0.15);
          }
          if (n.glow > 0.03) {
            var g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 3.2);
            g.addColorStop(0, "rgba(56,189,248,.45)");
            g.addColorStop(1, "rgba(56,189,248,0)");
            ctx.fillStyle = g;
            ctx.beginPath(); ctx.arc(n.x, n.y, n.r * 3.2, 0, Math.PI * 2); ctx.fill();
          }
          ctx.fillStyle = "#0b1220";
          ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fill();
          ctx.lineWidth = 1.6;
          ctx.strokeStyle = base;
          ctx.globalAlpha = 0.45 + Math.min(0.55, n.glow);
          ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.fillStyle = base;
          ctx.globalAlpha = 0.25 + Math.min(0.75, n.glow);
          ctx.beginPath(); ctx.arc(n.x, n.y, n.r * 0.45, 0, Math.PI * 2); ctx.fill();
          ctx.globalAlpha = 1;
          if (isOut) {
            ctx.font = "700 9px system-ui, sans-serif";
            ctx.fillStyle = ni === 0 ? "#fca5a5" : "#86efac";
            ctx.fillText(ni === 0 ? "FRAUD" : "SAFE", n.x, n.y + 3.5);
          }
        });
      });
      if (!reduced) raf = requestAnimationFrame(draw);
    }

    var idleTimer = null;
    function idle() {
      if (reduced) { draw(); return; }
      if (idleTimer) return;
      idleTimer = setInterval(function () {
        spawnPulse(Math.floor(Math.random() * (LAYERS.length - 1)), 0.5);
      }, 420);
    }

    function scan(onDone) {
      verdict = null;
      var li = 0;
      stage = 0;
      if (opts.onStage) opts.onStage(0);
      var iv = setInterval(function () {
        for (var k = 0; k < 14; k++) spawnPulse(li, 1);
        li++;
        stage = li;
        if (opts.onStage) opts.onStage(li);
        if (li >= LAYERS.length - 1) {
          clearInterval(iv);
          stage = -1;
          if (onDone) setTimeout(onDone, 350);
        }
      }, reduced ? 60 : 260);
    }

    function setVerdict(label, probability) {
      verdict = { label: label, probability: probability };
      var outLayer = nodes[nodes.length - 1];
      if (outLayer) outLayer.forEach(function (n) { n.glow = 1; });
      if (reduced) draw();
    }

    layout();
    draw();
    idle();
    window.addEventListener("resize", layout);
    return { scan: scan, setVerdict: setVerdict, idle: idle };
  }

  global.FraudNN = { attach: attach, LAYER_NAMES: LAYER_NAMES };
})(typeof window !== "undefined" ? window : globalThis);

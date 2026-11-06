# ruff: noqa: E501
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, results_dir

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tokenizer Scaling Calculator | TopoBPE</title>
<link rel="icon" href="data:,">
<style>
:root{--bg:#0b1020;--panel:#141b31;--ink:#e8edf7;--muted:#98a3ba;--accent:#5eead4;--warn:#fbbf24;--line:#26314d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:clamp(28px,4vw,44px);line-height:1.1;margin:0 0 8px}
p.lead{color:var(--muted);max-width:70ch}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:26px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}
label{display:block;color:var(--muted);font-size:13px;margin:12px 0 6px}
select,input[type=range],input[type=number]{width:100%}
select,input[type=number]{background:#0e1527;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:9px}
output{font-variant-numeric:tabular-nums}
.big{font-size:34px;font-weight:700;color:var(--accent);margin:8px 0 0}
.row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px dashed var(--line);padding:7px 0}
.row span:last-child{font-variant-numeric:tabular-nums;text-align:right}
.warn{color:var(--warn);font-size:13px;margin-top:10px}
small{color:var(--muted)}
svg{width:100%;height:auto;margin-top:12px}
footer{margin-top:34px;color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:16px}
a{color:var(--accent)}
code{background:#0e1527;padding:1px 5px;border-radius:4px}
</style>
</head>
<body>
<main>
<h1>Tokenizer scaling calculator</h1>
<p class="lead">Estimate bytes per token, token count and API cost from a fitted power law:
<code>bytes/token = a × vocab<sup>b</sup></code>. Coefficients are fit per corpus and tokenizer family on 8,192–32,768 tokens. Treat results beyond 32,768 as extrapolation, not measurement.</p>
<div class="grid">
<section class="panel">
<h2 style="margin:0 0 4px;font-size:18px">Inputs</h2>
<label for="corpus">Corpus / language</label>
<select id="corpus"></select>
<label for="family">Tokenizer family</label>
<select id="family"></select>
<label for="vocab">Vocabulary size: <output id="vocabOut"></output></label>
<input id="vocab" type="range" min="512" max="131072" step="512" value="32768">
<label for="bytes">Document size (bytes)</label>
<input id="bytes" type="number" min="1" step="1000" value="1000000">
<label for="price">Price (USD per 1M tokens)</label>
<input id="price" type="number" min="0" step="0.01" value="0.50">
<div class="warn" id="warn" hidden>You are outside the fitted range (8,192–32,768); this is an extrapolation.</div>
</section>
<section class="panel">
<h2 style="margin:0 0 4px;font-size:18px">Estimate</h2>
<div class="big" id="bpt">—</div>
<small>bytes per token (95% fit interval <span id="ci"></span>)</small>
<div class="row"><span>Tokens</span><span id="tokens">—</span></div>
<div class="row"><span>Cost at selected price</span><span id="cost">—</span></div>
<div class="row"><span>Fit R²</span><span id="r2">—</span></div>
<div class="row"><span>Fit residual σ (log)</span><span id="sigma">—</span></div>
<svg id="chart" viewBox="0 0 420 220" role="img" aria-label="Measured compression points and fitted power law"></svg>
</section>
</div>
<footer>
<p>Model: ordinary least squares on <code>log(bytes/token)</code> versus <code>log(vocab)</code> using six tokenizers per corpus. Measured points: 8,192, 16,384 and 32,768. 32k extrapolation from 8k/16k had MAPE 2.3% (BPE) and 1.9% (Unigram). A pooled cross-corpus model did <strong>not</strong> generalize (LOCO R² &lt; 0), so this calculator is corpus-specific by design.</p>
<p>Source: TopoBPE, <a href="docs/PROTOCOL.md">protocol</a> · <a href="index.html">project home</a> · <a href="preview.html">interactive preview</a></p>
</footer>
</main>
<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat();
function option(select, value, label) {
  const el = document.createElement('option');
  el.value = value; el.textContent = label; select.appendChild(el);
}
for (const corpus of Object.keys(DATA).sort()) option($('corpus'), corpus, corpus.replaceAll('_', ' '));
function syncFamilies() {
  const corpus = DATA[$('corpus').value];
  $('family').innerHTML = '';
  for (const family of Object.keys(corpus).sort()) option($('family'), family, family.toUpperCase());
}
function draw(entry, vocab) {
  const svg = $('chart');
  const w = 420, h = 220, pad = 34;
  const xs = entry.points.map(p => Math.log(p.vocab));
  const ys = entry.points.map(p => Math.log(p.compression));
  const minX = Math.min(...xs) - 0.4, maxX = Math.max(...xs) + 0.4;
  const minY = Math.min(...ys) - 0.1, maxY = Math.max(...ys) + 0.1;
  const sx = v => pad + (Math.log(v) - minX) / (maxX - minX) * (w - 2 * pad);
  const sy = v => h - pad - (Math.log(v) - minY) / (maxY - minY) * (h - 2 * pad);
  const line = [];
  for (let i = 0; i <= 60; i++) {
    const v = Math.exp(minX + (maxX - minX) * i / 60);
    line.push(`${sx(v).toFixed(1)},${sy(entry.a * Math.pow(v, entry.b)).toFixed(1)}`);
  }
  const dots = entry.points.map(p => `<circle cx="${sx(p.vocab).toFixed(1)}" cy="${sy(p.compression).toFixed(1)}" r="4" fill="#5eead4"/>`).join('');
  const marker = `<line x1="${sx(vocab).toFixed(1)}" y1="${pad}" x2="${sx(vocab).toFixed(1)}" y2="${h-pad}" stroke="#fbbf24" stroke-dasharray="4 4"/>`;
  const labels = entry.points.map(p => `<text x="${sx(p.vocab).toFixed(1)}" y="${h-12}" fill="#98a3ba" font-size="10" text-anchor="middle">${(p.vocab/1024).toFixed(0)}k</text>`).join('');
  svg.innerHTML = `<rect x="0" y="0" width="${w}" height="${h}" fill="none"/><polyline points="${line.join(' ')}" fill="none" stroke="#5eead4" stroke-width="2"/>${marker}${dots}${labels}<text x="${pad}" y="16" fill="#98a3ba" font-size="11">bytes/token (log scale)</text>`;
}
function update() {
  const corpus = $('corpus').value, family = $('family').value;
  const entry = DATA[corpus][family];
  const vocab = Number($('vocab').value);
  $('vocabOut').textContent = fmt.format(vocab);
  const bpt = entry.a * Math.pow(vocab, entry.b);
  const low = bpt * Math.exp(-1.96 * entry.sigma_log);
  const high = bpt * Math.exp(1.96 * entry.sigma_log);
  const tokens = Number($('bytes').value) / bpt;
  const cost = tokens / 1e6 * Number($('price').value);
  $('bpt').textContent = bpt.toFixed(3);
  $('ci').textContent = `${low.toFixed(2)}–${high.toFixed(2)}`;
  $('tokens').textContent = fmt.format(Math.round(tokens));
  $('cost').textContent = '$' + cost.toFixed(4);
  $('r2').textContent = entry.r2.toFixed(4);
  $('sigma').textContent = entry.sigma_log.toFixed(4);
  $('warn').hidden = vocab >= 8192 && vocab <= 32768;
  draw(entry, vocab);
}
$('corpus').addEventListener('change', () => { syncFamilies(); update(); });
$('family').addEventListener('change', update);
for (const id of ['vocab', 'bytes', 'price']) $(id).addEventListener('input', update);
syncFamilies(); update();
</script>
</body>
</html>
"""


def main() -> int:
    config = load_config("configs/default.yaml")
    out = results_dir(config)
    data = json.loads((out / "scaling_law.json").read_text())
    embedded: dict[str, dict[str, dict]] = {}
    for row in data["per_corpus"]:
        embedded.setdefault(row["corpus"], {})[row["family"]] = {
            "a": row["a"],
            "b": row["b"],
            "r2": row["r2"],
            "sigma_log": row["sigma_log"],
            "points": row["points"],
        }
    payload = json.dumps(embedded, separators=(",", ":"))
    Path("calculator.html").write_text(TEMPLATE.replace("__DATA__", payload))
    print(f"wrote calculator.html ({Path('calculator.html').stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

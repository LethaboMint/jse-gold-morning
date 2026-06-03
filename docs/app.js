const pct = (x) => {
  if (x == null || Number.isNaN(x)) return "—";
  const v = Number(x) * 100;
  const s = (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
  return s;
};

const retClass = (x) => {
  if (x == null || Number.isNaN(x) || Math.abs(x) < 1e-6) return "flat";
  return x > 0 ? "up" : "down";
};

const sigBadge = (s) => `<span class="sig ${s}">${s}</span>`;

async function loadSignals() {
  const res = await fetch(`signals.json?t=${Date.now()}`);
  if (!res.ok) throw new Error(`Could not load signals.json (${res.status})`);
  return res.json();
}

function renderMacro(data) {
  const el = document.getElementById("macro");
  const items = [
    { label: "Gold (t)", value: data.return_gold_t },
    { label: "GDX (t)", value: data.return_gdx_t },
    { label: "USD/ZAR (t)", value: data.return_zar_t },
  ];
  el.innerHTML = items
    .map(
      (i) => `
    <div class="card">
      <p class="card-label">${i.label}</p>
      <p class="card-value ${retClass(i.value)}">${pct(i.value)}</p>
    </div>`
    )
    .join("");
}

function renderSummary(signals) {
  const base = { LONG: 0, SHORT: 0, FLAT: 0 };
  const hi = { LONG: 0, SHORT: 0, FLAT: 0 };
  for (const r of signals) {
    base[r.signal] = (base[r.signal] || 0) + 1;
    hi[r.signal_high_conv] = (hi[r.signal_high_conv] || 0) + 1;
  }
  const actionable = signals.filter(
    (r) => r.signal_high_conv === "LONG" || r.signal_high_conv === "SHORT"
  );
  document.getElementById("summary").innerHTML = `
    <span class="chip">Base: <strong>${base.LONG}</strong> long · <strong>${base.SHORT}</strong> short · <strong>${base.FLAT}</strong> flat</span>
    <span class="chip">High conv.: <strong>${hi.LONG}</strong> long · <strong>${hi.SHORT}</strong> short · <strong>${hi.FLAT}</strong> flat</span>
    <span class="chip">Actionable: <strong>${actionable.length}</strong></span>
  `;
  return actionable;
}

function renderTable(signals) {
  const body = document.getElementById("miners-body");
  const order = ["HAR", "GFI", "ANG", "DRD", "PAN", "SSW"];
  const byMiner = Object.fromEntries(signals.map((r) => [r.miner, r]));
  body.innerHTML = order
    .map((m) => {
      const r = byMiner[m];
      if (!r) return "";
      const regime = r.regime_pass === "-" || !r.regime_pass ? "—" : r.regime_pass;
      return `<tr>
        <td><span class="miner-code">${m}</span></td>
        <td>${pct(r.pred_return_miner_t1)}</td>
        <td>${sigBadge(r.signal)}</td>
        <td>${sigBadge(r.signal_high_conv)}</td>
        <td>${regime}</td>
      </tr>`;
    })
    .join("");
}

function renderActive(actionable) {
  const panel = document.getElementById("active-panel");
  const list = document.getElementById("active-list");
  if (!actionable.length) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  list.innerHTML = actionable
    .map(
      (r) => `
    <li>
      ${sigBadge(r.signal_high_conv)}
      <span class="miner-code">${r.miner}</span>
      <span>Predicted ${pct(r.pred_return_miner_t1)}</span>
      <span class="muted">Regime: ${r.regime_pass}</span>
    </li>`
    )
    .join("");
}

function renderMeta(data) {
  const gen = data.generated_at_utc
    ? new Date(data.generated_at_utc).toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      })
    : "—";
  document.getElementById("meta").innerHTML = `
    <span class="meta-label">Signal date (features)</span>
    <strong>${data.signal_date || "—"}</strong>
    <span class="meta-label" style="margin-top:0.75rem">Last updated</span>
    <strong>${gen}</strong>
    <span class="meta-label" style="margin-top:0.5rem">Rules: ${data.rules_mode || "—"}</span>
  `;
}

async function main() {
  try {
    const data = await loadSignals();
    const signals = data.signals || [];
    renderMeta(data);
    renderMacro(data);
    const actionable = renderSummary(signals);
    renderTable(signals);
    renderActive(actionable);
  } catch (e) {
    document.querySelector(".page").insertAdjacentHTML(
      "afterbegin",
      `<div class="error"><strong>Could not load signals.</strong> ${e.message}</div>`
    );
  }
}

main();

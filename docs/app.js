const fmtPrice = (v, currency) => {
  if (v == null || Number.isNaN(v)) return "—";
  const n = Number(v);
  if (currency === "USD" && n > 100) return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

const fmtPct = (v) => {
  if (v == null || Number.isNaN(v)) return "—";
  const n = Number(v);
  return (n >= 0 ? "+" : "") + n.toFixed(2) + "%";
};

const pctClass = (v) => {
  if (v == null || Number.isNaN(v) || Math.abs(v) < 0.001) return "flat";
  return v > 0 ? "up" : "down";
};

const fmtPred = (x) => {
  if (x == null || Number.isNaN(x)) return "—";
  const v = Number(x) * 100;
  return (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
};

const sigBadge = (s) => `<span class="sig ${s}">${s}</span>`;

async function loadSignals() {
  const res = await fetch(`signals.json?t=${Date.now()}`);
  if (!res.ok) throw new Error(`Could not load signals.json (${res.status})`);
  return res.json();
}

function heroCard(item) {
  const chg = item.pct_change;
  return `
    <div class="card card-hero">
      <p class="card-label">${item.label} <span class="ticker">${item.ticker}</span></p>
      <p class="card-price">${fmtPrice(item.close, item.currency)} <span class="ccy">${item.currency || ""}</span></p>
      <p class="card-chg ${pctClass(chg)}">${fmtPct(chg)}</p>
      ${item.quote_date ? `<p class="card-date">Close · ${item.quote_date}</p>` : ""}
    </div>`;
}

function renderMacroUs(market) {
  const el = document.getElementById("macro-us");
  if (!market) {
    el.innerHTML = "<p class='muted'>No market data</p>";
    return;
  }
  el.innerHTML = [market.gold, market.gdx].filter(Boolean).map(heroCard).join("");
}

function renderMinerQuotes(market) {
  const el = document.getElementById("miners-quotes");
  const miners = market?.miners || [];
  el.innerHTML = miners
    .map(
      (m) => `
    <div class="card card-miner">
      <p class="card-label"><span class="miner-code">${m.miner}</span> <span class="ticker">${m.ticker}</span></p>
      <p class="card-price">${m.currency === "USD" ? "$" : "R "}${fmtPrice(m.close, m.currency)}</p>
      <p class="card-chg ${pctClass(m.pct_change)}">${fmtPct(m.pct_change)}</p>
      ${m.quote_date ? `<p class="card-date">${m.quote_date}</p>` : ""}
    </div>`
    )
    .join("");
}

function renderSummary(signals) {
  const hi = { LONG: 0, SHORT: 0, FLAT: 0 };
  for (const r of signals) hi[r.signal_high_conv] = (hi[r.signal_high_conv] || 0) + 1;
  const actionable = signals.filter(
    (r) => r.signal_high_conv === "LONG" || r.signal_high_conv === "SHORT"
  );
  document.getElementById("summary").innerHTML = `
    <span class="chip">High conv. long <strong>${hi.LONG}</strong></span>
    <span class="chip">High conv. short <strong>${hi.SHORT}</strong></span>
    <span class="chip">Flat / filtered <strong>${hi.FLAT}</strong></span>
    <span class="chip">Actionable <strong>${actionable.length}</strong></span>
  `;
  return actionable;
}

function renderForecast(signals) {
  const order = ["HAR", "GFI", "ANG", "DRD", "PAN", "SSW"];
  const by = Object.fromEntries(signals.map((r) => [r.miner, r]));
  document.getElementById("forecast-body").innerHTML = order
    .map((m) => {
      const r = by[m];
      if (!r) return "";
      const regime = r.regime_pass === "-" || !r.regime_pass ? "—" : r.regime_pass;
      return `<tr>
        <td><span class="miner-code">${m}</span></td>
        <td class="forecast-cell">${fmtPred(r.pred_return_miner_t1)}</td>
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
      <span>Forecast ${fmtPred(r.pred_return_miner_t1)}</span>
      <span class="muted">Last R ${fmtPrice(r.close, "ZAR")} (${fmtPct(r.pct_change)})</span>
    </li>`
    )
    .join("");
}

function renderMeta(data) {
  const gen = data.generated_at_utc
    ? new Date(data.generated_at_utc).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
    : "—";
  document.getElementById("meta").innerHTML = `
    <span class="meta-label">US signal date</span>
    <strong>${data.signal_date || "—"}</strong>
    <span class="meta-label" style="margin-top:0.75rem">Updated</span>
    <strong>${gen}</strong>
    <span class="meta-label" style="margin-top:0.5rem">${data.data_source || "yahoo_finance"}</span>
  `;
}

async function main() {
  try {
    const data = await loadSignals();
    const signals = data.signals || [];
    const market = data.market || {
      gold: {
        label: "Gold",
        ticker: "GC=F",
        currency: "USD",
        close: null,
        pct_change: data.return_gold_t != null ? data.return_gold_t * 100 : null,
      },
      gdx: {
        label: "GDX",
        ticker: "GDX",
        currency: "USD",
        close: null,
        pct_change: data.return_gdx_t != null ? data.return_gdx_t * 100 : null,
      },
      miners: signals.map((s) => ({
        miner: s.miner,
        ticker: s.yahoo_ticker || s.miner,
        close: s.close,
        pct_change: s.pct_change,
      })),
    };

    renderMeta(data);
    renderMacroUs(market);
    renderMinerQuotes(market);
    const actionable = renderSummary(signals);
    renderForecast(signals);
    renderActive(actionable);
  } catch (e) {
    document.querySelector(".page").insertAdjacentHTML(
      "afterbegin",
      `<div class="error"><strong>Could not load briefing.</strong> ${e.message}</div>`
    );
  }
}

main();

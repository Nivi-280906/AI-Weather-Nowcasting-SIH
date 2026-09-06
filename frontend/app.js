// ---------- Map (free, colorful, no API key) ----------
const API_BASE = "";
const WS_URL =
  location.protocol === "https:"
    ? "wss://ai-weather-nowcasting-sih.onrender.com/ws/live"
    : "ws://localhost:8000/ws/live";
const map = L.map("map", { zoomControl: true }).setView([22.5, 79.0], 5);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
  maxZoom: 18,
}).addTo(map);

const markers = {};        // station_id -> L.CircleMarker
const stationsById = {};   // station_id -> station record
let selectedStationId = null;
let detailsOpen = false;

function riskLevel(p) {
  if (p >= 0.70) return { key: "severe", color: "#dc2626" };
  if (p >= 0.50) return { key: "warning", color: "#ea580c" };
  if (p >= 0.30) return { key: "watch", color: "#ca8a04" };
  return { key: "low", color: "#16a34a" };
}

function fmtPct(p) { return `${Math.round(p * 100)}%`; }
function fmtDateTime(d) {
  return d.toLocaleString(undefined, {
    weekday: "short", day: "2-digit", month: "short",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}
function fmtDateTimeShort(d) {
  return d.toLocaleString(undefined, {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
  });
}

async function apiFetch(path, opts = {}) {
  return fetch(`${API_BASE}${path}`, opts);
}

async function loadStations() {
  const res = await fetch(`${API_BASE}/api/stations`); // public endpoint
  const stations = await res.json();
  stations.forEach((s) => {
    stationsById[s.id] = s;
    const marker = L.circleMarker([s.lat, s.lon], {
      radius: 8,
      color: "#16a34a",
      fillColor: "#16a34a",
      fillOpacity: 0.85,
      weight: 2,
    }).addTo(map);
    marker.bindTooltip(s.name, { direction: "top" });
    marker.on("click", () => selectStation(s.id));
    markers[s.id] = marker;
  });
}

function updateMarker(stationId, data) {
  const marker = markers[stationId];
  if (!marker) return;
  const lvl = riskLevel(data.flash_flood_prob);
  marker.setStyle({ color: lvl.color, fillColor: lvl.color, radius: 7 + data.flash_flood_prob * 10 });
}

function renderMetricBar(label, value, colorVar) {
  return `
    <div class="metric-row">
      <div class="metric-name">${label}</div>
      <div class="metric-bar-wrap"><div class="metric-bar" style="width:${Math.round(value*100)}%; background:${colorVar}"></div></div>
      <div class="metric-val">${fmtPct(value)}</div>
    </div>`;
}

function renderDrivers(attributions, task) {
  if (!attributions) return "";
  const entries = Object.entries(attributions)
    .map(([name, vals]) => [name, vals[task]])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);
  return `<ul class="driver-list">${entries
    .map(([name, val]) => `<li><span>${name}</span><span>${val >= 0 ? "+" : ""}${(val*100).toFixed(1)}%</span></li>`)
    .join("")}</ul>`;
}

function renderFeatureTable(features) {
  if (!features) return "";
  const rows = Object.entries(features)
    .map(([name, val]) => `<tr><td>${name}</td><td>${Number(val).toFixed(3)}</td></tr>`)
    .join("");
  return `<table class="feature-table">${rows}</table>`;
}

async function selectStation(stationId) {
  selectedStationId = stationId;
  detailsOpen = false;
  const station = stationsById[stationId];
  const panel = document.getElementById("station-detail");
  panel.innerHTML = `<h2>${station.name}</h2><p class="muted">Fetching fresh nowcast…</p>`;

  const res = await apiFetch(`/api/predict/${stationId}`);
  const pred = await res.json();
  updateMarker(stationId, pred);
  renderStationDetail(station, pred);
}

function renderStationDetail(station, pred) {
  const panel = document.getElementById("station-detail");
  const lvlFlood = riskLevel(pred.flash_flood_prob);

  const calculatedAt = new Date(pred.issued_at + "Z"); // server sends UTC-naive ISO, treat as UTC
  const eventStart = new Date(calculatedAt.getTime() + pred.valid_from_hr * 3600 * 1000);
  const eventEnd = new Date(calculatedAt.getTime() + pred.valid_to_hr * 3600 * 1000);

  panel.innerHTML = `
    <h2>${station.name} <span class="muted" style="font-weight:400">— ${station.district}, ${station.state}</span></h2>

    <div class="timebox">
      <div class="row"><span class="label">Calculated at</span><span class="val">${fmtDateTime(calculatedAt)}</span></div>
      <div class="row"><span class="label">Expected event window</span><span class="val">${fmtDateTimeShort(eventStart)} - ${fmtDateTimeShort(eventEnd)}</span></div>
    </div>

    ${renderMetricBar("Thunderstorm", pred.thunderstorm_prob, "#ea580c")}
    ${renderMetricBar("Cloudburst", pred.cloudburst_prob, "#2563eb")}
    ${renderMetricBar("Flash flood (incl. upstream)", pred.flash_flood_prob, lvlFlood.color)}

    <button class="details-toggle" id="details-toggle">Show full prediction details</button>
    <div class="details-body" id="details-body">
      <p class="muted">Station: elevation ${station.elevation_m}m, slope ${station.slope_deg} deg, stream order ${station.drainage_order}.</p>
      <p class="muted">Model uncertainty (MC-Dropout, +/-1 sigma): thunderstorm ${fmtPct(pred.thunderstorm_uncertainty)}, cloudburst ${fmtPct(pred.cloudburst_uncertainty)}, flash flood ${fmtPct(pred.flash_flood_uncertainty)}.</p>
      <p class="muted">Downstream-propagated component: <strong>${fmtPct(pred.propagated_flood_risk)}</strong> (DEM flow-routing novelty module).</p>

      <h2 style="margin-top:12px">Why this alert - top drivers</h2>
      <p class="muted" style="margin-bottom:2px">Cloudburst risk:</p>
      ${renderDrivers(pred.attributions, "cloudburst")}
      <p class="muted" style="margin:8px 0 2px">Flash-flood risk:</p>
      ${renderDrivers(pred.attributions, "flash_flood")}

      <h2 style="margin-top:12px">Raw atmospheric signature (last frame, peak values)</h2>
      ${renderFeatureTable(pred.features)}
    </div>
  `;

  document.getElementById("details-toggle").addEventListener("click", () => {
    detailsOpen = !detailsOpen;
    const body = document.getElementById("details-body");
    const btn = document.getElementById("details-toggle");
    body.classList.toggle("open", detailsOpen);
    btn.textContent = detailsOpen ? "Hide full prediction details" : "Show full prediction details";
  });
}

// ---------- Alerts + notifications ----------
let seenAlertIds = new Set();
let unreadCount = 0;

function updateBellBadge() {
  const badge = document.getElementById("bell-badge");
  if (unreadCount > 0) {
    badge.style.display = "inline-block";
    badge.textContent = unreadCount > 99 ? "99+" : unreadCount;
  } else {
    badge.style.display = "none";
  }
}

function renderAlerts(alerts) {
  const feed = document.getElementById("alert-feed");
  const dropdown = document.getElementById("bell-dropdown");

  if (!alerts.length) {
    feed.innerHTML = `<p class="muted">No active alerts. Monitoring continues in the background.</p>`;
    dropdown.innerHTML = `<p class="muted" style="padding:8px">No notifications yet.</p>`;
    return;
  }

  const rowHtml = (a) => {
    const created = new Date(a.created_at + "Z");
    const start = new Date(created.getTime() + a.valid_from_hr * 3600 * 1000);
    const end = new Date(created.getTime() + a.valid_to_hr * 3600 * 1000);
    return `
      <div class="alert-item ${a.severity}">
        <span class="badge ${a.severity}">${a.severity}</span>
        <div>${a.message}</div>
        <div class="meta">Calculated ${fmtDateTime(created)} - Expected ${fmtDateTimeShort(start)} to ${fmtDateTimeShort(end)}</div>
      </div>`;
  };

  feed.innerHTML = alerts.map(rowHtml).join("");
  dropdown.innerHTML = alerts.slice(0, 10).map(rowHtml).join("");

  alerts.forEach((a) => {
    if (!seenAlertIds.has(a.id)) {
      seenAlertIds.add(a.id);
      unreadCount += 1;
      if (a.severity === "SEVERE" && "Notification" in window && Notification.permission === "granted") {
        new Notification("Severe weather alert", { body: a.message });
      }
    }
  });
  updateBellBadge();
}

async function pollAlerts() {
  try {
    const res = await apiFetch(`/api/alerts?limit=25`);
    renderAlerts(await res.json());
  } catch (e) { /* ignore transient errors */ }
}

document.getElementById("bell-btn").addEventListener("click", () => {
  const dropdown = document.getElementById("bell-dropdown");
  dropdown.classList.toggle("open");
  if (dropdown.classList.contains("open")) {
    unreadCount = 0;
    updateBellBadge();
  }
});
document.addEventListener("click", (e) => {
  const wrap = document.querySelector(".bell-wrap");
  if (wrap && !wrap.contains(e.target)) {
    document.getElementById("bell-dropdown").classList.remove("open");
  }
});

if ("Notification" in window && Notification.permission === "default") {
  Notification.requestPermission();
}

// ---------- Live socket ----------
function connectSocket() {
  const statusEl = document.getElementById("conn-status");
  const ws = new WebSocket(WS_URL);

  ws.onopen = () => { statusEl.textContent = "Live - streaming nowcasts"; statusEl.classList.add("live"); };
  ws.onclose = () => {
    statusEl.textContent = "Reconnecting..."; statusEl.classList.remove("live");
    setTimeout(connectSocket, 3000);
  };
  ws.onerror = () => ws.close();

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === "nowcast_update") {
      msg.data.forEach((d) => updateMarker(d.station_id, d));
      pollAlerts();
    }
  };
}

(async function init() {
  await loadStations();
  pollAlerts();
  connectSocket();
})();

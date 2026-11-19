// URBIS app: review, view and export a city's slum map. Talks to scripts/urbis_app.py.
const map = L.map("map", { preferCanvas: true, zoomControl: true }).setView([0, 0], 2);
const imagery = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { maxZoom: 20, maxNativeZoom: 19, attribution: "Imagery Esri · Buildings Overture Maps (Google, Microsoft, OSM)" }).addTo(map);
const L_ = { outlines: L.layerGroup().addTo(map), houses: L.layerGroup().addTo(map), numbers: L.layerGroup().addTo(map),
             reviewed: L.layerGroup(), guess: L.layerGroup(), current: L.layerGroup().addTo(map), imagery };
let C = null, S = null, H = null, Q = [], qi = 0, houseLayer = null, poll = null;

// tabs
$$(".tabs button").forEach(b => b.onclick = () => tab(b.dataset.tab));
function showGuess() { if ($('[data-layer="guess"]').checked) L_.guess.addTo(map); }
function tab(t) {
  $$(".tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === t));
  $$("[data-pane]").forEach(p => p.hidden = p.dataset.pane !== t);
  if (t === "review" && !Q.length) loadQueue();
  if (t !== "review") { L_.current.clearLayers(); showGuess(); } else map.removeLayer(L_.guess);   // the reviewer judges plain imagery
  setTimeout(() => map.invalidateSize(), 0);
}
$$("[data-layer]").forEach(i => i.onchange = () => { const l = L_[i.dataset.layer]; i.checked ? l.addTo(map) : map.removeLayer(l); if (i.dataset.layer === "guess" && i.checked) loadGuess(); drawNumbers(); });
async function loadGuess() {
  const g = await api(`/city/${C}/guess`); L_.guess.clearLayers();
  L.geoJSON(g, { style: f => ({ color: "#000", weight: .5, fillColor: "#ffd60a", fillOpacity: .25 + 10 * (f.properties.rank - .95) }),
    onEachFeature: (f, l) => l.bindTooltip(`Model rank: top ${(100 * (1 - f.properties.rank)).toFixed(1)} % of the city`, { sticky: true }) }).addTo(L_.guess);
}

// city: set-up progress until ready, then the workspace
userMenu(); boot(new URLSearchParams(location.search).get("city"));
function delButton(b) {          // two clicks: the second confirms (the viewer has no confirm dialogs)
  b.onclick = async () => {
    if (!b.dataset.armed) { b.dataset.armed = 1; b.textContent = "Click again to delete"; setTimeout(() => { delete b.dataset.armed; b.textContent = b.dataset.label; }, 4000); return; }
    b.disabled = true; await api(`/city/${encodeURIComponent(new URLSearchParams(location.search).get("city"))}/delete`, { method: "POST" }); location.href = "/cities.html";
  };
  b.dataset.label = b.textContent;
}
delButton($("#bdel")); delButton($("#bcancel"));
async function boot(c) {
  let st;
  try { st = await api(`/city/${encodeURIComponent(c)}/status`); } catch (x) { $("#cname").textContent = "City not found"; return; }
  $("#cname").textContent = st.name; document.title = st.name + " · URBIS"; $("#simbanner").hidden = !/ · demo$/.test(st.name);
  const s = st.setup || {};
  if (s.step !== "ready") {
    $("#setup").hidden = false; $("main").hidden = true; $(".tabs").hidden = true;
    $("#cstate").textContent = s.step === "error" ? "set-up failed" : "setting up";
    $("#smsg").textContent = s.msg || "Waiting to start"; $("#sbar").style.width = (s.total ? 100 * s.done / s.total : 2) + "%";
    $("#sname").textContent = s.step === "error" ? `${st.name}: set-up failed` : `Setting up ${st.name}`;
    const order = ["queued", "grid", "buildings", "score", "ready"], k = order.indexOf(s.step);
    $$("#slist li").forEach(li => li.className = order.indexOf(li.dataset.s) < k ? "done" : li.dataset.s === s.step ? "now" : "");
    if (st.bbox && $("#simg").hidden) { $("#simg").src = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=${st.bbox.join(",")}&bboxSR=4326&imageSR=3857&size=520,520&format=jpg&f=image`; $("#simg").hidden = false; }
    if (s.step !== "error") setTimeout(() => boot(c), 4000); return;
  }
  $("#setup").hidden = true; $("main").hidden = false; $(".tabs").hidden = false; load(c, st);
}

async function load(c, st0) {
  C = c; Q = []; qi = 0;
  [L_.outlines, L_.houses, L_.numbers, L_.reviewed, L_.guess, L_.current].forEach(l => l.clearLayers());
  $("#cstate").textContent = "loading";
  const [st, s, h] = await Promise.all([st0 || api(`/city/${c}/status`), api(`/city/${c}/settlements`), api(`/city/${c}/houses`)]);
  S = s; H = h; map.invalidateSize(); drawSettlements(); drawHouses(); status(st); files(st.built); loadReviewed();
  const g = $('[data-layer="guess"]');   // no map yet: show the model's first guess so the city is never blank
  if (!S.features.length && !g.checked) { g.checked = true; g.onchange(); }
  if (S.features.length) map.fitBounds(L.geoJSON(S).getBounds(), { padding: [20, 20] });
  else if (st.bbox) map.fitBounds([[st.bbox[1], st.bbox[0]], [st.bbox[3], st.bbox[2]]]);
}

function status(s) {
  $("#cstate").textContent = s.stop ? "map complete" : s.checked ? "in review" : "not started";
  $("#tcount").textContent = fmt(s.checked);
  $("#stats").innerHTML = `<div><b>${fmt(s.settlements)}</b><span>settlements</span></div><div><b>${fmt(s.area_ha)}</b><span>hectares of slum</span></div><div><b>${s.buildings ? fmt(s.buildings) + "+" : "0"}</b><span>houses</span></div>`;
  $("#rstatus").innerHTML = !s.checked ? `The model has ranked all ${fmt(s.built_boxes)} built-up 100 m boxes; its first guess is shown in yellow on the map. Open <b>Review</b> and answer slum or not slum, most likely first. After 10 answers the settlement map draws itself.` : `${fmt(s.checked)} boxes checked (${(100 * s.share_checked).toFixed(1)} % of built-up boxes), ${fmt(s.yes)} slum.<br>`
    + (s.stop ? "<b>Review complete.</b> The map can be exported." : "Keep reviewing in Review; URBIS tells you when the map is complete.");
  $("#rprog").innerHTML = `${fmt(s.checked)} checked · ${fmt(s.yes)} slum` + (s.stop ? " · <b>review complete</b>" : "");
  $("#rbar").style.width = Math.min(100, 100 * s.share_checked / 0.1) + "%";
  $("#rbar").title = "share of the 10 % review budget used";
  job(s.job);
  $("#hsrc").textContent = s.houses_source === "URBIS roof model"
    ? "Houses drawn by the URBIS roof model on 0.3 m imagery, one outline per roof."
    : "Houses from building footprint maps (Google, Microsoft, OpenStreetMap). They merge tightly packed roofs, so read the counts as a minimum.";
  $("#hsrc").textContent += " Zoom in to see each house; zoom in further for house numbers.";
}

function drawSettlements() {
  L_.outlines.clearLayers();
  const style = w => ({ color: "#000", weight: w + 2, fill: false });
  L.geoJSON(S, { style: () => style(2), interactive: false }).addTo(L_.outlines);
  L.geoJSON(S, { style: () => ({ color: "#00e5ff", weight: 2.5, fillColor: "#00c8ff", fillOpacity: .18 }),
    onEachFeature: (f, l) => { l.on("click", () => pick(f)); l.bindTooltip(`Settlement ${f.properties.settlement}`, { sticky: true }); } }).addTo(L_.outlines);
  const tb = $("#stable tbody"); $("#scount").textContent = S.features.length; $("#sempty").hidden = !!S.features.length;
  tb.innerHTML = S.features.map(f => { const p = f.properties; return `<tr data-s="${p.settlement}"><td>${p.settlement}</td><td>${p.area_ha.toFixed(1)}</td><td>${fmt(p.buildings)}</td><td>${p.buildings_per_ha.toFixed(0)}</td></tr>`; }).join("");
  tb.querySelectorAll("tr").forEach(r => r.onclick = () => { const f = S.features.find(f => f.properties.settlement == r.dataset.s); pick(f); map.fitBounds(L.geoJSON(f).getBounds(), { padding: [30, 30] }); });
}

function pick(f) {
  const p = f.properties;
  $$("#stable tr").forEach(r => r.classList.toggle("on", r.dataset.s == p.settlement));
  $("#sel").innerHTML = `<b>Settlement ${p.settlement}</b><br>${p.area_ha.toFixed(1)} ha · ${fmt(p.buildings)}+ houses · ${p.buildings_per_ha.toFixed(0)} per ha<br>`
    + `${p.confirmed_boxes} confirmed boxes · <span class="mono">${p.lat}, ${p.lon}</span><br><a href="${p.google_maps}" target="_blank" rel="noopener">Open in Google Maps</a>`;
}

function drawHouses() {
  L_.houses.clearLayers();
  houseLayer = L.geoJSON(H, { style: { color: "#fff", weight: 1, fillColor: "#000", fillOpacity: .15 },
    onEachFeature: (f, l) => l.on("click", () => { const p = f.properties;
      $("#sel").innerHTML = `<b>House ${p.house}</b> in settlement ${p.settlement}<br>${p.area_m2} m² footprint · source ${p.source}`; }) });
  toggleHouses();
}
function toggleHouses() {
  if (!houseLayer) return;
  const on = map.getZoom() >= 16 && $('[data-layer="houses"]').checked;
  if (on && !L_.houses.hasLayer(houseLayer)) L_.houses.addLayer(houseLayer);
  if (!on && L_.houses.hasLayer(houseLayer)) L_.houses.removeLayer(houseLayer);
  drawNumbers();
}
function drawNumbers() {
  L_.numbers.clearLayers();
  if (!H || map.getZoom() < 19 || !$('[data-layer="numbers"]').checked) return;
  const b = map.getBounds(); let n = 0;
  for (const f of H.features) {
    const c = f.geometry.coordinates[0][0] && f.geometry.type === "Polygon" ? f.geometry.coordinates[0] : f.geometry.coordinates[0][0];
    let x = 0, y = 0; c.forEach(p => { x += p[0]; y += p[1]; }); x /= c.length; y /= c.length;
    if (!b.contains([y, x])) continue;
    L.marker([y, x], { interactive: false, icon: L.divIcon({ className: "", html: `<span class="hn">${f.properties.house}</span>`, iconSize: null }) }).addTo(L_.numbers);
    if (++n > 1500) break;
  }
}
map.on("zoomend moveend", toggleHouses);

async function loadReviewed() {
  const r = await api(`/city/${C}/reviewed`); L_.reviewed.clearLayers();
  L.geoJSON(r, { style: f => f.properties.answer === "yes" ? { color: "#fff", weight: 2, dashArray: "4 3", fill: false } : { color: "#000", weight: 1, fill: false }, interactive: false }).addTo(L_.reviewed);
}

// review
async function loadQueue() { Q = (await api(`/city/${C}/queue?n=40`)).features; qi = 0; show(); }
function show() {
  L_.current.clearLayers();
  if (qi >= Q.length) { $("#qpos").textContent = "—"; $("#qid").textContent = ""; if (Q.length) loadQueue(); return; }
  const f = Q[qi], b = L.geoJSON(f).getBounds(), c = b.getCenter();
  L.geoJSON(f, { style: { color: "#000", weight: 5, fill: false }, interactive: false }).addTo(L_.current);
  L.geoJSON(f, { style: { color: "#fff", weight: 2, fill: false }, interactive: false }).addTo(L_.current);
  map.fitBounds(b.pad(2.5), { maxZoom: 18 });
  $("#qmodel").textContent = f.properties.rank != null ? `Model: top ${Math.max(.1, 100 * (1 - f.properties.rank)).toFixed(1)} % most slum-like boxes in this city.` : "";
  $("#qpos").textContent = `${qi + 1} of ${Q.length}`; $("#qid").textContent = "id " + f.properties.box;
  $("#gmaps").href = `https://www.google.com/maps/@${c.lat.toFixed(6)},${c.lng.toFixed(6)},120m/data=!3m1!1e3`;
}
async function answer(a) {
  if (qi >= Q.length) return;
  const f = Q[qi]; qi++; show();
  if (a === "skip") return;
  await api(`/city/${C}/answer`, { method: "POST", body: JSON.stringify({ box: f.properties.box, answer: a }) });
  status(await api(`/city/${C}/status`)); if ($('[data-layer="reviewed"]').checked) loadReviewed();
}
$("#byes").onclick = () => answer("yes"); $("#bno").onclick = () => answer("no"); $("#bskip").onclick = () => answer("skip");
document.addEventListener("keydown", e => {
  if ($('[data-pane="review"]').hidden || e.target.tagName === "SELECT" || e.metaKey || e.ctrlKey) return;
  const k = e.key.toLowerCase();
  if (k === "y") answer("yes"); else if (k === "n") answer("no"); else if (k === "s") answer("skip"); else if (k === "z" && Q[qi]) map.fitBounds(L.geoJSON(Q[qi]).getBounds().pad(1), { maxZoom: 19 });
});
$("#brebuild").onclick = async () => { await api(`/city/${C}/rebuild`, { method: "POST" }); job("running"); };
function job(j) {
  $("#jstate").textContent = j ? (j === "running" ? "Rebuilding…" : j === "done" ? "Rebuilt." : j) : "";
  $("#brebuild").disabled = j === "running"; clearInterval(poll);
  if (j === "running") poll = setInterval(async () => { const s = await api(`/city/${C}/status`); if (s.job !== "running") { clearInterval(poll); load(C); } else job("running"); }, 5000);
}

// export
const FILES = [
  ["geojson", "GeoJSON", "Settlements and houses in one file. Web maps, QGIS, Python."],
  ["kml", "KML", "Settlement boundary lines for Google Earth."],
  ["gpkg", "GeoPackage", "Two layers, settlements and houses. QGIS, ArcGIS."],
  ["shp", "Shapefile (zip)", "Settlements and houses as separate shapefiles. ArcGIS, older GIS."],
  ["fgb", "FlatGeobuf (zip)", "Fast binary vector format for large house layers."],
  ["csv", "Settlements CSV", "One row per settlement: area, houses, density, location. Excel."],
  ["houses_csv", "Houses CSV", "One row per house with its footprint as WKT. Excel, databases."],
  ["tif", "GeoTIFF, 10 m", "Raster: pixel value = settlement number. Remote sensing tools."],
  ["report", "City report (HTML)", "Printable summary with a table of every settlement."],
];
function files(built) {
  $("#files").innerHTML = (built ? "" : `<li class="muted">Files are ready once the settlement map is drawn: answer 10 boxes in Review.</li>`)
    + FILES.map(([k, n, d]) => `<li><div><b>${n}</b><span class="muted">${d}</span></div>${built ? `<a class="btn" href="/api/city/${C}/export/${k}">Download</a>` : `<button class="btn" disabled>Download</button>`}</li>`).join("");
}

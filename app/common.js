// Shared by all URBIS pages: API calls (redirect to sign-in on 401), header user menu.
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const fmt = n => Number(n).toLocaleString("en");
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
async function api(p, o) {
  const r = await fetch("/api" + p, o && o.body ? { headers: { "Content-Type": "application/json" }, ...o } : o);
  if (r.status === 401 && !/^\/(login|signup)$/.test(p)) { location.href = "/login.html?next=" + encodeURIComponent(location.pathname + location.search); throw new Error("sign in"); }
  const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j;
}
async function userMenu() {
  const me = await api("/me"), el = $("#user");
  if (el) el.innerHTML = `<span class="muted-inv">${esc(me.user)}</span><button class="linkbtn" id="signout">Sign out</button>`;
  $("#signout") && ($("#signout").onclick = async () => { await api("/logout", { method: "POST" }); location.href = "/"; });
  return me.user;
}

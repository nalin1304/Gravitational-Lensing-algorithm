/* ================================================================
   App Router + Shared Utilities
   ================================================================ */

// ---- Auth State ----
const AUTH_KEY = "lensing_api_access_token";
const REFRESH_KEY = "lensing_api_refresh_token";

function getToken() { return localStorage.getItem(AUTH_KEY) || ""; }
function setToken(t) { localStorage.setItem(AUTH_KEY, t); updateUserDisplay(); }
function getRefreshToken() { return localStorage.getItem(REFRESH_KEY) || ""; }
function setRefreshToken(t) { localStorage.setItem(REFRESH_KEY, t); }
function clearAuth() { localStorage.removeItem(AUTH_KEY); localStorage.removeItem(REFRESH_KEY); updateUserDisplay(); }

function updateUserDisplay() {
  const el = document.getElementById("topbarUser");
  if (el) el.textContent = getToken() ? "● Authenticated" : "";
}

// ---- API Helpers ----

async function api(path, { method = "GET", body = null, auth = true } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (auth && getToken()) headers["Authorization"] = `Bearer ${getToken()}`;
  const opts = { method, headers };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    if (resp.status === 401 && auth) {
      // Try refresh
      const refreshToken = getRefreshToken();
      if (refreshToken) {
        const refreshResp = await fetch('/api/v1/auth/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken })
        });
        if (refreshResp.ok) {
          const refreshData = await refreshResp.json();
          setToken(refreshData.access_token);
          if (refreshData.refresh_token) setRefreshToken(refreshData.refresh_token);
          // Retry original request with new token
          const retryResp = await fetch(path, {
            method,
            headers: { ...headers, Authorization: `Bearer ${refreshData.access_token}` },
            body: body ? JSON.stringify(body) : undefined
          });
          if (!retryResp.ok) throw new Error(`API ${retryResp.status}`);
          if (retryResp.status === 204 || retryResp.headers.get('content-length') === '0') return null;
          return retryResp.json();
        }
      }
    }
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || err.error || resp.statusText);
  }
  if (resp.status === 204 || resp.headers.get('content-length') === '0') {
    return null;
  }
  return resp.json();
}

function fmtSci(v, d = 3) {
  const n = Number(v);
  return isNaN(n) ? "-" : (Math.abs(n) > 1e4 || (Math.abs(n) < 0.01 && n !== 0)) ? n.toExponential(d) : n.toFixed(d);
}

function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

// ---- Toast ----

let toastTimeout = null;
function toast(msg, type = "info") {
  let el = document.getElementById("toast");
  if (!el) { el = document.createElement("div"); el.id = "toast"; document.body.appendChild(el); }
  el.className = `toast toast-${type} show`;
  el.textContent = msg;
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => el.classList.remove("show"), 3500);
}

// ---- Loading ----

function showLoading(msg = "Loading...") {
  const ol = document.getElementById("loadingOverlay");
  const lt = document.getElementById("loadingText");
  if (ol) ol.classList.remove("hidden");
  if (lt) lt.textContent = msg;
}

function hideLoading() {
  const ol = document.getElementById("loadingOverlay");
  if (ol) ol.classList.add("hidden");
}

// ---- Health Check ----

async function checkHealth() {
  const dot = document.querySelector(".status-dot");
  const txt = document.querySelector(".status-text");
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);
    const resp = await fetch("/health", { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!resp.ok) throw new Error(resp.statusText);
    const h = await resp.json();
    if (dot) { dot.classList.add("online"); dot.classList.remove("offline"); }
    if (txt) txt.textContent = `Online · v${h.version}`;
    return h;
  } catch {
    // Only show offline if we're currently not marked online (prevent flicker)
    if (dot && !dot.classList.contains("online")) {
      dot.classList.add("offline");
    }
    if (txt && txt.textContent === "") txt.textContent = "Offline";
    return null;
  }
}

// ---- Router ----

const routes = {};
let currentPage = null;
const ASSET_VERSION = "2026-03-14-redesign";

function registerPage(name, mod) { routes[name] = mod; }

async function navigate() {
  const hash = location.hash || "#/";
  const path = hash.replace("#", "") || "/";
  const pageName = path === "/" ? "dashboard" : path.replace("/", "");

  // Update nav
  document.querySelectorAll(".nav-link").forEach(l => {
    l.classList.toggle("active", l.dataset.page === pageName);
  });

  const titles = {
    dashboard: "Dashboard", workbench: "Workbench", validation: "Validation",
    analyses: "Analyses", account: "Account", api: "API Explorer",
    survey: "Stage IV Survey", rigor: "Statistical Rigor", "pi-sbi": "PI-SBI",
    inference: "NUTS-HMC Inference", lensing: "Lensing Analysis"
  };
  const subtitles = {
    dashboard: "System overview & health",
    workbench: "NFW convergence maps & PINN inference",
    validation: "SLACS, calibration & ablation benchmarks",
    analyses: "Manage saved analyses & results",
    account: "Authentication & API keys",
    api: "Interactive OpenAPI request builder",
    survey: "Stage IV survey tools & ePSF",
    rigor: "Publication-grade statistical checks",
    "pi-sbi": "Physics-informed simulation-based inference",
    inference: "Differentiable NUTS-HMC sampler",
    lensing: "Critical curves, caustics & image solver"
  };
  document.getElementById("pageTitle").textContent = titles[pageName] || "Dashboard";
  const subEl = document.getElementById("pageSubtitle");
  if (subEl) subEl.textContent = subtitles[pageName] || "";

  // Cleanup previous page (e.g. purge Plotly plots)
  if (currentPage && routes[currentPage]?.cleanup) {
    try { routes[currentPage].cleanup(); } catch (_) {}
  }

  const app = document.getElementById("app");
  const mod = routes[pageName];
  if (!mod) { app.innerHTML = `<div class="empty-state"><p>Page not found</p></div>`; currentPage = null; return; }

  currentPage = pageName;
  app.style.opacity = "0";
  app.innerHTML = mod.render();
  requestAnimationFrame(() => { app.style.opacity = "1"; });
  if (mod.init) {
    try { await mod.init(); } catch (e) { console.error(`[${pageName}] init error:`, e); }
  }
}

// ---- Mobile Menu ----

function setupMenu() {
  const toggle = document.getElementById("menuToggle");
  const sidebar = document.getElementById("sidebar");
  if (toggle) toggle.addEventListener("click", () => sidebar.classList.toggle("open"));
  // Close on nav click (mobile)
  document.querySelectorAll(".nav-link").forEach(l => {
    l.addEventListener("click", () => sidebar.classList.remove("open"));
  });
}

// ---- Boot ----

async function boot() {
  // Load page modules
  const pageEntries = [
    ["dashboard", "dashboard.js"],
    ["workbench", "workbench.js"],
    ["validation", "validation.js"],
    ["analyses", "analyses.js"],
    ["account", "account.js"],
    ["api", "api-explorer.js"],
    ["survey", "survey.js"],
    ["rigor", "rigor.js"],
    ["pi-sbi", "pi_sbi.js"],
    ["inference", "inference.js"],
    ["lensing", "lensing.js"],
  ];

  const results = await Promise.allSettled(
    pageEntries.map(([, file]) => import(`/ui-static/pages/${file}?v=${ASSET_VERSION}`))
  );
  results.forEach((r, i) => {
    if (r.status === "fulfilled") registerPage(pageEntries[i][0], r.value);
    else console.error(`Failed to load ${pageEntries[i][1]}:`, r.reason);
  });

  setupMenu();
  updateUserDisplay();
  window.addEventListener("hashchange", navigate);
  navigate();
  checkHealth();
  const healthInterval = setInterval(checkHealth, 60000);
  window.addEventListener("beforeunload", () => clearInterval(healthInterval));
}

// Export shared utils for page modules
window.LensPINN = { api, fmtSci, esc, toast, showLoading, hideLoading, getToken, setToken, setRefreshToken, clearAuth, checkHealth };

boot();

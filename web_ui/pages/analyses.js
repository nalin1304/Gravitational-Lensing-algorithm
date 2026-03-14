/* Analyses Page — CRUD analyses, jobs, results */
const L = () => window.LensPINN;

let currentTab = "my";

export function render() {
    return `
    <div class="mb-20">
      <h2 style="font-size:1.5rem;font-weight:700;margin:0 0 8px 0">Analyses</h2>
      <p class="section-desc" style="margin:0">Manage your lensing analyses, view jobs, and explore public results.</p>
    </div>

    <div class="tabs mb-20">
      <button class="tab active" data-tab="my">My Analyses</button>
      <button class="tab" data-tab="public">Public Gallery</button>
      <button class="tab" data-tab="jobs">Jobs</button>
      <button class="tab" data-tab="results">Results</button>
    </div>

    <div id="aContent"></div>

    <div id="aCreateModal" class="card" style="display:none;margin-top:20px">
      <div class="card-header" style="display:flex;justify-content:space-between;align-items:center">
        <span class="card-title">Create New Analysis</span>
        <button id="aCloseModal" class="btn btn-sm btn-ghost">✕</button>
      </div>
      <div class="form-group">
        <label class="form-label">Name</label>
        <input id="aName" class="form-input" placeholder="My analysis" />
      </div>
      <div class="form-group">
        <label class="form-label">Type</label>
        <select id="aType" class="form-input">
          <option value="synthetic">Synthetic</option>
          <option value="real_data">Real Data</option>
          <option value="inference">Inference</option>
          <option value="batch">Batch</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Description</label>
        <textarea id="aDesc" class="form-input" rows="3" placeholder="Optional description"></textarea>
      </div>
      <div class="form-group">
        <label class="form-label">Config (JSON)</label>
        <textarea id="aConfig" class="form-input" rows="4">{"grid_size": 64, "profile_type": "NFW"}</textarea>
      </div>
      <button id="aSubmit" class="btn btn-primary btn-full">Create Analysis</button>
    </div>
  `;
}

async function loadTab(tab) {
    const P = L();
    const el = document.getElementById("aContent");
    currentTab = tab;
    P.showLoading("Loading…");

    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === tab));

    if (tab === "my") {
        if (!P.getToken()) {
            el.innerHTML = `<div class="empty-state"><p>Sign in to view your analyses</p><a href="#/account" class="btn btn-primary" style="margin-top:12px">Go to Account</a></div>`;
            return;
        }
        try {
            const raw = await P.api("/api/v1/analyses");
            const analyses = Array.isArray(raw) ? raw : (raw.analyses || []);
            if (!analyses.length) {
                el.innerHTML = `<div class="empty-state"><p>No analyses yet</p><button id="aNew" class="btn btn-primary" style="margin-top:12px">Create First Analysis</button></div>`;
                document.getElementById("aNew")?.addEventListener("click", showCreate);
                return;
            }
            el.innerHTML = `
        <div style="margin-bottom:14px"><button id="aNew" class="btn btn-primary btn-sm">+ New Analysis</button></div>
        <table class="data-table">
          <thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Created</th><th></th></tr></thead>
          <tbody>${analyses.map(a => `
            <tr><td style="font-weight:500">${P.esc(a.name)}</td><td><span class="badge badge-info">${P.esc(a.type)}</span></td>
            <td><span class="badge ${a.status === 'completed' ? 'badge-success' : 'badge-warning'}">${P.esc(a.status)}</span></td>
            <td>${new Date(a.created_at).toLocaleDateString()}</td>
            <td><button class="btn btn-sm btn-secondary aDeleteBtn" data-id="${P.esc(String(a.id))}">Delete</button></td></tr>
          `).join("")}</tbody>
        </table>`;
            document.getElementById("aNew")?.addEventListener("click", showCreate);
            document.querySelectorAll(".aDeleteBtn").forEach(btn => btn.addEventListener("click", async () => {
                if (!confirm("Delete this analysis?")) return;
                try {
                    await P.api(`/api/v1/analyses/${btn.dataset.id}`, { method: "DELETE" });
                    P.toast("Analysis deleted", "success");
                    loadTab("my");
                } catch (e) { P.toast(`Delete failed: ${e.message}`, "error"); }
            }));
        } catch (e) { el.innerHTML = `<div class="empty-state"><p>Error: ${P.esc(e.message)}</p></div>`; }
    } else if (tab === "public") {
        try {
            const raw = await P.api("/api/v1/analyses/public", { auth: false });
            const analyses = Array.isArray(raw) ? raw : (raw.analyses || []);
            if (!analyses.length) { el.innerHTML = `<div class="empty-state"><p>No public analyses available</p></div>`; return; }
            el.innerHTML = `<table class="data-table">
        <thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Created</th></tr></thead>
        <tbody>${analyses.map(a => `
          <tr><td style="font-weight:500">${P.esc(a.name)}</td><td><span class="badge badge-info">${P.esc(a.type)}</span></td>
          <td><span class="badge badge-success">${P.esc(a.status)}</span></td>
          <td>${new Date(a.created_at).toLocaleDateString()}</td></tr>
        `).join("")}</tbody></table>`;
        } catch (e) { el.innerHTML = `<div class="empty-state"><p>${P.esc(e.message)}</p></div>`; }
    } else if (tab === "jobs") {
        if (!P.getToken()) { el.innerHTML = `<div class="empty-state"><p>Sign in to view jobs</p></div>`; return; }
        try {
            const raw = await P.api("/api/v1/jobs");
            const jobs = Array.isArray(raw) ? raw : (raw.jobs || []);
            if (!jobs.length) { el.innerHTML = `<div class="empty-state"><p>No jobs found</p></div>`; return; }
            el.innerHTML = `<table class="data-table">
        <thead><tr><th>Job ID</th><th>Type</th><th>Status</th><th>Progress</th><th>Created</th></tr></thead>
        <tbody>${jobs.map(j => `
          <tr><td style="font-family:var(--mono);font-size:12px">${P.esc(j.job_id)}</td>
          <td>${P.esc(j.job_type)}</td>
          <td><span class="badge ${j.status === 'completed' ? 'badge-success' : j.status === 'failed' ? 'badge-danger' : 'badge-warning'}">${P.esc(j.status)}</span></td>
          <td>${(j.progress * 100).toFixed(0)}%</td>
          <td>${new Date(j.created_at).toLocaleDateString()}</td></tr>
        `).join("")}</tbody></table>`;
        } catch (e) { el.innerHTML = `<div class="empty-state"><p>${P.esc(e.message)}</p></div>`; }
    } else if (tab === "results") {
        if (!P.getToken()) { el.innerHTML = `<div class="empty-state"><p>Sign in to view results</p></div>`; return; }
        try {
            const raw = await P.api("/api/v1/results");
            const results = Array.isArray(raw) ? raw : (raw.results || []);
            if (!results.length) { el.innerHTML = `<div class="empty-state"><p>No results yet</p></div>`; return; }
            el.innerHTML = `<table class="data-table">
        <thead><tr><th>Type</th><th>Confidence</th><th>Created</th></tr></thead>
        <tbody>${results.map(r => `
          <tr><td>${P.esc(r.result_type)}</td>
          <td>${r.confidence_score != null ? (r.confidence_score * 100).toFixed(1) + '%' : '—'}</td>
          <td>${new Date(r.created_at).toLocaleDateString()}</td></tr>
        `).join("")}</tbody></table>`;
        } catch (e) { el.innerHTML = `<div class="empty-state"><p>${P.esc(e.message)}</p></div>`; }
    }
    P.hideLoading();
}

function showCreate() { document.getElementById("aCreateModal").style.display = "block"; }

export function init() {
    const P = L();

    document.querySelectorAll(".tab").forEach(t =>
        t.addEventListener("click", () => loadTab(t.dataset.tab))
    );

    document.getElementById("aCloseModal")?.addEventListener("click", () => {
        document.getElementById("aCreateModal").style.display = "none";
    });

    document.getElementById("aSubmit")?.addEventListener("click", async () => {
        try {
            let config;
            try { config = JSON.parse(document.getElementById("aConfig").value); } catch { P.toast("Invalid JSON config", "error"); return; }
            await P.api("/api/v1/analyses", {
                method: "POST", body: {
                    name: document.getElementById("aName").value,
                    type: document.getElementById("aType").value,
                    description: document.getElementById("aDesc").value,
                    config
                }
            });
            document.getElementById("aCreateModal").style.display = "none";
            P.toast("Analysis created!", "success");
            loadTab("my");
        } catch (e) { P.toast(`Create failed: ${e.message}`, "error"); }
    });

    loadTab("my");
}

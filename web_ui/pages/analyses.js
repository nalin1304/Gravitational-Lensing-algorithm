/* Analyses Page — CRUD analyses, jobs, results */
const L = () => window.LensPINN;

let currentTab = "my";

export function render() {
    return `
    <div class="tabs">
      <button class="tab active" data-tab="my">My Analyses</button>
      <button class="tab" data-tab="public">Public Gallery</button>
      <button class="tab" data-tab="jobs">Jobs</button>
      <button class="tab" data-tab="results">Results</button>
    </div>

    <div id="aContent"></div>

    <div id="aCreateModal" style="display:none" class="card" style="margin-top:20px;">
      <div class="card-header">
        <span class="card-title">Create New Analysis</span>
        <button id="aCloseModal" class="btn btn-sm btn-secondary">×</button>
      </div>
      <div class="form-group">
        <label class="form-label">Name</label>
        <input id="aName" class="form-input" placeholder="My analysis" />
      </div>
      <div class="form-group">
        <label class="form-label">Type</label>
        <select id="aType" class="form-select">
          <option value="synthetic">Synthetic</option>
          <option value="real_data">Real Data</option>
          <option value="inference">Inference</option>
          <option value="batch">Batch</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Description</label>
        <textarea id="aDesc" class="form-textarea" placeholder="Optional description"></textarea>
      </div>
      <div class="form-group">
        <label class="form-label">Config (JSON)</label>
        <textarea id="aConfig" class="form-textarea">{"grid_size": 64, "profile_type": "NFW"}</textarea>
      </div>
      <button id="aSubmit" class="btn btn-primary">Create Analysis</button>
    </div>
  `;
}

async function loadTab(tab) {
    const P = L();
    const el = document.getElementById("aContent");
    currentTab = tab;

    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === tab));

    if (tab === "my") {
        if (!P.getToken()) {
            el.innerHTML = `<div class="empty-state"><p>Sign in to view your analyses</p><a href="#/account" class="btn btn-primary" style="margin-top:12px">Go to Account</a></div>`;
            return;
        }
        try {
            const data = await P.api("/api/v1/analyses");
            if (!data.analyses?.length) {
                el.innerHTML = `<div class="empty-state"><p>No analyses yet</p><button id="aNew" class="btn btn-primary" style="margin-top:12px">Create First Analysis</button></div>`;
                document.getElementById("aNew")?.addEventListener("click", showCreate);
                return;
            }
            el.innerHTML = `
        <div style="margin-bottom:14px"><button id="aNew" class="btn btn-primary btn-sm">+ New Analysis</button></div>
        <table class="data-table">
          <thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Created</th></tr></thead>
          <tbody>${data.analyses.map(a => `
            <tr><td style="font-weight:500">${P.esc(a.name)}</td><td><span class="badge badge-info">${P.esc(a.type)}</span></td>
            <td><span class="badge ${a.status === 'completed' ? 'badge-success' : 'badge-warning'}">${P.esc(a.status)}</span></td>
            <td>${new Date(a.created_at).toLocaleDateString()}</td></tr>
          `).join("")}</tbody>
        </table>`;
            document.getElementById("aNew")?.addEventListener("click", showCreate);
        } catch (e) { el.innerHTML = `<div class="empty-state"><p>Error: ${P.esc(e.message)}</p></div>`; }
    } else if (tab === "public") {
        try {
            const data = await P.api("/api/v1/analyses/public", { auth: false });
            if (!data.analyses?.length) { el.innerHTML = `<div class="empty-state"><p>No public analyses available</p></div>`; return; }
            el.innerHTML = `<table class="data-table">
        <thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Created</th></tr></thead>
        <tbody>${data.analyses.map(a => `
          <tr><td style="font-weight:500">${P.esc(a.name)}</td><td><span class="badge badge-info">${P.esc(a.type)}</span></td>
          <td><span class="badge badge-success">${P.esc(a.status)}</span></td>
          <td>${new Date(a.created_at).toLocaleDateString()}</td></tr>
        `).join("")}</tbody></table>`;
        } catch (e) { el.innerHTML = `<div class="empty-state"><p>${P.esc(e.message)}</p></div>`; }
    } else if (tab === "jobs") {
        if (!P.getToken()) { el.innerHTML = `<div class="empty-state"><p>Sign in to view jobs</p></div>`; return; }
        try {
            const data = await P.api("/api/v1/jobs");
            if (!data.jobs?.length) { el.innerHTML = `<div class="empty-state"><p>No jobs found</p></div>`; return; }
            el.innerHTML = `<table class="data-table">
        <thead><tr><th>Job ID</th><th>Type</th><th>Status</th><th>Progress</th><th>Created</th></tr></thead>
        <tbody>${data.jobs.map(j => `
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
            const data = await P.api("/api/v1/results");
            if (!data.results?.length) { el.innerHTML = `<div class="empty-state"><p>No results yet</p></div>`; return; }
            el.innerHTML = `<table class="data-table">
        <thead><tr><th>Type</th><th>Confidence</th><th>Created</th></tr></thead>
        <tbody>${data.results.map(r => `
          <tr><td>${P.esc(r.result_type)}</td>
          <td>${r.confidence_score != null ? (r.confidence_score * 100).toFixed(1) + '%' : '—'}</td>
          <td>${new Date(r.created_at).toLocaleDateString()}</td></tr>
        `).join("")}</tbody></table>`;
        } catch (e) { el.innerHTML = `<div class="empty-state"><p>${P.esc(e.message)}</p></div>`; }
    }
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

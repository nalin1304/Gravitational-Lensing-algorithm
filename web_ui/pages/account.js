/* Account Page — Auth, Profile, API Keys */
const L = () => window.LensPINN;

export function render() {
    const loggedIn = !!L().getToken();

    if (!loggedIn) {
        return `
      <div class="auth-container">
        <div class="tabs mb-16">
          <button class="tab active" data-atab="login">Sign In</button>
          <button class="tab" data-atab="register">Register</button>
        </div>

        <div id="acLoginForm" class="auth-card">
          <h2 class="auth-title">Welcome Back</h2>
          <p class="auth-sub">Sign in to access your analyses and API keys</p>
          <div class="form-group">
            <label class="form-label">Username or Email</label>
            <input id="acUser" class="form-input" placeholder="username" autocomplete="username" />
          </div>
          <div class="form-group">
            <label class="form-label">Password</label>
            <input id="acPass" class="form-input" type="password" placeholder="password" autocomplete="current-password" />
          </div>
          <button id="acLoginBtn" class="btn btn-primary" style="width:100%;margin-top:6px">Sign In</button>
        </div>

        <div id="acRegForm" class="auth-card" style="display:none">
          <h2 class="auth-title">Create Account</h2>
          <p class="auth-sub">Register to save analyses and manage API keys</p>
          <div class="form-group">
            <label class="form-label">Email</label>
            <input id="acRegEmail" class="form-input" type="email" placeholder="you@example.com" />
          </div>
          <div class="form-group">
            <label class="form-label">Username</label>
            <input id="acRegUser" class="form-input" placeholder="username (min 3 chars)" />
          </div>
          <div class="form-group">
            <label class="form-label">Full Name</label>
            <input id="acRegName" class="form-input" placeholder="Optional" />
          </div>
          <div class="form-group">
            <label class="form-label">Password</label>
            <input id="acRegPass" class="form-input" type="password" placeholder="min 8 characters" />
          </div>
          <button id="acRegBtn" class="btn btn-primary" style="width:100%;margin-top:6px">Register</button>
        </div>
      </div>
    `;
    }

    return `
    <div class="grid-2 gap-20">
      <div class="card">
        <div class="card-header">
          <span class="card-title">Profile</span>
          <button id="acLogout" class="btn btn-sm btn-danger">Sign Out</button>
        </div>
        <div id="acProfile"><p class="section-desc">Loading...</p></div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Usage Statistics</span>
        </div>
        <div id="acStats"><p class="section-desc">Loading...</p></div>
      </div>
    </div>

    <div class="card" style="margin-top:20px">
      <div class="card-header">
        <span class="card-title">API Keys</span>
        <button id="acNewKey" class="btn btn-sm btn-primary">+ Create Key</button>
      </div>
      <div id="acKeys"><p class="section-desc">Loading...</p></div>

      <div id="acKeyForm" style="display:none;margin-top:16px;padding-top:16px;border-top:1px solid var(--border)">
        <div class="form-group">
          <label class="form-label">Key Name</label>
          <input id="acKeyName" class="form-input" placeholder="my-api-key" />
        </div>
        <button id="acKeySubmit" class="btn btn-primary btn-sm">Generate Key</button>
      </div>
    </div>
  `;
}

export async function init() {
    const P = L();

    if (!P.getToken()) {
        // Auth tabs
        document.querySelectorAll("[data-atab]").forEach(t =>
            t.addEventListener("click", () => {
                const tab = t.dataset.atab;
                document.querySelectorAll("[data-atab]").forEach(x => x.classList.toggle("active", x.dataset.atab === tab));
                document.getElementById("acLoginForm").style.display = tab === "login" ? "block" : "none";
                document.getElementById("acRegForm").style.display = tab === "register" ? "block" : "none";
            })
        );

        // Login
        document.getElementById("acLoginBtn")?.addEventListener("click", async () => {
            const username = document.getElementById("acUser").value;
            const password = document.getElementById("acPass").value;
            if (!username || !password) { P.toast("Enter credentials", "error"); return; }
            try {
                const form = new URLSearchParams(); form.append("username", username); form.append("password", password);
                const resp = await fetch("/api/v1/auth/login", { method: "POST", body: form });
                if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || "Login failed"); }
                const data = await resp.json();
                P.setToken(data.access_token);
                P.setRefreshToken(data.refresh_token);
                P.toast("Signed in!", "success");
                location.hash = "#/account"; // re-render
                window.dispatchEvent(new HashChangeEvent("hashchange"));
            } catch (e) { P.toast(e.message, "error"); }
        });

        // Register
        document.getElementById("acRegBtn")?.addEventListener("click", async () => {
            try {
                await P.api("/api/v1/auth/register", {
                    method: "POST", auth: false, body: {
                        email: document.getElementById("acRegEmail").value,
                        username: document.getElementById("acRegUser").value,
                        full_name: document.getElementById("acRegName").value,
                        password: document.getElementById("acRegPass").value,
                    }
                });
                P.toast("Account created! Sign in now.", "success");
                document.querySelector("[data-atab='login']")?.click();
            } catch (e) { P.toast(e.message, "error"); }
        });
        return;
    }

    // Logged in — load profile
    document.getElementById("acLogout")?.addEventListener("click", () => {
        P.clearAuth();
        P.toast("Signed out", "info");
        window.dispatchEvent(new HashChangeEvent("hashchange"));
    });

    try {
        const user = await P.api("/api/v1/auth/me");
        document.getElementById("acProfile").innerHTML = `
      <div class="metric-row"><span class="metric-key">Username</span><span class="metric-val">${P.esc(user.username)}</span></div>
      <div class="metric-row"><span class="metric-key">Email</span><span class="metric-val">${P.esc(user.email)}</span></div>
      <div class="metric-row"><span class="metric-key">Role</span><span class="metric-val"><span class="badge badge-purple">${P.esc(user.role)}</span></span></div>
      <div class="metric-row"><span class="metric-key">Verified</span><span class="metric-val">${user.is_verified ? '✓' : '✗'}</span></div>
      <div class="metric-row"><span class="metric-key">Member Since</span><span class="metric-val">${new Date(user.created_at).toLocaleDateString()}</span></div>
    `;
    } catch (e) { document.getElementById("acProfile").innerHTML = `<p class="section-desc">Error: ${P.esc(e.message)}</p>`; }

    try {
        const stats = await P.api("/api/v1/user-stats");
        document.getElementById("acStats").innerHTML = `
      <div class="metric-row"><span class="metric-key">Analyses</span><span class="metric-val">${stats.analyses_count}</span></div>
      <div class="metric-row"><span class="metric-key">Jobs Run</span><span class="metric-val">${stats.jobs_count}</span></div>
      <div class="metric-row"><span class="metric-key">Results</span><span class="metric-val">${stats.results_count}</span></div>
    `;
    } catch { document.getElementById("acStats").innerHTML = `<p class="section-desc">Stats unavailable</p>`; }

    // API Keys
    document.getElementById("acNewKey")?.addEventListener("click", () => {
        document.getElementById("acKeyForm").style.display = document.getElementById("acKeyForm").style.display === "none" ? "block" : "none";
    });

    async function loadApiKeys() {
        const keys = await P.api('/api/v1/auth/api-keys');
        const container = document.getElementById('acKeys');
        if (!container) return;
        if (!keys || keys.length === 0) {
            container.innerHTML = '<p class="dim">No API keys yet. Create one below.</p>';
            return;
        }
        container.innerHTML = keys.map(k => `
            <div class="key-row" style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.1)">
                <div>
                    <strong>${P.esc(k.name || 'Unnamed')}</strong>
                    <span class="dim" style="margin-left:8px">${k.key_prefix}…</span>
                    <span class="dim" style="margin-left:8px">Created: ${k.created_at ? k.created_at.slice(0,10) : '—'}</span>
                </div>
                <button class="btn btn-sm btn-danger" onclick="revokeKey(${k.id})">Revoke</button>
            </div>
        `).join('');
    }

    window.revokeKey = async function revokeKey(keyId) {
        if (!confirm('Revoke this API key? This cannot be undone.')) return;
        await P.api(`/api/v1/auth/api-keys/${keyId}`, { method: 'DELETE' });
        P.toast('API key revoked', 'success');
        loadApiKeys();
    };

    loadApiKeys();

    document.getElementById("acKeySubmit")?.addEventListener("click", async () => {
        try {
            const key = await P.api("/api/v1/auth/api-keys", {
                method: "POST", body: {
                    name: document.getElementById("acKeyName").value, scopes: ["read", "write"]
                }
            });
            P.toast("API key created! Copy it now — you won't see it again.", "success");
            document.getElementById("acKeys").innerHTML = `
        <div class="metric-row" style="background:var(--accent-dim);padding:10px;border-radius:var(--radius-sm)">
          <span class="metric-key">New Key</span>
          <span class="metric-val" style="font-size:12px;word-break:break-all">${P.esc(key.api_key || key.key_prefix)}</span>
        </div>`;
        } catch (e) { P.toast(e.message, "error"); }
    });
}

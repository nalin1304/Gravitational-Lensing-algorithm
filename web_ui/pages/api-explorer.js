/* API Explorer Page — OpenAPI-driven endpoint browser */
const L = () => window.LensPINN;

let endpoints = [];
let openApiSpec = null;

export function render() {
    return `
    <div class="grid-12 gap-20">
      <div class="col-span-4">
        <div class="card" style="position:sticky;top:0">
          <div class="card-header">
            <span class="card-title">Endpoints</span>
            <span class="badge badge-info" id="apiCount">—</span>
          </div>
          <div id="apiList" style="max-height:calc(100vh - 200px);overflow-y:auto">
            <p class="section-desc">Loading OpenAPI schema...</p>
          </div>
        </div>
      </div>

      <div class="col-span-8">
        <div class="card mb-16">
          <div class="card-header"><span class="card-title">Request Builder</span></div>
          <div class="form-group">
            <label class="form-label">Selected Endpoint</label>
            <input id="apiEndpoint" class="form-input" readonly placeholder="Select an endpoint from the list" />
          </div>
          <div class="grid-2" style="gap:12px">
            <div class="form-group">
              <label class="form-label">Path Parameters (JSON)</label>
              <textarea id="apiPath" class="form-textarea" rows="3">{}</textarea>
            </div>
            <div class="form-group">
              <label class="form-label">Query Parameters (JSON)</label>
              <textarea id="apiQuery" class="form-textarea" rows="3">{}</textarea>
            </div>
          </div>
          <div class="form-group">
            <label class="form-label">Request Body (JSON)</label>
            <textarea id="apiBody" class="form-textarea" rows="5">{}</textarea>
          </div>
          <button id="apiRun" class="btn btn-primary">Send Request</button>
        </div>

        <div class="card">
          <div class="card-header"><span class="card-title">Response</span></div>
          <pre id="apiResponse" style="max-height:400px">No request sent yet.</pre>
        </div>
      </div>
    </div>
  `;
}

function methodColor(m) {
    const c = { get: "badge-info", post: "badge-success", put: "badge-warning", patch: "badge-warning", delete: "badge-danger" };
    return c[m] || "badge-info";
}

function resolveSchema(schema, spec, depth = 0) {
    if (!schema || depth > 5) return schema;
    if (schema.$ref) {
        const ref = schema.$ref.replace("#/components/schemas/", "");
        return resolveSchema(spec.components?.schemas?.[ref] || {}, spec, depth + 1);
    }
    return schema;
}

function buildExample(schema, spec, depth = 0) {
    if (!schema || depth > 4) return null;
    const s = resolveSchema(schema, spec, depth);
    if (s.type === "object" && s.properties) {
        const obj = {};
        for (const [k, v] of Object.entries(s.properties)) obj[k] = buildExample(v, spec, depth + 1);
        return obj;
    }
    if (s.type === "array") return [buildExample(s.items || {}, spec, depth + 1)];
    if (s.type === "string") return s.example || "string";
    if (s.type === "number" || s.type === "integer") return s.example || 0;
    if (s.type === "boolean") return false;
    return null;
}

async function loadSchema() {
    const P = L();
    try {
        openApiSpec = await P.api("/openapi.json", { auth: false });
        endpoints = [];
        for (const [path, methods] of Object.entries(openApiSpec.paths || {})) {
            for (const [method, detail] of Object.entries(methods)) {
                if (["get", "post", "put", "patch", "delete"].includes(method)) {
                    endpoints.push({ path, method: method.toUpperCase(), summary: detail.summary || "", tags: detail.tags || [], detail });
                }
            }
        }

        document.getElementById("apiCount").textContent = `${endpoints.length} endpoints`;

        const grouped = {};
        for (const ep of endpoints) {
            const tag = ep.tags[0] || "Other";
            if (!grouped[tag]) grouped[tag] = [];
            grouped[tag].push(ep);
        }

        let html = "";
        for (const [tag, eps] of Object.entries(grouped)) {
            html += `<div style="margin-bottom:12px">
        <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;padding:0 4px">${P.esc(tag)}</div>`;
            for (const ep of eps) {
                html += `<button class="api-ep-btn" data-path="${ep.path}" data-method="${ep.method}" style="
          display:flex;align-items:center;gap:8px;width:100%;text-align:left;padding:7px 8px;border-radius:var(--radius-sm);
          background:transparent;border:none;color:var(--text-secondary);font-size:12px;cursor:pointer;margin-bottom:2px;
          transition:background var(--transition)
        ">
          <span class="badge ${methodColor(ep.method.toLowerCase())}" style="min-width:40px;justify-content:center;font-size:10px">${ep.method}</span>
          <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--mono);font-size:11px">${P.esc(ep.path)}</span>
        </button>`;
            }
            html += `</div>`;
        }

        document.getElementById("apiList").innerHTML = html;

        // Click handlers
        document.querySelectorAll(".api-ep-btn").forEach(btn => {
            btn.addEventListener("click", () => selectEndpoint(btn.dataset.path, btn.dataset.method));
            btn.addEventListener("mouseenter", () => btn.style.background = "var(--bg-hover)");
            btn.addEventListener("mouseleave", () => btn.style.background = "transparent");
        });
    } catch (e) {
        document.getElementById("apiList").innerHTML = `<p class="section-desc">Could not load schema: ${P.esc(e.message)}</p>`;
    }
}

function selectEndpoint(path, method) {
    const P = L();
    document.getElementById("apiEndpoint").value = `${method} ${path}`;
    const ep = endpoints.find(e => e.path === path && e.method === method);
    if (!ep) return;

    // Build example body
    const bodySchema = ep.detail?.requestBody?.content?.["application/json"]?.schema;
    if (bodySchema) {
        const example = buildExample(bodySchema, openApiSpec);
        document.getElementById("apiBody").value = JSON.stringify(example, null, 2);
    } else {
        document.getElementById("apiBody").value = "{}";
    }

    document.getElementById("apiPath").value = "{}";
    document.getElementById("apiQuery").value = "{}";
    document.getElementById("apiResponse").textContent = `Ready to send ${method} ${path}`;
}

export async function init() {
    const P = L();
    await loadSchema();

    document.getElementById("apiRun")?.addEventListener("click", async () => {
        const raw = document.getElementById("apiEndpoint").value;
        if (!raw) { P.toast("Select an endpoint first", "error"); return; }
        const [method, ...pathParts] = raw.split(" ");
        let path = pathParts.join(" ");

        try {
            const pathParams = JSON.parse(document.getElementById("apiPath").value || "{}");
            for (const [k, v] of Object.entries(pathParams)) path = path.replace(`{${k}}`, encodeURIComponent(v));

            const queryParams = JSON.parse(document.getElementById("apiQuery").value || "{}");
            const qs = new URLSearchParams(queryParams).toString();
            if (qs) path += `?${qs}`;

            let body = null;
            if (["POST", "PUT", "PATCH"].includes(method)) {
                body = JSON.parse(document.getElementById("apiBody").value || "{}");
            }

            P.showLoading(`${method} ${path}...`);
            const result = await P.api(path, { method, body });
            document.getElementById("apiResponse").textContent = JSON.stringify(result, null, 2);
            P.toast("Request successful", "success");
        } catch (e) {
            document.getElementById("apiResponse").textContent = `Error: ${e.message}`;
            P.toast(`Request failed: ${e.message}`, "error");
        } finally { P.hideLoading(); }
    });
}

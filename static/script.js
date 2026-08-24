const THREAD_STORAGE_KEY = "travel_thread_id";

const agentDefinitions = [
    ["supervisor", "Supervisor"],
    ["flight_agent", "Flights"],
    ["hotel_agent", "Hotels"],
    ["weather_agent", "Weather"],
    ["budget_agent", "Budget"],
    ["itinerary_agent", "Itinerary"],
    ["human_review", "Human Review"],
];

const tabDefinitions = [
    ["flight_results", "Flights"],
    ["hotel_results", "Hotels"],
    ["weather_results", "Weather"],
    ["budget_results", "Budget"],
    ["itinerary", "Draft Itinerary"],
];

let currentThreadId = localStorage.getItem(THREAD_STORAGE_KEY) || null;
let latestAnswerMarkdown = "";
let latestResponse = null;
let activeTabKey = "flight_results";

function byId(id) {
    return document.getElementById(id);
}

function setPrompt(text) {
    byId("userInput").value = text;
    byId("userInput").focus();
}

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderMarkdown(target, markdown) {
    const content = String(markdown || "").trim();

    if (!content) {
        target.innerHTML = "<p class=\"muted\">No output returned for this section.</p>";
        return;
    }

    if (window.marked && window.DOMPurify) {
        target.innerHTML = DOMPurify.sanitize(marked.parse(content), {
            USE_PROFILES: { html: true },
        });
        return;
    }

    target.textContent = content;
}

function showElement(element, shouldShow) {
    element.classList.toggle("hidden", !shouldShow);
}

function setLoading(isLoading) {
    const sendBtn = byId("sendBtn");
    const btnText = byId("btnText");
    const btnLoader = byId("btnLoader");

    sendBtn.disabled = isLoading;
    byId("approveBtn").disabled = isLoading;
    byId("revisionBtn").disabled = isLoading;

    btnText.classList.toggle("hidden", isLoading);
    btnLoader.classList.toggle("hidden", !isLoading);
    showElement(byId("loadingPanel"), isLoading);
}

function showError(message) {
    const errorBox = byId("errorBox");
    errorBox.textContent = message;
    showElement(errorBox, true);
}

function hideError() {
    const errorBox = byId("errorBox");
    errorBox.textContent = "";
    showElement(errorBox, false);
}

function resetTrip() {
    currentThreadId = null;
    latestAnswerMarkdown = "";
    latestResponse = null;
    activeTabKey = "flight_results";
    localStorage.removeItem(THREAD_STORAGE_KEY);

    byId("userInput").value = "";
    byId("revisionFeedback").value = "";
    hideError();

    ["workflowSection", "specialistSection", "reviewSection", "finalSection", "loadingPanel"].forEach((id) => {
        showElement(byId(id), false);
    });

    byId("threadInfo").textContent = "Thread ID: -";
    byId("resultBox").innerHTML = "";
    byId("draftItinerary").innerHTML = "";
    byId("agentList").innerHTML = "";
}

function getAgentState(agentKey, data) {
    if (agentKey === "supervisor") {
        return "completed";
    }

    if (agentKey === "human_review") {
        if (data.requires_approval) {
            return "waiting for review";
        }

        if (data.approved !== null && data.approved !== undefined) {
            return "completed";
        }

        return "not selected";
    }

    const selected = Array.isArray(data.selected_agents) && data.selected_agents.includes(agentKey);

    if (!selected) {
        return "not selected";
    }

    if (data.specialist_statuses && data.specialist_statuses[agentKey]) {
        const status = String(data.specialist_statuses[agentKey]).toUpperCase();
        if (status === "DEGRADED") return "degraded";
        if (status === "FAILED") return "failed";
        if (status === "COMPLETED") return "completed";
        if (status === "NOT_SELECTED") return "not selected";
    }

    // Heuristic fallbacks if specialist_statuses is absent:
    if (agentKey === "flight_agent") {
        const text = String(data.flight_results || "");
        if (text.includes("unavailable") || text.includes("temporarily unavailable")) return "degraded";
        if (text) return "completed";
    }
    if (agentKey === "hotel_agent") {
        const text = String(data.hotel_results || "");
        if (text.includes("temporarily unavailable")) return "degraded";
        if (text) return "completed";
    }
    if (agentKey === "weather_agent") {
        const text = String(data.weather_results || "");
        if (text.includes("temporarily unavailable")) return "degraded";
        if (text) return "completed";
    }
    if (agentKey === "budget_agent" && data.budget_results) return "completed";
    if (agentKey === "itinerary_agent" && data.itinerary) return "completed";

    return "selected";
}

function renderAgents(data) {
    const agentList = byId("agentList");

    agentList.innerHTML = agentDefinitions.map(([key, label]) => {
        const state = getAgentState(key, data);
        return `
            <div class="agent-row agent-${state.replaceAll(" ", "-")}">
                <span>${escapeHtml(label)}</span>
                <strong>${escapeHtml(state)}</strong>
            </div>
        `;
    }).join("");
}

function renderSupervisor(data) {
    const guardrailBox = byId("guardrailBox");
    const allowed = data.guardrail_allowed !== false;

    guardrailBox.className = allowed ? "guardrail-box guardrail-ok" : "guardrail-box guardrail-blocked";
    guardrailBox.textContent = allowed
        ? "Travel scope accepted."
        : (data.guardrail_reason || "This request is outside TripBandhu's travel scope.");

    const constraints = data.trip_constraints || {};
    const rows = [
        ["Origin", constraints.origin],
        ["Destination", constraints.destination],
        ["Duration", constraints.duration],
        ["Budget", constraints.budget],
        ["Travel Style", constraints.travel_style],
        ["Preferences", Array.isArray(constraints.special_preferences) ? constraints.special_preferences.join(", ") : constraints.special_preferences],
    ];

    byId("constraintsGrid").innerHTML = rows.map(([label, value]) => `
        <div class="constraint-item">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value || "Not specified")}</strong>
        </div>
    `).join("");

    let reasoning = data.supervisor_reasoning || "No supervisor reasoning returned.";
    if (reasoning.toLowerCase().includes("except none")) {
        const activeSpecialists = (data.selected_agents || [])
            .map((name) => name.replace("_agent", ""))
            .join(", ");
        reasoning = `Coordinating ${activeSpecialists} specialists for this travel plan.`;
    }

    byId("supervisorReasoning").textContent = reasoning;
}

function renderTabs(data) {
    const tabs = byId("tabs");
    const availableTabs = tabDefinitions.filter(([key]) => Boolean(data[key]));
    const visibleTabs = availableTabs.length ? availableTabs : tabDefinitions;

    if (!visibleTabs.some(([key]) => key === activeTabKey)) {
        activeTabKey = visibleTabs[0][0];
    }

    tabs.innerHTML = visibleTabs.map(([key, label]) => `
        <button type="button" class="${key === activeTabKey ? "active" : ""}" data-tab="${escapeHtml(key)}" role="tab" aria-selected="${key === activeTabKey}">
            ${escapeHtml(label)}
        </button>
    `).join("");

    renderMarkdown(byId("tabPanel"), data[activeTabKey]);
}

function renderReview(data) {
    const requiresApproval = Boolean(data.requires_approval);
    showElement(byId("reviewSection"), requiresApproval);

    if (!requiresApproval) {
        return;
    }

    byId("approvalRequest").textContent = data.approval_request || "Please review this draft before finalizing.";
    renderMarkdown(byId("draftItinerary"), data.itinerary || data.answer);
}

function renderFinal(data) {
    const isFinal = !data.requires_approval && data.guardrail_allowed !== false && Boolean(data.answer);

    showElement(byId("finalSection"), isFinal);

    if (!isFinal) {
        return;
    }

    latestAnswerMarkdown = data.answer;
    byId("threadInfo").textContent = `Thread ID: ${data.thread_id || "-"}`;
    renderMarkdown(byId("resultBox"), data.answer);
}

function renderGuardrailOnly(data) {
    if (data.guardrail_allowed !== false) {
        return false;
    }

    showElement(byId("workflowSection"), true);
    showElement(byId("specialistSection"), false);
    showElement(byId("reviewSection"), false);
    showElement(byId("finalSection"), false);
    showError(data.guardrail_reason || "TripBandhu can only help with travel-planning requests.");
    return true;
}

function renderResponse(data) {
    latestResponse = data;
    currentThreadId = data.thread_id || currentThreadId;

    if (currentThreadId) {
        localStorage.setItem(THREAD_STORAGE_KEY, currentThreadId);
    }

    hideError();
    showElement(byId("workflowSection"), true);

    renderAgents(data);
    renderSupervisor(data);

    if (renderGuardrailOnly(data)) {
        return;
    }

    showElement(byId("specialistSection"), true);
    renderTabs(data);
    renderReview(data);
    renderFinal(data);
}

async function postJson(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
        const code = data.error_code ? `${data.error_code}: ` : "";
        throw new Error(`${code}${data.error || "Request failed."}`);
    }

    return data;
}

async function sendMessage() {
    hideError();

    const message = byId("userInput").value.trim();

    if (!message) {
        showError("Please enter your travel request first.");
        return;
    }

    setLoading(true);

    try {
        const data = await postJson("/api/travel", {
            message,
            thread_id: currentThreadId,
        });

        renderResponse(data);
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

async function resumeTrip(approved) {
    hideError();

    if (!currentThreadId) {
        showError("No active trip thread is available to resume.");
        return;
    }

    const feedback = byId("revisionFeedback").value.trim();

    if (!approved && !feedback) {
        showError("Add revision feedback before requesting changes.");
        return;
    }

    setLoading(true);

    try {
        const data = await postJson("/api/travel/resume", {
            thread_id: currentThreadId,
            approved,
            feedback,
        });

        renderResponse(data);
        byId("revisionFeedback").value = "";
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

function copyResult() {
    const text = byId("resultBox").innerText;

    if (!text) {
        return;
    }

    navigator.clipboard.writeText(text)
        .then(() => {
            const copyBtn = byId("copyBtn");
            const oldText = copyBtn.textContent;
            copyBtn.textContent = "Copied";
            setTimeout(() => {
                copyBtn.textContent = oldText;
            }, 1400);
        })
        .catch(() => {
            showError("Could not copy result.");
        });
}

async function downloadPDF() {
    // Primary source: in-memory markdown from the last API response.
    // Fallback: read plaintext from the rendered result box (survives page refresh
    // but loses markdown formatting — still produces a valid readable PDF).
    let content = latestAnswerMarkdown;
    if (!content) {
        const resultBox = byId("resultBox");
        content = resultBox ? (resultBox.innerText || "").trim() : "";
    }

    if (!content) {
        alert("No final travel plan is available to download yet.\nPlease run a trip query first.");
        return;
    }

    const downloadBtn = byId("downloadBtn");
    const oldText = downloadBtn.textContent;
    downloadBtn.textContent = "Preparing PDF...";
    downloadBtn.disabled = true;

    const titleMatch = content.match(/^#+ (.+)/m);
    const title = titleMatch ? titleMatch[1].trim() : "TripBandhu Travel Plan";

    hideError();

    try {
        const response = await fetch("/api/travel/download-pdf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: content, title }),
        });

        if (!response.ok) {
            let message = "Could not generate the itinerary PDF.";
            try {
                const errorPayload = await response.json();
                message = errorPayload.error || message;
            } catch (_error) {
                // The stable fallback above is used for a non-JSON proxy response.
            }
            throw new Error(message);
        }

        const contentType = response.headers.get("Content-Type") || "";
        if (!contentType.toLowerCase().includes("application/pdf")) {
            throw new Error("The server returned an invalid PDF response.");
        }

        const blob = await response.blob();
        if (blob.size < 5) {
            throw new Error("The generated PDF was empty.");
        }

        const disposition = response.headers.get("Content-Disposition") || "";
        const filenameMatch = disposition.match(/filename="?([^";]+)"?/i);
        const filename = filenameMatch ? filenameMatch[1] : "tripbandhu-travel-plan.pdf";
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
        console.error("[PDF] Download failed:", error);
        showError(error.message || "Could not download the itinerary PDF.");
    } finally {
        downloadBtn.textContent = oldText;
        downloadBtn.disabled = false;
    }
}

function bindQuickLoadPresets() {
    document.querySelectorAll(".quick-load").forEach((btn) => {
        btn.addEventListener("click", (event) => {
            event.preventDefault();
            const prompt = btn.getAttribute("data-prompt") || btn.dataset.prompt;
            if (prompt) {
                setPrompt(prompt);
            }
        });
    });
}

document.addEventListener("click", (event) => {
    const tabButton = event.target.closest("[data-tab]");

    if (tabButton && latestResponse) {
        activeTabKey = tabButton.dataset.tab;
        renderTabs(latestResponse);
    }
});

document.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "Enter") {
        sendMessage();
    }
});

byId("sendBtn").addEventListener("click", sendMessage);
byId("newTripBtn").addEventListener("click", resetTrip);
byId("approveBtn").addEventListener("click", () => resumeTrip(true));
byId("revisionBtn").addEventListener("click", () => resumeTrip(false));
byId("copyBtn").addEventListener("click", copyResult);
byId("downloadBtn").addEventListener("click", downloadPDF);

bindQuickLoadPresets();

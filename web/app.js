const apiInput = document.querySelector("#api-url");
const tokenInput = document.querySelector("#api-token");
const statusEl = document.querySelector("#connection-status");
const form = document.querySelector("#analysis-form");
const submitButton = document.querySelector("#analyze-button");
const resultCard = document.querySelector("#result-card");

apiInput.value = localStorage.getItem("clickbait-api-url") || apiInput.value;

function apiBase() {
  const parsed = new URL(apiInput.value.trim());
  const isLoopback = ["localhost", "127.0.0.1", "[::1]"].includes(parsed.hostname);
  if (!isLoopback && parsed.protocol !== "https:") {
    throw new Error("Use HTTPS for a remote API address.");
  }
  if (isLoopback && !["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("The local API address must use HTTP or HTTPS.");
  }
  localStorage.setItem("clickbait-api-url", parsed.origin);
  return parsed.origin;
}

function localAddressSpace(url) {
  const host = new URL(url).hostname;
  return ["localhost", "127.0.0.1", "[::1]"].includes(host) ? { targetAddressSpace: "loopback" } : {};
}

function showStatus(message, state = "") {
  statusEl.textContent = message;
  statusEl.className = `connection-status ${state}`.trim();
}

async function checkConnection() {
  try {
    const base = apiBase();
    showStatus("Checking local FastAPI…");
    const response = await fetch(`${base}/health`, { ...localAddressSpace(base), method: "GET" });
    if (!response.ok) throw new Error(`FastAPI returned ${response.status}.`);
    const health = await response.json();
    if (!health.models_ready) {
      showStatus("FastAPI is running, but its models did not load. Check its console.", "offline");
      return false;
    }
    showStatus("Connected · local models are ready.", "online");
    return true;
  } catch (error) {
    showStatus(`${error.message} Allow local network access if your browser asks, then try again.`, "offline");
    return false;
  }
}

document.querySelector("#check-connection").addEventListener("click", checkConnection);

function percent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function setScore(name, value) {
  const numeric = Math.max(0, Math.min(1, Number(value)));
  document.querySelector(`#${name}-score`).textContent = percent(numeric);
  document.querySelector(`#${name}-bar`).style.width = `${numeric * 100}%`;
}

function showResult(data) {
  const isClickbait = Number(data.final_prediction) === 1;
  const label = document.querySelector("#result-label");
  label.textContent = isClickbait ? "Likely clickbait" : "Likely not clickbait";
  label.className = isClickbait ? "is-clickbait" : "is-not-clickbait";
  document.querySelector("#result-title").textContent = data.title || "";
  document.querySelector("#fusion-score").textContent = percent(data.fusion_score);
  document.querySelector("#score-ring").style.borderColor = isClickbait ? "#805c3c" : "#657b43";
  setScore("text", data.text_score);
  setScore("thumbnail", data.thumbnail_score);
  setScore("metadata", data.metadata_score);
  const factsContainer = document.querySelector("#video-facts");
  factsContainer.replaceChildren();
  const facts = [
    ["VIDEO ID", data.video_id],
    ["DURATION", data.duration],
    ["VIEWS", data.views],
    ["LIKES", data.likes],
  ];
  for (const [labelText, value] of facts) {
    if (value !== null && value !== undefined && value !== "") {
      const fact = document.createElement("div");
      const label = document.createElement("span");
      label.textContent = labelText;
      fact.append(label, document.createTextNode(` ${String(value)}`));
      factsContainer.append(fact);
    }
  }
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const token = tokenInput.value.trim();
  if (!token) {
    document.querySelector("#connection-settings").open = true;
    showStatus("Paste the local API token shown in the FastAPI console.", "offline");
    tokenInput.focus();
    return;
  }

  submitButton.disabled = true;
  submitButton.querySelector("span:first-child").textContent = "Analyzing…";
  resultCard.classList.add("hidden");
  try {
    const base = apiBase();
    const response = await fetch(`${base}/analyze`, {
      ...localAddressSpace(base),
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ video_url: document.querySelector("#video-url").value.trim() }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Analysis failed (${response.status}).`);
    showStatus("Connected · local models are ready.", "online");
    showResult(payload);
  } catch (error) {
    showStatus(`${error.message} Allow local network access if your browser asks, then try again.`, "offline");
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span:first-child").textContent = "Analyze video";
  }
});

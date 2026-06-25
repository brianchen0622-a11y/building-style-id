import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

// Run entirely in-browser, never touch any server for inference.
env.allowLocalModels = false;

// Surface otherwise-silent failures (e.g. inside the WASM loader) so the
// status text never just freezes on "loading" with no explanation.
window.addEventListener("error", (e) => {
  console.error("window error:", e.error || e.message);
  setStatus(`發生錯誤：${e.message}`);
});
window.addEventListener("unhandledrejection", (e) => {
  console.error("unhandled rejection:", e.reason);
  setStatus(`發生錯誤：${e.reason?.message || e.reason}`);
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch((err) => {
      console.error("service worker registration failed:", err);
    });
  });
}

const fileInput = document.getElementById("file-input");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const previewFrame = document.getElementById("preview-frame");
const previewImage = document.getElementById("preview-image");
const retakeBtn = document.getElementById("retake-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const cameraBtn = document.getElementById("camera-btn");
const galleryBtn = document.getElementById("gallery-btn");
const cameraOverlay = document.getElementById("camera-overlay");
const cameraVideo = document.getElementById("camera-video");
const cameraCloseBtn = document.getElementById("camera-close-btn");
const cameraShutterBtn = document.getElementById("camera-shutter-btn");
const cameraCanvas = document.getElementById("camera-canvas");
const statusSection = document.getElementById("status-section");
const statusText = document.getElementById("status-text");
const resultsSection = document.getElementById("results-section");
const resultsHeading = document.getElementById("results-heading");
const resultsList = document.getElementById("results-list");
const detailCard = document.getElementById("detail-card");
const detailName = document.getElementById("detail-name");
const detailEra = document.getElementById("detail-era");
const detailGeometry = document.getElementById("detail-geometry");
const detailStructure = document.getElementById("detail-structure");
const detailExamples = document.getElementById("detail-examples");

const engineRadios = document.querySelectorAll('input[name="engine"]');
const geminiKeySection = document.getElementById("gemini-key-section");
const geminiKeyInput = document.getElementById("gemini-key-input");
const geminiKeySaveBtn = document.getElementById("gemini-key-save-btn");
const landmarkCallout = document.getElementById("landmark-callout");
const landmarkName = document.getElementById("landmark-name");
const landmarkConfidence = document.getElementById("landmark-confidence");

const GEMINI_KEY_STORAGE = "gemini_api_key";
const ENGINE_STORAGE = "engine_choice";

function getGeminiKey() {
  return localStorage.getItem(GEMINI_KEY_STORAGE) || "";
}

function setGeminiKey(key) {
  if (key) localStorage.setItem(GEMINI_KEY_STORAGE, key);
  else localStorage.removeItem(GEMINI_KEY_STORAGE);
}

function getEngineChoice() {
  return localStorage.getItem(ENGINE_STORAGE) || "clip";
}

function setEngineChoice(value) {
  localStorage.setItem(ENGINE_STORAGE, value);
}

function syncEngineUI() {
  const choice = getEngineChoice();
  engineRadios.forEach((radio) => {
    radio.checked = radio.value === choice;
  });
  geminiKeySection.hidden = choice === "clip";
}

engineRadios.forEach((radio) => {
  radio.addEventListener("change", () => {
    if (radio.checked) {
      setEngineChoice(radio.value);
      syncEngineUI();
    }
  });
});

geminiKeyInput.value = getGeminiKey();
geminiKeySaveBtn.addEventListener("click", () => {
  setGeminiKey(geminiKeyInput.value.trim());
  geminiKeySaveBtn.textContent = "已儲存";
  setTimeout(() => {
    geminiKeySaveBtn.textContent = "儲存";
  }, 1200);
});

syncEngineUI();

let stylesById = new Map();
let labelToStyle = new Map();
let candidateLabels = [];
let classifierPromise = null;
let currentImageUrl = null;
let inferenceImageUrl = null;
let cameraStream = null;

// Wikipedia's public REST API is free, requires no key, and is CORS-enabled
// for browser fetches. We look up a thumbnail by page title instead of
// hardcoding image URLs we can't verify ahead of time.
const wikiThumbCache = new Map();
async function fetchWikiThumbnail(title) {
  if (wikiThumbCache.has(title)) return wikiThumbCache.get(title);
  try {
    const res = await fetch(
      `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`,
    );
    if (!res.ok) throw new Error(`wiki lookup failed: ${res.status}`);
    const data = await res.json();
    const url = data.thumbnail?.source || null;
    wikiThumbCache.set(title, url);
    return url;
  } catch (err) {
    console.warn("wiki thumbnail fetch failed for", title, err);
    wikiThumbCache.set(title, null);
    return null;
  }
}

// CLIP resizes internally to 224x224 anyway, so feeding it a multi-megapixel
// phone photo just wastes time on canvas decode/resize inside the WASM
// pipeline. Downscale client-side first so inference stays fast.
function resizeImageForInference(imageUrl, maxDim = 384) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
      const w = Math.max(1, Math.round(img.naturalWidth * scale));
      const h = Math.max(1, Math.round(img.naturalHeight * scale));
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error("圖片縮圖失敗"));
          return;
        }
        resolve(URL.createObjectURL(blob));
      }, "image/jpeg", 0.85);
    };
    img.onerror = () => reject(new Error("圖片載入失敗，無法縮圖"));
    img.src = imageUrl;
  });
}

async function loadStyles() {
  const res = await fetch("./data/styles.json");
  const styles = await res.json();
  for (const style of styles) {
    stylesById.set(style.id, style);
    labelToStyle.set(style.label, style);
  }
  candidateLabels = styles.map((s) => s.label);
}

function setStatus(text) {
  statusText.textContent = text;
  statusSection.hidden = false;
}

function getClassifier() {
  if (!classifierPromise) {
    setStatus("正在連線下載模型 (Xenova/clip-vit-base-patch32)...");
    classifierPromise = pipeline(
      "zero-shot-image-classification",
      "Xenova/clip-vit-base-patch32",
      {
        progress_callback: (progress) => {
          console.log("model progress:", progress);
          if (progress.status === "initiate") {
            setStatus(`準備下載：${progress.file}`);
          } else if (progress.status === "download") {
            setStatus(`開始下載：${progress.file}`);
          } else if (progress.status === "progress") {
            const pct = typeof progress.progress === "number" ? progress.progress.toFixed(0) : "0";
            const loadedMB = progress.loaded ? (progress.loaded / 1024 / 1024).toFixed(1) : "?";
            const totalMB = progress.total ? (progress.total / 1024 / 1024).toFixed(1) : "?";
            setStatus(`模型下載中 (${progress.file})：${pct}% (${loadedMB}MB / ${totalMB}MB)`);
          } else if (progress.status === "done") {
            setStatus(`下載完成：${progress.file}，準備載入...`);
          } else if (progress.status === "ready") {
            setStatus("模型準備完成，開始分析圖片...");
          }
        },
      },
    ).catch((err) => {
      classifierPromise = null; // allow retry on next click
      throw err;
    });
  }
  return classifierPromise;
}

function useImage(url) {
  if (currentImageUrl) URL.revokeObjectURL(currentImageUrl);
  if (inferenceImageUrl) {
    URL.revokeObjectURL(inferenceImageUrl);
    inferenceImageUrl = null;
  }
  currentImageUrl = url;
  previewImage.src = currentImageUrl;
  previewFrame.hidden = false;
  uploadPlaceholder.hidden = true;
  analyzeBtn.disabled = false;
  resultsSection.hidden = true;
  detailCard.hidden = true;
  landmarkCallout.hidden = true;
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  useImage(URL.createObjectURL(file));
});

retakeBtn.addEventListener("click", () => {
  previewFrame.hidden = true;
  uploadPlaceholder.hidden = false;
  analyzeBtn.disabled = true;
  resultsSection.hidden = true;
  detailCard.hidden = true;
  landmarkCallout.hidden = true;
});

galleryBtn.addEventListener("click", () => fileInput.click());

// In-page camera capture so the shutter is one tap, with no OS picker
// sheet in between (iOS Safari always shows that sheet for <input
// capture>, so a real getUserMedia preview is the only way around it).
async function openCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
  } catch (err) {
    console.error("camera access failed:", err);
    setStatus(`無法開啟相機：${err.message || err}，請改用「從相簿選擇」`);
    return;
  }
  cameraVideo.srcObject = cameraStream;
  cameraOverlay.hidden = false;
}

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraOverlay.hidden = true;
}

cameraBtn.addEventListener("click", openCamera);
cameraCloseBtn.addEventListener("click", closeCamera);

cameraShutterBtn.addEventListener("click", () => {
  const w = cameraVideo.videoWidth;
  const h = cameraVideo.videoHeight;
  if (!w || !h) return;
  cameraCanvas.width = w;
  cameraCanvas.height = h;
  cameraCanvas.getContext("2d").drawImage(cameraVideo, 0, 0, w, h);
  cameraCanvas.toBlob(
    (blob) => {
      if (!blob) return;
      closeCamera();
      useImage(URL.createObjectURL(blob));
    },
    "image/jpeg",
    0.9,
  );
});

function createResultsRenderer(refs) {
  function showDetail(style) {
    if (!style) {
      refs.detailCard.hidden = true;
      return;
    }
    refs.detailName.textContent = style.name;
    refs.detailEra.textContent = style.era || "";
    refs.detailGeometry.textContent = style.geometry;
    refs.detailStructure.textContent = style.structure;

    refs.detailExamples.innerHTML = "";
    (style.examples || []).forEach((ex) => {
      const li = document.createElement("li");

      const thumbPlaceholder = document.createElement("div");
      thumbPlaceholder.className = "example-thumb-placeholder";
      li.appendChild(thumbPlaceholder);

      const textWrap = document.createElement("div");
      textWrap.className = "example-text";
      const nameSpan = document.createElement("span");
      nameSpan.textContent = ex.name;
      const locSpan = document.createElement("span");
      locSpan.className = "example-location";
      locSpan.textContent = ex.location;
      textWrap.appendChild(nameSpan);
      textWrap.appendChild(locSpan);
      li.appendChild(textWrap);

      refs.detailExamples.appendChild(li);

      if (ex.wiki) {
        fetchWikiThumbnail(ex.wiki).then((url) => {
          if (!url) return;
          const img = document.createElement("img");
          img.className = "example-thumb";
          img.src = url;
          img.alt = ex.name;
          img.loading = "lazy";
          thumbPlaceholder.replaceWith(img);
        });
      }
    });

    refs.detailCard.hidden = false;
  }

  function render(output) {
    const top = output.slice(0, 5);
    refs.resultsList.innerHTML = "";

    top.forEach((item, index) => {
      const style = labelToStyle.get(item.label);
      const li = document.createElement("li");
      li.className = index === 0 ? "active" : "";
      const nameSpan = document.createElement("span");
      nameSpan.textContent = style ? style.name : item.label;
      const scoreSpan = document.createElement("span");
      scoreSpan.className = "score";
      scoreSpan.textContent = `${(item.score * 100).toFixed(1)}%`;
      li.appendChild(nameSpan);
      li.appendChild(scoreSpan);
      li.addEventListener("click", () => {
        refs.resultsList.querySelectorAll("li").forEach((el) => el.classList.remove("active"));
        li.classList.add("active");
        showDetail(style);
      });
      refs.resultsList.appendChild(li);
    });

    refs.resultsSection.hidden = false;
    if (top.length > 0) {
      showDetail(labelToStyle.get(top[0].label));
    }
  }

  return { render, showDetail };
}

const resultsRenderer = createResultsRenderer({
  resultsSection,
  resultsList,
  detailCard,
  detailName,
  detailEra,
  detailGeometry,
  detailStructure,
  detailExamples,
});

// CLIP only ever scores the fixed style label set, so when both engines run
// we average per-label scores; a label only one engine returned just keeps
// that engine's score instead of being penalized for the other's silence.
function mergeScores(clipOutput, geminiOutput) {
  const clipMap = new Map((clipOutput || []).map((item) => [item.label, item.score]));
  const geminiMap = new Map((geminiOutput || []).map((item) => [item.label, item.score]));
  const labels = new Set([...clipMap.keys(), ...geminiMap.keys()]);

  const merged = [...labels].map((label) => {
    const clipScore = clipMap.get(label);
    const geminiScore = geminiMap.get(label);
    const score = clipScore != null && geminiScore != null
      ? (clipScore + geminiScore) / 2
      : (clipScore ?? geminiScore);
    return { label, score };
  });

  merged.sort((a, b) => b.score - a.score);
  return merged;
}

function showLandmark(landmark) {
  if (!landmark || !landmark.name) {
    landmarkCallout.hidden = true;
    return;
  }
  landmarkName.textContent = landmark.name;
  landmarkConfidence.textContent =
    typeof landmark.confidence === "number" ? `（信心 ${(landmark.confidence * 100).toFixed(0)}%）` : "";
  landmarkCallout.hidden = false;
}

function blobUrlToBase64(url) {
  return fetch(url)
    .then((res) => res.blob())
    .then(
      (blob) =>
        new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result.split(",")[1]);
          reader.onerror = () => reject(new Error("圖片轉換失敗"));
          reader.readAsDataURL(blob);
        }),
    );
}

// Ask Gemini to score every candidate label itself (rather than just name
// the top guess) so the result list can render the same ranked-list UI as
// the local CLIP classifier, and to separately call out a specific famous
// building if it recognizes one (CLIP's fixed style labels can't do that).
async function classifyWithGemini(imageUrl, apiKey) {
  const base64 = await blobUrlToBase64(imageUrl);
  const prompt =
    "你是建築辨識助手，請完成兩項任務：\n" +
    "1. 從以下風格清單中，針對每一個風格給一個 0 到 1 的信心分數，評估這張建築物照片符合該風格的程度：\n" +
    candidateLabels.join("、") +
    "\n2. 判斷這張照片是否為你能辨識出的具體知名建築（例如某座地標、某棟著名大樓）。如果可以，給出建築名稱與 0 到 1 的信心分數；如果無法判斷出具體建築，landmark 請省略或設為 null。\n" +
    "label 必須完全等於清單中的字串，不要加任何說明文字。";

  // Natural-language formatting instructions alone don't reliably keep the
  // same JSON shape across calls (Gemini occasionally wraps/unwraps the
  // "styles" array differently for the same prompt+image). A responseSchema
  // makes the API itself enforce the structure instead of just asking nicely.
  const responseSchema = {
    type: "OBJECT",
    properties: {
      styles: {
        type: "ARRAY",
        items: {
          type: "OBJECT",
          properties: {
            label: { type: "STRING" },
            score: { type: "NUMBER" },
          },
          required: ["label", "score"],
        },
      },
      landmark: {
        type: "OBJECT",
        nullable: true,
        properties: {
          name: { type: "STRING" },
          confidence: { type: "NUMBER" },
        },
      },
    },
    required: ["styles"],
  };

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${encodeURIComponent(apiKey)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [
          {
            parts: [{ text: prompt }, { inline_data: { mime_type: "image/jpeg", data: base64 } }],
          },
        ],
        generationConfig: { responseMimeType: "application/json", responseSchema },
      }),
    },
  );

  if (!res.ok) {
    const errBody = await res.text().catch(() => "");
    throw new Error(`Gemini API 錯誤 (${res.status})：${errBody.slice(0, 200)}`);
  }

  const data = await res.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error("Gemini 沒有回傳結果");

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new Error("Gemini 回傳格式無法解析");
  }

  const validLabels = new Set(candidateLabels);
  const styles = Array.isArray(parsed.styles)
    ? parsed.styles
        .filter((item) => item && validLabels.has(item.label) && typeof item.score === "number")
        .sort((a, b) => b.score - a.score)
    : [];

  if (styles.length === 0) throw new Error("Gemini 回傳結果無法對應已知風格");

  let landmark = null;
  if (parsed.landmark && typeof parsed.landmark.name === "string" && parsed.landmark.name.trim()) {
    landmark = {
      name: parsed.landmark.name.trim(),
      confidence: typeof parsed.landmark.confidence === "number" ? parsed.landmark.confidence : null,
    };
  }

  return { styles, landmark };
}

async function runClip() {
  setStatus("模型載入中（第一次使用需要下載模型，請稍候）...");
  const classifier = await getClassifier();

  setStatus("正在縮圖以加速分析...");
  if (!inferenceImageUrl) {
    inferenceImageUrl = await resizeImageForInference(currentImageUrl);
  }

  setStatus("正在使用本機 CLIP 分析建築風格...");
  console.time("clip-inference");
  const output = await classifier(inferenceImageUrl, candidateLabels);
  console.timeEnd("clip-inference");
  console.log("classifier output:", output);

  return output;
}

async function runGemini() {
  const apiKey = getGeminiKey();
  if (!apiKey) {
    throw new Error("請先在上方輸入並儲存 Gemini API Key");
  }
  setStatus("正在使用 Gemini 分析建築風格...");
  return classifyWithGemini(currentImageUrl, apiKey);
}

analyzeBtn.addEventListener("click", async () => {
  if (!currentImageUrl) return;
  const engine = getEngineChoice();
  analyzeBtn.disabled = true;
  resultsSection.hidden = true;
  detailCard.hidden = true;
  landmarkCallout.hidden = true;

  let slowHintTimer = null;
  if (engine !== "gemini") {
    slowHintTimer = setTimeout(() => {
      setStatus("正在分析建築風格...（第一次分析在較舊的裝置上可能需要 30 秒以上，請耐心等候）");
    }, 8000);
  }

  const errors = [];
  let clipOutput = null;
  let geminiOutput = null;
  let landmark = null;

  try {
    if (engine === "clip" || engine === "both") {
      try {
        clipOutput = await runClip();
      } catch (err) {
        console.error(err);
        errors.push(err.message || String(err));
      }
    }
    if (engine === "gemini" || engine === "both") {
      try {
        const result = await runGemini();
        geminiOutput = result.styles;
        landmark = result.landmark;
      } catch (err) {
        console.error(err);
        errors.push(err.message || String(err));
      }
    }
  } finally {
    if (slowHintTimer) clearTimeout(slowHintTimer);
    analyzeBtn.disabled = false;
  }

  const usedBoth = engine === "both" && clipOutput && geminiOutput;
  const merged = usedBoth ? mergeScores(clipOutput, geminiOutput) : (clipOutput || geminiOutput);

  if (merged) {
    resultsHeading.textContent = usedBoth ? "辨識結果（CLIP + Gemini 平均）" : "辨識結果";
    resultsRenderer.render(merged);
    showLandmark(landmark);
  }

  if (errors.length > 0) {
    setStatus(merged ? `部分辨識失敗：${errors.join("；")}` : `辨識失敗：${errors.join("；")}`);
  } else {
    statusSection.hidden = true;
  }
});

loadStyles().catch((err) => {
  console.error(err);
  statusText.textContent = "知識庫載入失敗，請重新整理頁面";
  statusSection.hidden = false;
});

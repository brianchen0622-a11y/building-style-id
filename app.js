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
const resultsList = document.getElementById("results-list");
const detailCard = document.getElementById("detail-card");
const detailName = document.getElementById("detail-name");
const detailEra = document.getElementById("detail-era");
const detailGeometry = document.getElementById("detail-geometry");
const detailStructure = document.getElementById("detail-structure");
const detailExamples = document.getElementById("detail-examples");

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

analyzeBtn.addEventListener("click", async () => {
  if (!currentImageUrl) return;
  analyzeBtn.disabled = true;
  resultsSection.hidden = true;
  detailCard.hidden = true;
  setStatus("模型載入中（第一次使用需要下載模型，請稍候）...");

  let slowHintTimer = null;
  try {
    const classifier = await getClassifier();

    setStatus("正在縮圖以加速分析...");
    if (!inferenceImageUrl) {
      inferenceImageUrl = await resizeImageForInference(currentImageUrl);
    }

    setStatus("正在分析建築風格...");
    slowHintTimer = setTimeout(() => {
      setStatus("正在分析建築風格...（第一次分析在較舊的裝置上可能需要 30 秒以上，請耐心等候）");
    }, 8000);

    console.time("clip-inference");
    console.log("running classifier on", inferenceImageUrl);
    const output = await classifier(inferenceImageUrl, candidateLabels);
    console.timeEnd("clip-inference");
    console.log("classifier output:", output);

    renderResults(output);
    statusSection.hidden = true;
  } catch (err) {
    console.error(err);
    setStatus(`辨識失敗：${err.message || err}`);
  } finally {
    if (slowHintTimer) clearTimeout(slowHintTimer);
    analyzeBtn.disabled = false;
  }
});

function renderResults(output) {
  const top = output.slice(0, 5);
  resultsList.innerHTML = "";

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
      resultsList.querySelectorAll("li").forEach((el) => el.classList.remove("active"));
      li.classList.add("active");
      showDetail(style);
    });
    resultsList.appendChild(li);
  });

  resultsSection.hidden = false;
  statusSection.hidden = true;
  if (top.length > 0) {
    showDetail(labelToStyle.get(top[0].label));
  }
}

function showDetail(style) {
  if (!style) {
    detailCard.hidden = true;
    return;
  }
  detailName.textContent = style.name;
  detailEra.textContent = style.era || "";
  detailGeometry.textContent = style.geometry;
  detailStructure.textContent = style.structure;

  detailExamples.innerHTML = "";
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

    detailExamples.appendChild(li);

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

  detailCard.hidden = false;
}

loadStyles().catch((err) => {
  console.error(err);
  statusText.textContent = "知識庫載入失敗，請重新整理頁面";
  statusSection.hidden = false;
});

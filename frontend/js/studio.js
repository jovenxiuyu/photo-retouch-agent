/**
 * 生图工作台 —— 一句话进，一张真图出。
 *
 * 三条规则贯穿整个文件，改动时不要破坏它们：
 *
 *  1. 不自动重试。每一次生成就是一次真实付费请求，失败只把错误显示出来，
 *     绝不偷偷再发一次。
 *  2. 先报价、后花钱。点「生成」只调用 `/studio/quote`（不联网、不花钱），
 *     只有人在费用弹窗里点了「确认并生成」，才会带 `confirmPaid` 调
 *     `/studio/render`。点「取消」不会产生任何请求。
 *  3. 服务端回来的文字（提示词回显、报错、模型名）一律用 textContent 写入，
 *     不拼进 innerHTML。
 */

import { API_BASE, assetUrl, uploadPhoto } from "./api.js";

const EDIT = "edit";
const TEXT = "text2image";

/** 服务端起不来的兜底值，正常会被 `/studio/status` 覆盖。 */
const FALLBACK_MAX_RENDERS = 4;
const FALLBACK_MAX_UPLOAD_MB = 10;

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

/** 常用提示词示例。改图示例都写清「保留什么」，这样结果更容易对照检查。 */
const PROMPT_CHIPS = {
  [EDIT]: [
    {
      label: "换成海边清晨",
      text: "把背景换成清晨的海边，光线柔和自然，人物面部、发型和衣服保持不变。",
    },
    {
      label: "换米白针织外套",
      text: "把身上的衣服换成米白色针织外套，领口和版型自然贴合，人物面部保持不变。",
    },
    {
      label: "胶片质感暖调",
      text: "把整张照片调成胶片质感、暖色调，轻微颗粒感，肤色自然不过曝。",
    },
    {
      label: "去掉画面杂物",
      text: "去掉画面里多余的杂物和背景路人，保持主体人物和背景结构不变。",
    },
  ],
  [TEXT]: [
    {
      label: "海边黄昏写真",
      text: "一位女性站在海边黄昏的沙滩上，逆光轮廓，浅景深，电影感，写真级人像摄影。",
    },
    {
      label: "北欧风客厅",
      text: "北欧风格客厅，原木地板，大面积落地窗自然光，米白沙发和绿植，室内摄影。",
    },
    {
      label: "手冲咖啡静物",
      text: "一杯手冲咖啡放在木质桌面上，晨光斜射，旁边散落咖啡豆和手冲壶，静物特写。",
    },
    {
      label: "产品白底图",
      text: "一只简约陶瓷马克杯放在纯白背景上，柔和影棚光，商业产品摄影。",
    },
  ],
};

const state = {
  configured: false,
  models: [],
  maxRenders: FALLBACK_MAX_RENDERS,
  maxUploadMb: FALLBACK_MAX_UPLOAD_MB,
  mode: EDIT,
  source: null,
  busy: false,
  lastRequest: null,
  lastResult: null,
};

const dom = {};

function cacheDom() {
  const ids = [
    "providerPill",
    "providerText",
    "sourcePanel",
    "sourceHint",
    "dropzone",
    "sourcePreview",
    "dropzoneEmpty",
    "fileInput",
    "sourceMeta",
    "sourceMetaText",
    "clearSource",
    "modeSwitch",
    "promptInput",
    "promptChips",
    "modelSelect",
    "modelNote",
    "sizeLabel",
    "sizeSelect",
    "negativeInput",
    "countSelect",
    "costPreview",
    "generateButton",
    "resultHint",
    "resultStage",
    "compare",
    "compareSource",
    "compareResult",
    "compareResultCaption",
    "resultFacts",
    "resultActions",
    "downloadButton",
    "reuseButton",
    "historyTitle",
    "history",
    "confirmBackdrop",
    "confirmFacts",
    "confirmCancel",
    "confirmAccept",
  ];
  for (const id of ids) dom[id] = document.getElementById(id);
}

// ------------------------------------------------------------------ 小工具

/** 服务端文案只经这里进入界面：永远当纯文本，绝不解析成 HTML。 */
function text(tag, value, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = value;
  return node;
}

function renderPairs(list, rows) {
  list.replaceChildren();
  for (const [label, value] of rows) {
    if (value === null || value === undefined || value === "") continue;
    list.append(text("dt", label), text("dd", String(value)));
  }
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", { hour12: false });
}

function filenameOf(url) {
  const clean = String(url || "").split("?")[0];
  return clean.slice(clean.lastIndexOf("/") + 1) || "studio.png";
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload && payload.detail;
    throw new Error(
      typeof detail === "string" && detail ? detail : `请求失败（HTTP ${response.status}）`
    );
  }
  if (payload === null) throw new Error("服务端返回了无法解析的内容。");
  return payload;
}

function post(body) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// -------------------------------------------------------------------- 状态

function modelById(modelId) {
  return state.models.find((model) => model.id === modelId) || null;
}

function modelsForMode(mode) {
  return state.models.filter((model) => model.mode === mode);
}

function currentModel() {
  return modelById(dom.modelSelect.value);
}

function sourceReady() {
  return state.mode === TEXT || Boolean(state.source);
}

function canGenerate() {
  return (
    state.configured &&
    !state.busy &&
    dom.promptInput.value.trim().length > 0 &&
    sourceReady()
  );
}

function refreshGenerateButton() {
  dom.generateButton.disabled = !canGenerate();
  if (state.busy) {
    dom.generateButton.textContent = "正在生成…";
    return;
  }
  if (!state.configured) {
    dom.generateButton.textContent = "还没配置 API Key";
    return;
  }
  dom.generateButton.textContent = "生成（需确认费用）";
}

function setBusy(busy) {
  state.busy = busy;
  dom.generateButton.classList.toggle("is-busy", busy);
  refreshGenerateButton();
}

/**
 * 把「要点几次付费请求」讲清楚。这里只报次数，不编单价：每张的真实价格
 * 以硅基流动账单为准，写死一个数字就会和账单对不上。
 */
function refreshCostPreview() {
  const model = currentModel();
  const count = Number(dom.countSelect.value || 1);
  const node = dom.costPreview;
  node.classList.remove("is-error");

  if (!state.configured) {
    node.textContent = "还没读到 SILICONFLOW_API_KEY，生成按钮暂时不可用。";
    return;
  }
  if (!model) {
    node.textContent = "先选一个模型。";
    return;
  }
  if (!dom.promptInput.value.trim()) {
    node.textContent = "先写下你要什么。";
    return;
  }
  if (state.mode === EDIT && !state.source) {
    node.textContent = "改图模式要先上传一张输入图。";
    return;
  }

  node.replaceChildren();
  node.append(
    text("span", `将用 `),
    text("strong", model.label),
    text(
      "span",
      `${state.mode === EDIT ? "改图" : "文生图"}，共 ${count} 次真实付费请求。`
    ),
    document.createElement("br"),
    text("span", "点「生成」会先弹出费用确认，确认后才真的发出请求。")
  );
}

function showCostError(message) {
  dom.costPreview.classList.add("is-error");
  dom.costPreview.textContent = message;
}

// -------------------------------------------------------------------- 输入图

function clearSource() {
  if (state.source && state.source.objectUrl) URL.revokeObjectURL(state.source.objectUrl);
  state.source = null;
  dom.sourcePreview.hidden = true;
  dom.sourcePreview.removeAttribute("src");
  dom.dropzoneEmpty.hidden = false;
  dom.sourceMeta.hidden = true;
  dom.sourceMetaText.textContent = "";
  refreshGenerateButton();
  refreshCostPreview();
}

function setSourcePreview(label, objectUrl) {
  dom.sourcePreview.src = objectUrl;
  dom.sourcePreview.hidden = false;
  dom.dropzoneEmpty.hidden = true;
  dom.sourceMeta.hidden = false;
  dom.sourceMetaText.textContent = label;
}

async function acceptFile(file) {
  if (!file) return;
  if (state.mode === TEXT) return;
  if (!ACCEPTED_TYPES.includes(file.type)) {
    dom.sourceMeta.hidden = false;
    dom.sourceMetaText.textContent = "只支持 JPG / PNG / WebP。";
    return;
  }
  if (file.size > state.maxUploadMb * 1024 * 1024) {
    dom.sourceMeta.hidden = false;
    dom.sourceMetaText.textContent = `图片超过 ${state.maxUploadMb}MB，换一张小点的。`;
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  setSourcePreview("正在上传…", objectUrl);
  dom.dropzone.classList.add("is-disabled");
  try {
    const asset = await uploadPhoto(file);
    if (state.source && state.source.objectUrl) URL.revokeObjectURL(state.source.objectUrl);
    state.source = {
      imageId: asset.imageId,
      url: asset.url,
      filename: asset.filename,
      width: asset.width,
      height: asset.height,
      bytes: asset.sizeBytes,
      objectUrl,
    };
    setSourcePreview(
      `${asset.filename} · ${asset.width}×${asset.height} · ${formatBytes(asset.sizeBytes)}`,
      objectUrl
    );
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    dom.sourcePreview.hidden = true;
    dom.dropzoneEmpty.hidden = false;
    dom.sourceMeta.hidden = false;
    dom.sourceMetaText.textContent = `上传失败：${error.message}`;
  } finally {
    dom.dropzone.classList.remove("is-disabled");
    refreshGenerateButton();
    refreshCostPreview();
  }
}

// ---------------------------------------------------------------- 模式与参数

function fillModelOptions() {
  const models = modelsForMode(state.mode);
  dom.modelSelect.replaceChildren();
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label;
    dom.modelSelect.append(option);
  }
  if (models.length) dom.modelSelect.value = models[0].id;
  refreshModelNote();
}

function refreshModelNote() {
  const model = currentModel();
  if (!model) {
    dom.modelNote.textContent = "";
    return;
  }
  dom.modelNote.textContent = model.note || "";
}

function fillSizeOptions() {
  const model = currentModel();
  const sizes = (model && model.imageSizes) || [];
  const showSize = state.mode === TEXT && sizes.length > 0;
  dom.sizeLabel.hidden = !showSize;
  dom.sizeSelect.hidden = !showSize;
  dom.sizeSelect.replaceChildren();
  for (const size of sizes) {
    const option = document.createElement("option");
    option.value = size;
    option.textContent = size;
    dom.sizeSelect.append(option);
  }
}

function fillCountOptions() {
  const max = Math.max(1, state.maxRenders || FALLBACK_MAX_RENDERS);
  const previous = Number(dom.countSelect.value || 1);
  dom.countSelect.replaceChildren();
  for (let index = 1; index <= max; index += 1) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index} 张`;
    dom.countSelect.append(option);
  }
  dom.countSelect.value = String(Math.min(Math.max(previous, 1), max));
}

function fillChips() {
  dom.promptChips.replaceChildren();
  for (const chip of PROMPT_CHIPS[state.mode]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.textContent = chip.label;
    button.addEventListener("click", () => {
      dom.promptInput.value = chip.text;
      dom.promptInput.focus();
      refreshGenerateButton();
      refreshCostPreview();
    });
    dom.promptChips.append(button);
  }
}

/**
 * 切模式。文生图模式下输入图会被禁用而不是隐藏：后端在文生图时根本不发送
 * 输入图，与其让上传静静失效，不如把这件事写在界面上。
 */
function setMode(mode) {
  state.mode = mode;
  for (const button of dom.modeSwitch.querySelectorAll(".mode-button")) {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  }

  const isEdit = mode === EDIT;
  dom.sourceHint.textContent = isEdit ? "改图必填" : "文生图不使用输入图";
  dom.dropzone.classList.toggle("is-disabled", !isEdit);

  fillModelOptions();
  fillSizeOptions();
  fillChips();
  refreshGenerateButton();
  refreshCostPreview();
}

// -------------------------------------------------------------------- 生成

function buildRequestBody() {
  const model = currentModel();
  const count = Number(dom.countSelect.value || 1);
  const body = {
    prompt: dom.promptInput.value.trim(),
    mode: state.mode,
    modelId: model ? model.id : null,
    negativePrompt: dom.negativeInput.value.trim(),
    count,
  };
  if (state.mode === EDIT && state.source) body.sourceImageId = state.source.imageId;
  if (state.mode === TEXT && dom.sizeSelect.value) body.imageSize = dom.sizeSelect.value;
  return body;
}

function previewRequest(request) {
  return {
    prompt: request.prompt,
    mode: request.mode,
    modelId: request.modelId,
    negativePrompt: request.negativePrompt,
    count: request.count,
  };
}

function openConfirm(quote, request) {
  renderPairs(dom.confirmFacts, [
    ["模型", quote.modelLabel],
    ["方式", quote.mode === EDIT ? "在原图上改" : "凭空生成"],
    ["成片张数", `${quote.count} 张`],
    ["付费请求", `${quote.paidRequests} 次`],
    ["画布尺寸", quote.imageSize || ""],
    ["输入图", state.source ? state.source.filename : ""],
    ["说明", quote.note],
  ]);
  dom.confirmBackdrop.hidden = false;
  dom.confirmAccept.disabled = false;
  dom.confirmAccept.focus();
  state.lastRequest = request;
}

function closeConfirm() {
  dom.confirmBackdrop.hidden = true;
}

async function requestQuote() {
  if (!canGenerate()) return;
  const request = buildRequestBody();
  dom.generateButton.classList.add("is-busy");
  dom.generateButton.disabled = true;
  try {
    const quote = await api("/studio/quote", post(request));
    openConfirm(quote, request);
  } catch (error) {
    showCostError(error.message);
  } finally {
    dom.generateButton.classList.remove("is-busy");
    refreshGenerateButton();
  }
}

function showStage(node) {
  if (factsNote) {
    factsNote.remove();
    factsNote = null;
  }
  dom.resultStage.replaceChildren(node);
  dom.resultStage.hidden = false;
  dom.compare.hidden = true;
}

function showGenerating(count) {
  const wrapper = document.createElement("div");
  wrapper.append(
    text("div", "", "spinner"),
    text("p", `正在生成，共 ${count} 张，每张都是一次真实付费请求…`, "empty-state")
  );
  dom.resultHint.textContent = "生成中";
  showStage(wrapper);
  dom.resultActions.hidden = true;
  dom.resultFacts.hidden = true;
}

function showError(message) {
  dom.resultHint.textContent = "出错了";
  showStage(text("p", message, "empty-state is-error"));
}

function showCompare(item, quote, sourceUrl) {
  dom.compareSource.src = sourceUrl || "";
  dom.compareSource.hidden = !sourceUrl;
  dom.compareResult.src = assetUrl(item.url);
  dom.compareResultCaption.textContent = `${item.modelLabel} · ${item.width}×${item.height}`;
  // 文生图没有原图，左右并排会空出一栏，这时收成单栏。
  dom.compare.classList.toggle("is-single", !sourceUrl);
  dom.compare.hidden = false;
  dom.resultStage.hidden = true;
  dom.resultHint.textContent = "生成完成";
}

/** 上一次渲染留下的失败提示，重新生成前必须先摘掉，否则会越堆越多。 */
let factsNote = null;

function renderFacts(item, quote, extraErrors) {
  renderPairs(dom.resultFacts, [
    ["模型", item.modelLabel],
    ["方式", item.mode === EDIT ? "在原图上改" : "凭空生成"],
    ["尺寸", `${item.width}×${item.height}`],
    ["文件大小", formatBytes(item.bytes)],
    ["随机种子", item.seed],
    ["任务 ID", item.requestId],
    ["生成时间", formatTime(item.generatedAt)],
    ["付费请求", `${quote.paidRequests} 次`],
  ]);
  dom.resultFacts.hidden = false;
  if (factsNote) {
    factsNote.remove();
    factsNote = null;
  }
  if (extraErrors && extraErrors.length) {
    factsNote = text(
      "p",
      `部分张数失败（未重试）：${extraErrors.join("；")}`,
      "confirm-warning"
    );
    dom.resultFacts.after(factsNote);
  }
}

async function runRender() {
  const request = state.lastRequest;
  if (!request) return;
  closeConfirm();
  setBusy(true);
  showGenerating(request.count);

  const body = { ...request, confirmPaid: true };
  try {
    const result = await api("/studio/render", post(body));
    state.lastResult = result;
    const item = result.items[0];
    showCompare(item, result.quote, state.source ? state.source.objectUrl : "");
    renderFacts(item, result.quote, result.errors);
    dom.downloadButton.href = assetUrl(item.url);
    dom.downloadButton.setAttribute("download", filenameOf(item.url));
    dom.resultActions.hidden = false;
    await loadHistory();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

function reuseLastRequest() {
  const request = state.lastRequest || (state.lastResult && state.lastResult.quote);
  if (!request) return;
  if (request.prompt) dom.promptInput.value = request.prompt;
  if (request.negativePrompt !== undefined) dom.negativeInput.value = request.negativePrompt;
  if (request.modelId && modelById(request.modelId)) {
    const model = modelById(request.modelId);
    if (model.mode !== state.mode) setMode(model.mode);
    dom.modelSelect.value = request.modelId;
    refreshModelNote();
    fillSizeOptions();
  }
  refreshGenerateButton();
  refreshCostPreview();
  dom.promptInput.focus();
}

// -------------------------------------------------------------------- 历史

async function loadHistory() {
  let entries = [];
  try {
    entries = await api("/studio/history?limit=24");
  } catch {
    entries = [];
  }
  dom.history.replaceChildren();
  if (!entries.length) {
    dom.historyTitle.hidden = true;
    return;
  }
  dom.historyTitle.hidden = false;
  for (const entry of entries) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    button.title = `${entry.model || ""} · ${formatTime(entry.generatedAt)}`;
    const image = document.createElement("img");
    image.src = assetUrl(entry.url);
    image.alt = entry.imageId;
    image.loading = "lazy";
    button.append(image);
    button.addEventListener("click", () => {
      const shown = document.createElement("img");
      shown.src = assetUrl(entry.url);
      shown.alt = entry.imageId;
      dom.resultHint.textContent = "历史成片";
      showStage(shown);
      renderPairs(dom.resultFacts, [
        ["模型", entry.model],
        ["尺寸", `${entry.width}×${entry.height}`],
        ["文件大小", formatBytes(entry.bytes)],
        ["任务 ID", entry.requestId],
        ["生成时间", formatTime(entry.generatedAt)],
      ]);
      dom.resultFacts.hidden = false;
      dom.downloadButton.href = assetUrl(entry.url);
      dom.downloadButton.setAttribute("download", filenameOf(entry.url));
      dom.resultActions.hidden = false;
    });
    dom.history.append(button);
  }
}

// -------------------------------------------------------------------- 初始化

async function loadStatus() {
  try {
    const status = await api("/studio/status");
    state.configured = Boolean(status.configured);
    state.models = status.models || [];
    state.maxRenders = status.maxRenders || FALLBACK_MAX_RENDERS;
    state.maxUploadMb = status.maxUploadMb || FALLBACK_MAX_UPLOAD_MB;
    dom.providerPill.classList.toggle("is-ok", state.configured);
    dom.providerPill.classList.toggle("is-bad", !state.configured);
    dom.providerText.textContent = state.configured
      ? `SiliconFlow 已连接 · ${status.models.length} 个模型`
      : "没有读到 API Key";
  } catch (error) {
    dom.providerPill.classList.add("is-bad");
    dom.providerText.textContent = `连不上服务：${error.message}`;
    state.configured = false;
  }
  fillCountOptions();
  setMode(EDIT);
}

function wireDropzone() {
  dom.dropzone.addEventListener("click", () => {
    if (state.mode === TEXT) return;
    dom.fileInput.click();
  });
  dom.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (state.mode !== TEXT) dom.fileInput.click();
    }
  });
  dom.dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (state.mode !== TEXT) dom.dropzone.classList.add("is-dragging");
  });
  dom.dropzone.addEventListener("dragleave", () => {
    dom.dropzone.classList.remove("is-dragging");
  });
  dom.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dom.dropzone.classList.remove("is-dragging");
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    acceptFile(file);
  });
  dom.fileInput.addEventListener("change", () => {
    acceptFile(dom.fileInput.files && dom.fileInput.files[0]);
    dom.fileInput.value = "";
  });
  dom.clearSource.addEventListener("click", clearSource);
  document.addEventListener("paste", (event) => {
    if (state.mode === TEXT) return;
    const items = event.clipboardData && event.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.type && item.type.startsWith("image/")) {
        acceptFile(item.getAsFile());
        break;
      }
    }
  });
}

function wireControls() {
  dom.modeSwitch.addEventListener("click", (event) => {
    const button = event.target.closest(".mode-button");
    if (button && button.dataset.mode !== state.mode) setMode(button.dataset.mode);
  });
  dom.promptInput.addEventListener("input", () => {
    refreshGenerateButton();
    refreshCostPreview();
  });
  dom.negativeInput.addEventListener("input", refreshCostPreview);
  dom.countSelect.addEventListener("change", refreshCostPreview);
  dom.modelSelect.addEventListener("change", () => {
    refreshModelNote();
    fillSizeOptions();
    refreshCostPreview();
  });
  dom.generateButton.addEventListener("click", requestQuote);
  dom.confirmCancel.addEventListener("click", closeConfirm);
  dom.confirmAccept.addEventListener("click", () => {
    dom.confirmAccept.disabled = true;
    runRender();
  });
  dom.reuseButton.addEventListener("click", reuseLastRequest);
  dom.confirmBackdrop.addEventListener("click", (event) => {
    if (event.target === dom.confirmBackdrop) closeConfirm();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !dom.confirmBackdrop.hidden) closeConfirm();
  });
}

function init() {
  cacheDom();
  wireDropzone();
  wireControls();
  refreshGenerateButton();
  refreshCostPreview();
  loadStatus();
  loadHistory();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

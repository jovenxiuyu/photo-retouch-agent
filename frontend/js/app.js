import { state, on, categories, allTools, toolMap } from "./state.js";

import {
  analyzePhoto,
  assetUrl,
  createJob,
  createPlans,
  createStylingProposals,
  getJob,
  getProviderCapabilities,
  getStylingProviders,
  refineJob,
  uploadPhoto,
} from "./api.js";

import {
  designStyling,
  estimateStylingCost,
  executeAiRetouch,
  executeStyling,
  normalizeStylingProposals,
  refineStyling,
  validateModelSelection,
} from "./ai-retouch.mjs";

import {
  buildExportFilename,
  canvasToBlob,
  triggerBlobDownload,
} from "./export-image.mjs";

import { renderLocalRetouch } from "./local-retouch.mjs";


const elements = {
  categoryTitle: document.querySelector("#categoryTitle"),
  quickTags: document.querySelector("#quickTags"),
  adjustmentList: document.querySelector("#adjustmentList"),
  toolSearch: document.querySelector("#toolSearch"),
  localPanel: document.querySelector("#localPanel"),
  aiPanel: document.querySelector("#aiPanel"),
  modeNote: document.querySelector("#modeNote"),
  previewCanvas: document.querySelector("#previewCanvas"),
  beforeImage: document.querySelector("#beforeImage"),
  afterLayer: document.querySelector("#afterLayer"),
  compareLine: document.querySelector("#compareLine"),
  compareRange: document.querySelector("#compareRange"),
  compareButton: document.querySelector("#compareButton"),
  changeList: document.querySelector("#changeList"),
  changeCount: document.querySelector("#changeCount"),
  historyState: document.querySelector("#historyState"),
  footerStatus: document.querySelector("#footerStatus"),
  canvasStatus: document.querySelector("#canvasStatus"),
  fileInput: document.querySelector("#fileInput"),
  uploadButton: document.querySelector("#uploadButton"),
  exportButton: document.querySelector("#exportButton"),
  resetButton: document.querySelector("#resetButton"),
  resetCategoryButton: document.querySelector("#resetCategoryButton"),
  undoButton: document.querySelector("#undoButton"),
  aiPrompt: document.querySelector("#aiPrompt"),
  promptCount: document.querySelector("#promptCount"),
  generateButton: document.querySelector("#generateButton"),
  floatingTip: document.querySelector("#floatingTip"),
  aiFeedback: document.querySelector("#aiFeedback"),
  modelConfigState: document.querySelector("#modelConfigState"),
  brainProviderSelect: document.querySelector("#brainProviderSelect"),
  actionProviderSelect: document.querySelector("#actionProviderSelect"),
  brainApiKeyField: document.querySelector("#brainApiKeyField"),
  actionApiKeyField: document.querySelector("#actionApiKeyField"),
  brainApiKeyInput: document.querySelector("#brainApiKeyInput"),
  actionApiKeyInput: document.querySelector("#actionApiKeyInput"),
  brainWorkspaceField: document.querySelector("#brainWorkspaceField"),
  actionWorkspaceField: document.querySelector("#actionWorkspaceField"),
  brainWorkspaceInput: document.querySelector("#brainWorkspaceInput"),
  actionWorkspaceInput: document.querySelector("#actionWorkspaceInput"),
  documentName: document.querySelector("#documentName"),
  stylingSource: document.querySelector("#stylingSource"),
  stylingDesignButton: document.querySelector("#stylingDesignButton"),
  stylingTemplateButton: document.querySelector("#stylingTemplateButton"),
  stylingCostNote: document.querySelector("#stylingCostNote"),
  stylingGrid: document.querySelector("#stylingGrid"),
  stylingRefine: document.querySelector("#stylingRefine"),
  stylingInstruction: document.querySelector("#stylingInstruction"),
  stylingRefineButton: document.querySelector("#stylingRefineButton"),
  stylingFeedback: document.querySelector("#stylingFeedback"),
};


const aiApi = {
  uploadPhoto,
  analyzePhoto,
  createPlans,
  createJob,
  createStylingProposals,
  getJob,
  refineJob,
};


function populateProviderSelect(select, providers, preferredId) {
  const previous = select.value;
  select.replaceChildren(
    ...providers.map((provider) => {
      const option = document.createElement("option");
      option.value = provider.id;
      option.textContent = `${provider.label} · ${provider.model}`;
      return option;
    }),
  );
  const availableIds = new Set(providers.map((p) => p.id));
  select.value = availableIds.has(previous)
    ? previous
    : availableIds.has(preferredId)
      ? preferredId
      : providers[0]?.id || "";
}


function selectedProvider(layer) {
  const isBrain = layer === "brain";
  const providers = isBrain ? state.capabilities.brainProviders : state.capabilities.actionProviders;
  const select = isBrain ? elements.brainProviderSelect : elements.actionProviderSelect;
  return providers.find((p) => p.id === select.value);
}


function modelRequest(layer) {
  const isBrain = layer === "brain";
  const providerName = isBrain ? elements.brainProviderSelect.value : elements.actionProviderSelect.value;
  const directApiKey = (isBrain ? elements.brainApiKeyInput.value : elements.actionApiKeyInput.value).trim();
  const brainProvider = elements.brainProviderSelect.value;
  const sharedKeyActions = {
    qwen: ["qwen", "wan"],
    doubao: ["seedream"],
    siliconflow: ["siliconflow"],
  };
  const canReuseBrainKey =
    !isBrain && (sharedKeyActions[brainProvider] || []).includes(providerName);
  return {
    name: providerName,
    apiKey: directApiKey || (canReuseBrainKey ? elements.brainApiKeyInput.value.trim() : ""),
    workspaceId: (isBrain ? elements.brainWorkspaceInput.value : elements.actionWorkspaceInput.value).trim(),
  };
}


function syncModelConfig() {
  const brain = selectedProvider("brain");
  const action = selectedProvider("action");
  elements.brainApiKeyField.hidden = !brain?.requiresApiKey;
  elements.actionApiKeyField.hidden = !action?.requiresApiKey;
  elements.brainWorkspaceField.hidden = !brain?.workspaceSupported;
  elements.actionWorkspaceField.hidden = !action?.workspaceSupported;
  elements.brainApiKeyInput.placeholder = brain?.configured ? "服务端已配置，可留空" : "请输入本次会话 Key";
  elements.actionApiKeyInput.placeholder = action?.configured ? "服务端已配置，可留空" : "可留空复用同平台视觉 Key";

  const br = modelRequest("brain");
  const ar = modelRequest("action");
  const ready = brain && action &&
    (!brain.requiresApiKey || brain.configured || br.apiKey) &&
    (!action.requiresApiKey || action.configured || ar.apiKey);
  elements.modelConfigState.textContent = ready ? "已就绪" : "需要配置";
  elements.modelConfigState.dataset.state = ready ? "ready" : "error";
}


async function loadProviderCapabilities() {
  try {
    state.capabilities = await getProviderCapabilities();
    populateProviderSelect(elements.brainProviderSelect, state.capabilities.brainProviders || [], "qwen");
    populateProviderSelect(elements.actionProviderSelect, state.capabilities.actionProviders || [], "qwen");
    syncModelConfig();
  } catch (error) {
    elements.modelConfigState.textContent = "服务不可用";
    elements.modelConfigState.dataset.state = "error";
    elements.aiFeedback.textContent = error.message;
    elements.aiFeedback.className = "ai-feedback error";
  }
}


async function currentSourceFile() {
  if (state.sourceFile) return state.sourceFile;
  const src = elements.beforeImage.currentSrc || elements.beforeImage.src;
  if (!src) throw new Error("请先选择一张照片。");
  const response = await fetch(src);
  if (!response.ok) throw new Error("无法读取当前照片，请重新选择图片。");
  const blob = await response.blob();
  const extension = blob.type === "image/jpeg" ? "jpg" : "png";
  state.sourceFile = new File([blob], `portrait.${extension}`, {
    type: blob.type || "image/png",
  });
  return state.sourceFile;
}


function drawImageToPreview(image) {
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  elements.previewCanvas.width = width;
  elements.previewCanvas.height = height;
  const ctx = elements.previewCanvas.getContext("2d", { alpha: false });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(image, 0, 0, width, height);
}


function showAiResult(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.addEventListener("load", () => {
      state.aiResultImage = img;
      state.aiResultUrl = url;
      state.displayingAiResult = true;
      drawImageToPreview(img);
      resolve();
    });
    img.addEventListener("error", () => reject(new Error("AI 结果图片加载失败。")));
    img.src = url;
  });
}


async function exportCurrentImage(event) {
  event.preventDefault();
  const isAi = state.mode === "ai" && state.displayingAiResult && state.aiResultUrl;
  elements.footerStatus.textContent = "正在准备导出图片";
  try {
    let blob;
    if (isAi) {
      const response = await fetch(state.aiResultUrl);
      if (!response.ok) throw new Error("无法读取 AI 结果图片。");
      blob = await response.blob();
    } else {
      blob = await canvasToBlob(elements.previewCanvas, "image/png");
    }
    const fileName = buildExportFilename(state.sourceName || "portrait", isAi ? "ai" : "local", blob.type);
    triggerBlobDownload(blob, fileName);
    elements.footerStatus.textContent = `已导出 ${fileName}`;
  } catch (error) {
    elements.footerStatus.textContent = error.message;
  }
}


function renderCategory() {
  const cat = categories[state.category];
  elements.categoryTitle.textContent = cat.title;
  const tools = [...cat.tools];

  elements.quickTags.innerHTML = tools
    .map(
      (tool) => `
        <button class="tool-tag ${state.values[tool.id] > 0 ? "active" : ""}"
                data-tool-tag="${tool.id}" type="button">${tool.label}</button>`,
    ).join("");

  elements.adjustmentList.innerHTML = tools
    .map((tool) => {
      const v = state.values[tool.id];
      return `
        <label class="adjustment">
          <span class="adjustment-head">
            <span class="adjustment-label"><i>${tool.short}</i><span>${tool.label}</span></span>
            <b class="adjustment-value" data-value-for="${tool.id}">${v}</b>
          </span>
          <span class="slider-row">
            <span>−</span>
            <input type="range" min="0" max="100" value="${v}" data-tool-range="${tool.id}"
                   style="--range-progress: ${v}%" />
            <span>＋</span>
          </span>
        </label>`;
    }).join("");

  elements.quickTags.querySelectorAll("[data-tool-tag]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.toolTag;
      const prev = state.values[id];
      state.history = [...state.history, { id, value: prev }];
      state.values = { ...state.values, [id]: prev > 0 ? 0 : 20 };
      renderCategory();
      updatePreview();
    });
  });

  elements.adjustmentList.querySelectorAll("[data-tool-range]").forEach((input) => {
    input.addEventListener("pointerdown", () => {
      input.dataset.startValue = String(state.values[input.dataset.toolRange]);
    });
    input.addEventListener("input", () => {
      const id = input.dataset.toolRange;
      state.values = { ...state.values, [id]: Number(input.value) };
      input.style.setProperty("--range-progress", `${input.value}%`);
      const valEl = elements.adjustmentList.querySelector(`[data-value-for="${id}"]`);
      if (valEl) valEl.textContent = input.value;
      const tag = elements.quickTags.querySelector(`[data-tool-tag="${id}"]`);
      if (tag) tag.classList.toggle("active", Number(input.value) > 0);
      updatePreview();
    });
    input.addEventListener("change", () => {
      const start = Number(input.dataset.startValue ?? 0);
      if (start !== Number(input.value)) {
        state.history = [...state.history, { id: input.dataset.toolRange, value: start }];
      }
    });
  });
}


let renderFrame = 0;

function scheduleLocalRender() {
  if (!elements.beforeImage.complete || !elements.beforeImage.naturalWidth) return;
  if (state.mode === "ai" && state.displayingAiResult) return;
  window.cancelAnimationFrame(renderFrame);
  renderFrame = window.requestAnimationFrame(() => {
    renderLocalRetouch(elements.previewCanvas, elements.beforeImage, state.values);
  });
}


function updatePreview() {
  scheduleLocalRender();
  renderChanges();
}


function renderChanges() {
  const changes = Object.entries(state.values)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]);
  elements.changeCount.textContent = `${changes.length} 项`;
  elements.changeList.innerHTML = changes.length
    ? changes.slice(0, 5).map(([id, v]) => {
        const t = toolMap[id];
        return `<div class="change-item"><span>${t.short}</span><div><strong>${t.label}</strong><small>${t.hint}</small></div><b>+${v}</b></div>`;
      }).join("")
    : `<div class="change-item"><span>原</span><div><strong>尚未调整</strong><small>选择左侧标签开始美化</small></div><b>0</b></div>`;
  elements.historyState.textContent = changes.length
    ? changes.slice(0, 4).map(([id, v]) => `${toolMap[id].label} ${v}`).join(" · ")
    : "尚未添加美化参数";
}


function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  const isLocal = mode === "local";
  elements.localPanel.hidden = !isLocal;
  elements.aiPanel.hidden = isLocal;
  elements.modeNote.innerHTML = isLocal
    ? '<span class="local-dot"></span>本地实时预览 · 图片无需上传模型'
    : '<span class="local-dot"></span>云端 AI 精修 · 生成前会确认参数';
  elements.canvasStatus.classList.toggle("ai", !isLocal);
  elements.canvasStatus.innerHTML = isLocal
    ? "<span></span>实时预览已开启"
    : state.displayingAiResult ? "<span></span>AI 美化结果" : "<span></span>AI 模式 · 当前显示手动预览";
  elements.footerStatus.textContent = isLocal ? "实时美化已就绪" : "AI 精修已就绪";
  if (isLocal) {
    state.displayingAiResult = false;
    scheduleLocalRender();
  } else if (state.aiResultImage) {
    state.displayingAiResult = true;
    drawImageToPreview(state.aiResultImage);
  }
}


function resetTools(tools) {
  const snapshot = {};
  tools.forEach((t) => { snapshot[t.id] = state.values[t.id]; });
  const newValues = { ...state.values };
  tools.forEach((t) => { newValues[t.id] = 0; });
  state.values = newValues;
  state.history = [...state.history, { snapshot }];
  renderCategory();
  updatePreview();
}


function applyPreset(preset) {
  const presets = {
    natural: { smooth: 24, skinTone: 16, whiten: 6, sculpt: 8, eyeBags: 10 },
    clear: { smooth: 18, skinTone: 20, whiten: 18, brightEyes: 14, clarity: 12 },
    sculpt: { smooth: 12, sculpt: 28, jawline: 14, noseBridge: 12, contrast: 10 },
    warm: { smooth: 20, skinTone: 24, blush: 14, warmth: 18, lipColor: 9 },
  };
  state.values = { ...state.values, ...presets[preset] };
  updatePreview();
}


function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}


function currentStylingProposals() {
  return Array.isArray(state.stylingProposals) ? state.stylingProposals : [];
}


function selectedStylingProposal() {
  return currentStylingProposals().find((look) => look.planId === state.selectedStylingId) || null;
}


function selectStylingProposal(planId) {
  state.selectedStylingId = planId;
  elements.stylingGrid.querySelectorAll("[data-styling-card]").forEach((card) => {
    card.classList.toggle("active", card.dataset.stylingCard === planId);
  });
  updateStylingCostNote();
}


function updateStylingCostNote() {
  const looks = currentStylingProposals();
  if (!looks.length) return;
  const mode = stylingDesignMode();
  const cost = estimateStylingCost(looks, { selectedCount: 1, mode });
  const selected = selectedStylingProposal();
  elements.stylingCostNote.textContent = selected
    ? `已选「${selected.name || selected.planId}」：${cost.note} 未点击生成前不会产生任何请求。`
    : cost.note;
}


function stylingDesignMode() {
  return currentStylingProposals()[0]?.designMode === "ai" ? "ai" : "template";
}


function designSourceLabel(proposal) {
  const isAi = proposal.designMode === "ai";
  const provider = proposal.sourceProvider ? ` · ${proposal.sourceProvider}` : "";
  return isAi
    ? { text: `AI 个性化设计${provider}`, className: "styling-source-badge ai" }
    : { text: "免费模板预览", className: "styling-source-badge template" };
}


function referenceSourceLabel(proposal) {
  const presets = Array.isArray(proposal.referencePresets) ? proposal.referencePresets : [];
  if (!presets.length) return "";
  const source = proposal.stylingVocabularySource || presets[0].sourceFile || "stylekit";
  return `风格库来源：${source}`;
}


function renderStylingCards(proposals) {
  const looks = Array.isArray(proposals) && proposals.length ? proposals : currentStylingProposals();
  elements.stylingGrid.innerHTML = looks
    .map((look) => {
      const active = look.planId === state.selectedStylingId;
      const presets = Array.isArray(look.referencePresets) ? look.referencePresets : [];
      const source = designSourceLabel(look);
      const presetChips = presets
        .slice(0, 6)
        .map((preset) => `<span class="styling-chip">${escapeHtml(preset.nameZh || preset.name || preset.presetId)}</span>`)
        .join("");
      return `
        <article class="styling-card ${active ? "active" : ""}" data-styling-card="${escapeHtml(look.planId)}" data-design-mode="${escapeHtml(look.designMode || "template")}">
          <header class="styling-card-head">
            <strong>${escapeHtml(look.name || look.title)}</strong>
            <em>${escapeHtml(referenceSourceLabel(look) || "stylekit")}</em>
          </header>
          <span class="${source.className}">${escapeHtml(source.text)}</span>
          <span class="styling-card-direction">${escapeHtml(look.designDirection || look.designAxis || "")}</span>
          <dl>
            <dt>发型</dt><dd>${escapeHtml(look.hairPlan)}</dd>
            <dt>妆容</dt><dd>${escapeHtml(look.makeupPlan)}</dd>
            <dt>穿搭</dt><dd>${escapeHtml(look.outfitPlan)}</dd>
            <dt>配饰</dt><dd>${escapeHtml(look.accessoryPlan || "无")}</dd>
            <dt>配色</dt><dd>${escapeHtml(look.palette || "")}</dd>
          </dl>
          <p class="styling-why">为什么适合当前人物：${escapeHtml(look.whyItFits || look.designReason || "")}</p>
          ${presetChips ? `<div class="styling-chips">${presetChips}</div>` : ""}
          <button type="button" data-styling-generate="${escapeHtml(look.planId)}">生成这套造型</button>
        </article>`;
    })
    .join("");

  elements.stylingGrid.querySelectorAll("[data-styling-card]").forEach((card) => {
    card.addEventListener("click", () => selectStylingProposal(card.dataset.stylingCard));
  });
  elements.stylingGrid.querySelectorAll("[data-styling-generate]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      selectStylingProposal(btn.dataset.stylingGenerate);
      generateStyling(btn.dataset.stylingGenerate);
    });
  });
}


function setStylingBusy(busy) {
  state.stylingBusy = busy;
  elements.stylingDesignButton.disabled = busy;
  if (elements.stylingTemplateButton) elements.stylingTemplateButton.disabled = busy;
  if (elements.stylingRefineButton) elements.stylingRefineButton.disabled = busy;
}


async function designStylingFlow(mode = "template") {
  if (state.stylingBusy) return;
  const isAi = mode === "ai";
  const userInstruction = (elements.stylingInstruction?.value || "").trim();
  setStylingBusy(true);
  elements.stylingFeedback.textContent = isAi
    ? "正在调用 AI 分析人物视觉特征并个性化设计三套造型（预计 1 次模型调用）。"
    : "正在使用免费模板组合生成三套造型预览（0 次模型调用，不产生任何费用）。";
  elements.stylingFeedback.className = "ai-feedback";

  try {
    const brain = modelRequest("brain");
    const action = modelRequest("action");
    validateModelSelection(state.capabilities, brain, action);
    const file = await currentSourceFile();
    const result = await designStyling({ file, brain, action, mode, userInstruction, api: aiApi });
    state.stylingPhoto = result.photo;
    state.stylingAnalysis = result.analysis;
    state.stylingProposals = result.proposals;
    state.selectedStylingId = result.proposals[0].planId;
    state.generatedStylingJobId = null;
    elements.stylingGrid.hidden = false;
    elements.stylingRefine.hidden = true;
    renderStylingCards(result.proposals);
    selectStylingProposal(state.selectedStylingId);
    elements.stylingFeedback.textContent = isAi
      ? `AI 个性化设计完成 ${result.proposals.length} 套造型（1 次模型调用），选择一套再生成图片。`
      : `已生成 ${result.proposals.length} 套免费模板预览（0 次模型调用），选择一套再生成图片。`;
    elements.stylingFeedback.className = "ai-feedback success";
  } catch (error) {
    // A failed AI design must surface as an error — never silently fall back to
    // the local templates while pretending the AI produced the result.
    elements.stylingFeedback.textContent = isAi
      ? `AI 个性化设计失败：${error.message}（未生成或使用任何模板方案，可改用「免费模板预览」。）`
      : error.message;
    elements.stylingFeedback.className = "ai-feedback error";
  } finally {
    setStylingBusy(false);
  }
}


function describeFaceLock(faceLock) {
  if (!faceLock || typeof faceLock.ok !== "boolean") return "";
  const mode = faceLock.diagnostics?.mode;
  const suffix = mode ? `（${mode}）` : "";
  return faceLock.ok
    ? ` · 严格保脸校验通过${suffix}`
    : ` · 保脸校验未通过（${faceLock.status}）`;
}

function describeStylingFailure(error) {
  const faceLock = error?.faceLock;
  if (faceLock && faceLock.ok === false) {
    // The styling pixels may look nice, but the original face could not be
    // restored with confidence -- report the identity failure honestly.
    return `保脸失败（${faceLock.status}）：${faceLock.message || "未能把原始五官无痕还原。"} 生成图未通过严格人脸锁定校验，请人工检查保护区域或调整掩膜后重试。`;
  }
  return error?.message || "AI 造型生成失败。";
}

async function generateStyling(planId) {
  if (state.stylingBusy) return;
  const proposal = currentStylingProposals().find((look) => look.planId === planId);
  if (!proposal) return;
  if (!state.stylingPhoto) {
    elements.stylingFeedback.textContent = "请先让 AI 设计三套造型。";
    elements.stylingFeedback.className = "ai-feedback error";
    return;
  }

  const cost = estimateStylingCost(currentStylingProposals(), {
    selectedCount: 1,
    mode: stylingDesignMode(),
  });
  const userInstruction = (elements.stylingInstruction?.value || "").trim();
  setStylingBusy(true);
  elements.stylingFeedback.textContent = `即将生成「${proposal.name || planId}」：预计 ${cost.totalRequests} 次付费图像编辑请求（￥按服务商计费）。`;
  elements.stylingFeedback.className = "ai-feedback";
  elements.footerStatus.textContent = "AI 造型生成中";

  try {
    const brain = modelRequest("brain");
    const action = modelRequest("action");
    const result = await executeStyling({
      photo: state.stylingPhoto,
      proposals: currentStylingProposals(),
      selectedId: planId,
      userInstruction,
      brain,
      action,
      api: aiApi,
    });
    const resultUrl = assetUrl(result.job.outputUrls[0]);
    state.mode = "ai";
    await showAiResult(resultUrl);
    elements.exportButton.href = resultUrl;
    elements.canvasStatus.innerHTML = `<span></span>造型结果 · ${escapeHtml(result.proposal.name || planId)}`;
    elements.footerStatus.textContent = "造型结果已生成";
    state.generatedStylingJobId = result.job.jobId;
    elements.stylingRefine.hidden = false;
    const lockBadge = describeFaceLock(result.job.faceLock);
    elements.stylingFeedback.textContent = `「${result.proposal.name || planId}」生成成功 · ${result.job.modelName || ""}${lockBadge}。可继续提出局部修改。`;
    elements.stylingFeedback.className = "ai-feedback success";
  } catch (error) {
    elements.footerStatus.textContent = "AI 造型失败";
    elements.stylingFeedback.textContent = describeStylingFailure(error);
    elements.stylingFeedback.className = "ai-feedback error";
  } finally {
    setStylingBusy(false);
  }
}


async function refineStylingFlow() {
  if (state.stylingBusy) return;
  const userInstruction = (elements.stylingInstruction?.value || "").trim();
  if (!state.generatedStylingJobId) {
    elements.stylingFeedback.textContent = "请先生成一套造型，再进行二次修改。";
    elements.stylingFeedback.className = "ai-feedback error";
    return;
  }
  if (!userInstruction) {
    elements.stylingFeedback.textContent = "请填写你的修改要求，例如“头发再短一点”。";
    elements.stylingFeedback.className = "ai-feedback error";
    return;
  }

  setStylingBusy(true);
  elements.stylingFeedback.textContent = "即将基于上一版结果发起 1 次付费图像编辑请求。";
  elements.stylingFeedback.className = "ai-feedback";

  try {
    const brain = modelRequest("brain");
    const action = modelRequest("action");
    const result = await refineStyling({
      jobId: state.generatedStylingJobId,
      userInstruction,
      brain,
      action,
      api: aiApi,
    });
    const resultUrl = assetUrl(result.job.outputUrls[0]);
    state.mode = "ai";
    await showAiResult(resultUrl);
    elements.exportButton.href = resultUrl;
    elements.canvasStatus.innerHTML = "<span></span>造型二次修改结果";
    elements.footerStatus.textContent = "二次修改完成";
    state.generatedStylingJobId = result.job.jobId;
    elements.stylingInstruction.value = "";
    elements.stylingFeedback.textContent = "二次修改已生成，可继续提出下一条要求。";
    elements.stylingFeedback.className = "ai-feedback success";
  } catch (error) {
    elements.stylingFeedback.textContent = error.message;
    elements.stylingFeedback.className = "ai-feedback error";
  } finally {
    setStylingBusy(false);
  }
}


async function loadStylingVocabulary() {
  if (!elements.stylingSource) return;
  try {
    const info = await getStylingProviders();
    const vocab = info.stylingVocabulary || {};
    const files = Array.isArray(vocab.loadedFiles) ? vocab.loadedFiles.length : 0;
    elements.stylingSource.textContent = vocab.available
      ? `风格库已加载 · ${vocab.sourceLabel || "stylekit"} · ${files} 个文件`
      : "风格库不可用，将使用内置兜底词汇";
    const cost = estimateStylingCost(
      [{ estimatedPaidRequests: info.estimatedRequestsPerGeneration || 1 }],
      { selectedCount: 1, mode: "ai" },
    );
    if (elements.stylingCostNote) {
      const aiRequests = Number(info.stylingAiDesignRequests ?? cost.designRequests) || 1;
      elements.stylingCostNote.textContent =
        `AI 个性化设计：1 次模型调用（约 ${aiRequests} 次付费请求，一次性生成三套方案）；免费模板预览 0 次调用。出图只对选中的一套发起 ${cost.requestsPerLook} 次图像编辑请求。`;
    }
  } catch (error) {
    elements.stylingSource.textContent = "风格库状态读取失败";
  }
}


function init() {
  document.querySelectorAll("[data-category]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.category = btn.dataset.category;
      elements.toolSearch.value = "";
      document.querySelectorAll("[data-category]").forEach((b) => b.classList.toggle("active", b === btn));
      renderCategory();
    });
  });

  document.querySelectorAll("[data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  elements.toolSearch.addEventListener("input", renderCategory);
  elements.compareRange.addEventListener("input", () => {
    const v = Number(elements.compareRange.value);
    elements.afterLayer.style.clipPath = `inset(0 0 0 ${v}%)`;
    elements.compareLine.style.left = `${v}%`;
    elements.compareRange.style.setProperty("--range-progress", `${v}%`);
  });

  elements.compareButton.addEventListener("click", () => {
    state.originalOnly = !state.originalOnly;
    elements.afterLayer.style.opacity = state.originalOnly ? "0" : "1";
    elements.compareLine.hidden = state.originalOnly;
    elements.compareButton.lastChild.textContent = state.originalOnly ? " 显示美化" : " 查看原图";
  });

  elements.resetCategoryButton.addEventListener("click", () => {
    resetTools(categories[state.category].tools);
  });
  elements.resetButton.addEventListener("click", () => {
    resetTools(allTools);
  });
  elements.undoButton.addEventListener("click", () => {
    const prev = state.history.pop();
    if (!prev) return;
    if (prev.snapshot) {
      state.values = { ...state.values, ...prev.snapshot };
    } else {
      state.values = { ...state.values, [prev.id]: prev.value };
    }
    renderCategory();
    updatePreview();
  });

  elements.uploadButton.addEventListener("click", () => elements.fileInput.click());
  elements.fileInput.addEventListener("change", () => {
    const [file] = elements.fileInput.files;
    if (!file) return;
    const url = URL.createObjectURL(file);
    state.sourceFile = file;
    state.sourceName = file.name;
    state.aiResultImage = null;
    state.aiResultUrl = "";
    state.displayingAiResult = false;
    elements.beforeImage.src = url;
    elements.exportButton.href = url;
    if (elements.documentName) elements.documentName.textContent = file.name;
    elements.footerStatus.textContent = "新照片已载入";
  });

  elements.beforeImage.addEventListener("error", () => {
    if (elements.documentName) elements.documentName.textContent = "请选择照片";
    elements.footerStatus.textContent = "请通过左侧 + 按钮选择照片";
  });

  elements.beforeImage.addEventListener("load", scheduleLocalRender);

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.selectedPreset = btn.dataset.preset;
      document.querySelectorAll("[data-preset]").forEach((b) => b.classList.toggle("active", b === btn));
    });
  });

  elements.aiPrompt.addEventListener("input", () => {
    elements.promptCount.textContent = String(elements.aiPrompt.value.length);
  });

  elements.brainProviderSelect.addEventListener("change", syncModelConfig);
  elements.actionProviderSelect.addEventListener("change", syncModelConfig);
  elements.brainApiKeyInput.addEventListener("input", syncModelConfig);
  elements.actionApiKeyInput.addEventListener("input", syncModelConfig);
  elements.exportButton.addEventListener("click", exportCurrentImage);

  state.stylingProposals = state.stylingProposals || [];
  state.selectedStylingId = state.selectedStylingId ?? null;
  state.stylingBusy = false;
  state.stylingPhoto = state.stylingPhoto || null;
  state.generatedStylingJobId = state.generatedStylingJobId ?? null;
  elements.stylingDesignButton.addEventListener("click", () => designStylingFlow("ai"));
  if (elements.stylingTemplateButton) {
    elements.stylingTemplateButton.addEventListener("click", () => designStylingFlow("template"));
  }
  elements.stylingRefineButton.addEventListener("click", refineStylingFlow);

  elements.generateButton.addEventListener("click", async () => {
    if (state.aiBusy) return;
    const label = elements.generateButton.querySelector("b");
    const brain = modelRequest("brain");
    const action = modelRequest("action");
    state.aiBusy = true;
    elements.generateButton.classList.add("loading");
    label.textContent = "AI 正在分析并精修…";
    elements.footerStatus.textContent = "AI 精修生成中";
    elements.aiFeedback.textContent = "正在上传并分析照片，请保持页面打开。";
    elements.aiFeedback.className = "ai-feedback";

    try {
      validateModelSelection(state.capabilities, brain, action);
      const file = await currentSourceFile();
      const result = await executeAiRetouch({
        file, values: state.values, preset: state.selectedPreset,
        userInstruction: elements.aiPrompt.value, brain, action, api: aiApi,
      });
      const resultUrl = assetUrl(result.job.outputUrls[0]);
      await showAiResult(resultUrl);
      elements.exportButton.href = resultUrl;
      elements.canvasStatus.innerHTML = `<span></span>AI 结果 · ${result.job.modelName}`;
      elements.footerStatus.textContent = "AI 美化结果已生成";
      elements.aiFeedback.textContent = `${result.job.brainModel} + ${result.job.modelName} · 生成成功`;
      elements.aiFeedback.className = "ai-feedback success";
      label.textContent = "重新生成 AI 美化效果";
    } catch (error) {
      elements.footerStatus.textContent = "AI 美化失败";
      elements.aiFeedback.textContent = error.message;
      elements.aiFeedback.className = "ai-feedback error";
      label.textContent = "重新尝试 AI 美化";
    } finally {
      state.aiBusy = false;
      elements.generateButton.classList.remove("loading");
    }
  });

  document.querySelector("#aiShortcut").addEventListener("click", () => setMode("ai"));
  if (elements.floatingTip) {
    elements.floatingTip.querySelector("button")?.addEventListener("click", () => {
      elements.floatingTip.hidden = true;
    });
  }

  renderCategory();
  updatePreview();
  elements.compareRange.dispatchEvent(new Event("input"));
  if (elements.beforeImage.complete && elements.beforeImage.naturalWidth) scheduleLocalRender();
  loadProviderCapabilities();
  loadStylingVocabulary();
}

init();

import assert from "node:assert/strict";
import test from "node:test";

import {
  buildStylingInstruction,
  designStyling,
  estimateStylingCost,
  executeStyling,
  normalizeStylingProposals,
  refineStyling,
} from "../js/ai-retouch.mjs";

function makeLook(id, overrides = {}) {
  return {
    planId: id,
    name: `造型 ${id}`,
    designAxis: id,
    hairPlan: `${id} 发型：耳下短发 / 层次 / 发色`,
    makeupPlan: `${id} 妆容：眉 / 眼 / 唇 / 底妆`,
    outfitPlan: `${id} 穿搭：款式 / 材质 / 颜色 / 搭配`,
    accessoryPlan: `${id} 配饰`,
    palette: `${id} 配色`,
    identityAnchors: ["保留圆润脸型轮廓", "保留原本肤色基调"],
    whyItFits: "贴合当前人物脸型与发量",
    expectedChanges: ["发型 → 短发"],
    editPrompt: `Restyle into ${id}. change the hair from current to ${id} hair; change the makeup from current to ${id} makeup; replace the current top with ${id} outfit.`,
    estimatedPaidRequests: 1,
    ...overrides,
  };
}

const THREE_LOOKS = [
  makeLook("urban_short"),
  makeLook("artistic_wave"),
  makeLook("bold_street"),
];

test("normalizeStylingProposals requires three distinct prompts", () => {
  assert.equal(normalizeStylingProposals(THREE_LOOKS).length, 3);

  assert.throws(() => normalizeStylingProposals(THREE_LOOKS.slice(0, 2)), /三套/);

  assert.throws(
    () =>
      normalizeStylingProposals([
        makeLook("a", { editPrompt: "same prompt" }),
        makeLook("b", { editPrompt: "same prompt" }),
        makeLook("c"),
      ]),
    /各不相同/,
  );

  assert.throws(
    () => normalizeStylingProposals([
      makeLook("a"),
      makeLook("b", { hairPlan: "" }),
      makeLook("c"),
    ]),
    /发型/,
  );
});

test("buildStylingInstruction merges the look, anchors and user text", () => {
  const instruction = buildStylingInstruction(THREE_LOOKS[0], "头发再短一点");
  assert.match(instruction, /urban_short/);
  assert.match(instruction, /保留本人辨识特征/);
  assert.match(instruction, /保留圆润脸型轮廓/);
  assert.match(instruction, /用户补充要求：头发再短一点/);
  assert.match(instruction, /不保证百分之百/);
});

test("estimateStylingCost reports requests and the free design phase", () => {
  const one = estimateStylingCost(THREE_LOOKS, { selectedCount: 1 });
  assert.equal(one.totalRequests, 1);
  assert.match(one.note, /免费生成/);
  assert.match(one.note, /1 次/);

  const two = estimateStylingCost(THREE_LOOKS, { selectedCount: 2 });
  assert.equal(two.totalRequests, 2);
});

test("designs three looks, then generates only the selected look from the original", async () => {
  const calls = [];
  let submittedPrompt = "";
  let submittedInstruction = "";
  const api = {
    async uploadPhoto(file) {
      calls.push(["upload", file.name]);
      return { imageId: "img_original" };
    },
    async analyzePhoto(imageId, brain, action) {
      calls.push(["analyze", imageId, brain.name, action.name]);
      return { imageId, domainType: "portrait" };
    },
    async createStylingProposals(analysis) {
      calls.push(["proposals", analysis.imageId]);
      return THREE_LOOKS;
    },
    async createJob(sourceImageId, plan, instruction, brain, action) {
      calls.push(["job", sourceImageId, plan.planId, brain.apiKey]);
      submittedPrompt = plan.editPrompt;
      submittedInstruction = instruction;
      return {
        jobId: "job_1",
        status: "succeeded",
        outputUrls: ["/data/outputs/styled.jpg"],
      };
    },
    async refineJob(jobId, instruction) {
      calls.push(["refine", jobId, instruction]);
      return {
        jobId: "job_2",
        status: "succeeded",
        outputUrls: ["/data/outputs/styled-2.jpg"],
      };
    },
  };

  const brain = { name: "local", apiKey: "" };
  const action = { name: "mock", apiKey: "" };

  const { photo, proposals } = await designStyling({
    file: { name: "portrait.png" },
    brain,
    action,
    api,
  });
  assert.equal(proposals.length, 3);

  const result = await executeStyling({
    photo,
    proposals,
    selectedId: "bold_street",
    userInstruction: "",
    brain,
    action,
    api,
  });

  // upload → analyze → proposals → exactly one job, always from the original.
  assert.deepEqual(calls.map((call) => call[0]), [
    "upload",
    "analyze",
    "proposals",
    "job",
  ]);
  assert.equal(calls[3][1], "img_original");
  assert.equal(calls[3][2], "bold_street");
  assert.equal(result.proposal.planId, "bold_street");
  assert.equal(result.job.outputUrls[0], "/data/outputs/styled.jpg");

  // The prompt handed to the image editor only describes the chosen look.
  assert.match(submittedPrompt, /bold_street/);
  assert.doesNotMatch(submittedPrompt, /urban_short/);
  assert.doesNotMatch(submittedPrompt, /artistic_wave/);
  assert.match(submittedPrompt, /change the hair from current to bold_street hair/);
  assert.match(submittedPrompt, /保留本人辨识特征/);
  assert.equal(submittedInstruction, "");

  const refined = await refineStyling({
    jobId: result.job.jobId,
    userInstruction: "衣服换成皮衣",
    brain,
    action,
    api,
  });
  assert.equal(calls[4][0], "refine");
  assert.equal(calls[4][1], "job_1");
  assert.equal(calls[4][2], "衣服换成皮衣");
  assert.equal(refined.job.outputUrls[0], "/data/outputs/styled-2.jpg");

  // The user's own words ride along once, and only for the look being edited.
  const withExtra = await executeStyling({
    photo,
    proposals,
    selectedId: "artistic_wave",
    userInstruction: "头发再短一点",
    brain,
    action,
    api,
  });
  assert.equal(withExtra.proposal.planId, "artistic_wave");
  assert.match(submittedPrompt, /用户补充要求：头发再短一点/);
  assert.match(submittedPrompt, /artistic_wave/);
  assert.doesNotMatch(submittedPrompt, /bold_street/);
  assert.equal(submittedInstruction, "头发再短一点");
});

test("executeStyling rejects an unknown look and a missing photo", async () => {
  const api = {
    async createJob() {
      throw new Error("should not be called");
    },
  };

  await assert.rejects(
    executeStyling({
      photo: { imageId: "img_original" },
      proposals: THREE_LOOKS,
      selectedId: "nope",
      api,
    }),
    /请选择一套/,
  );

  await assert.rejects(
    executeStyling({ photo: null, proposals: THREE_LOOKS, selectedId: "urban_short", api }),
    /缺少原图/,
  );
});

test("refineStyling requires an instruction", async () => {
  await assert.rejects(
    refineStyling({ jobId: "job_1", userInstruction: "   ", api: {} }),
    /修改要求/,
  );
  await assert.rejects(refineStyling({ api: {} }), /先生成一套造型/);
});

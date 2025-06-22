import http from "k6/http";
import { check } from "k6";
import { Rate } from "k6/metrics";
import { SharedArray } from "k6/data";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.2/index.js";
const logger = {
  info: (...args) => console.log("[INFO]", ...args),
  error: (...args) => console.error("[ERROR]", ...args),
};

const failRate = new Rate("failed_requests");

const params = {
  rps: __ENV.RPS ? parseInt(__ENV.RPS) : 10,
  modelName: __ENV.MODEL_NAME,
  maxTokensMean: __ENV.MAX_TOKENS_MEAN ? parseInt(__ENV.MAX_TOKENS_MEAN) : 128,
  maxTokensStdDev: __ENV.MAX_TOKENS_STD ? parseInt(__ENV.MAX_TOKENS_STD) : 20,
  promptLenMean: __ENV.PROMPT_LEN_MEAN ? parseInt(__ENV.PROMPT_LEN_MEAN) : 100,
  promptLenStdDev: __ENV.PROMPT_LEN_STD ? parseInt(__ENV.PROMPT_LEN_STD) : 30,
  duration: __ENV.DURATION ? __ENV.DURATION : "10s",
  promptType: __ENV.PROMPT_TYPE || "random",
  resultsDir: __ENV.RESULTS_DIR || "./load_test_results",
  proxyURL: "http://localhost:9000" + __ENV.API_ROUTE,
};

let prompts = [];
if (params.promptType === "code" && params.resultsDir.concat("/prompts.json")) {
  prompts = new SharedArray("prompts", function () {
    return JSON.parse(open(params.resultsDir.concat("/prompts.json")));
  });
}

export const options = {
  scenarios: {
    streaming: {
      executor: "constant-arrival-rate",
      rate: params.rps,
      timeUnit: "1s",
      duration: params.duration,
      preAllocatedVUs: 500,
    },
  },
};

function generatePrompt(prompt_length) {
  if (params.promptType === "code" && prompts.length > 0) {
    const idx = Math.floor(Math.random() * prompts.length);
    return prompts[idx].slice(0, prompt_length);
  }

  const text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
  return text
    .repeat(Math.ceil(prompt_length / text.length))
    .slice(0, prompt_length);
}

function boxMullerTransform() {
  const u1 = Math.random();
  const u2 = Math.random();

  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);

  return { z0, z1 };
}

function getNormallyDistributedRandomNumber(mean, stddev) {
  const { z0, _ } = boxMullerTransform();

  return z0 * stddev + mean;
}

export default function () {
  const normDistTokens = getNormallyDistributedRandomNumber(
    params.maxTokensMean,
    params.maxTokensStdDev,
  );
  const normDistPrompts = getNormallyDistributedRandomNumber(
    params.promptLenMean,
    params.promptLenStdDev,
  );

  const maxTokens = Math.max(1, Math.round(normDistTokens));
  // We assume that 1 character is approximately equivalent to 4 tokens.
  const promptLen = Math.max(10, Math.round(normDistPrompts)) * 4;

  const payload = {
    model: params.modelName,
    messages: [
      {
        role: "user",
        content: generatePrompt(promptLen),
      },
    ],
    max_tokens: maxTokens,
    temperature: 1,
  };

  const headers = {
    "Content-Type": "application/json",
  };

  const response = http.post(params.proxyURL, JSON.stringify(payload), {
    headers: headers,
    timeout: "120s",
  });

  check(response, {
    "is status 200": (r) => r.status === 200,
  }) || failRate.add(1);

  if (response.body) {
    try {
      const responseData = JSON.parse(response.body);
    } catch (error) {
      logger.error("Error parsing JSON response:", error);
    }
  } else {
    logger.error("Response body is null or undefined.");
  }
}

export function handleSummary(data) {
  data.metrics["end_to_end_latency"] = data.metrics["http_req_duration"];

  for (const key in data.metrics) {
    if (
      key.startsWith("iteration") ||
      key.startsWith("data") ||
      key.startsWith("http")
    ) {
      delete data.metrics[key];
    }
  }

  return {
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}

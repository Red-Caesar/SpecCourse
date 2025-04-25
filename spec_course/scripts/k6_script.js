import http from "k6/http";
import { check } from "k6";
import { Rate, Trend } from "k6/metrics";

const endToEndLatency = new Trend("end_to_end_latency");
const failRate = new Rate("failed_requests");

const params = {
  rps: __ENV.RPS ? parseInt(__ENV.RPS) : 10,
  modelName: __ENV.MODEL_NAME,
  endpointUrl: __ENV.ENDPOINT_URL,
  maxTokensMean: __ENV.MAX_TOKENS_MEAN ? parseInt(__ENV.MAX_TOKENS_MEAN) : 128,
  maxTokensStdDev: __ENV.MAX_TOKENS_STD ? parseInt(__ENV.MAX_TOKENS_STD) : 20,
  promptLenMean: __ENV.PROMPT_LEN_MEAN ? parseInt(__ENV.PROMPT_LEN_MEAN) : 100,
  promptLenStdDev: __ENV.PROMPT_LEN_STD ? parseInt(__ENV.PROMPT_LEN_STD) : 30,
  duration: __ENV.DURATION ? __ENV.DURATION : "10s",
};

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

function generatePrompt(length) {
  const text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
  return text.repeat(Math.ceil(length / text.length)).slice(0, length);
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
  const promptLen = Math.max(10, Math.round(normDistPrompts));

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

  const startTime = new Date().getTime();

  const response = http.post(params.endpointUrl, JSON.stringify(payload), {
    headers: headers,
    timeout: "60s",
  });

  const endTime = new Date().getTime();
  const latency = (endTime - startTime) / 1000;

  endToEndLatency.add(latency);

  check(response, {
    "is status 200": (r) => r.status === 200,
  }) || failRate.add(1);
}

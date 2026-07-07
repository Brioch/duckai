type ModelDefaults = {
  reasoning_effort?: DuckAIReasoningEffort;
  valid_reasoning_efforts?: Array<DuckAIReasoningEffort>;
};

export type DuckAIReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high";
export const DUCKAI_REASONING_EFFORTS = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
];

export const DUCKAI_MODELS: Record<string, ModelDefaults> = {
  "gpt-4o-mini": {},
  "gpt-5-mini": {
    reasoning_effort: "minimal",
    valid_reasoning_efforts: ["minimal", "low"],
  },
  "tinfoil/gpt-oss-120b": {
    reasoning_effort: "low",
    valid_reasoning_efforts: ["low"],
  },
  "meta-llama/Llama-4-Scout-17B-16E-Instruct": {},
  "claude-haiku-4-5": {
    reasoning_effort: "none",
    valid_reasoning_efforts: ["none", "low"],
  },
  "mistralai/Mistral-Small-24B-Instruct-2501": {},
} as const;

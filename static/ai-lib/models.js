// AI Models Configuration Library
// This module provides a plug-and-play interface for different AI models
// It defines available models, their capabilities, and strategies for model selection.

// Text length thresholds for model selection (used for complexity analysis)
export const TEXT_LENGTH_THRESHOLDS = {
  SHORT: 100, // 0-100 characters: use smaller models
  MEDIUM: 500, // 101-500 characters: use medium models
  LONG: 1000, // 501-1000 characters: use larger models
  VERY_LONG: 2000, // 1000+ characters: use best models
};

// Model registry - defines all available models and their properties
export const MODEL_REGISTRY = {
  // Groq Cloud Models
  "llama3-8b-8192": {
    provider: "groq",
    name: "Llama 3 8B",
    maxTokens: 8192,
    costPer1kTokens: 0.00005, // USD
    speed: "fast",
    quality: "good",
    bestFor: ["short", "medium"],
    capabilities: ["chat", "analysis", "summarization"],
    contextConfig: {
      maxContextMessages: 3,
      maxTokens: 4000,
      contextStrategy: "recent",
    },
  },
  "llama3-70b-8192": {
    provider: "groq",
    name: "Llama 3 70B",
    maxTokens: 8192,
    costPer1kTokens: 0.00059, // USD
    speed: "medium",
    quality: "excellent",
    bestFor: ["medium", "long", "very_long"],
    capabilities: ["chat", "analysis", "summarization", "creation", "ideation"],
    contextConfig: {
      maxContextMessages: 6,
      maxTokens: 8000,
      contextStrategy: "smart",
    },
  },
  "mixtral-8x7b-32768": {
    provider: "groq",
    name: "Mixtral 8x7B",
    maxTokens: 32768,
    costPer1kTokens: 0.00024, // USD
    speed: "medium",
    quality: "excellent",
    bestFor: ["long", "very_long"],
    capabilities: [
      "chat",
      "analysis",
      "summarization",
      "creation",
      "ideation",
      "conversion",
    ],
    contextConfig: {
      maxContextMessages: 8,
      maxTokens: 16000,
      contextStrategy: "comprehensive",
    },
  },
  "gpt-4": {
    provider: "openai",
    name: "GPT-4",
    maxTokens: 8192,
    costPer1kTokens: 0.03, // USD (input)
    speed: "medium",
    quality: "excellent",
    bestFor: ["medium", "long", "very_long"],
    capabilities: [
      "chat",
      "analysis",
      "summarization",
      "creation",
      "ideation",
      "conversion",
    ],
    contextConfig: {
      maxContextMessages: 10,
      maxTokens: 6000,
      contextStrategy: "comprehensive",
    },
  },
  "claude-3-sonnet": {
    provider: "anthropic",
    name: "Claude 3 Sonnet",
    maxTokens: 200000,
    costPer1kTokens: 0.03, // USD
    speed: "medium",
    quality: "excellent",
    bestFor: ["long", "very_long"],
    capabilities: [
      "chat",
      "analysis",
      "summarization",
      "creation",
      "ideation",
      "conversion",
    ],
    contextConfig: {
      maxContextMessages: 12,
      maxTokens: 100000,
      contextStrategy: "comprehensive",
    },
  },
};

// Default model mappings for different complexity levels
// Used to select a model based on the complexity of the user query
export const DEFAULT_MODELS = {
  SHORT: "llama3-8b-8192",
  MEDIUM: "llama3-70b-8192",
  LONG: "mixtral-8x7b-32768",
  VERY_LONG: "llama3-70b-8192",
};

// Model selection strategies for different optimization goals
export const MODEL_SELECTION_STRATEGIES = {
  // Cost-optimized: prioritize cheaper models
  COST_OPTIMIZED: {
    SHORT: "llama3-8b-8192",
    MEDIUM: "llama3-8b-8192",
    LONG: "llama3-70b-8192",
    VERY_LONG: "llama3-70b-8192",
  },
  // Quality-optimized: prioritize better models
  QUALITY_OPTIMIZED: {
    SHORT: "llama3-70b-8192",
    MEDIUM: "llama3-70b-8192",
    LONG: "mixtral-8x7b-32768",
    VERY_LONG: "claude-3-sonnet",
  },
  // Speed-optimized: prioritize faster models
  SPEED_OPTIMIZED: {
    SHORT: "llama3-8b-8192",
    MEDIUM: "llama3-8b-8192",
    LONG: "llama3-8b-8192",
    VERY_LONG: "llama3-70b-8192",
  },
  // Balanced: good balance of cost, quality, and speed
  BALANCED: {
    SHORT: "llama3-8b-8192",
    MEDIUM: "llama3-70b-8192",
    LONG: "mixtral-8x7b-32768",
    VERY_LONG: "llama3-70b-8192",
  },
};

// ModelManager class: handles model selection and config lookup
export class ModelManager {
  /**
   * Construct a new ModelManager with a given strategy.
   * @param {string} strategy - Model selection strategy (default: BALANCED).
   */
  constructor(strategy = "BALANCED") {
    this.strategy = strategy;
    this.models =
      MODEL_SELECTION_STRATEGIES[strategy] ||
      MODEL_SELECTION_STRATEGIES.BALANCED;
  }

  /**
   * Get the model for a given complexity level.
   * @param {string} complexity - Complexity level (SHORT, MEDIUM, etc).
   * @returns {string} - Model ID.
   */
  getModelForComplexity(complexity) {
    return this.models[complexity] || this.models.SHORT;
  }

  /**
   * Get the configuration for a model by ID.
   * @param {string} modelId - Model ID.
   * @returns {object} - Model config object.
   */
  getModelConfig(modelId) {
    return MODEL_REGISTRY[modelId] || MODEL_REGISTRY["llama3-8b-8192"];
  }

  /**
   * Check if a model supports a specific capability (e.g., 'analysis').
   * @param {string} modelId - Model ID.
   * @param {string} capability - Capability string.
   * @returns {boolean}
   */
  modelSupportsCapability(modelId, capability) {
    const config = this.getModelConfig(modelId);
    return config.capabilities.includes(capability);
  }

  /**
   * Get all available model IDs.
   * @returns {Array<string>} - List of model IDs.
   */
  getAvailableModels() {
    return Object.keys(MODEL_REGISTRY);
  }

  /**
   * Get all models for a given provider.
   * @param {string} provider - Provider name.
   * @returns {Array<object>} - List of model configs.
   */
  getModelsByProvider(provider) {
    return Object.entries(MODEL_REGISTRY)
      .filter(([_, config]) => config.provider === provider)
      .map(([id, config]) => ({ id, ...config }));
  }

  /**
   * Estimate the cost of a model call.
   * @param {string} modelId - Model ID.
   * @param {number} inputTokens - Number of input tokens.
   * @param {number} outputTokens - Number of output tokens (optional).
   * @returns {number} - Estimated cost in USD.
   */
  estimateCost(modelId, inputTokens, outputTokens = 0) {
    const config = this.getModelConfig(modelId);
    const inputCost = (inputTokens / 1000) * config.costPer1kTokens;
    const outputCost = (outputTokens / 1000) * (config.costPer1kTokens * 2); // Output typically costs more
    return inputCost + outputCost;
  }

  /**
   * Change the model selection strategy.
   * @param {string} strategy - New strategy name.
   */
  setStrategy(strategy) {
    if (MODEL_SELECTION_STRATEGIES[strategy]) {
      this.strategy = strategy;
      this.models = MODEL_SELECTION_STRATEGIES[strategy];
    }
  }

  /**
   * Get the current model selection strategy.
   * @returns {string} - Strategy name.
   */
  getStrategy() {
    return this.strategy;
  }
}

// Export default model manager instance
let _defaultModelManager = null;

export function getDefaultModelManager() {
  if (!_defaultModelManager) {
    _defaultModelManager = new ModelManager("BALANCED");
  }
  return _defaultModelManager;
}

// For backward compatibility
export const defaultModelManager = getDefaultModelManager();

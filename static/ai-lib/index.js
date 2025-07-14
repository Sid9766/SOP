// AI Library - Main Interface
// This module provides a unified plug-and-play interface for AI functionality
// It acts as the main orchestrator, connecting all AI components (model selection, task classification, prompt management, context management, LLM client, and chat title generation).

import {
  ModelManager,
  MODEL_REGISTRY,
  MODEL_SELECTION_STRATEGIES,
  getDefaultModelManager,
} from "./models.js";

import {
  TaskClassifier,
  TASK_TYPES,
  TextComplexityAnalyzer,
  getDefaultTaskClassifier,
} from "./taskClassifier.js";

import {
  PromptManager,
  SYSTEM_PROMPTS,
  getDefaultPromptManager,
} from "./prompts.js";

import {
  ContextManager,
  CONTEXT_STRATEGIES,
  getDefaultContextManager,
} from "./contextManager.js";

import { LLMClient, getDefaultLLMClient } from "./llmClient.js";

import {
  ChatTitleGenerator,
  getDefaultChatTitleGenerator,
} from "./chatTitleGenerator.js";

// Main AI orchestrator class: central entry point for all AI operations
export class AIOrchestrator {
  /**
   * Construct a new AIOrchestrator instance, wiring up all subcomponents.
   * @param {object} options - Optional custom components for advanced use.
   */
  constructor(options = {}) {
    // Model manager handles model selection and config
    this.modelManager = options.modelManager || getDefaultModelManager();
    // Task classifier determines the type of user query
    this.taskClassifier = options.taskClassifier || getDefaultTaskClassifier();
    // Prompt manager generates system prompts and manages prompt templates
    this.promptManager = options.promptManager || getDefaultPromptManager();
    // Context manager manages chat history and context window
    this.contextManager = options.contextManager || getDefaultContextManager();
    // LLM client handles API calls to GroqCloud or other providers
    this.llmClient = options.llmClient || getDefaultLLMClient();
    // Chat title generator creates smart chat titles
    this.chatTitleGenerator =
      options.chatTitleGenerator || getDefaultChatTitleGenerator();

    // Link model manager to other components for consistent model selection
    this.taskClassifier.setModelManager(this.modelManager);
    this.contextManager.setModelManager(this.modelManager);

    // Avoid circular dependency by setting orchestrator in chat title generator
    if (this.chatTitleGenerator && !this.chatTitleGenerator.aiOrchestrator) {
      this.chatTitleGenerator.setAIOrchestrator(this);
    }
  }

  /**
   * Main method to process a user query with optional chat history.
   * Handles classification, prompt building, context selection, and LLM call.
   * @param {string} queryText - The user's query.
   * @param {object|null} chatHistory - The chat history (see contextManager for format).
   * @returns {Promise<object>} - The LLM response and metadata.
   */
  async processQuery(queryText, chatHistory = null) {
    console.log("AI Orchestrator processing query:", queryText);

    // Step 1: Classify the query to determine task type and model
    const classification = await this.taskClassifier.classifyQuery(queryText);
    console.log("Final query classification:", classification);

    // Step 2: Get prompt and LLM parameters based on classification
    const maxTokens = this.promptManager.getTokenLimit(
      classification.task,
      classification.complexity
    );
    const temperature = this.promptManager.getTemperature(
      classification.task,
      classification.complexity,
      0
    );
    const systemPrompt = this.promptManager.getSystemPrompt(
      classification.task,
      maxTokens,
      classification.complexity
    );

    // Step 3: Process chat history into context messages
    // (contextManager expects chatHistory to be an object with a .messages array)
    const contextMessages = this.contextManager.processChatHistory(chatHistory);
    // Select the most relevant context messages for the model
    const modelConfig = this.modelManager.getModelConfig(classification.model);
    const selectedContext = this.contextManager.selectContextMessages(
      contextMessages,
      classification.model,
      modelConfig.contextConfig.maxContextMessages
    );
    // Optimize context to fit within token limits
    const optimizedContext = this.contextManager.optimizeContext(
      selectedContext,
      classification.model,
      maxTokens
    );

    // Step 4: Build the messages array for the LLM API
    const messages = this.contextManager.buildMessagesArray(
      systemPrompt,
      optimizedContext,
      queryText
    );

    // Step 5: Call the LLM with all parameters
    const result = await this.llmClient.callLLM({
      model: classification.model,
      messages,
      maxTokens,
      temperature,
      task: classification.task,
      complexity: classification.complexity,
      reason: classification.reason,
      contextMessages: optimizedContext,
    });

    // Attach detailed metadata for debugging and analysis
    result.metadata = {
      classification,
      contextStats: this.contextManager.getContextStats(
        optimizedContext,
        classification.model
      ),
      modelConfig: modelConfig,
      promptInfo: {
        maxTokens,
        temperature,
        systemPrompt: systemPrompt, // Full prompt for transparency
      },
    };

    return result;
  }

  /**
   * Generate a chat title for a new conversation.
   * @param {string} firstMessage - The first user message.
   * @returns {Promise<string>} - The generated chat title.
   */
  async generateChatTitle(firstMessage) {
    return await this.chatTitleGenerator.generateChatTitle(firstMessage);
  }

  /**
   * Dynamically update the chat title as the conversation evolves.
   * @param {Array} messages - The chat messages.
   * @param {string} currentTitle - The current chat title.
   * @returns {Promise<string>} - The updated chat title.
   */
  async updateChatTitleDynamically(messages, currentTitle) {
    return await this.chatTitleGenerator.updateChatTitleDynamically(
      messages,
      currentTitle
    );
  }

  /**
   * Configure the orchestrator with custom strategies or components.
   * @param {object} options - Custom configuration options.
   */
  configure(options) {
    if (options.modelStrategy) {
      this.modelManager.setStrategy(options.modelStrategy);
    }

    if (options.customModelManager) {
      this.modelManager = options.customModelManager;
      this.taskClassifier.setModelManager(this.modelManager);
      this.contextManager.setModelManager(this.modelManager);
    }

    if (options.customTaskClassifier) {
      this.taskClassifier = options.customTaskClassifier;
    }

    if (options.customPromptManager) {
      this.promptManager = options.customPromptManager;
    }

    if (options.customContextManager) {
      this.contextManager = options.customContextManager;
    }

    if (options.customLLMClient) {
      this.llmClient = options.customLLMClient;
    }

    if (options.customChatTitleGenerator) {
      this.chatTitleGenerator = options.customChatTitleGenerator;
    }
  }

  /**
   * Get the current orchestrator configuration (for debugging/UI).
   * @returns {object} - Current configuration details.
   */
  getConfiguration() {
    return {
      modelStrategy: this.modelManager.getStrategy(),
      availableModels: this.modelManager.getAvailableModels(),
      availableTasks: this.taskClassifier.getAvailableTasks(),
      availablePromptTypes: this.promptManager.getAvailablePromptTypes(),
    };
  }

  /**
   * Add a custom model to the registry.
   * @param {string} modelId - The model identifier.
   * @param {object} config - The model configuration.
   */
  addCustomModel(modelId, config) {
    MODEL_REGISTRY[modelId] = config;
  }

  /**
   * Add a custom task type.
   * @param {string} taskType - The task type identifier.
   * @param {object} config - The task configuration.
   */
  addCustomTask(taskType, config) {
    TASK_TYPES[taskType] = config;
  }

  /**
   * Add a custom system prompt for a task type.
   * @param {string} taskType - The task type identifier.
   * @param {function} promptFunction - The prompt generator function.
   */
  addCustomPrompt(taskType, promptFunction) {
    this.promptManager.addSystemPrompt(taskType, promptFunction);
  }

  /**
   * Test if a model is available (calls the LLM with a test prompt).
   * @param {string} modelId - The model identifier.
   * @returns {Promise<boolean>} - True if the model responds successfully.
   */
  async testModel(modelId) {
    return await this.llmClient.testModel(modelId);
  }

  /**
   * Estimate the cost of a given model usage.
   * @param {string} modelId - The model identifier.
   * @param {number} inputTokens - Number of input tokens.
   * @param {number} outputTokens - Number of output tokens (optional).
   * @returns {number} - Estimated cost in USD.
   */
  estimateCost(modelId, inputTokens, outputTokens = 0) {
    return this.modelManager.estimateCost(modelId, inputTokens, outputTokens);
  }
}

/**
 * Factory function to create an AI orchestrator with custom configuration.
 * @param {object} config - Configuration options for the orchestrator.
 * @returns {AIOrchestrator} - The configured orchestrator instance.
 */
export function createAIOrchestrator(config = {}) {
  const options = {
    modelStrategy: config.modelStrategy || "BALANCED",
    customModelManager: config.customModelManager,
    customTaskClassifier: config.customTaskClassifier,
    customPromptManager: config.customPromptManager,
    customContextManager: config.customContextManager,
    customLLMClient: config.customLLMClient,
    customChatTitleGenerator: config.customChatTitleGenerator,
  };

  const orchestrator = new AIOrchestrator(options);

  if (config.modelStrategy) {
    orchestrator.configure({ modelStrategy: config.modelStrategy });
  }

  return orchestrator;
}

// Pre-configured orchestrators for common use cases
export const AIOrchestrators = {
  // Cost-optimized: prioritize cheaper models
  CostOptimized: () =>
    createAIOrchestrator({ modelStrategy: "COST_OPTIMIZED" }),

  // Quality-optimized: prioritize better models
  QualityOptimized: () =>
    createAIOrchestrator({ modelStrategy: "QUALITY_OPTIMIZED" }),

  // Speed-optimized: prioritize faster models
  SpeedOptimized: () =>
    createAIOrchestrator({ modelStrategy: "SPEED_OPTIMIZED" }),

  // Balanced: good balance of cost, quality, and speed
  Balanced: () => createAIOrchestrator({ modelStrategy: "BALANCED" }),
};

// Export all components for advanced usage
export {
  // Models
  ModelManager,
  MODEL_REGISTRY,
  MODEL_SELECTION_STRATEGIES,
  getDefaultModelManager,

  // Task Classification
  TaskClassifier,
  TASK_TYPES,
  TextComplexityAnalyzer,
  getDefaultTaskClassifier,

  // Prompts
  PromptManager,
  SYSTEM_PROMPTS,
  getDefaultPromptManager,

  // Context Management
  ContextManager,
  CONTEXT_STRATEGIES,
  getDefaultContextManager,

  // LLM Client
  LLMClient,
  getDefaultLLMClient,

  // Chat Title Generator
  ChatTitleGenerator,
  getDefaultChatTitleGenerator,
};

// Export default orchestrator instance
let _defaultAIOrchestrator = null;

export function getDefaultAIOrchestrator() {
  if (!_defaultAIOrchestrator) {
    _defaultAIOrchestrator = new AIOrchestrator();
  }
  return _defaultAIOrchestrator;
}

// For backward compatibility
export const defaultAIOrchestrator = getDefaultAIOrchestrator();

// Utility functions for API key management
export function setApiKey(provider, key) {
  getDefaultLLMClient().setApiKey(provider, key);
}

export function getApiKey(provider) {
  return getDefaultLLMClient().getApiKey(provider);
}

// Usage:
// import { setApiKey } from 'ai-lib';
// setApiKey('groq', 'YOUR_GROQCLOUD_API_KEY');

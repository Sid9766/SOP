// Prompts Library
// This module manages system prompts and dynamic prompt generation for different task types.
// It also provides logic for dynamic token and temperature settings based on task and complexity.

// System prompts for each task type (functions that return a prompt string)
export const SYSTEM_PROMPTS = {
  LLM_Summary: (maxTokens, complexity) => {
    const basePrompt = `You are an expert at summarizing text. You have ${maxTokens} tokens available.`;
    if (complexity === "LONG" || complexity === "VERY_LONG") {
      return `${basePrompt} For this long-form content, provide a comprehensive summary that captures the main points, key insights, and important details. Structure your response with clear sections if appropriate. Aim for a detailed but concise summary.`;
    }
    return `${basePrompt} Provide a concise, clear summary within this limit. Respond in no more than 3 sentences.`;
  },
  LLM_Creation: (maxTokens, complexity) => {
    const basePrompt = `You are a creative assistant. You have ${maxTokens} tokens available.`;
    if (complexity === "LONG" || complexity === "VERY_LONG") {
      return `${basePrompt} For this substantial input, generate creative content that demonstrates depth and nuance. Consider multiple perspectives and provide rich, detailed responses.`;
    }
    return `${basePrompt} Generate creative content, but keep your response within this limit. Do not exceed 100 words.`;
  },
  LLM_Ideation: (maxTokens, complexity) => {
    const basePrompt = `You are a brainstorming assistant. You have ${maxTokens} tokens available.`;
    if (complexity === "LONG" || complexity === "VERY_LONG") {
      return `${basePrompt} For this complex topic, provide a comprehensive list of ideas, suggestions, and approaches. Consider different angles and provide detailed explanations for key ideas.`;
    }
    return `${basePrompt} Provide a list of ideas or suggestions, but keep your response concise. Do not exceed 5 bullet points.`;
  },
  LLM_Analysis: (maxTokens, complexity) => {
    const basePrompt = `You are an analytical assistant. You have ${maxTokens} tokens available.`;
    if (complexity === "LONG" || complexity === "VERY_LONG") {
      return `${basePrompt} For this detailed content, provide a thorough analysis that examines multiple aspects, identifies patterns, and offers nuanced insights. Structure your response with clear sections and provide evidence-based conclusions.`;
    }
    return `${basePrompt} Analyze the input and provide a clear, structured response within this limit. Limit your answer to 5 sentences.`;
  },
  LLM_Converter: (maxTokens, complexity) => {
    const basePrompt = `You are a conversion assistant. You have ${maxTokens} tokens available.`;
    if (complexity === "LONG" || complexity === "VERY_LONG") {
      return `${basePrompt} For this substantial content, perform a thorough conversion or transformation that maintains the depth and nuance of the original while adapting it to the requested format.`;
    }
    return `${basePrompt} Convert or transform the input as requested, and keep your response concise. Do not exceed 100 words.`;
  },
  LLM_Default: (maxTokens, complexity) => {
    const basePrompt = `You have ${maxTokens} tokens available.`;
    if (complexity === "LONG" || complexity === "VERY_LONG") {
      return `${basePrompt} For this substantial input, provide a comprehensive and thoughtful response that addresses the complexity of the content. Consider multiple aspects and provide detailed insights.`;
    }
    return `${basePrompt} Please keep your response concise and within this limit. Respond in no more than 3 sentences.`;
  },
};

// Dynamic token limits based on task and complexity
export function getDynamicTokenLimit(task, complexity) {
  // Base token limits for each task type
  const baseTokens =
    {
      LLM_Summary: 2000,
      LLM_Creation: 1500,
      LLM_Ideation: 1200,
      LLM_Analysis: 2500,
      LLM_Converter: 1500,
      LLM_Default: 1500,
    }[task] || 1500;
  // Adjust based on complexity
  switch (complexity) {
    case "SHORT":
      return Math.min(baseTokens, 1000);
    case "MEDIUM":
      return baseTokens;
    case "LONG":
      return Math.min(baseTokens * 1.5, 3000);
    case "VERY_LONG":
      return Math.min(baseTokens * 2, 4000);
    default:
      return baseTokens;
  }
}

// Dynamic temperature adjustment based on task, complexity, and context length
export function getDynamicTemperature(task, complexity, contextLength = 0) {
  const baseTemps = {
    LLM_Summary: 0.3,
    LLM_Creation: 0.7,
    LLM_Ideation: 0.5,
    LLM_Analysis: 0.2,
    LLM_Converter: 0.4,
    LLM_Default: 0.3,
  };
  const baseTemp = baseTemps[task] || baseTemps.LLM_Default;
  // Adjust temperature for longer conversations or complex tasks
  if (contextLength > 4) {
    // Reduce temperature for longer conversations to maintain consistency
    return Math.max(baseTemp * 0.8, 0.1);
  } else if (complexity === "LONG" || complexity === "VERY_LONG") {
    // Slightly increase temperature for complex tasks
    return Math.min(baseTemp * 1.1, 0.9);
  }
  return baseTemp;
}

// Prompt manager class: manages system prompts and dynamic settings
export class PromptManager {
  constructor() {
    this.systemPrompts = SYSTEM_PROMPTS;
  }
  /**
   * Get the system prompt for a given task, token limit, and complexity.
   * @param {string} task - Task type.
   * @param {number} maxTokens - Token limit.
   * @param {string} complexity - Complexity level.
   * @returns {string} - System prompt string.
   */
  getSystemPrompt(task, maxTokens, complexity) {
    const promptFn = this.systemPrompts[task] || this.systemPrompts.LLM_Default;
    return promptFn(maxTokens, complexity);
  }
  /**
   * Get the token limit for a task and complexity.
   */
  getTokenLimit(task, complexity) {
    return getDynamicTokenLimit(task, complexity);
  }
  /**
   * Get the temperature for a task, complexity, and context length.
   */
  getTemperature(task, complexity, contextLength) {
    return getDynamicTemperature(task, complexity, contextLength);
  }
  /**
   * Add a custom system prompt for a task type.
   */
  addSystemPrompt(taskType, promptFunction) {
    this.systemPrompts[taskType] = promptFunction;
  }
  /**
   * Remove a custom system prompt for a task type.
   */
  removeSystemPrompt(taskType) {
    if (this.systemPrompts[taskType] && !SYSTEM_PROMPTS[taskType]) {
      delete this.systemPrompts[taskType];
    }
  }
  /**
   * Get all available prompt types.
   */
  getAvailablePromptTypes() {
    return Object.keys(this.systemPrompts);
  }
}

// Export default prompt manager instance (singleton)
let _defaultPromptManager = null;

export function getDefaultPromptManager() {
  if (!_defaultPromptManager) {
    _defaultPromptManager = new PromptManager();
  }
  return _defaultPromptManager;
}

// For backward compatibility
export const defaultPromptManager = getDefaultPromptManager();

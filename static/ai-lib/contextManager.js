// Context Management Library
// This module handles conversation context optimization and token management
// It is responsible for selecting, optimizing, and formatting chat history for LLM calls.

import { defaultModelManager } from "./models.js";

// Context selection strategies for different use cases
export const CONTEXT_STRATEGIES = {
  RECENT: "recent", // Only the most recent messages
  SMART: "smart", // First message + recent messages
  COMPREHENSIVE: "comprehensive", // Full context with smart selection
  MINIMAL: "minimal", // Minimal context for cost efficiency
};

// Context manager class: manages chat history and context window
export class ContextManager {
  /**
   * Construct a new ContextManager instance.
   * @param {object} modelManager - The model manager for model-specific configs.
   */
  constructor(modelManager = defaultModelManager) {
    this.modelManager = modelManager;
  }

  /**
   * Select context messages based on the model's strategy and max allowed messages.
   * @param {Array} messages - All available context messages.
   * @param {string} model - The model ID.
   * @param {number} maxContextMessages - Max messages allowed by the model.
   * @returns {Array} - Selected context messages.
   */
  selectContextMessages(messages, model, maxContextMessages) {
    if (!messages || messages.length === 0) return [];

    const config = this.modelManager.getModelConfig(model);
    const strategy = config.contextConfig.contextStrategy;

    switch (strategy) {
      case CONTEXT_STRATEGIES.RECENT:
        return this.selectRecentMessages(messages, maxContextMessages);
      case CONTEXT_STRATEGIES.SMART:
        return this.selectSmartMessages(messages, maxContextMessages);
      case CONTEXT_STRATEGIES.COMPREHENSIVE:
        return this.selectComprehensiveMessages(messages, maxContextMessages);
      case CONTEXT_STRATEGIES.MINIMAL:
        return this.selectMinimalMessages(messages, maxContextMessages);
      default:
        return this.selectRecentMessages(messages, maxContextMessages);
    }
  }

  /**
   * Select only the most recent messages.
   */
  selectRecentMessages(messages, maxContextMessages) {
    return messages.slice(-maxContextMessages);
  }

  /**
   * Select the first message and the most recent messages.
   */
  selectSmartMessages(messages, maxContextMessages) {
    if (messages.length <= maxContextMessages) {
      return messages;
    }
    const recentMessages = messages.slice(-maxContextMessages + 1);
    const firstMessage = messages[0];
    return [firstMessage, ...recentMessages];
  }

  /**
   * Select a comprehensive set: first, middle, and recent messages.
   */
  selectComprehensiveMessages(messages, maxContextMessages) {
    if (messages.length <= maxContextMessages) {
      return messages;
    }
    // Take first message, middle message, and recent messages
    const first = messages[0];
    const middle = messages[Math.floor(messages.length / 2)];
    const recent = messages.slice(-maxContextMessages + 2);
    return [first, middle, ...recent];
  }

  /**
   * Select a minimal set of context messages for cost efficiency.
   */
  selectMinimalMessages(messages, maxContextMessages) {
    const minimalCount = Math.min(maxContextMessages, 2);
    return messages.slice(-minimalCount);
  }

  /**
   * Optimize context to stay within token limits for the model.
   * Truncates messages if needed to fit the allowed context window.
   */
  optimizeContext(contextMessages, model, maxTokens) {
    const config = this.modelManager.getModelConfig(model);
    const maxContextTokens = Math.floor(maxTokens * 0.7); // Reserve 30% for response

    let totalTokens = 0;
    const optimizedMessages = [];

    // Start from most recent messages and work backwards
    for (let i = contextMessages.length - 1; i >= 0; i--) {
      const message = contextMessages[i];
      const messageTokens = this.estimateTokenCount(message.content);

      if (totalTokens + messageTokens <= maxContextTokens) {
        optimizedMessages.unshift(message);
        totalTokens += messageTokens;
      } else {
        // If we can't fit the full message, truncate it
        const remainingTokens = maxContextTokens - totalTokens;
        const maxChars = remainingTokens * 4;
        if (maxChars > 100) {
          // Only add if we have meaningful space
          const truncatedMessage = {
            ...message,
            content: message.content.substring(0, maxChars) + "...",
          };
          optimizedMessages.unshift(truncatedMessage);
        }
        break;
      }
    }
    return optimizedMessages;
  }

  /**
   * Roughly estimate token count for a given text (1 token ≈ 4 characters).
   */
  estimateTokenCount(text) {
    return Math.ceil(text.length / 4);
  }

  /**
   * Build the messages array for the LLM API, including system prompt, context, and user query.
   */
  buildMessagesArray(systemPrompt, contextMessages, userQuery) {
    const messages = [
      { role: "system", content: systemPrompt },
      ...contextMessages,
      { role: "user", content: userQuery },
    ];
    return messages;
  }

  /**
   * Convert chat history (object with .messages array) into context messages for the LLM.
   * Expects each message to have .user and optionally .response fields.
   */
  processChatHistory(chatHistory) {
    const contextMessages = [];
    if (chatHistory && chatHistory.messages) {
      for (const msg of chatHistory.messages) {
        contextMessages.push({ role: "user", content: msg.user });
        if (msg.response) {
          contextMessages.push({ role: "assistant", content: msg.response });
        }
      }
    }
    return contextMessages;
  }

  /**
   * Get statistics about the context (token usage, strategy, etc).
   */
  getContextStats(contextMessages, model) {
    const totalTokens = contextMessages.reduce(
      (sum, msg) => sum + this.estimateTokenCount(msg.content),
      0
    );
    const config = this.modelManager.getModelConfig(model);
    const maxContextTokens = config.contextConfig.maxTokens;
    const usagePercentage = (totalTokens / maxContextTokens) * 100;
    return {
      messageCount: contextMessages.length,
      totalTokens,
      maxContextTokens,
      usagePercentage,
      strategy: config.contextConfig.contextStrategy,
    };
  }

  /**
   * Set the model manager (for dynamic model config changes).
   */
  setModelManager(modelManager) {
    this.modelManager = modelManager;
  }
}

// Export default context manager instance
let _defaultContextManager = null;

export function getDefaultContextManager() {
  if (!_defaultContextManager) {
    _defaultContextManager = new ContextManager();
  }
  return _defaultContextManager;
}

// For backward compatibility
export const defaultContextManager = getDefaultContextManager();

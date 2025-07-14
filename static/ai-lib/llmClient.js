// LLM Client Library
// This module handles API calls to different LLM providers (GroqCloud, OpenAI, Anthropic, etc.)
// It abstracts the details of endpoint URLs, API keys, and request/response formatting.

// LLM client class: manages all LLM API interactions
export class LLMClient {
  /**
   * Construct a new LLMClient instance.
   * Initializes endpoints and API key storage.
   */
  constructor() {
    this.endpoints = {
      groq: "https://api.groq.com/openai/v1/chat/completions", // GroqCloud API endpoint
      openai: "/api/openai", // Placeholder for future OpenAI endpoint
      anthropic: "/api/anthropic", // Placeholder for future Anthropic endpoint
    };
    this.apiKeys = {
      groq: null,
      openai: null,
      anthropic: null,
    };
  }

  /**
   * Set the API key for a provider.
   * @param {string} provider - Provider name (e.g., 'groq').
   * @param {string} key - API key value.
   */
  setApiKey(provider, key) {
    this.apiKeys[provider] = key;
  }

  /**
   * Get the API key for a provider.
   * @param {string} provider - Provider name.
   * @returns {string|null} - API key value.
   */
  getApiKey(provider) {
    return this.apiKeys[provider];
  }

  /**
   * Main method to call an LLM with the given parameters.
   * Handles endpoint selection, API key, and request formatting.
   * @param {object} params - LLM call parameters (model, messages, etc).
   * @returns {Promise<object>} - LLM response or error object.
   */
  async callLLM(params) {
    const {
      model,
      messages,
      maxTokens,
      temperature,
      task,
      complexity,
      reason,
      contextMessages,
    } = params;

    console.log("Calling LLM with params:", {
      model,
      task,
      temperature,
      maxTokens,
      complexity,
      contextMessages: contextMessages?.length || 0,
    });

    try {
      // Determine provider from model name
      const provider = this.getProviderFromModel(model);
      const endpoint = this.endpoints[provider] || this.endpoints.groq;
      const apiKey = this.getApiKey(provider);

      // Prepare headers (add Authorization if API key is set)
      const headers = {
        "Content-Type": "application/json",
      };
      if (apiKey) {
        headers["Authorization"] = `Bearer ${apiKey}`;
      }

      // Make the API call
      const response = await fetch(endpoint, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model,
          max_tokens: maxTokens,
          temperature,
          messages,
        }),
      });

      // Handle non-OK responses
      if (!response.ok) {
        const errorText = await response.text();
        return this.createErrorResponse("LLM call failed", errorText, params);
      }

      // Parse and return the successful response
      const result = await response.json();
      return this.createSuccessResponse(result, params);
    } catch (error) {
      // Catch network or other errors
      return this.createErrorResponse("LLM call error", error.message, params);
    }
  }

  /**
   * Determine the provider name from the model name.
   * @param {string} model - Model identifier.
   * @returns {string} - Provider name.
   */
  getProviderFromModel(model) {
    if (model.includes("llama") || model.includes("mixtral")) {
      return "groq";
    } else if (model.includes("gpt")) {
      return "openai";
    } else if (model.includes("claude")) {
      return "anthropic";
    }
    return "groq"; // Default to groq
  }

  /**
   * Format a successful LLM response with metadata.
   */
  createSuccessResponse(result, params) {
    return {
      ...result,
      temperature: params.temperature,
      model: params.model,
      task: params.task,
      complexity: params.complexity,
      reason: params.reason,
      contextMessages: params.contextMessages?.length || 0,
      success: true,
    };
  }

  /**
   * Format an error response with details and metadata.
   */
  createErrorResponse(error, details, params) {
    return {
      error,
      details,
      temperature: params.temperature,
      model: params.model,
      task: params.task,
      complexity: params.complexity,
      reason: params.reason,
      contextMessages: params.contextMessages?.length || 0,
      success: false,
    };
  }

  /**
   * Test if a model is available by making a test call.
   * @param {string} model - Model identifier.
   * @returns {Promise<boolean>} - True if the model responds successfully.
   */
  async testModel(model) {
    try {
      const testParams = {
        model,
        messages: [{ role: "user", content: "Hello" }],
        maxTokens: 10,
        temperature: 0.1,
        task: "LLM_Default",
        complexity: "SHORT",
        reason: "Test",
        contextMessages: [],
      };
      const result = await this.callLLM(testParams);
      return result.success;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get available models from a provider (not implemented for GroqCloud).
   */
  async getAvailableModels(provider = "groq") {
    try {
      const response = await fetch(`/api/models?provider=${provider}`);
      if (response.ok) {
        return await response.json();
      }
      return [];
    } catch (error) {
      console.error("Error fetching available models:", error);
      return [];
    }
  }

  /**
   * Set a custom endpoint for a provider.
   */
  setEndpoint(provider, endpoint) {
    this.endpoints[provider] = endpoint;
  }

  /**
   * Get the endpoint for a provider.
   */
  getEndpoint(provider) {
    return this.endpoints[provider];
  }
}

// Export default LLM client instance (singleton)
let _defaultLLMClient = null;

export function getDefaultLLMClient() {
  if (!_defaultLLMClient) {
    _defaultLLMClient = new LLMClient();
  }
  return _defaultLLMClient;
}

// For backward compatibility
export const defaultLLMClient = getDefaultLLMClient();

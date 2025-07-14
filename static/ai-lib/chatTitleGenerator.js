// Chat Title Generator
// This module handles intelligent chat title generation using the AI library.
// It can generate, update, and clean chat titles based on conversation content.

// Chat title generator class: manages all chat title logic
export class ChatTitleGenerator {
  /**
   * Construct a new ChatTitleGenerator instance.
   * @param {object|null} aiOrchestrator - Optional orchestrator for LLM calls.
   */
  constructor(aiOrchestrator = null) {
    this.aiOrchestrator = aiOrchestrator;
  }

  /**
   * Set the AI orchestrator (for dependency injection or circular avoidance).
   */
  setAIOrchestrator(aiOrchestrator) {
    this.aiOrchestrator = aiOrchestrator;
  }

  /**
   * Get the AI orchestrator, loading it if needed (avoids circular import).
   */
  async getAIOrchestrator() {
    if (!this.aiOrchestrator) {
      const { getDefaultAIOrchestrator } = await import("./index.js");
      this.aiOrchestrator = getDefaultAIOrchestrator();
    }
    return this.aiOrchestrator;
  }

  /**
   * Generate a chat title using the AI library and the first message.
   * @param {string} firstMessage - The first user message.
   * @returns {Promise<string>} - The generated chat title.
   */
  async generateChatTitle(firstMessage) {
    if (!firstMessage || firstMessage.length < 10) {
      return "New Chat";
    }
    try {
      // Get AI orchestrator
      const aiOrchestrator = await this.getAIOrchestrator();
      // Use the AI orchestrator to generate a title
      const titlePrompt = `Generate a short, descriptive title (3-6 words) for a chat conversation that starts with this message: "${firstMessage.substring(
        0,
        200
      )}${firstMessage.length > 200 ? "..." : ""}"

Title:`;
      const result = await aiOrchestrator.processQuery(titlePrompt);
      // Debug: Log the result structure
      console.log("Chat title generation result:", result);
      // Extract title from the response - handle different response formats
      let title = null;
      if (result.success) {
        // Try different response formats
        if (result.response) {
          title = result.response;
          console.log("Found title in result.response:", title);
        } else if (
          result.choices &&
          result.choices.length > 0 &&
          result.choices[0].message
        ) {
          title = result.choices[0].message.content;
          console.log(
            "Found title in result.choices[0].message.content:",
            title
          );
        } else if (result.content) {
          title = result.content;
          console.log("Found title in result.content:", title);
        } else {
          console.warn(
            "No title found in result, result keys:",
            Object.keys(result)
          );
        }
      } else {
        console.warn("Result not successful:", result.error || result.details);
      }
      if (title) {
        const cleanTitle = this.cleanTitle(title.trim());
        console.log("Final clean title:", cleanTitle);
        return cleanTitle || "New Chat";
      } else {
        console.warn("Failed to generate chat title with AI, using fallback");
        return this.generateFallbackTitle(firstMessage);
      }
    } catch (error) {
      console.error("Error generating chat title:", error);
      return this.generateFallbackTitle(firstMessage);
    }
  }

  /**
   * Generate a fallback title using simple text processing.
   * @param {string} message - The first user message.
   * @returns {string} - Fallback chat title.
   */
  generateFallbackTitle(message) {
    if (!message || message.length < 10) {
      return "New Chat";
    }
    // Extract first sentence or first 50 characters
    const firstSentence = message.split(/[.!?]/)[0].trim();
    const title =
      firstSentence.length > 50
        ? firstSentence.substring(0, 50) + "..."
        : firstSentence;
    return title || "New Chat";
  }

  /**
   * Clean up the generated title (remove quotes, trim, limit length).
   * @param {string} title - The raw title string.
   * @returns {string} - Cleaned title.
   */
  cleanTitle(title) {
    if (!title) return "New Chat";
    return title
      .replace(/^["']|["']$/g, "") // Remove surrounding quotes
      .replace(/\s+/g, " ") // Normalize spaces
      .trim()
      .substring(0, 50); // Limit to 50 characters
  }

  /**
   * Generate a title based on the content of the chat (first few messages).
   * @param {Array} messages - Chat messages (with .user fields).
   * @returns {Promise<string>} - Generated or fallback title.
   */
  async generateTitleFromChatContent(messages) {
    if (!messages || messages.length === 0) {
      return "New Chat";
    }
    try {
      // Get AI orchestrator
      const aiOrchestrator = await this.getAIOrchestrator();
      // Combine first few messages for context
      const contextMessages = messages.slice(0, 3);
      const contextText = contextMessages
        .map((msg) => msg.user)
        .join(" ")
        .substring(0, 300);
      const titlePrompt = `Based on this chat conversation, generate a short, descriptive title (3-6 words):\n\n${contextText}\n\nTitle:`;
      const result = await aiOrchestrator.processQuery(titlePrompt);
      // Extract title from the response - handle different response formats
      let title = null;
      if (result.success) {
        // Try different response formats
        if (result.response) {
          title = result.response;
        } else if (
          result.choices &&
          result.choices.length > 0 &&
          result.choices[0].message
        ) {
          title = result.choices[0].message.content;
        } else if (result.content) {
          title = result.content;
        }
      }
      if (title) {
        return this.cleanTitle(title.trim());
      } else {
        return this.generateFallbackTitle(contextText);
      }
    } catch (error) {
      console.error("Error generating title from chat content:", error);
      return this.generateFallbackTitle(messages[0]?.user || "");
    }
  }

  /**
   * Dynamically update the chat title as the conversation grows.
   * @param {Array} messages - Chat messages.
   * @param {string} currentTitle - Current chat title.
   * @returns {Promise<string>} - Updated or current title.
   */
  async updateChatTitleDynamically(messages, currentTitle) {
    if (!messages || messages.length < 3) {
      return currentTitle;
    }
    // Only update if we have significantly more messages
    if (messages.length % 5 === 0) {
      // Update every 5 messages
      try {
        const newTitle = await this.generateTitleFromChatContent(messages);
        return newTitle;
      } catch (error) {
        console.error("Error updating chat title dynamically:", error);
        return currentTitle;
      }
    }
    return currentTitle;
  }
}

// Export default instance (singleton)
let _defaultChatTitleGenerator = null;

export function getDefaultChatTitleGenerator() {
  if (!_defaultChatTitleGenerator) {
    _defaultChatTitleGenerator = new ChatTitleGenerator();
  }
  return _defaultChatTitleGenerator;
}

// For backward compatibility
export const defaultChatTitleGenerator = getDefaultChatTitleGenerator();

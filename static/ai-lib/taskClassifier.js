// Task Classification Library
// This module handles determining the appropriate task type and model for queries.
// It analyzes the user's query, classifies the task, and selects the best model.

import { TEXT_LENGTH_THRESHOLDS, defaultModelManager } from "./models.js";

// Task type definitions (each with keywords, description, and default settings)
export const TASK_TYPES = {
  LLM_Summary: {
    name: "Summary",
    description: "Summarize or condense information",
    keywords: [
      "summary",
      "summarise",
      "summarize",
      "condense",
      "brief",
      "overview",
    ],
    maxTokens: 2000,
    temperature: 0.3,
  },
  LLM_Creation: {
    name: "Creation",
    description: "Generate creative content",
    keywords: ["create", "generate", "write", "compose", "craft", "develop"],
    maxTokens: 1500,
    temperature: 0.7,
  },
  LLM_Ideation: {
    name: "Ideation",
    description: "Brainstorm ideas and suggestions",
    keywords: [
      "ideate",
      "brainstorm",
      "suggest",
      "ideas",
      "options",
      "alternatives",
    ],
    maxTokens: 1200,
    temperature: 0.5,
  },
  LLM_Analysis: {
    name: "Analysis",
    description: "Analyze, explain, or evaluate content",
    keywords: [
      "analyze",
      "analyse",
      "analysis",
      "explain",
      "compare",
      "evaluate",
      "assess",
      "examine",
      "study",
      "review",
    ],
    maxTokens: 2500,
    temperature: 0.2,
  },
  LLM_Converter: {
    name: "Converter",
    description: "Convert or transform content",
    keywords: [
      "convert",
      "transform",
      "translate",
      "reformat",
      "adapt",
      "modify",
    ],
    maxTokens: 1500,
    temperature: 0.4,
  },
  LLM_Default: {
    name: "General",
    description: "General conversation and assistance",
    keywords: [],
    maxTokens: 1500,
    temperature: 0.3,
  },
};

// Text complexity analyzer: determines the complexity of a query
export class TextComplexityAnalyzer {
  /**
   * Analyze a text string and return complexity info.
   * @param {string} text - The text to analyze.
   * @returns {object} - Complexity, reason, and stats.
   */
  static analyze(text) {
    if (!text || typeof text !== "string") {
      return {
        complexity: "SHORT",
        reason: "Empty or invalid text",
        stats: {
          length: 0,
          wordCount: 0,
          sentenceCount: 0,
          paragraphCount: 0,
          complexityScore: 0,
        },
      };
    }
    const length = text.length;
    const wordCount = text.split(/\s+/).length;
    const sentenceCount = text.split(/[.!?]+/).length - 1;
    const paragraphCount = text.split(/\n\s*\n/).length;
    // Calculate a simple complexity score
    const complexityScore =
      length / 100 + // Length factor
      wordCount / 20 + // Word density factor
      sentenceCount * 2 + // Sentence complexity factor
      paragraphCount * 3; // Structural complexity factor
    // Determine complexity level
    let complexity, reason;
    if (length <= TEXT_LENGTH_THRESHOLDS.SHORT) {
      complexity = "SHORT";
      reason = `Short text (${length} chars, ${wordCount} words)`;
    } else if (length <= TEXT_LENGTH_THRESHOLDS.MEDIUM) {
      complexity = "MEDIUM";
      reason = `Medium text (${length} chars, ${wordCount} words)`;
    } else if (length <= TEXT_LENGTH_THRESHOLDS.LONG) {
      complexity = "LONG";
      reason = `Long text (${length} chars, ${wordCount} words, ${sentenceCount} sentences)`;
    } else {
      complexity = "VERY_LONG";
      reason = `Very long text (${length} chars, ${wordCount} words, ${paragraphCount} paragraphs)`;
    }
    // Check for analytical keywords that might warrant model upgrade
    const analyticalKeywords = [
      "analyze",
      "analyse",
      "compare",
      "evaluate",
      "summarize",
      "summarise",
      "explain",
      "discuss",
      "research",
      "study",
      "review",
      "assess",
    ];
    const hasAnalyticalContent = analyticalKeywords.some((keyword) =>
      text.toLowerCase().includes(keyword)
    );
    if (hasAnalyticalContent && complexity === "SHORT") {
      complexity = "MEDIUM";
      reason += " - Upgraded due to analytical keywords";
    }
    return {
      complexity,
      reason,
      stats: {
        length,
        wordCount,
        sentenceCount,
        paragraphCount,
        complexityScore,
      },
    };
  }
}

// Task classifier: determines the task type and model for a query
export class TaskClassifier {
  /**
   * Construct a new TaskClassifier instance.
   * @param {object} modelManager - The model manager for model selection.
   */
  constructor(modelManager = defaultModelManager) {
    this.modelManager = modelManager;
  }
  /**
   * Classify a query and determine task type and model.
   * @param {string} queryText - The user's query.
   * @returns {Promise<object>} - Classification result.
   */
  async classifyQuery(queryText) {
    if (!queryText) {
      return {
        task: "LLM_Default",
        model: this.modelManager.getModelForComplexity("SHORT"),
        complexity: "SHORT",
        reason: "Empty query",
      };
    }
    // Analyze text complexity
    const complexityAnalysis = TextComplexityAnalyzer.analyze(queryText);
    console.log("Text complexity analysis:", complexityAnalysis);
    // Determine task type by keywords
    let task = this.determineTaskType(queryText);
    // If no keyword matches, use GroqCloud llama3-8b-8192 to classify
    if (task === "LLM_Default") {
      const prompt =
        "Classify the following user query into one of the following task types: " +
        "LLM_Summary, LLM_Creation, LLM_Ideation, LLM_Analysis, LLM_Converter, or LLM_Default. " +
        "If the query asks to 'analyze', 'analyse', 'analysis', 'explain', 'compare', 'evaluate', 'assess', or similar, classify as LLM_Analysis. " +
        "Only return the task type.\n\nUser query: " +
        queryText +
        "\nTask type:";
      try {
        const response = await fetch("/api/llm", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "llama3-8b-8192",
            messages: [
              {
                role: "system",
                content:
                  "You are a helpful AI assistant that classifies user queries.",
              },
              { role: "user", content: prompt },
            ],
            max_tokens: 10,
            temperature: 0.0,
          }),
        });
        if (response.ok) {
          const result = await response.json();
          if (result.choices && result.choices.length > 0) {
            task = result.choices[0].message.content.trim();
          }
        }
      } catch (e) {
        // Fallback to LLM_Default
        task = "LLM_Default";
      }
    }
    // Get appropriate model
    const model = this.getModelForTask(task, complexityAnalysis.complexity);
    return {
      task,
      model,
      complexity: complexityAnalysis.complexity,
      reason: complexityAnalysis.reason,
      stats: complexityAnalysis.stats,
    };
  }
  /**
   * Determine task type based on keywords in the query.
   * @param {string} queryText - The user's query.
   * @returns {string} - Task type identifier.
   */
  determineTaskType(queryText) {
    const query = queryText.toLowerCase().trim();
    // Check for task-specific keywords
    for (const [taskType, taskConfig] of Object.entries(TASK_TYPES)) {
      if (taskType === "LLM_Default") continue; // Skip default, check others first
      for (const keyword of taskConfig.keywords) {
        if (query.includes(keyword)) {
          return taskType;
        }
      }
    }
    // If no specific keywords found, use AI classification
    return "LLM_Default";
  }
  /**
   * Get the appropriate model for a task and complexity.
   * @param {string} task - Task type.
   * @param {string} complexity - Complexity level.
   * @returns {string} - Model ID.
   */
  getModelForTask(task, complexity) {
    // Get base model for complexity
    let model = this.modelManager.getModelForComplexity(complexity);
    // Override for complex analysis tasks
    if (
      task === "LLM_Analysis" &&
      (complexity === "LONG" || complexity === "VERY_LONG")
    ) {
      // Use better models for complex analysis
      const betterModels = [
        "mixtral-8x7b-32768",
        "llama3-70b-8192",
        "claude-3-sonnet",
      ];
      for (const betterModel of betterModels) {
        if (
          this.modelManager.modelSupportsCapability(betterModel, "analysis")
        ) {
          model = betterModel;
          break;
        }
      }
    }
    return model;
  }
  /**
   * Get the config for a task type.
   */
  getTaskConfig(taskType) {
    return TASK_TYPES[taskType] || TASK_TYPES.LLM_Default;
  }
  /**
   * Get all available task types.
   */
  getAvailableTasks() {
    return Object.keys(TASK_TYPES);
  }
  /**
   * Set the model manager (for dynamic model config changes).
   */
  setModelManager(modelManager) {
    this.modelManager = modelManager;
  }
}

// Export default task classifier instance
let _defaultTaskClassifier = null;

export function getDefaultTaskClassifier() {
  if (!_defaultTaskClassifier) {
    _defaultTaskClassifier = new TaskClassifier();
  }
  return _defaultTaskClassifier;
}

// For backward compatibility
export const defaultTaskClassifier = getDefaultTaskClassifier();

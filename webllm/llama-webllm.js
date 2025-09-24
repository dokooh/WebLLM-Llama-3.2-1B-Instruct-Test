/**
 * WebLLM Implementation for Llama-3.2-1B-Instruct
 * This script handles the initialization and interaction with the Llama model
 */

import * as webllm from "@mlc-ai/web-llm";

class LlamaWebLLM {
    constructor() {
        this.engine = null;
        this.isLoading = false;
        this.isReady = false;
        this.modelId = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
    }

    /**
     * Initialize the WebLLM engine with Llama-3.2-1B-Instruct
     */
    async initialize() {
        if (this.isLoading || this.isReady) {
            return;
        }

        try {
            this.isLoading = true;
            this.updateStatus("Initializing WebLLM engine...");

            // Create the engine with progress callback
            this.engine = new webllm.MLCEngine();
            
            this.updateStatus("Loading Llama-3.2-1B-Instruct model...");
            
            // Initialize with the specific Llama model
            await this.engine.reload(this.modelId, {
                temperature: 0.7,
                top_p: 0.9,
            });

            this.isReady = true;
            this.isLoading = false;
            this.updateStatus("✅ Llama-3.2-1B-Instruct is ready!");
            this.enableUI();

        } catch (error) {
            console.error("Failed to initialize WebLLM:", error);
            this.updateStatus(`❌ Failed to initialize: ${error.message}`);
            this.isLoading = false;
        }
    }

    /**
     * Validate and truncate message if too long
     */
    validateMessage(message, maxLength = 8000) {
        if (message.length > maxLength) {
            console.warn(`⚠️ Message truncated from ${message.length} to ${maxLength} characters`);
            return message.substring(0, maxLength) + "\n\n[Message truncated due to length limit]";
        }
        return message;
    }

    /**
     * Generate a response using the Llama model
     */
    async generateResponse(prompt, onUpdate = null) {
        if (!this.isReady) {
            throw new Error("Model not ready. Please initialize first.");
        }

        // Validate and truncate if necessary
        const validatedPrompt = this.validateMessage(prompt);
        
        if (validatedPrompt !== prompt) {
            this.updateStatus("⚠️ Long message truncated for processing");
        }

        try {
            this.updateStatus("🧠 Generating response...");
            
            const messages = [
                {
                    role: "user", 
                    content: validatedPrompt
                }
            ];

            let fullResponse = "";

            // Create completion with streaming
            const completion = await this.engine.chat.completions.create({
                messages: messages,
                temperature: 0.7,
                max_tokens: 512,
                stream: true
            });

            // Process streaming response
            for await (const chunk of completion) {
                const delta = chunk.choices[0]?.delta?.content;
                if (delta) {
                    fullResponse += delta;
                    if (onUpdate) {
                        onUpdate(fullResponse);
                    }
                }
            }

            this.updateStatus("✅ Response generated successfully!");
            return fullResponse;

        } catch (error) {
            console.error("Error generating response:", error);
            this.updateStatus(`❌ Error: ${error.message}`);
            throw error;
        }
    }

    /**
     * Get model information
     */
    getModelInfo() {
        return {
            modelId: this.modelId,
            isReady: this.isReady,
            isLoading: this.isLoading
        };
    }

    /**
     * Reset the conversation
     */
    async resetConversation() {
        if (this.isReady) {
            await this.engine.resetChat();
            this.updateStatus("🔄 Conversation reset");
        }
    }

    /**
     * Update status in the UI
     */
    updateStatus(message) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
            console.log("Status:", message);
        }
    }

    /**
     * Enable UI controls when model is ready
     */
    enableUI() {
        const sendButton = document.getElementById('sendButton');
        const promptInput = document.getElementById('promptInput');
        const resetButton = document.getElementById('resetButton');

        if (sendButton) sendButton.disabled = false;
        if (promptInput) promptInput.disabled = false;
        if (resetButton) resetButton.disabled = false;
    }

    /**
     * Disable UI controls
     */
    disableUI() {
        const sendButton = document.getElementById('sendButton');
        const promptInput = document.getElementById('promptInput');
        const resetButton = document.getElementById('resetButton');

        if (sendButton) sendButton.disabled = true;
        if (promptInput) promptInput.disabled = true;
        if (resetButton) resetButton.disabled = true;
    }
}

// Export for global use
window.LlamaWebLLM = LlamaWebLLM;
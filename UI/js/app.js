// Enhanced ChatBot Pro - JavaScript Implementation
class ChatBot {
    constructor() {
        this.chatHistory = [];
        this.intents = {
            greetings: {
                patterns: ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "what's up", "how are you"],
                responses: ["Hello! I'm your AI assistant with Wikipedia integration. Ask me about any topic!", "Hi there! I can help you find information about anything. What would you like to learn?", "Hey! I'm here to help you explore knowledge. What are you curious about?"]
            },
            wikipedia_search: {
                patterns: ["what is", "who is", "tell me about", "explain", "define", "information about", "facts about", "search for", "look up", "when was", "where is", "how does"],
                responses: ["Let me search Wikipedia for that information...", "I'll find comprehensive information about that topic...", "Searching Wikipedia for details..."]
            },
            science_tech: {
                keywords: ["artificial intelligence", "machine learning", "quantum physics", "space", "technology", "science", "computer", "internet", "biology", "chemistry", "physics", "mathematics"],
                response: "That's a fascinating scientific topic! Let me find detailed information for you..."
            },
            history: {
                keywords: ["history", "war", "ancient", "medieval", "renaissance", "historical", "empire", "civilization", "revolution"],
                response: "History is fascinating! I'll search for comprehensive information about this topic..."
            },
            geography: {
                keywords: ["country", "city", "capital", "continent", "ocean", "mountain", "river", "desert", "island", "geography"],
                response: "That's an interesting geographical topic! Let me find information about that location..."
            },
            thanks: {
                patterns: ["thank you", "thanks", "appreciate it", "that's helpful", "great", "awesome", "perfect"],
                responses: ["You're welcome! Feel free to ask about any other topics.", "Happy to help! What else would you like to know?", "Glad I could help! I'm here for any other questions."]
            },
            goodbye: {
                patterns: ["bye", "goodbye", "see you", "farewell", "take care", "until next time"],
                responses: ["Goodbye! It was great helping you learn something new today!", "See you later! Come back anytime for more information!", "Take care! I'm here whenever you need help with any topic."]
            }
        };

        this.initializeElements();
        this.setupModal();
        this.attachEventListeners();
    }

    initializeElements() {
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.infoBtn = document.getElementById('infoBtn');
        this.infoModal = document.getElementById('infoModal');
        this.modalClose = document.getElementById('modalClose');
        this.modalBackdrop = document.getElementById('modalBackdrop');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.charCounter = document.getElementById('charCounter');
    }

    setupModal() {
        // Ensure modal starts hidden
        if (this.infoModal) {
            this.infoModal.classList.add('hidden');
        }
        // Reset body overflow
        document.body.style.overflow = '';
    }

    attachEventListeners() {
        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Character counter
        this.chatInput.addEventListener('input', () => {
            const length = this.chatInput.value.length;
            this.charCounter.textContent = `${length}/500`;

            if (length >= 450) {
                this.charCounter.style.color = 'var(--color-warning)';
            } else if (length >= 400) {
                this.charCounter.style.color = 'var(--color-info)';
            } else {
                this.charCounter.style.color = 'var(--color-text-secondary)';
            }
        });

        // Quick actions
        this.clearBtn.addEventListener('click', () => this.clearChat());
//        this.exportBtn.addEventListener('click', () => this.exportHistory());

        // Modal events with proper event handling
        this.infoBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.showModal();
        });

        this.modalClose.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.hideModal();
        });

        this.modalBackdrop.addEventListener('click', (e) => {
            if (e.target === this.modalBackdrop) {
                this.hideModal();
            }
        });

        // Example queries
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('example-btn')) {
                e.preventDefault();
                const query = e.target.getAttribute('data-query');
                if (query) {
                    this.chatInput.value = query;
                    this.sendMessage();
                }
            }
        });

        // ESC key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.infoModal && !this.infoModal.classList.contains('hidden')) {
                this.hideModal();
            }
        });

        // Focus input on page load
        setTimeout(() => {
            this.chatInput.focus();
        }, 100);
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        // Add user message
        this.addMessage(message, 'user');
        this.chatInput.value = '';
        this.charCounter.textContent = '0/500';
        this.charCounter.style.color = 'var(--color-text-secondary)';

        // Show typing indicator
        this.showTypingIndicator();

        // Process message and get response
        try {
            const response = await this.processMessage(message);
            this.hideTypingIndicator();
            this.addMessage(response.text, 'bot', response.link);
        } catch (error) {
            console.error('Error processing message:', error);
            this.hideTypingIndicator();
            this.addMessage("I apologize, but I encountered an error while processing your request. Please try again.", 'bot');
        }

        // Focus input for next message
        this.chatInput.focus();
    }

    addMessage(text, sender, link = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message--${sender}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        const messageText = document.createElement('p');
        messageText.className = 'message-text';
        messageText.textContent = text;
        bubble.appendChild(messageText);

        if (link) {
            const linkElement = document.createElement('a');
            linkElement.href = link;
            linkElement.className = 'wikipedia-link';
            linkElement.textContent = '📖 Read more on Wikipedia';
            linkElement.target = '_blank';
            linkElement.rel = 'noopener noreferrer';
            bubble.appendChild(linkElement);
        }

        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        bubble.appendChild(timestamp);

        messageDiv.appendChild(bubble);
        this.chatMessages.appendChild(messageDiv);

        // Store in history
        this.chatHistory.push({
            text,
            sender,
            timestamp: new Date().toISOString(),
            link
        });

        // Scroll to bottom
        this.scrollToBottom();
    }

    async processMessage(message) {
        const lowerMessage = message.toLowerCase();

        // Check for greetings
        if (this.matchesPatterns(lowerMessage, this.intents.greetings.patterns)) {
            return {
                text: this.getRandomResponse(this.intents.greetings.responses)
            };
        }

        // Check for thanks
        if (this.matchesPatterns(lowerMessage, this.intents.thanks.patterns)) {
            return {
                text: this.getRandomResponse(this.intents.thanks.responses)
            };
        }

        // Check for goodbye
        if (this.matchesPatterns(lowerMessage, this.intents.goodbye.patterns)) {
            return {
                text: this.getRandomResponse(this.intents.goodbye.responses)
            };
        }

        // Check if it's a Wikipedia search query
        if (this.matchesPatterns(lowerMessage, this.intents.wikipedia_search.patterns) ||
            this.containsKeywords(lowerMessage, this.intents.science_tech.keywords) ||
            this.containsKeywords(lowerMessage, this.intents.history.keywords) ||
            this.containsKeywords(lowerMessage, this.intents.geography.keywords)) {

            return await this.searchWikipedia(message);
        }

        // Default Wikipedia search for any other query
        return await this.searchWikipedia(message);
    }

    matchesPatterns(message, patterns) {
        return patterns.some(pattern => message.includes(pattern));
    }

    containsKeywords(message, keywords) {
        return keywords.some(keyword => message.includes(keyword.toLowerCase()));
    }

    getRandomResponse(responses) {
        return responses[Math.floor(Math.random() * responses.length)];
    }

    async searchWikipedia(query) {
        try {
            // Extract search term from query
            const searchTerm = this.extractSearchTerm(query);

            // Search Wikipedia
            const searchUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(searchTerm)}`;

            const response = await fetch(searchUrl, {
                headers: {
                    'Accept': 'application/json',
                    'User-Agent': 'ChatBot/1.0 (https://example.com/contact)'
                }
            });

            if (response.ok) {
                const data = await response.json();

                if (data.extract) {
                    let responseText = data.extract;

                    // Truncate if too long
                    if (responseText.length > 500) {
                        responseText = responseText.substring(0, 500) + '...';
                    }

                    return {
                        text: responseText,
                        link: data.content_urls ? data.content_urls.desktop.page : `https://en.wikipedia.org/wiki/${encodeURIComponent(searchTerm)}`
                    };
                } else {
                    // Try alternative search if direct lookup fails
                    return await this.alternativeSearch(searchTerm);
                }
            } else {
                return await this.alternativeSearch(searchTerm);
            }
        } catch (error) {
            console.error('Wikipedia search error:', error);
            return {
                text: "I'm having trouble accessing Wikipedia right now. Please try rephrasing your question or try again later."
            };
        }
    }

    async alternativeSearch(searchTerm) {
        try {
            // Use Wikipedia's search API
            const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch=${encodeURIComponent(searchTerm)}&origin=*`;

            const response = await fetch(searchUrl);
            const data = await response.json();

            if (data.query && data.query.search && data.query.search.length > 0) {
                const firstResult = data.query.search[0];
                const title = firstResult.title;

                // Get page summary
                const summaryUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`;
                const summaryResponse = await fetch(summaryUrl);

                if (summaryResponse.ok) {
                    const summaryData = await summaryResponse.json();
                    let responseText = summaryData.extract || firstResult.snippet.replace(/<[^>]*>/g, '');

                    if (responseText.length > 500) {
                        responseText = responseText.substring(0, 500) + '...';
                    }

                    return {
                        text: responseText,
                        link: `https://en.wikipedia.org/wiki/${encodeURIComponent(title)}`
                    };
                }
            }

            return {
                text: `I couldn't find specific information about "${searchTerm}" on Wikipedia. Try being more specific or using different keywords.`
            };
        } catch (error) {
            return {
                text: "I'm having trouble searching Wikipedia right now. Please try again later."
            };
        }
    }

    extractSearchTerm(query) {
        const lowerQuery = query.toLowerCase();

        // Remove common question words and patterns
        let searchTerm = lowerQuery
            .replace(/^(what is|who is|tell me about|explain|define|information about|facts about|search for|look up|when was|where is|how does)\s*/i, '')
            .replace(/\?$/, '')
            .trim();

        // If no meaningful term extracted, use the original query
        if (!searchTerm || searchTerm.length < 2) {
            searchTerm = query.trim();
        }

        return searchTerm;
    }

    showTypingIndicator() {
        if (this.typingIndicator) {
            this.typingIndicator.classList.remove('hidden');
            this.scrollToBottom();
        }
    }

    hideTypingIndicator() {
        if (this.typingIndicator) {
            this.typingIndicator.classList.add('hidden');
        }
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }

    clearChat() {
        // Keep the welcome card, remove other messages
        const messages = this.chatMessages.querySelectorAll('.message');
        messages.forEach(message => message.remove());

        this.chatHistory = [];

        // Add confirmation message
        this.addMessage("Chat cleared! Feel free to ask me anything.", 'bot');
    }

//    exportHistory() {
//        if (this.chatHistory.length === 0) {
//            this.addMessage("No chat history to export yet. Start a conversation first!", 'bot');
//            return;
//        }
//
//        const exportData = {
//            timestamp: new Date().toISOString(),
//            chatHistory: this.chatHistory,
//            metadata: {
//                botName: "Enhanced ChatBot Pro",
//                version: "1.0",
//                totalMessages: this.chatHistory.length
//            }
//        };
//
//        const dataStr = JSON.stringify(exportData, null, 2);
//        const dataBlob = new Blob([dataStr], { type: 'application/json' });
//        const url = URL.createObjectURL(dataBlob);
//
//        const link = document.createElement('a');
//        link.href = url;
//        link.download = `chatbot-history-${new Date().toISOString().split('T')[0]}.json`;
//        document.body.appendChild(link);
//        link.click();
//        document.body.removeChild(link);
//        URL.revokeObjectURL(url);
//
//        this.addMessage("Chat history exported successfully! Check your downloads folder.", 'bot');
//    }

    showModal() {
        if (this.infoModal) {
            this.infoModal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';

            // Focus the close button for accessibility
            setTimeout(() => {
                this.modalClose.focus();
            }, 100);
        }
    }

    hideModal() {
        if (this.infoModal) {
            this.infoModal.classList.add('hidden');
            document.body.style.overflow = '';

            // Return focus to the info button
            this.infoBtn.focus();
        }
    }
}

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatBot();
});





































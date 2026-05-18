import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import re
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import wikipedia
import requests
from urllib.parse import quote
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# app = Flask(__name__, static_folder='static', template_folder='templates')
app = Flask(
    __name__,
    static_folder='UI',
    template_folder='UI/templates'
)
CORS(app)

class EnhancedChatbotAPI:
    def __init__(self):
        try:
            # Load model and data
            self.model = load_model('models/chatbot_model_enhanced.h5')
            with open('data/intents_enhanced_wikipedia.json', 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
            self.words = pickle.load(open('models/words_enhanced.pkl','rb'))
            self.classes = pickle.load(open('models/classes_enhanced.pkl','rb'))

            # Enhanced features
            self.sessions = {}  # Store conversation history by session
            self.confidence_threshold = 0.25

            # Entity patterns for extraction
            self.entity_patterns = {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                'order_id': r'\b(order|#)\s*[A-Za-z0-9]{6,}\b',
                'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                'price': r'\$\d+(\.\d{2})?'
            }

            print("Enhanced Chatbot API initialized successfully!")

        except Exception as e:
            print(f"Error initializing chatbot: {e}")
            self.model = None

    def clean_up_sentence(self, sentence):
        """Enhanced sentence preprocessing"""
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words, show_details=False):
        """Create bag of words array"""
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0]*len(words)
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    bag[i] = 1
                    if show_details:
                        print(f"Found in bag: {w}")
        return np.array(bag)

    def predict_class(self, sentence):
        """Predict intent with enhanced confidence handling"""
        if not self.model:
            return []

        p = self.bow(sentence, self.words, show_details=False)
        res = self.model.predict(np.array([p]))[0]

        results = [[i,r] for i,r in enumerate(res) if r > self.confidence_threshold]
        results.sort(key=lambda x: x[1], reverse=True)

        return_list = []
        for r in results:
            return_list.append({
                "intent": self.classes[r[0]], 
                "probability": str(r[1])
            })
        return return_list

    def search_wikipedia(self, query, max_sentences=3):
        """Search Wikipedia and return summary"""
        try:
            # Set language to English
            wikipedia.set_lang("en")

            # Search for the topic
            search_results = wikipedia.search(query, results=5)
            if not search_results:
                return "I couldn't find any information about that topic on Wikipedia."

            # Get the page summary
            page_title = search_results[0]
            summary = wikipedia.summary(page_title, sentences=max_sentences)

            # Get the page URL
            page = wikipedia.page(page_title)
            url = page.url

            response = f"Here's what I found about **{page_title}** on Wikipedia:\n\n"
            response += f"{summary}\n\n"
            response += f"🔗 **Read more:** [Wikipedia Article]({url})"

            return response

        except wikipedia.DisambiguationError as e:
            # Handle disambiguation
            try:
                # Try the first option from disambiguation
                summary = wikipedia.summary(e.options[0], sentences=max_sentences)
                page = wikipedia.page(e.options[0])
                url = page.url

                response = f"Here's what I found about **{e.options[0]}** on Wikipedia:\n\n"
                response += f"{summary}\n\n"
                response += f"🔗 **Read more:** [Wikipedia Article]({url})\n\n"
                response += f"💡 *Note: There were multiple topics with similar names. I showed you information about {e.options[0]}.*"

                return response
            except:
                return f"I found multiple topics related to '{query}'. Could you be more specific? Some options include: {', '.join(e.options[:5])}"

        except wikipedia.PageError:
            return f"I couldn't find a Wikipedia page for '{query}'. Could you try rephrasing your search?"

        except Exception as e:
            return f"Sorry, I encountered an error while searching Wikipedia. Please try again."

    def extract_search_query(self, text):
        """Extract the search query from user text"""
        # Remove common question words and extract the main topic
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'tell', 'me', 'about', 'is', 'are', 'was', 'were']

        # Clean the text
        cleaned_text = text.lower()

        # Remove question words from the beginning
        words = cleaned_text.split()
        filtered_words = []

        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w\s]', '', word)
            if word and word not in question_words:
                filtered_words.append(word)

        # Join the remaining words
        query = ' '.join(filtered_words)

        # If query is too short, use the original text
        if len(query.strip()) < 2:
            query = re.sub(r'[^\w\s]', '', text.lower())

        return query.strip()

    def get_session_history(self, session_id):
        """Get or create session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'conversation_history': [],
                'context': '',
                'user_name': ''
            }
        return self.sessions[session_id]

    def get_response(self, user_input, session_id):
        """Get chatbot response with Wikipedia integration"""
        session = self.get_session_history(session_id)

        # Check for name extraction
        name_patterns = [r"my name is (\w+)", r"I'm (\w+)", r"call me (\w+)"]
        for pattern in name_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                session['user_name'] = match.group(1).title()
                return f"Nice to meet you, {session['user_name']}! How can I help you today?"

        # Predict intent
        ints = self.predict_class(user_input)

        if not ints:
            return "I'm not sure I understand. Could you please rephrase that?"

        tag = ints[0]['intent']
        confidence = float(ints[0]['probability'])

        # Add to conversation history
        session['conversation_history'].append({
            'user': user_input,
            'intent': tag,
            'confidence': confidence,
            'timestamp': datetime.now()
        })

        # Handle Wikipedia search intents
        if tag == 'wikipedia_search' or confidence < 0.4:  # Low confidence might mean they want to search
            # Check if it's a question or information request
            search_indicators = ['what', 'who', 'when', 'where', 'tell me about', 'information about', 'explain', 'define']
            if any(indicator in user_input.lower() for indicator in search_indicators) or tag == 'wikipedia_search':
                query = self.extract_search_query(user_input)
                if query:
                    return self.search_wikipedia(query)

        # Handle regular intents
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])

                # Personalize response if we know the user's name
                if session['user_name'] and '{name}' in response:
                    response = response.replace('{name}', session['user_name'])

                # Context-aware responses
                if tag == 'greeting' and len(session['conversation_history']) > 1:
                    response = "Welcome back! " + response

                session['context'] = intent.get('context', '')
                return response

        # If no specific intent found, try Wikipedia search as fallback
        query = self.extract_search_query(user_input)
        if query and len(query) > 2:
            return self.search_wikipedia(query)

        return "I'm not sure about that. Could you try asking in a different way?"

# Initialize the chatbot
chatbot_api = EnhancedChatbotAPI()
#
# @app.route('/')
# def index():
#     """Serve the main chat interface"""
#     return render_template('index.html')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        response = chatbot_api.get_response(user_message, session_id)

        return jsonify({
            'response': response,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    # return send_from_directory('static', filename)
    return send_from_directory('UI', filename)

if __name__ == '__main__':
    print("Starting Enhanced ChatBot Server with Wikipedia Integration...")
    print("Access the chatbot at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5001)

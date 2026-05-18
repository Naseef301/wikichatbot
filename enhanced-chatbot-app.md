# Enhanced ChatBot Pro - Complete Project Files

This document contains all the files for your Enhanced ChatBot Pro with Wikipedia integration. Create the following directory structure and copy the respective code into each file.

## Project Structure
```
enhanced-chatbot-pro/
├── 📄 app.py
├── 📄 train_enhanced.py
├── 📄 setup.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📁 data/
│   └── 📄 intents_enhanced_wikipedia.json
├── 📁 templates/
│   └── 📄 index.html
├── 📁 static/
│   ├── 📁 css/
│   │   └── 📄 styles.css
│   └── 📁 js/
│       └── 📄 script.js
└── 📁 models/ (created during training)
```

---

## 📄 app.py (Root Directory)

```python
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

app = Flask(__name__, static_folder='static', template_folder='templates')
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

@app.route('/')
def index():
    """Serve the main chat interface"""
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
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("Starting Enhanced ChatBot Server with Wikipedia Integration...")
    print("Access the chatbot at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
```

---

## 📄 train_enhanced.py (Root Directory)

```python
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import random
import os

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data"""
    required_data = ['punkt', 'wordnet', 'averaged_perceptron_tagger']
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else 
                          f'taggers/{data}' if data == 'averaged_perceptron_tagger' else 
                          f'corpora/{data}')
        except LookupError:
            print(f"Downloading {data}...")
            nltk.download(data)

class EnhancedChatbotTrainer:
    def __init__(self, intents_file='data/intents_enhanced_wikipedia.json'):
        """Initialize the trainer with enhanced features"""
        self.intents_file = intents_file
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '.', ',', '!', "'", '"', ';', ':', '-', '(', ')']
        
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Download NLTK data
        download_nltk_data()

    def load_data(self):
        """Load and parse the intents JSON file"""
        try:
            with open(self.intents_file, 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
            print(f"✓ Loaded {len(self.intents['intents'])} intents from {self.intents_file}")
            
            # Display intent statistics
            total_patterns = sum(len(intent['patterns']) for intent in self.intents['intents'])
            print(f"✓ Total patterns: {total_patterns}")
            
        except FileNotFoundError:
            print(f"❌ Error: Intents file '{self.intents_file}' not found!")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in intents file: {e}")
            raise

    def preprocess_data(self):
        """Enhanced preprocessing with better tokenization"""
        print("\n🔄 Preprocessing data...")
        
        pattern_count = 0
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word
                w = nltk.word_tokenize(pattern.lower())
                self.words.extend(w)
                
                # Add documents in the corpus
                self.documents.append((w, intent['tag']))
                pattern_count += 1

                # Add to classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # Lemmatize, lower each word and remove duplicates
        self.words = [lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))

        # Sort classes
        self.classes = sorted(list(set(self.classes)))

        print(f"✓ Documents: {len(self.documents)}")
        print(f"✓ Classes: {len(self.classes)}")
        print(f"✓ Unique lemmatized words: {len(self.words)}")
        
        # Display classes
        print(f"\nClasses: {', '.join(self.classes)}")

        # Save preprocessed data
        pickle.dump(self.words, open('models/words_enhanced.pkl', 'wb'))
        pickle.dump(self.classes, open('models/classes_enhanced.pkl', 'wb'))
        print("✓ Preprocessed data saved")

    def create_training_data(self):
        """Create training data with improved bag-of-words representation"""
        print("\n🔄 Creating training data...")

        training = []
        output_empty = [0] * len(self.classes)

        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            
            # Lemmatize each word
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

            # Create bag of words array
            for word in self.words:
                bag.append(1) if word in pattern_words else bag.append(0)

            # Output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            training.append([bag, output_row])

        # Shuffle and convert to numpy array
        random.shuffle(training)
        training = np.array(training, dtype=object)

        # Create train and test lists
        self.train_x = list(training[:, 0])
        self.train_y = list(training[:, 1])

        print(f"✓ Training data created: {len(self.train_x)} samples")

    def build_model(self):
        """Build an enhanced neural network model"""
        print("\n🔄 Building enhanced model...")

        model = Sequential()
        
        # Input layer
        model.add(Dense(256, input_shape=(len(self.train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        
        # Hidden layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        
        # Output layer
        model.add(Dense(len(self.train_y[0]), activation='softmax'))

        # Compile model
        adam = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', 
                     optimizer=adam, 
                     metrics=['accuracy'])

        print("✓ Model architecture built")
        model.summary()
        
        return model

    def train_model(self, epochs=200, batch_size=8, validation_split=0.1):
        """Train the model with enhanced parameters"""
        print(f"\n🔄 Training model for {epochs} epochs...")

        model = self.build_model()

        # Train the model
        history = model.fit(np.array(self.train_x), 
                           np.array(self.train_y), 
                           epochs=epochs, 
                           batch_size=batch_size, 
                           verbose=1,
                           validation_split=validation_split)

        # Save model
        model.save('models/chatbot_model_enhanced.h5')
        print("\n✓ Enhanced model saved successfully!")

        # Save training history
        with open('models/training_history.json', 'w') as f:
            history_dict = {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']] if 'val_loss' in history.history else [],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']] if 'val_accuracy' in history.history else []
            }
            json.dump(history_dict, f, indent=2)

        return model, history

    def evaluate_model(self, model):
        """Evaluate the model and show performance metrics"""
        print("\n🔄 Evaluating model...")
        
        # Evaluate on training data
        loss, accuracy = model.evaluate(np.array(self.train_x), np.array(self.train_y), verbose=0)
        print(f"✓ Training accuracy: {accuracy:.4f}")
        print(f"✓ Training loss: {loss:.4f}")
        
        # Test with sample predictions
        print("\n🔄 Testing sample predictions...")
        test_sentences = [
            "hello",
            "what is artificial intelligence",
            "tell me about space",
            "goodbye",
            "thank you"
        ]
        
        for sentence in test_sentences:
            # Simple prediction test
            words = nltk.word_tokenize(sentence.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            
            bag = [0] * len(self.words)
            for w in words:
                for i, word in enumerate(self.words):
                    if word == w:
                        bag[i] = 1
            
            prediction = model.predict(np.array([bag]))[0]
            predicted_class = self.classes[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            print(f"  '{sentence}' -> {predicted_class} (confidence: {confidence:.4f})")

    def train_chatbot(self, epochs=200):
        """Main training pipeline"""
        print("🚀 Starting Enhanced Chatbot Training with Wikipedia Integration...")
        print("=" * 60)
        
        try:
            # Load and preprocess data
            self.load_data()
            self.preprocess_data()
            self.create_training_data()
            
            # Train model
            model, history = self.train_model(epochs=epochs)
            
            # Evaluate model
            self.evaluate_model(model)
            
            print("\n" + "=" * 60)
            print("🎉 Enhanced Chatbot Training Complete!")
            print("\nFiles created:")
            print("  📁 models/chatbot_model_enhanced.h5")
            print("  📁 models/words_enhanced.pkl") 
            print("  📁 models/classes_enhanced.pkl")
            print("  📁 models/training_history.json")
            print("\n✅ Ready to run the chatbot server!")
            
            return model, history
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise

def main():
    """Main function to run the training"""
    print("Enhanced ChatBot Pro - Training Script")
    print("=====================================")
    
    # Check if intents file exists
    intents_file = 'data/intents_enhanced_wikipedia.json'
    if not os.path.exists(intents_file):
        print(f"❌ Intents file not found: {intents_file}")
        print("Please make sure the intents file is in the correct location.")
        return
    
    # Initialize trainer
    trainer = EnhancedChatbotTrainer(intents_file)
    
    # Train the chatbot
    try:
        model, history = trainer.train_chatbot(epochs=200)
        print("\n🎯 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")

if __name__ == "__main__":
    main()
```

---

## 📄 requirements.txt (Root Directory)

```
# Enhanced ChatBot Pro - Dependencies

# Core ML and NLP libraries
tensorflow>=2.13.0
keras>=2.13.1
numpy>=1.24.3
nltk>=3.8.1
scikit-learn>=1.3.0

# Web framework and API
Flask>=2.3.3
Flask-CORS>=4.0.0

# Wikipedia integration
wikipedia>=1.4.0
requests>=2.31.0

# Data processing
pandas>=2.0.3
json5>=0.9.14

# Development and testing
pytest>=7.4.0
python-dotenv>=1.0.0

# Optional: For advanced features
# speechrecognition>=3.10.0  # For voice input
# pyttsx3>=2.90  # For text-to-speech
# matplotlib>=3.7.2  # For training visualization
# seaborn>=0.12.2  # For enhanced plots
```

---

## 📄 setup.py (Root Directory)

```python
#!/usr/bin/env python3
"""
Enhanced ChatBot Pro - Setup Script
Automatically organizes files and sets up the project structure
"""

import os
import shutil
import sys
from pathlib import Path

class ChatBotSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.required_dirs = [
            'data',
            'models', 
            'templates',
            'static/css',
            'static/js',
            'static/images',
            'tests',
            'docs'
        ]
        
        self.file_mappings = {
            # Source file -> Target directory
            'intents_enhanced_wikipedia.json': 'data/',
            'intents_comprehensive.json': 'data/intents_backup.json',  # backup original
            'index.html': 'templates/',
            'styles.css': 'static/css/',
            'script.js': 'static/js/',
        }
    
    def create_directories(self):
        """Create the required directory structure"""
        print("🏗️  Creating project directories...")
        
        for directory in self.required_dirs:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {directory}/")
    
    def organize_files(self):
        """Move files to their correct locations"""
        print("\n📁 Organizing files...")
        
        for source_file, target_location in self.file_mappings.items():
            source_path = self.project_root / source_file
            
            if source_path.exists():
                target_path = self.project_root / target_location
                
                # Handle renaming (like backup files)
                if target_location.endswith('.json') and '/' in target_location:
                    target_path = self.project_root / target_location
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    target_path = target_path / source_file
                
                try:
                    if target_path.exists():
                        print(f"⚠️  {target_location} already exists, skipping {source_file}")
                    else:
                        shutil.move(str(source_path), str(target_path))
                        print(f"✓ Moved: {source_file} -> {target_location}")
                except Exception as e:
                    print(f"❌ Error moving {source_file}: {e}")
            else:
                print(f"⚠️  File not found: {source_file}")
    
    def create_env_file(self):
        """Create example environment file"""
        print("\n⚙️  Creating environment configuration...")
        
        env_content = """# Enhanced ChatBot Pro - Environment Configuration
# Copy this file to .env and modify as needed

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_PORT=5000
FLASK_HOST=127.0.0.1

# Chatbot Configuration
CONFIDENCE_THRESHOLD=0.25
MAX_WIKIPEDIA_SENTENCES=3
SESSION_TIMEOUT=3600

# Wikipedia API Configuration
WIKIPEDIA_LANGUAGE=en
WIKIPEDIA_REQUEST_TIMEOUT=10

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=app.log

# Security (generate your own secret key)
SECRET_KEY=your-secret-key-here-change-this-in-production
"""
        
        env_path = self.project_root / '.env.example'
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✓ Created: .env.example")
    
    def create_gitignore(self):
        """Create .gitignore file"""
        print("\n📝 Creating .gitignore...")
        
        gitignore_content = """# Enhanced ChatBot Pro - Git Ignore File

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
chatbot_env/
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project Specific
.env
*.log
models/*.h5
models/*.pkl
models/training_history.json

# Temporary
*.tmp
*.temp
temp/

# Testing
.coverage
htmlcov/
.pytest_cache/
.tox/

# Documentation
docs/_build/
"""
        
        gitignore_path = self.project_root / '.gitignore'
        if not gitignore_path.exists():
            with open(gitignore_path, 'w', encoding='utf-8') as f:
                f.write(gitignore_content)
            print("✓ Created: .gitignore")
        else:
            print("⚠️  .gitignore already exists")
    
    def verify_setup(self):
        """Verify the setup is correct"""
        print("\n🔍 Verifying setup...")
        
        # Check if key files exist in correct locations
        key_files = [
            'data/intents_enhanced_wikipedia.json',
            'templates/index.html', 
            'static/css/styles.css',
            'static/js/script.js',
            'app.py',
            'train_enhanced.py',
            'requirements.txt'
        ]
        
        all_good = True
        for file_path in key_files:
            if (self.project_root / file_path).exists():
                print(f"✓ {file_path}")
            else:
                print(f"❌ {file_path} (missing)")
                all_good = False
        
        return all_good
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🚀 Enhanced ChatBot Pro - Setup Script")
        print("=" * 50)
        print(f"Setting up project in: {self.project_root}")
        
        try:
            # Create directories
            self.create_directories()
            
            # Organize files
            self.organize_files()
            
            # Create configuration files
            self.create_env_file()
            self.create_gitignore()
            
            # Verify setup
            setup_ok = self.verify_setup()
            
            print("\n" + "=" * 50)
            if setup_ok:
                print("🎉 Setup completed successfully!")
                print("\n📋 Next steps:")
                print("1. Install dependencies: pip install -r requirements.txt")
                print("2. Train the model: python train_enhanced.py")
                print("3. Start the server: python app.py")
                print("4. Open browser: http://localhost:5000")
            else:
                print("⚠️  Setup completed with some issues.")
                print("Please check the missing files above.")
                
        except Exception as e:
            print(f"\n❌ Setup failed with error: {e}")
            sys.exit(1)

def main():
    """Main entry point"""
    setup = ChatBotSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()
```

---

## 📄 data/intents_enhanced_wikipedia.json

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi", "Hello", "Hey", "Good morning", "Good afternoon", 
        "Good evening", "What's up", "How are you", "Howdy", 
        "Greetings", "Sup", "Yo", "Hi there", "Hello there", 
        "Hey there", "Good day", "Nice to meet you", "Salutations"
      ],
      "responses": [
        "Hello! I'm your AI assistant with Wikipedia integration. I can help you with questions, search for information, or just have a chat. What would you like to know?",
        "Hi there! I'm here to help you find information, answer questions, or search Wikipedia for detailed knowledge. How can I assist you today?",
        "Hey! Welcome! I can help you with various topics and search Wikipedia for comprehensive information. What are you curious about?",
        "Greetings! I'm your intelligent assistant ready to help with questions and Wikipedia searches. What can I do for you?"
      ],
      "context": ""
    },
    {
      "tag": "wikipedia_search",
      "patterns": [
        "search for", "tell me about", "what is", "who is", "when was", "when were", "when did",
        "where is", "where was", "how does", "how do", "how did", "explain", "define", 
        "information about", "facts about", "wikipedia", "look up", "find information",
        "research", "learn about", "details about", "history of", "biography of",
        "what are", "what were", "how to", "why does", "why did", "why is", "why are",
        "can you tell me", "I want to know", "I need to know", "help me understand",
        "give me information", "show me information", "find me information"
      ],
      "responses": [
        "I'll search Wikipedia for that information. Let me find what you're looking for...",
        "Searching Wikipedia for comprehensive information about your query...",
        "Let me look that up on Wikipedia for you...",
        "I'll find detailed information about that topic from Wikipedia..."
      ],
      "context": "wikipedia_search"
    },
    {
      "tag": "science_tech",
      "patterns": [
        "artificial intelligence", "machine learning", "quantum physics", "quantum computing",
        "space exploration", "climate change", "renewable energy", "solar power", "wind energy",
        "biotechnology", "nanotechnology", "robotics", "cryptocurrency", "bitcoin",
        "blockchain", "internet of things", "5G", "virtual reality", "augmented reality",
        "DNA", "genetics", "evolution", "black hole", "neutron star", "galaxy",
        "telescope", "satellite", "rocket", "spacecraft", "mars", "moon landing",
        "nuclear power", "fusion", "chemistry", "physics", "biology", "mathematics"
      ],
      "responses": [
        "That's a fascinating topic in science and technology! Let me search for comprehensive information...",
        "Science and technology topics are exciting! I'll find detailed information for you...",
        "Great question about modern science and tech! Searching for in-depth details..."
      ],
      "context": "wikipedia_search"
    },
    {
      "tag": "history_culture",
      "patterns": [
        "ancient", "medieval", "renaissance", "world war", "civilization", "empire",
        "revolution", "historical", "culture", "tradition", "mythology", "legend",
        "historical figure", "historical event", "dynasty", "kingdom", "republic",
        "democracy", "ancient egypt", "ancient greece", "roman empire", "vikings",
        "crusades", "napoleon", "hitler", "churchill", "gandhi", "lincoln",
        "shakespeare", "leonardo da vinci", "mozart", "beethoven", "picasso"
      ],
      "responses": [
        "History and culture are rich topics! Let me find comprehensive information...",
        "That's an interesting historical or cultural topic! Searching for details...",
        "Great question about history and culture! I'll look up detailed information..."
      ],
      "context": "wikipedia_search"
    },
    {
      "tag": "geography_travel",
      "patterns": [
        "country", "city", "capital", "continent", "ocean", "mountain", "river",
        "desert", "forest", "island", "population", "tourism", "landmarks",
        "monuments", "national park", "geography", "climate", "weather patterns",
        "Paris", "London", "New York", "Tokyo", "Sydney", "Rome", "Cairo",
        "Mount Everest", "Amazon", "Sahara", "Nile", "Pacific Ocean", "Atlantic Ocean"
      ],
      "responses": [
        "Geography and travel topics are wonderful! Let me search for comprehensive information...",
        "That sounds like an interesting place or geographical feature! Searching for details...",
        "Great question about geography! I'll find detailed information for you..."
      ],
      "context": "wikipedia_search"
    },
    {
      "tag": "animals_nature",
      "patterns": [
        "animals", "wildlife", "endangered species", "mammals", "birds", "reptiles",
        "fish", "insects", "ecosystem", "biodiversity", "conservation", "natural habitat",
        "lion", "tiger", "elephant", "whale", "shark", "eagle", "penguin",
        "dinosaur", "extinct animals", "evolution", "natural selection", "adaptation"
      ],
      "responses": [
        "Animals and nature are fascinating topics! Let me find detailed information...",
        "That's a wonderful question about the natural world! Searching for comprehensive details...",
        "Great interest in wildlife and nature! I'll look up detailed information for you..."
      ],
      "context": "wikipedia_search"
    },
    {
      "tag": "bot_info",
      "patterns": [
        "What are you?", "Are you a robot?", "Are you human?", "What can you do?",
        "Tell me about yourself", "Your capabilities", "How do you work?",
        "What are your features?", "What's your purpose?", "Are you AI?",
        "Can you search wikipedia?", "What information can you find?"
      ],
      "responses": [
        "I'm an enhanced AI chatbot with Wikipedia integration! I can answer questions, search for information on any topic, and help you learn about virtually anything.",
        "I'm your AI assistant with access to Wikipedia's vast knowledge base. I can help you find comprehensive information on almost any topic you're curious about!",
        "I'm an intelligent chatbot that combines conversational AI with Wikipedia search capabilities. I'm here to help you learn and find information!",
        "I'm your enhanced AI companion! I can chat, answer questions, and search Wikipedia for detailed information on any topic you're interested in."
      ],
      "context": ""
    },
    {
      "tag": "help",
      "patterns": [
        "Help", "I need help", "Can you help me?", "Support", "Assistance",
        "How do you work?", "Commands", "What can I ask?", "How to use",
        "Instructions", "Guide", "Tutorial"
      ],
      "responses": [
        "I'm here to help! You can ask me questions about any topic, and I'll search Wikipedia for comprehensive answers. Try asking 'Tell me about [topic]' or just ask any question!",
        "I can assist you with information on virtually any topic! Ask me about science, history, geography, people, events, or anything else you're curious about.",
        "I'm your information assistant! I can search Wikipedia and provide detailed answers. What would you like to learn about today?",
        "You can ask me anything! I specialize in finding information from Wikipedia. Try questions like 'What is...?', 'Who was...?', 'Tell me about...', or 'Explain...'"
      ],
      "context": "help"
    },
    {
      "tag": "learning_education",
      "patterns": [
        "homework help", "study", "learn", "education", "school", "university",
        "college", "research paper", "assignment", "exam", "test", "quiz",
        "subject", "course", "curriculum", "academic", "student", "teacher",
        "mathematics", "science", "history", "literature", "philosophy"
      ],
      "responses": [
        "I'd be happy to help with your learning! I can search for educational information on any topic. What subject are you studying?",
        "Learning is wonderful! I can help you find comprehensive information for your studies. What topic do you need help with?",
        "Education support is one of my specialties! I can search for detailed information to help with your studies."
      ],
      "context": "wikipedia_search"
    },
    {
      "tag": "famous_people",
      "patterns": [
        "biography", "life story", "famous person", "celebrity", "historical figure",
        "scientist", "inventor", "artist", "musician", "writer", "politician",
        "athlete", "actor", "director", "philosopher", "leader", "pioneer"
      ],
      "responses": [
        "I can find biographical information about famous people! Who would you like to learn about?",
        "Biographies are fascinating! I can search for detailed information about any notable person. Who interests you?",
        "Great question about notable people! I can find comprehensive biographical information. Who should I look up?"
      ],
      "context": "wikipedia_search"
    },
    {
      "tag": "goodbye",
      "patterns": [
        "Bye", "Goodbye", "See you later", "See you soon", "Talk to you later",
        "Catch you later", "Farewell", "Take care", "Until next time",
        "I'm leaving", "Gotta go", "I have to go", "Time to leave"
      ],
      "responses": [
        "Goodbye! It was great helping you learn new things today. Come back anytime for more information!",
        "See you later! Don't hesitate to return if you need help with any topic!",
        "Farewell! I hope I was able to help you discover something interesting today!",
        "Take care! Remember, I'm here whenever you need information or have questions!"
      ],
      "context": ""
    },
    {
      "tag": "thanks",
      "patterns": [
        "Thanks", "Thank you", "Thank you so much", "I appreciate it",
        "Thanks a lot", "Much appreciated", "That's helpful", "Perfect",
        "Awesome", "Great", "Excellent", "You're the best", "Amazing",
        "That was useful", "Very informative", "Learned something new"
      ],
      "responses": [
        "You're very welcome! I'm glad I could help you find the information you needed!",
        "Happy to help! Feel free to ask me about any other topics you're curious about!",
        "You're welcome! I enjoy helping people learn new things. What else would you like to know?",
        "My pleasure! I'm here whenever you need information or have questions!"
      ],
      "context": ""
    },
    {
      "tag": "current_events",
      "patterns": [
        "news", "current events", "recent", "today", "latest", "happening now",
        "breaking news", "updates", "what's new", "recent developments"
      ],
      "responses": [
        "For current events and breaking news, I recommend checking reliable news sources. However, I can search Wikipedia for background information on recent events and topics.",
        "I don't have access to real-time news, but I can provide background information from Wikipedia about current topics and events.",
        "While I can't provide live news updates, I can search Wikipedia for comprehensive background information on current events and topics."
      ],
      "context": "wikipedia_search"
    },
    {
      "tag": "joke",
      "patterns": [
        "Tell me a joke", "Joke", "Make me laugh", "Something funny",
        "Humor", "Be funny", "Entertain me", "Fun", "Funny"
      ],
      "responses": [
        "Why don't scientists trust atoms? Because they make up everything! Want to learn more about atoms from Wikipedia?",
        "What do you call a chatbot that loves Wikipedia? A know-it-all! Speaking of which, what would you like to learn about?",
        "Why did the AI go to Wikipedia? To expand its knowledge base! What topic can I help you explore today?",
        "What's a chatbot's favorite type of music? Algo-rhythms! Now, what interesting topic shall we explore together?"
      ],
      "context": ""
    },
    {
      "tag": "fallback",
      "patterns": [],
      "responses": [
        "I'm not sure I understand exactly, but I can try searching Wikipedia for information. Could you rephrase your question or tell me what topic you'd like to learn about?",
        "I didn't quite catch that, but I'm great at finding information! What topic would you like me to search for?",
        "Let me help you find information instead! What subject or topic are you interested in learning about?",
        "I can search Wikipedia for almost any topic! What would you like to know more about?"
      ],
      "context": "wikipedia_search"
    }
  ]
}
```

---

## 📄 templates/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced ChatBot Pro - Wikipedia Integration</title>
    <link rel="stylesheet" href="static/css/styles.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="header-content">
                <div class="bot-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="header-text">
                    <h1>Enhanced ChatBot Pro</h1>
                    <p>AI Assistant with Wikipedia Integration</p>
                </div>
                <div class="header-actions">
                    <button class="info-btn" id="infoBtn" title="How to use">
                        <i class="fas fa-info-circle"></i>
                    </button>
                </div>
            </div>
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="welcome-message">
                <div class="welcome-card">
                    <i class="fas fa-sparkles"></i>
                    <h3>Welcome to Enhanced ChatBot Pro!</h3>
                    <p>I'm your AI assistant with Wikipedia integration. Ask me anything and I'll help you learn!</p>
                    <div class="example-queries">
                        <span class="query-example" data-query="Tell me about artificial intelligence">AI</span>
                        <span class="query-example" data-query="What is quantum physics?">Quantum Physics</span>
                        <span class="query-example" data-query="Who was Leonardo da Vinci?">Leonardo da Vinci</span>
                        <span class="query-example" data-query="Explain black holes">Black Holes</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-animation">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span class="typing-text">Bot is typing...</span>
            </div>
        </div>

        <div class="chat-input-container">
            <div class="input-wrapper">
                <input type="text" 
                       id="messageInput" 
                       class="chat-input" 
                       placeholder="Ask me anything... (e.g., Tell me about space exploration)" 
                       maxlength="500"
                       autocomplete="off">
                <button id="sendButton" class="send-button" title="Send message">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
            <div class="input-footer">
                <div class="character-count">
                    <span id="charCount">0</span>/500
                </div>
                <div class="quick-actions">
                    <button class="quick-btn" data-action="clear" title="Clear chat">
                        <i class="fas fa-trash"></i>
                    </button>
                    <button class="quick-btn" data-action="export" title="Export chat">
                        <i class="fas fa-download"></i>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Info Modal -->
    <div class="modal" id="infoModal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>How to Use Enhanced ChatBot Pro</h2>
                <button class="close-btn" id="closeModal">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <div class="info-section">
                    <h3><i class="fas fa-question-circle"></i> What can I ask?</h3>
                    <ul>
                        <li>Ask about any topic: "Tell me about [topic]"</li>
                        <li>Ask questions: "What is...?", "Who was...?", "How does...?"</li>
                        <li>Request explanations: "Explain quantum physics"</li>
                        <li>Get biographical info: "Who was Einstein?"</li>
                    </ul>
                </div>
                <div class="info-section">
                    <h3><i class="fas fa-lightbulb"></i> Example Queries</h3>
                    <div class="example-grid">
                        <span class="example-item">What is artificial intelligence?</span>
                        <span class="example-item">Tell me about Mars</span>
                        <span class="example-item">Who was Shakespeare?</span>
                        <span class="example-item">Explain climate change</span>
                        <span class="example-item">What are black holes?</span>
                        <span class="example-item">History of the Internet</span>
                    </div>
                </div>
                <div class="info-section">
                    <h3><i class="fas fa-magic"></i> Features</h3>
                    <ul>
                        <li>Wikipedia integration for comprehensive answers</li>
                        <li>Natural language understanding</li>
                        <li>Conversation memory within session</li>
                        <li>Export chat history</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <script src="static/js/script.js"></script>
</body>
</html>
```

---

## 📄 static/css/styles.css

```css
/* Enhanced ChatBot Pro - Styles */

/* Reset and Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
    line-height: 1.6;
}

/* Main Chat Container */
.chat-container {
    width: 100%;
    max-width: 900px;
    height: 90vh;
    min-height: 600px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
}

/* Header Styles */
.chat-header {
    background: linear-gradient(135deg, #4CAF50, #45a049);
    color: white;
    padding: 20px 25px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.header-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.bot-avatar {
    width: 50px;
    height: 50px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 15px;
    font-size: 24px;
}

.header-text h1 {
    font-size: 24px;
    margin-bottom: 5px;
    font-weight: 600;
}

.header-text p {
    font-size: 14px;
    opacity: 0.9;
}

.header-actions {
    margin-left: auto;
}

.info-btn {
    background: rgba(255, 255, 255, 0.2);
    border: none;
    color: white;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    cursor: pointer;
    font-size: 18px;
    transition: all 0.3s ease;
}

.info-btn:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: scale(1.05);
}

/* Messages Container */
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    background: linear-gradient(to bottom, #f8f9fa, #ffffff);
    scroll-behavior: smooth;
}

.chat-messages::-webkit-scrollbar {
    width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
}

/* Welcome Message */
.welcome-message {
    text-align: center;
    margin-bottom: 20px;
}

.welcome-card {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-radius: 15px;
    padding: 30px;
    color: #1565c0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.welcome-card i {
    font-size: 48px;
    margin-bottom: 15px;
    color: #4CAF50;
}

.welcome-card h3 {
    font-size: 24px;
    margin-bottom: 10px;
    color: #1565c0;
}

.welcome-card p {
    font-size: 16px;
    margin-bottom: 20px;
    color: #1976d2;
}

.example-queries {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
}

.query-example {
    background: #4CAF50;
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 2px solid transparent;
}

.query-example:hover {
    background: #45a049;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
}

/* Individual Messages */
.message {
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    animation: fadeInUp 0.3s ease;
}

.message.user {
    justify-content: flex-end;
}

.message-content {
    max-width: 75%;
    padding: 15px 20px;
    border-radius: 20px;
    font-size: 15px;
    line-height: 1.5;
    position: relative;
    word-wrap: break-word;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.message.bot .message-content {
    background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
    color: #1565c0;
    border-bottom-left-radius: 8px;
    border: 1px solid #bbdefb;
}

.message.user .message-content {
    background: linear-gradient(135deg, #4CAF50, #66bb6a);
    color: white;
    border-bottom-right-radius: 8px;
}

.message-content strong {
    font-weight: 600;
}

.message-content a {
    color: inherit;
    text-decoration: none;
    border-bottom: 1px dotted currentColor;
    transition: all 0.3s ease;
}

.message-content a:hover {
    border-bottom-style: solid;
    opacity: 0.8;
}

.message-time {
    font-size: 12px;
    opacity: 0.7;
    margin-top: 8px;
    text-align: right;
}

.message.bot .message-time {
    text-align: left;
}

/* Typing Indicator */
.typing-indicator {
    display: none;
    padding: 15px 20px;
    margin: 0 20px 20px;
}

.typing-animation {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #f5f5f5;
    padding: 15px 20px;
    border-radius: 20px;
    border-bottom-left-radius: 8px;
    max-width: 150px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.typing-dots {
    display: flex;
    gap: 4px;
}

.typing-dots span {
    width: 8px;
    height: 8px;
    background-color: #4CAF50;
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }
.typing-dots span:nth-child(3) { animation-delay: 0s; }

.typing-text {
    font-size: 12px;
    color: #666;
    font-style: italic;
}

/* Input Container */
.chat-input-container {
    padding: 20px 25px;
    background: white;
    border-top: 1px solid #e0e0e0;
}

.input-wrapper {
    display: flex;
    gap: 10px;
    margin-bottom: 10px;
}

.chat-input {
    flex: 1;
    padding: 15px 20px;
    border: 2px solid #e0e0e0;
    border-radius: 25px;
    font-size: 15px;
    outline: none;
    transition: all 0.3s ease;
    background: #f8f9fa;
}

.chat-input:focus {
    border-color: #4CAF50;
    background: white;
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
}

.send-button {
    width: 50px;
    height: 50px;
    background: linear-gradient(135deg, #4CAF50, #45a049);
    color: white;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    font-size: 18px;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
}

.send-button:hover:not(:disabled) {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
}

.send-button:disabled {
    background: #ccc;
    cursor: not-allowed;
    box-shadow: none;
    transform: none;
}

.input-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: #666;
}

.character-count {
    opacity: 0.7;
}

.quick-actions {
    display: flex;
    gap: 8px;
}

.quick-btn {
    background: none;
    border: none;
    color: #666;
    cursor: pointer;
    padding: 5px 8px;
    border-radius: 5px;
    transition: all 0.3s ease;
    font-size: 14px;
}

.quick-btn:hover {
    background: #f0f0f0;
    color: #4CAF50;
}

/* Modal Styles */
.modal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(5px);
}

.modal-content {
    background-color: white;
    margin: 5% auto;
    border-radius: 15px;
    width: 90%;
    max-width: 600px;
    max-height: 80vh;
    overflow-y: auto;
    animation: slideIn 0.3s ease;
}

.modal-header {
    padding: 25px;
    background: linear-gradient(135deg, #4CAF50, #45a049);
    color: white;
    border-radius: 15px 15px 0 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.modal-header h2 {
    margin: 0;
    font-size: 24px;
}

.close-btn {
    background: none;
    border: none;
    color: white;
    font-size: 24px;
    cursor: pointer;
    padding: 5px;
    border-radius: 50%;
    transition: all 0.3s ease;
}

.close-btn:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: scale(1.1);
}

.modal-body {
    padding: 25px;
}

.info-section {
    margin-bottom: 25px;
}

.info-section h3 {
    color: #4CAF50;
    margin-bottom: 15px;
    font-size: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.info-section ul {
    list-style: none;
    padding: 0;
}

.info-section li {
    padding: 8px 0;
    padding-left: 20px;
    position: relative;
}

.info-section li:before {
    content: "•";
    color: #4CAF50;
    font-weight: bold;
    position: absolute;
    left: 0;
}

.example-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
}

.example-item {
    background: #f8f9fa;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 14px;
    color: #333;
    border-left: 3px solid #4CAF50;
}

/* Animations */
@keyframes typing {
    0%, 80%, 100% {
        transform: scale(0.8);
        opacity: 0.5;
    }
    40% {
        transform: scale(1);
        opacity: 1;
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-50px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Responsive Design */
@media (max-width: 768px) {
    body {
        padding: 10px;
    }
    
    .chat-container {
        height: 95vh;
        border-radius: 15px;
    }
    
    .chat-header {
        padding: 15px 20px;
    }
    
    .header-text h1 {
        font-size: 20px;
    }
    
    .bot-avatar {
        width: 40px;
        height: 40px;
        font-size: 20px;
    }
    
    .chat-messages {
        padding: 15px;
    }
    
    .message-content {
        max-width: 85%;
        padding: 12px 16px;
        font-size: 14px;
    }
    
    .welcome-card {
        padding: 20px;
    }
    
    .welcome-card h3 {
        font-size: 20px;
    }
    
    .example-queries {
        flex-direction: column;
        align-items: center;
    }
    
    .query-example {
        padding: 10px 20px;
    }
    
    .chat-input-container {
        padding: 15px 20px;
    }
    
    .chat-input {
        font-size: 16px; /* Prevents zoom on iOS */
    }
    
    .modal-content {
        width: 95%;
        margin: 10% auto;
    }
    
    .modal-header {
        padding: 20px;
    }
    
    .modal-body {
        padding: 20px;
    }
}

@media (max-width: 480px) {
    .header-content {
        flex-direction: column;
        text-align: center;
        gap: 10px;
    }
    
    .header-actions {
        margin-left: 0;
    }
    
    .message-content {
        max-width: 90%;
    }
    
    .input-wrapper {
        flex-direction: column;
        gap: 15px;
    }
    
    .send-button {
        align-self: center;
        width: 60px;
        height: 60px;
        font-size: 20px;
    }
    
    .example-grid {
        grid-template-columns: 1fr;
    }
}

/* Print Styles */
@media print {
    body {
        background: white;
        padding: 0;
    }
    
    .chat-container {
        box-shadow: none;
        border: 1px solid #ddd;
    }
    
    .chat-input-container,
    .typing-indicator,
    .header-actions {
        display: none;
    }
}

/* High Contrast Mode */
@media (prefers-contrast: high) {
    .message.bot .message-content {
        border: 2px solid #333;
    }
    
    .message.user .message-content {
        border: 2px solid #fff;
    }
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
```

---

## 📄 static/js/script.js

```javascript
// Enhanced ChatBot Pro - JavaScript
class ChatBot {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.messageHistory = [];
        this.isTyping = false;
        
        this.initializeElements();
        this.bindEvents();
        this.setupWelcomeMessage();
    }
    
    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeElements() {
        // Main elements
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.charCount = document.getElementById('charCount');
        
        // Modal elements
        this.infoBtn = document.getElementById('infoBtn');
        this.infoModal = document.getElementById('infoModal');
        this.closeModal = document.getElementById('closeModal');
        
        // Quick action buttons
        this.quickButtons = document.querySelectorAll('.quick-btn');
        
        // Example queries
        this.queryExamples = document.querySelectorAll('.query-example');
    }
    
    bindEvents() {
        // Send message events
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Character counter
        this.messageInput.addEventListener('input', () => this.updateCharCount());
        
        // Modal events
        this.infoBtn.addEventListener('click', () => this.showModal());
        this.closeModal.addEventListener('click', () => this.hideModal());
        this.infoModal.addEventListener('click', (e) => {
            if (e.target === this.infoModal) this.hideModal();
        });
        
        // Quick actions
        this.quickButtons.forEach(btn => {
            btn.addEventListener('click', () => this.handleQuickAction(btn.dataset.action));
        });
        
        // Example queries
        this.queryExamples.forEach(example => {
            example.addEventListener('click', () => {
                this.messageInput.value = example.dataset.query;
                this.sendMessage();
            });
        });
        
        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.hideModal();
        });
        
        // Auto-focus input
        this.messageInput.focus();
    }
    
    setupWelcomeMessage() {
        // Add initial welcome message to history
        this.messageHistory.push({
            type: 'bot',
            message: 'Welcome! I\'m your AI assistant with Wikipedia integration.',
            timestamp: new Date()
        });
    }
    
    updateCharCount() {
        const length = this.messageInput.value.length;
        this.charCount.textContent = length;
        
        // Visual feedback for character limit
        if (length > 450) {
            this.charCount.style.color = '#f44336';
        } else if (length > 400) {
            this.charCount.style.color = '#ff9800';
        } else {
            this.charCount.style.color = '#666';
        }
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;
        
        // Add user message to UI
        this.addMessage(message, true);
        
        // Clear input and disable send button
        this.messageInput.value = '';
        this.updateCharCount();
        this.sendButton.disabled = true;
        
        // Show typing indicator
        this.showTyping();
        
        try {
            const response = await this.callAPI(message);
            
            // Simulate typing delay for better UX
            await this.delay(1000 + Math.random() * 1000);
            
            this.hideTyping();
            
            if (response.response) {
                this.addMessage(response.response);
            } else {
                this.addMessage('Sorry, I encountered an error. Please try again.');
            }
            
        } catch (error) {
            console.error('Error:', error);
            this.hideTyping();
            this.addMessage('Sorry, I couldn\'t connect to the server. Please check your connection and try again.');
        } finally {
            this.sendButton.disabled = false;
            this.messageInput.focus();
        }
    }
    
    async callAPI(message) {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: this.sessionId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
        
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        // Process content for better display
        const processedContent = this.processMessageContent(content, isUser);
        
        messageDiv.innerHTML = `
            <div class="message-content">
                ${processedContent}
                <div class="message-time">${timeStr}</div>
            </div>
        `;
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Add to history
        this.messageHistory.push({
            type: isUser ? 'user' : 'bot',
            message: content,
            timestamp: now
        });
        
        // Animate message appearance
        requestAnimationFrame(() => {
            messageDiv.style.opacity = '0';
            messageDiv.style.transform = 'translateY(20px)';
            messageDiv.offsetHeight; // Force reflow
            messageDiv.style.transition = 'all 0.3s ease';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        });
    }
    
    processMessageContent(content, isUser) {
        if (isUser) {
            // Simple HTML escaping for user messages
            return this.escapeHtml(content);
        }
        
        // Process bot messages for better formatting
        let processed = content;
        
        // Convert **text** to <strong>text</strong>
        processed = processed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Convert markdown-style links [text](url) to HTML links
        processed = processed.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
        
        // Convert newlines to <br>
        processed = processed.replace(/\n/g, '<br>');
        
        // Add emoji support for certain keywords
        processed = processed.replace(/🔗/g, '<i class="fas fa-external-link-alt"></i>');
        processed = processed.replace(/💡/g, '<i class="fas fa-lightbulb"></i>');
        
        return processed;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    showTyping() {
        this.isTyping = true;
        this.typingIndicator.style.display = 'block';
        this.scrollToBottom();
    }
    
    hideTyping() {
        this.isTyping = false;
        this.typingIndicator.style.display = 'none';
    }
    
    scrollToBottom() {
        requestAnimationFrame(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        });
    }
    
    showModal() {
        this.infoModal.style.display = 'block';
        document.body.style.overflow = 'hidden';
        
        // Focus trap for accessibility
        const focusableElements = this.infoModal.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (focusableElements.length > 0) {
            focusableElements[0].focus();
        }
    }
    
    hideModal() {
        this.infoModal.style.display = 'none';
        document.body.style.overflow = 'auto';
        this.messageInput.focus();
    }
    
    handleQuickAction(action) {
        switch (action) {
            case 'clear':
                this.clearChat();
                break;
            case 'export':
                this.exportChat();
                break;
            default:
                console.warn('Unknown quick action:', action);
        }
    }
    
    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            // Remove all messages except welcome
            const messages = this.chatMessages.querySelectorAll('.message');
            messages.forEach(msg => msg.remove());
            
            // Reset history
            this.messageHistory = [];
            
            // Show welcome message again
            this.setupWelcomeMessage();
            
            // Create new session
            this.sessionId = this.generateSessionId();
        }
    }
    
    exportChat() {
        if (this.messageHistory.length === 0) {
            alert('No messages to export.');
            return;
        }
        
        const exportData = {
            sessionId: this.sessionId,
            exportDate: new Date().toISOString(),
            messages: this.messageHistory.map(msg => ({
                type: msg.type,
                message: msg.message,
                timestamp: msg.timestamp.toISOString()
            }))
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `chatbot-conversation-${new Date().toISOString().slice(0, 10)}.json`;
        link.click();
        
        // Clean up
        URL.revokeObjectURL(link.href);
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    // Utility method for text-to-speech (optional feature)
    speak(text) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.8;
            utterance.pitch = 1;
            speechSynthesis.speak(utterance);
        }
    }
    
    // Handle connection status
    checkConnection() {
        if (navigator.onLine) {
            document.body.classList.remove('offline');
        } else {
            document.body.classList.add('offline');
            this.addMessage('⚠️ You appear to be offline. Please check your internet connection.');
        }
    }
}

// Notification system
class NotificationSystem {
    static show(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Add to document
        document.body.appendChild(notification);
        
        // Animate in
        requestAnimationFrame(() => {
            notification.style.transform = 'translateY(0)';
            notification.style.opacity = '1';
        });
        
        // Auto remove
        setTimeout(() => {
            notification.style.transform = 'translateY(-100px)';
            notification.style.opacity = '0';
            setTimeout(() => document.body.removeChild(notification), 300);
        }, duration);
    }
}

// Error handler for uncaught errors
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
    NotificationSystem.show('An unexpected error occurred. Please refresh the page.', 'error');
});

// Connection status monitoring
window.addEventListener('online', () => {
    NotificationSystem.show('Connection restored!', 'success');
});

window.addEventListener('offline', () => {
    NotificationSystem.show('You are now offline.', 'warning');
});

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Add loading animation
    document.body.classList.add('loading');
    
    // Initialize chatbot with small delay for smooth loading
    setTimeout(() => {
        window.chatBot = new ChatBot();
        document.body.classList.remove('loading');
        
        // Add welcome animation
        const welcomeCard = document.querySelector('.welcome-card');
        if (welcomeCard) {
            welcomeCard.style.animation = 'fadeInUp 0.6s ease 0.3s both';
        }
    }, 100);
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K to focus input
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('messageInput').focus();
    }
    
    // Ctrl/Cmd + L to clear chat
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        if (window.chatBot) {
            window.chatBot.clearChat();
        }
    }
});

// Export for external use
window.ChatBotAPI = {
    sendMessage: (message) => {
        if (window.chatBot) {
            window.chatBot.messageInput.value = message;
            window.chatBot.sendMessage();
        }
    },
    clearChat: () => {
        if (window.chatBot) {
            window.chatBot.clearChat();
        }
    },
    exportChat: () => {
        if (window.chatBot) {
            window.chatBot.exportChat();
        }
    }
};
```

---

## 📄 README.md

```markdown
# Enhanced ChatBot Pro with Wikipedia Integration

A modern, intelligent chatbot with Wikipedia search capabilities, built using TensorFlow/Keras for natural language processing and Flask for the web API. Features a beautiful, responsive web interface with real-time messaging.

## 🚀 Features

- **🧠 AI-Powered Conversations**: Advanced neural network for natural language understanding
- **📖 Wikipedia Integration**: Real-time Wikipedia search and information retrieval
- **💬 Modern Web Interface**: Responsive, mobile-friendly chat interface
- **⚡ Real-time Messaging**: Instant responses with typing indicators
- **🎨 Beautiful UI**: Gradient backgrounds, smooth animations, and modern design
- **📱 Mobile Responsive**: Works perfectly on all devices
- **💾 Export Conversations**: Save chat history as JSON files
- **🔍 Smart Search**: Intelligent query processing and topic extraction
- **🎯 Multiple Intent Categories**: Science, history, geography, education, and more

## 🛠️ Quick Setup

### Step 1: Create Project Structure

Create the main project directory and all subdirectories:

```bash
mkdir enhanced-chatbot-pro
cd enhanced-chatbot-pro

# Create subdirectories
mkdir -p data models templates static/css static/js tests docs
```

### Step 2: Create Files

Copy all the code from this document into their respective files in the correct directories.

### Step 3: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv chatbot_env
source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
python train_enhanced.py
```

### Step 5: Start the Server

```bash
python app.py
```

### Step 6: Open in Browser

Navigate to `http://localhost:5000`

## 💬 Example Usage

- **"Tell me about artificial intelligence"**
- **"What is quantum physics?"**
- **"Who was Leonardo da Vinci?"**
- **"Explain black holes"**
- **"History of the Internet"**

## 📁 Project Structure Summary

```
enhanced-chatbot-pro/
├── 📄 app.py                              # Flask server with Wikipedia API
├── 📄 train_enhanced.py                   # Enhanced training script
├── 📄 setup.py                           # Automated setup script
├── 📄 requirements.txt                    # Python dependencies
├── 📄 README.md                          # This documentation
├── 📁 data/
│   └── 📄 intents_enhanced_wikipedia.json # Enhanced intents
├── 📁 templates/
│   └── 📄 index.html                     # Main chat interface
├── 📁 static/
│   ├── 📁 css/
│   │   └── 📄 styles.css                 # Modern stylesheet
│   └── 📁 js/
│       └── 📄 script.js                  # Chat functionality
└── 📁 models/                            # Generated during training
    ├── 📄 chatbot_model_enhanced.h5      # Trained model
    ├── 📄 words_enhanced.pkl             # Vocabulary
    └── 📄 classes_enhanced.pkl           # Intent classes
```

## 🎯 Key Features

✅ **Wikipedia Integration** - Real-time search and comprehensive answers  
✅ **Modern Web UI** - Beautiful, responsive interface  
✅ **Enhanced AI** - 15 intent categories with 267+ patterns  
✅ **Real-time Chat** - Typing indicators and smooth animations  
✅ **Export Feature** - Save conversations as JSON  
✅ **Mobile Friendly** - Works on all devices  
✅ **Error Handling** - Graceful fallbacks  
✅ **Session Management** - Conversation history and context  

## 🔧 Troubleshooting

1. **NLTK Data Missing**: Run `python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"`
2. **Port in Use**: Change port in `app.py` or kill existing process
3. **Dependencies**: Make sure all packages in `requirements.txt` are installed
4. **Model Files**: Ensure training completed successfully and model files exist

## 📞 Support

Open an issue on GitHub or check the documentation for common problems and solutions.

---

**Made with ❤️ for learning and education**

Happy chatting! 🤖✨
```

---

## 🚀 Installation Instructions

1. **Create the project directory structure** as shown above
2. **Copy each file's content** into the appropriate location
3. **Run the setup commands**:
   ```bash
   pip install -r requirements.txt
   python train_enhanced.py
   python app.py
   ```
4. **Open http://localhost:5000** in your browser

The complete Enhanced ChatBot Pro with Wikipedia integration is now ready for download and use! 🎉
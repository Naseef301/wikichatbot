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

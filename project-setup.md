# Enhanced ChatBot Pro - Project Setup Instructions

## 📁 Folder Structure

Create this exact folder structure on your computer:

```
enhanced-chatbot-pro/
├── 📄 app.py                              
├── 📄 train_enhanced.py                   
├── 📄 requirements.txt                    
├── 📄 setup.py (optional)
├── 📄 README.md (optional)
│
├── 📁 data/
│   └── 📄 intents_enhanced_wikipedia.json
│
├── 📁 templates/
│   └── 📄 index.html
│
├── 📁 static/
│   ├── 📁 css/
│   │   └── 📄 styles.css
│   └── 📁 js/
│       └── 📄 script.js
│
└── 📁 models/ (will be created automatically during training)
    ├── 📄 chatbot_model_enhanced.h5 (created by training)
    ├── 📄 words_enhanced.pkl (created by training)
    └── 📄 classes_enhanced.pkl (created by training)
```

## 🛠️ Step-by-Step Setup

### Step 1: Create Main Folder
```bash
mkdir enhanced-chatbot-pro
cd enhanced-chatbot-pro
```

### Step 2: Create Sub-folders
```bash
mkdir data
mkdir templates
mkdir static
mkdir static/css
mkdir static/js
mkdir models
```

### Step 3: Download and Place Files

**Root Directory Files:**
- Save `app.py` in the main folder
- Save `train_enhanced.py` in the main folder  
- Save `requirements.txt` in the main folder

**Data Folder:**
- Rename `intents-wikipedia.json` to `intents_enhanced_wikipedia.json`
- Put it in the `data/` folder

**Templates Folder:**
- Put `index.html` in the `templates/` folder

**Static Folder:**
- Put `styles.css` in the `static/css/` folder
- Put `script.js` in the `static/js/` folder

### Step 4: File Placement Summary

```
enhanced-chatbot-pro/
├── app.py                              ← Download and save here
├── train_enhanced.py                   ← Download and save here  
├── requirements.txt                    ← Download and save here
│
├── data/
│   └── intents_enhanced_wikipedia.json ← Rename and save here
│
├── templates/
│   └── index.html                      ← Download and save here
│
└── static/
    ├── css/
    │   └── styles.css                  ← Download and save here
    └── js/
        └── script.js                   ← Download and save here
```

## 🚀 Quick Setup Commands

### Windows:
```cmd
mkdir enhanced-chatbot-pro
cd enhanced-chatbot-pro
mkdir data templates static static\css static\js models
```

### Mac/Linux:
```bash
mkdir enhanced-chatbot-pro
cd enhanced-chatbot-pro
mkdir -p data templates static/css static/js models
```

## ⚠️ Important Notes:

1. **File Names Must Match Exactly:**
   - `intents_enhanced_wikipedia.json` (not `intents-wikipedia.json`)
   - `app.py` (in root directory)
   - `styles.css` (in static/css/ folder)
   - `script.js` (in static/js/ folder)

2. **Folder Structure Must Match:**
   - HTML goes in `templates/`
   - CSS goes in `static/css/`
   - JS goes in `static/js/`
   - Data goes in `data/`

3. **Models Folder:**
   - Will be created automatically when you run training
   - Don't worry if it's empty initially

## 🎯 After Setup:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   ```bash
   python train_enhanced.py
   ```

3. **Start the Server:**
   ```bash
   python app.py
   ```

4. **Open Browser:**
   ```
   http://localhost:5000
   ```

## ✅ Verification Checklist:

- [ ] Created main `enhanced-chatbot-pro` folder
- [ ] Created all subfolders (data, templates, static/css, static/js, models)
- [ ] Placed `app.py` in root directory
- [ ] Placed `train_enhanced.py` in root directory
- [ ] Placed `requirements.txt` in root directory
- [ ] Placed `intents_enhanced_wikipedia.json` in data/ folder
- [ ] Placed `index.html` in templates/ folder
- [ ] Placed `styles.css` in static/css/ folder
- [ ] Placed `script.js` in static/js/ folder

That's it! Follow this structure exactly and your chatbot will work perfectly! 🤖✨
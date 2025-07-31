import re
import streamlit as st
from transformers import pipeline
import torch
import os
from PIL import Image

# Set page config with green theme
st.set_page_config(
    page_title="Nigerian Pidgin-English Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for green theme
st.markdown("""
    <style>
    :root {
        --primary: #2e7d32;
        --secondary: #4caf50;
        --light: #e8f5e9;
        --dark: #1b5e20;
    }
    
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #ffffff 100%);
    }
    
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
        border: 1px solid var(--primary);
        border-radius: 8px;
        padding: 10px;
    }
    
    .stButton>button {
        background-color: var(--primary) !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: var(--dark) !important;
        transform: scale(1.05);
    }
    
    .feedback-btn {
        margin: 5px;
        min-width: 100px;
    }
    
    .card {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid var(--primary);
    }
    
    .header {
        background: linear-gradient(90deg, #1b5e20 0%, #4caf50 100%);
        color: white;
        padding: 20px 0;
        border-radius: 0 0 20px 20px;
        margin-bottom: 30px;
    }
    
    .logo-title {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .method-tag {
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
    }
    
    .memory-tag { background-color: #ffeb3b; color: #333; }
    .rule-tag { background-color: #4caf50; color: white; }
    .model-tag { background-color: #2196f3; color: white; }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.9rem;
        margin-top: 40px;
        border-top: 1px solid #e0e0e0;
    }
    
    .translation-box {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
    }
    
    .history-item {
        padding: 10px;
        border-bottom: 1px solid #e0e0e0;
        transition: background-color 0.3s;
    }
    
    .history-item:hover {
        background-color: #f1f8e9;
    }
    </style>
    """, unsafe_allow_html=True)

# Define translation rules
translation_rules = {
    "pidgin_to_english": {
        r'\bhow far\b': 'how are you',
        r'\babeg\b': 'please',
        r'\bna\b': 'is',
        r'\byou fit\b': 'can you',
        r'\bdey go\b': 'going',
        r'\bi\b': "I'm",
        r'\bi dey\b': "I am",
        r'\bwetin\b': 'what',
        r'\bwetin dey happen\b': 'what is happening',
        r'\bwetin happen\b': 'what happened',
        r'\be don do\b': "that's enough",
        r'\byou sabi\b': 'do you know',
        r'\bsabi\b': 'know',
        r'\bthey sabi\b': 'they know',
        r'\bhe sabi\b': 'he know',
        r'\bshe sabi\b': 'she know',
        r'\bi sabi sabi\b': 'I know very well',
        r'\bi no sabi\b': "I don't know",
        r'\bi no know\b': "I don't know",
        r'\be be like say\b': 'it seems like',
        r'\bno wahala\b': 'no problem',
        r'\byou wan chop\b': 'do you want to eat',
        r'\bna dem\b': 'they are',
        r'\bmake we go\b': "let's go",
        r'\bna wetin\b': 'that is what',
        r'\buna\b': 'you all',
        r'\bchai\b': 'oh no',
        r'\bpopo\b': 'police',
    },
    "english_to_pidgin": {
        r'\bhow are you\b': 'how far',
        r'\bplease\b': 'abeg',
        r'\bis\b': 'na',
        r'\bcan you\b': 'you fit',
        r'\bgoing\b': 'dey go',
        r"\bI'm\b": 'i',
        r'\bI am\b': 'i dey',
        r'\bwhat is happening\b': 'wetin dey happen',
        r'\bwhat happened\b': 'wetin happen',
        r'\bwhat\b': 'wetin',
        r"\bthat's enough\b": 'e don do',
        r'\bdo you know\b': 'you sabi',
        r'\bknow\b': 'sabi',
        r'\bdont\b': 'no',
        r'\bthey know\b': 'they sabi',
        r'\bhe know\b': 'he sabi',
        r'\bshe know\b': 'she sabi',
        r'\bi know\b': 'i sabi',
        r"\bI don't know\b": 'i no sabi',
        r'\bit seems like\b': 'e be like say',
        r'\bno problem\b': 'no wahala',
        r'\bdo you want to eat\b': 'you wan chop',
        r'\bthey are\b': 'na dem',
        r"\blet's go\b": 'make we go',
        r'\bthat is what\b': 'na wetin',
        r'\byou all\b': 'una',
        r'\boh no\b': 'chai',
        r'\bpolice\b': 'popo',
    }
}

# Initialize session state
if 'memory' not in st.session_state:
    st.session_state.memory = {}
if 'history' not in st.session_state:
    st.session_state.history = []
if 'translation_result' not in st.session_state:
    st.session_state.translation_result = None
if 'translation_method' not in st.session_state:
    st.session_state.translation_method = ""
if 'feedback_requested' not in st.session_state:
    st.session_state.feedback_requested = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'current_input' not in st.session_state:
    st.session_state.current_input = ""
if 'current_direction' not in st.session_state:
    st.session_state.current_direction = ""
if 'attempted_methods' not in st.session_state:
    st.session_state.attempted_methods = set()
if 'approved_translations' not in st.session_state:
    st.session_state.approved_translations = set()

# Memory system
MEM_FILE = "mem.txt"

def load_memory():
    """Load approved translations from memory file"""
    if not os.path.exists(MEM_FILE):
        return {}
    
    translations = {}
    try:
        with open(MEM_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|||')
                if len(parts) == 3:
                    direction, src, tgt = parts
                    translations[(direction, src)] = tgt
                    st.session_state.approved_translations.add((direction, src, tgt))
    except:
        pass
    return translations

def save_to_memory(direction, src_text, tgt_text):
    """Save approved translation to memory file only if it's not already saved"""
    if (direction, src_text, tgt_text) in st.session_state.approved_translations:
        return
        
    try:
        with open(MEM_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{direction}|||{src_text}|||{tgt_text}\n")
        st.session_state.memory[(direction, src_text)] = tgt_text
        st.session_state.approved_translations.add((direction, src_text, tgt_text))
        st.toast("✅ Translation saved to memory!")
    except Exception as e:
        st.error(f"Error saving to memory: {str(e)}")

# Load models with caching
@st.cache_resource(show_spinner=False)
def load_models():
    """Load translation models with caching"""
    with st.spinner("Loading translation models... This may take a minute"):
        pidgin_to_english = pipeline(
            "translation",
            model="Xara2west/pidgin-to-english-translator-final09"
        )
        english_to_pidgin = pipeline(
            "translation",
            model="Xara2west/pidgin-translator-final06"
        )
    return pidgin_to_english, english_to_pidgin

# Load models on first run
if not st.session_state.models_loaded:
    pidgin_to_english_model, english_to_pidgin_model = load_models()
    st.session_state.models = (pidgin_to_english_model, english_to_pidgin_model)
    st.session_state.memory = load_memory()
    st.session_state.models_loaded = True

def apply_rules(text, rules):
    """Apply translation rules with regex patterns"""
    original_text = text
    for pattern, replacement in rules.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text, text != original_text

def translate_pidgin_to_english(text, force_method=None):
    """Translate Pidgin to English with memory first approach"""
    # Check memory first unless we're forcing a method
    if force_method != "memory":
        mem_key = ("pidgin_to_english", text)
        if mem_key in st.session_state.memory:
            return st.session_state.memory[mem_key], "memory"
    
    # Apply rules if we're not forcing model-based
    if force_method != "model-based":
        rule_translation, changed = apply_rules(text, translation_rules["pidgin_to_english"])
        if changed:
            return rule_translation, "rule-based"
    
    # Use model if we're not forcing rule-based
    if force_method != "rule-based":
        model_output = st.session_state.models[0](text)[0]['translation_text']
        return model_output, "model-based"
    
    # If we get here, return the original text
    return text, "none"

def translate_english_to_pidgin(text, force_method=None):
    """Translate English to Pidgin with memory first approach"""
    # Check memory first unless we're forcing a method
    if force_method != "memory":
        mem_key = ("english_to_pidgin", text)
        if mem_key in st.session_state.memory:
            return st.session_state.memory[mem_key], "memory"
    
    # Apply rules if we're not forcing model-based
    if force_method != "model-based":
        rule_translation, changed = apply_rules(text, translation_rules["english_to_pidgin"])
        if changed:
            return rule_translation, "rule-based"
    
    # Use model if we're not forcing rule-based
    if force_method != "rule-based":
        model_output = st.session_state.models[1](text)[0]['translation_text']
        return model_output, "model-based"
    
    # If we get here, return the original text
    return text, "none"

def handle_translation(direction, text):
    """Handle translation with feedback logic"""
    st.session_state.current_input = text
    st.session_state.current_direction = direction
    st.session_state.attempted_methods = set()  # Reset attempted methods
    
    if direction == "pidgin_to_english":
        translated, method = translate_pidgin_to_english(text)
    else:
        translated, method = translate_english_to_pidgin(text)
    
    st.session_state.attempted_methods.add(method)
    
    # Add to history
    st.session_state.history.insert(0, {
        "direction": "Pidgin to English" if direction == "pidgin_to_english" else "English to Pidgin",
        "input": text,
        "output": translated,
        "method": method
    })
    
    # Keep only last 10 items in history
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[:10]
    
    return translated, method

def handle_feedback(feedback):
    """Handle user feedback on translations"""
    if feedback == 'good':
        # Only save to memory when user explicitly marks as good
        save_to_memory(st.session_state.current_direction, 
                     st.session_state.current_input, 
                     st.session_state.translation_result)
        st.session_state.feedback_requested = False
        st.session_state.attempted_methods = set()
    else:
        # If user says bad, try the other available methods
        st.session_state.feedback_requested = True
        
        # Determine which methods haven't been tried yet
        available_methods = {"memory", "rule-based", "model-based"}
        remaining_methods = available_methods - st.session_state.attempted_methods
        
        if remaining_methods:
            # Try the next available method
            if st.session_state.current_direction == "pidgin_to_english":
                translated, method = translate_pidgin_to_english(
                    st.session_state.current_input,
                    force_method=next(iter(remaining_methods))
                )
            else:
                translated, method = translate_english_to_pidgin(
                    st.session_state.current_input,
                    force_method=next(iter(remaining_methods))
                )
            
            # Update the translation result and method
            st.session_state.translation_result = translated
            st.session_state.translation_method = method
            st.session_state.attempted_methods.add(method)
            
            # Update the history with the new attempt
            st.session_state.history[0]["output"] = translated
            st.session_state.history[0]["method"] = method
        else:
            st.warning("All translation methods have been attempted. Unable to provide a better translation.")

# UI Layout
st.markdown("""
    <div class="header">
        <div class="container">
            <div class="logo-title">
                <h1>🇳🇬 Nigerian Pidgin-English Translator</h1>
            </div>
            <p>Translate between Nigerian Pidgin and English with AI-powered accuracy</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Translation Panel")
    
    # Direction selection
    direction = st.radio("Translation Direction:", 
                         ["Pidgin to English", "English to Pidgin"],
                         horizontal=True,
                         index=0,
                         label_visibility="collapsed")
    
    # Text input
    input_text = st.text_area("Enter text to translate:", 
                             height=150,
                             placeholder="Type your text here...",
                             key="input_text")
    
    # Translate button
    translate_btn = st.button("Translate", type="primary", use_container_width=True)
    
    # Translation result
    if translate_btn and input_text:
        try:
            if direction == "Pidgin to English":
                direction_key = "pidgin_to_english"
            else:
                direction_key = "english_to_pidgin"
                
            translation, method = handle_translation(direction_key, input_text)
            
            st.session_state.translation_result = translation
            st.session_state.translation_method = method
            st.session_state.feedback_requested = True
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
    
    # Display translation result
    if st.session_state.translation_result:
        method = st.session_state.translation_method
        method_tag = {
            "memory": "memory-tag",
            "rule-based": "rule-tag",
            "model-based": "model-tag"
        }.get(method, "")
        
        method_label = {
            "memory": "Memory",
            "rule-based": "Rule-based",
            "model-based": "AI Model"
        }.get(method, "Unknown")
        
        st.markdown(f"<div class='translation-box'><h4>Translation Result <span class='method-tag {method_tag}'>{method_label}</span></h4><p style='font-size: 18px;'>{st.session_state.translation_result}</p></div>", 
                   unsafe_allow_html=True)
        
        # Feedback section
        if st.session_state.feedback_requested:
            st.subheader("Is this translation accurate?")
            col_fb1, col_fb2 = st.columns(2)
            with col_fb1:
                if st.button("👍 Good Translation", 
                            use_container_width=True, 
                            key="good_btn",
                            type="primary"):
                    handle_feedback('good')
            with col_fb2:
                if st.button("👎 Needs Improvement", 
                            use_container_width=True, 
                            key="bad_btn",
                            type="secondary"):
                    handle_feedback('bad')

with col2:
    # Memory section
    st.subheader("Memory Insights")
    st.caption("Approved translations saved for future use")
    
    if st.session_state.memory:
        mem_count = len(st.session_state.memory)
        st.metric("Stored Translations", mem_count)
        
        # Show some memory entries
        st.write("**Recent Memory Entries:**")
        for i, ((direction, src), tgt) in enumerate(list(st.session_state.memory.items())[:5]):
            st.markdown(f"""
                <div class="history-item">
                    <div><strong>{src}</strong> → {tgt}</div>
                    <small>{direction.replace('_', ' ').title()}</small>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No translations saved in memory yet. Approve good translations to build memory!")
    
    # Translation history
    st.subheader("Recent Translations")
    if st.session_state.history:
        for item in st.session_state.history:
            method_tag = {
                "memory": "memory-tag",
                "rule-based": "rule-tag",
                "model-based": "model-tag"
            }.get(item['method'], "")
            
            method_label = {
                "memory": "Memory",
                "rule-based": "Rule",
                "model-based": "AI"
            }.get(item['method'], "?")
            
            st.markdown(f"""
                <div class="history-item">
                    <div><strong>{item['input']}</strong> → {item['output']}</div>
                    <div>
                        <small>{item['direction']} • 
                        <span class='method-tag {method_tag}'>{method_label}</span></small>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Your translation history will appear here")

# How it works section
st.subheader("How It Works")
st.markdown("""
<div class="card">
    <div class="row">
        <div class="col">
            <h4>1. Memory First</h4>
            <p>The system first checks your approved translations in memory for an exact match.</p>
        </div>
        <div class="col">
            <h4>2. Rule-Based Translation</h4>
            <p>If no memory match, it applies predefined translation rules for common phrases.</p>
        </div>
        <div class="col">
            <h4>3. AI Model Translation</h4>
            <p>For unmatched phrases, it uses advanced AI models for context-aware translation.</p>
        </div>
        <div class="col">
            <h4>4. Continuous Improvement</h4>
            <p>You can approve translations to add to memory for future use.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Nigerian Pidgin-English Translator • Powered by Transformers AI • 
    <a href="https://huggingface.co/Xara2west" target="_blank">Model Source</a></p>
    <p>Approved translations are saved to memory for future use</p>
</div>
""", unsafe_allow_html=True)
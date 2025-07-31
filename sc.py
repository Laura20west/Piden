import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize models and tokenizers
@st.cache_resource
def load_models():
    # Pidgin to English model
    pe_tokenizer = T5Tokenizer.from_pretrained("Xara2west/pidgin-to-english-translator-final09")
    pe_model = T5ForConditionalGeneration.from_pretrained("Xara2west/pidgin-to-english-translator-final09")
    
    # English to Pidgin model
    ep_tokenizer = T5Tokenizer.from_pretrained("Xara2west/pidgin-translator-final06")
    ep_model = T5ForConditionalGeneration.from_pretrained("Xara2west/pidgin-translator-final06")
    
    return {
        "pidgin_to_english": (pe_tokenizer, pe_model),
        "english_to_pidgin": (ep_tokenizer, ep_model)
    }

models = load_models()

def translate(text, direction):
    if direction == "Pidgin to English":
        tokenizer, model = models["pidgin_to_english"]
        prefix = "translate Pidgin to English: "
    else:
        tokenizer, model = models["english_to_pidgin"]
        prefix = "translate English to Pidgin: "
    
    # Encode with task-specific prefix
    inputs = tokenizer(
        prefix + text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    # Generate translation with increased max_length
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=512,
        num_beams=5,
        early_stopping=True,
        repetition_penalty=2.5,
        length_penalty=1.0,
        no_repeat_ngram_size=3
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("🇳🇬 Pidgin-English Translator")
st.caption("Powered by Hugging Face Transformers")

# Initialize session states
if 'full_sentence' not in st.session_state:
    st.session_state.full_sentence = ""
if 'new_session' not in st.session_state:
    st.session_state.new_session = False

# Translation direction
direction = st.radio(
    "Select translation direction:",
    ("Pidgin to English", "English to Pidgin"),
    horizontal=True
)

# Input text - use a key that we can control
text_input_key = "text_input_" + str(st.session_state.new_session)
text = st.text_area("Enter text to translate:", height=150, key=text_input_key)

col1, col2 = st.columns(2)

with col1:
    if st.button("Translate"):
        if text.strip():
            with st.spinner("Translating..."):
                try:
                    result = translate(text, direction)
                    # Add to full sentence
                    if st.session_state.full_sentence:
                        st.session_state.full_sentence += " " + result
                    else:
                        st.session_state.full_sentence = result
                    
                    st.subheader("Current Translation:")
                    st.success(result)
                    
                    st.subheader("Full Sentence:")
                    st.info(st.session_state.full_sentence)
                except Exception as e:
                    st.error(f"Translation failed: {str(e)}")
        else:
            st.warning("Please enter text to translate")

with col2:
    if st.button("New Sentence"):
        # Clear the full sentence and trigger a new session
        st.session_state.full_sentence = ""
        st.session_state.new_session = not st.session_state.new_session
        st.rerun()

st.markdown("---")
st.info("**Model Details:**\n"
        "- Pidgin → English: [Xara2west/pidgin-to-english-translator-final09](https://huggingface.co/Xara2west/pidgin-to-english-translator-final09)\n"
        "- English → Pidgin: [Xara2west/pidgin-translator-final06](https://huggingface.co/Xara2west/pidgin-translator-final06)\n\n"
        "Both models are T5-based text generation models with 60.5M parameters")
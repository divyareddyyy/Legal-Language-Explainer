import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------------
# 1. Load Pretrained AI Model
# -----------------------------------

MODEL_NAME = "google/flan-t5-base"

print("Loading AI model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("AI Model Loaded Successfully!")

# -----------------------------------
# 2. Split Long Legal Text
# -----------------------------------

def split_text(text, max_tokens=400):
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(
            tokens[i:i + max_tokens],
            skip_special_tokens=True
        )
        chunks.append(chunk)

    return chunks

# -----------------------------------
# 3. AI Text Processing Function
# -----------------------------------

def generate_response(prompt, max_length=220):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=4,
            do_sample=False,
            early_stopping=True
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

# -----------------------------------
# 4. Simplify Legal Language
# -----------------------------------

def simplify_legal_text(legal_text):

    if not legal_text or not legal_text.strip():
        return "Please enter some legal text.", "", ""

    chunks = split_text(legal_text.strip())

    simplified_parts = []

    for chunk in chunks:
        prompt = (
            "Rewrite the following legal text in simple, "
            "clear English. Preserve all important legal "
            "meanings, conditions, dates, amounts and obligations. "
            "Do not invent information.\n\n"
            "Legal text: " + chunk
        )

        result = generate_response(prompt)
        simplified_parts.append(result)

    simplified_text = "\n\n".join(simplified_parts)

    # --------------------------------
    # 5. Generate Summary
    # --------------------------------

    summary_prompt = (
        "Summarize the following simplified legal text "
        "in simple English. Mention the main purpose, "
        "important conditions, rights and responsibilities. "
        "Do not add facts.\n\n"
        + simplified_text[:2500]
    )

    summary = generate_response(
        summary_prompt,
        max_length=180
    )

    # --------------------------------
    # 6. Display Results
    # --------------------------------

    return (
        simplified_text,
        summary,
        "Important: This is an AI-generated explanation, "
        "not professional legal advice. Always verify "
        "the original document and consult a qualified lawyer."
    )

# -----------------------------------
# 7. Create User Interface
# -----------------------------------

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # ⚖️ AI Legal-Language Plain-English Explainer

        ### Understand complicated legal documents easily!

        Enter legal language below and let AI explain it
        in simple English.
        """
    )

    legal_input = gr.Textbox(
        label="Enter Legal Text",
        placeholder="Paste your legal agreement, contract or legal clause here...",
        lines=8
    )

    explain_button = gr.Button(
        "Simplify Legal Language",
        variant="primary"
    )

    simplified_output = gr.Textbox(
        label="Simple English Explanation",
        lines=8
    )

    summary_output = gr.Textbox(
        label="Short Summary",
        lines=4
    )

    disclaimer_output = gr.Textbox(
        label="Legal Disclaimer",
        lines=2
    )

    explain_button.click(
        fn=simplify_legal_text,
        inputs=legal_input,
        outputs=[
            simplified_output,
            summary_output,
            disclaimer_output
        ]
    )

    gr.Markdown(
        """
        **Technology Used:** Python, NLP, Transformers,
        Google FLAN-T5 and Gradio.

        **Note:** AI explanations may be inaccurate.
        Always check the original legal document.
        """
    )

# -----------------------------------
# 8. Launch Application
# -----------------------------------

demo.launch(share=True)

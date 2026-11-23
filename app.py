import os
import re
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import PyPDF2
from io import BytesIO
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for Azure AI Inference client
client = None
phi4_deployment = None


def find_cred_json(start_path: str) -> str | None:
    """
    Recursively looks for 'cred.json' under start_path.
    Returns the first match or None if not found.
    """
    base = Path(start_path)
    print(f"🔎 Searching for cred.json under: {base.resolve()}")
    for candidate in base.rglob('cred.json'):
        print(f"✅ Found cred.json at: {candidate}")
        return str(candidate)
    return None


def initialize_azure_client():
    """Initialize the Azure AI Inference client"""
    global client, phi4_deployment

    try:
        # 1. Locate cred.json anywhere beneath the current directory
        cwd = os.getcwd()
        file_path = find_cred_json(cwd)
        if not file_path:
            raise FileNotFoundError("cred.json not found under the current directory")

        # 2. Load and parse the JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        # 3. Extract configuration
        endpoint = cfg.get("ENDPOINT", "")
        api_key = cfg.get("API_KEY", "")
        phi4_deployment = cfg.get("MODEL_DEPLOYMENT_NAME", "Phi-4")

        print(f"Endpoint:                   {endpoint}")
        print(f"Model Deployment Name:      {phi4_deployment}")

        # 4. Create the ChatCompletionsClient
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
            api_version="2024-05-01-preview"
        )
        print("✅ ChatCompletionsClient created successfully!")
        return True

    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON error in cred.json: {e}")
        return False
    except KeyError as e:
        print(f"❌ Missing key in config: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def chat_with_phi4_rag(user_question, retrieved_doc):
    """RAG-style call that works for both Phi-4 and Phi-4-reasoning."""
    global client, phi4_deployment

    if not client:
        return "Error: Azure AI Inference client not initialized"

    is_reasoning = "reasoning" in (phi4_deployment or "").lower()

    system_prompt = (
        "You are a helpful assistant that answers questions strictly from the "
        "document the user provides. Do not add general knowledge that is not in "
        "the document. If the document does not contain the answer, say so. "
        "Keep the final answer short and clear, with no disclaimers."
    )

    # Put the document in the USER message - Phi-4-reasoning pays much more
    # attention to it there than in the system message.
    user_prompt = (
        "Use ONLY the document below to answer the question.\n\n"
        "### DOCUMENT START ###\n"
        f"{retrieved_doc}\n"
        "### DOCUMENT END ###\n\n"
        f"Question: {user_question}"
    )

    try:
        response = client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ],
            model=phi4_deployment,
            # Reasoning models need room to think AND answer
            max_tokens=4096 if is_reasoning else 800,
            # Microsoft's recommended sampling for Phi-4-reasoning
            temperature=0.8 if is_reasoning else 0.3,
            top_p=0.95,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        # Phi-4-reasoning returns: [<think>] ...thinking... </think> final answer
        # The opening <think> is often missing, so keep only what comes after
        # the LAST </think>.
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1]
        elif "<think>" in content:
            # Thinking started but never finished -> ran out of tokens
            content = ""
        content = content.strip()

        if not content:
            return (f"Error: model returned no answer (finish_reason={choice.finish_reason}). "
                    "Try increasing max_tokens.")

        return content

    except Exception as e:
        return f"Error communicating with Azure AI Inference: {str(e)}"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(BytesIO(file.read()))

            return jsonify({
                'success': True,
                'text': pdf_text
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/ask-question', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        pdf_text = data.get('context', '')
        question = data.get('question', '')

        if not pdf_text or not question:
            return jsonify({'error': 'Missing context or question'}), 400

        # Use Azure AI Inference for chat completion
        answer = chat_with_phi4_rag(question, pdf_text)

        return jsonify({
            'answer': answer,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize Azure AI Inference client on startup
    if initialize_azure_client():
        print("🚀 Flask app starting with Azure AI Inference integration...")
        app.run(debug=True)
    else:
        print("❌ Failed to initialize Azure AI Inference client. Please check your cred.json file.")
        print("App will not start without proper Azure configuration.")

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import PyPDF2
from io import BytesIO
import requests
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# API Gateway endpoint URL
API_GATEWAY_URL = 'https://8atev3w62g.execute-api.us-east-1.amazonaws.com/default/gen-ai'  # Replace with your API Gateway URL

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
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
            
            # Store the PDF text in the session
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
        
        # Prepare payload for API Gateway
        payload = {
            'context': pdf_text,
            'question': question
        }
        
        # Make request to API Gateway
        response = requests.post(
            API_GATEWAY_URL,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        # Check response status
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Error processing request'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
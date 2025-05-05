from flask_cors import CORS
from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)
# Initialize summarizer with efficient settings
summarizer = pipeline(
    "summarization",
    model="linydub/bart-large-samsum",
    device=-1,  # Force CPU usage
    torch_dtype="auto"
)

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Get and validate input
        text = request.json.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        word_count = len(text.split())
        
        # Handle different input lengths
        if word_count < 10:
            return jsonify({
                "warning": "Input too short for meaningful summarization",
                "original_text": text,
                "word_count": word_count
            })
            
        # Calculate dynamic length limits
        max_len = max(10, min(62, word_count // 2))  # 10-62 words, max half of input
        min_len = max(2, word_count // 4)  # At least 2 words
        
        # Process summary
        summary = summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        
        return jsonify({
            "summary": summary[0]['summary_text'],
            "original_length": word_count,
            "summary_length": len(summary[0]['summary_text'].split())
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Summarization failed"
        }), 500

if __name__ == '__main__':
    # Configure for production
    app.run(
        host='0.0.0.0',
        port=5002,
        threaded=True
    )
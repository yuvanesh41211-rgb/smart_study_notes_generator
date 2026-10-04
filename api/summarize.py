"""
Vercel serverless version of the Smart Study Notes Generator.

Vercel functions are too small to hold PyTorch and the 1.6 GB BART model,
so this endpoint sends the summarization step to the Hugging Face
Inference API (same facebook/bart-large-cnn model). Everything else -
validation, chunking, statistics and key points - reuses app.py.

Requires the HF_TOKEN environment variable (a free Hugging Face access token).

POST /api/summarize   body: {"text": "..."}
"""

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import (  # noqa: E402
    MODEL_NAME,
    validate_input,
    generate_summary,
    count_words,
    reduction_percentage,
    extract_key_points,
)

API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"

# BART averages a little over one token per English word. Over-estimating
# keeps each chunk safely under the model's 1024-token limit.
TOKENS_PER_WORD = 1.4


class ApproxTokenizer:
    """Stands in for the BART tokenizer, which needs the transformers library."""

    def encode(self, text, add_special_tokens=True):
        count = int(len(text.split()) * TOKENS_PER_WORD) + (2 if add_special_tokens else 0)
        return [0] * count


class RemoteSummarizer:
    """Behaves like a transformers summarization pipeline, but calls the HF API."""

    tokenizer = ApproxTokenizer()

    def __init__(self, token):
        self.token = token

    def __call__(self, text, max_length, min_length, do_sample=False, truncation=True):
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": do_sample,
            },
        }
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=55) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Hugging Face API returned {err.code}: {detail}")


class handler(BaseHTTPRequestHandler):
    def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            text = json.loads(self.rfile.read(length) or b"{}").get("text", "")
        except (ValueError, AttributeError):
            return self._send(400, {"error": "Request body must be JSON like {\"text\": \"...\"}."})

        if not isinstance(text, str):
            return self._send(400, {"error": "'text' must be a string."})

        error = validate_input(text)
        if error:
            return self._send(400, {"error": error})

        token = os.environ.get("HF_TOKEN")
        if not token:
            return self._send(500, {"error": "Server is missing the HF_TOKEN environment variable."})

        try:
            summary = generate_summary(RemoteSummarizer(token), text)
        except Exception as err:
            return self._send(502, {"error": f"Something went wrong while summarizing ({err})."})

        if not summary:
            return self._send(502, {"error": "The model returned an empty summary. Please try different text."})

        original_count = count_words(text)
        summary_count = count_words(summary)
        return self._send(200, {
            "summary": summary,
            "original_word_count": original_count,
            "summary_word_count": summary_count,
            "reduction_percentage": round(reduction_percentage(original_count, summary_count), 2),
            "key_points": extract_key_points(summary),
        })

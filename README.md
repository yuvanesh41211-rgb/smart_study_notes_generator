# Smart Study Notes Generator

## Aim

Students often need to revise long paragraphs from textbooks and notes. The aim of this project is to build a simple Python application that takes a paragraph of text and uses a pre-trained Generative AI summarization model (**BART**, from Hugging Face Transformers) to turn it into short study notes: a concise summary, a list of key points, and statistics showing how much the text was shortened.

## Features

- Generates a short summary of any paragraph using the pre-trained `facebook/bart-large-cnn` model
- Displays key points as bullet points (one per summary sentence)
- Shows the original word count and the summary word count
- Calculates the text reduction percentage
- Accepts multi-line paragraphs (press Enter on an empty line to finish)
- Handles long input by splitting it into chunks that fit the model's limit
- Handles errors with clear messages: empty input, very short input, non-text input, model loading failures and summarization errors
- Runs fully offline after the first model download. No API key and no paid services.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3 | Programming language |
| Hugging Face Transformers (4.x) | Provides `pipeline("summarization")` and the pre-trained model |
| PyTorch | Deep learning backend that runs the model |
| BART (`facebook/bart-large-cnn`) | Pre-trained encoder-decoder model fine-tuned for news summarization |

> **Note:** `transformers` version 5 removed the `"summarization"` pipeline, so `requirements.txt` pins `transformers>=4.40,<5`.

## Project Structure

```
smart_study_notes_generator/
├── app.py               # Main program
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── sample_inputs.txt    # Three test paragraphs (AI, Cloud, Cybersecurity)
```

## Installation

```
python -m venv venv
```

Windows:

```
venv\Scripts\activate
```

Then:

```
pip install -r requirements.txt
```

The first time the program runs, it downloads the BART model (about 1.6 GB). Later runs load it from the local cache.

## How to Run

```
python app.py
```

Paste a paragraph, then press **Enter on an empty line** to generate the notes. After each result, the program asks whether you want to summarize another paragraph.

## Working

```
User Input
   → Input Validation (empty / too short / not text)
   → BART Summarization Model
   → Generated Summary
   → Word Count (original and summary)
   → Reduction Percentage
   → Key Points
```

1. **Model loading:** `pipeline("summarization", model="facebook/bart-large-cnn")` loads the tokenizer and model.
2. **Input and validation:** the paragraph is read and checked. It must not be empty, must contain at least 30 words, and must be mostly real words rather than numbers or symbols.
3. **Summarization:** the text is passed to BART. The summary length is set automatically from the input length (roughly 25–50% of the input, at most 150 tokens). `do_sample=False` makes the output deterministic.
4. **Statistics:**
   - Word count = number of whitespace-separated words
   - Reduction % = ((Original Word Count − Summary Word Count) / Original Word Count) × 100
5. **Key points:** the summary is split into sentences, and each sentence is shown as a bullet point. No second AI model is used.

### Long input handling

BART can process at most **1024 tokens** (roughly 700–750 English words) at a time. To handle longer input without crashing or losing text:

- The text is split into sentences, and sentences are grouped into chunks of at most 900 tokens.
- Each chunk is summarized separately, and the chunk summaries are joined together.
- `truncation=True` is a safety net if a single sentence is longer than the limit.
- If a chunk's summary reaches the length limit and stops mid-sentence, the unfinished sentence is removed.

Limitation: each chunk is summarized independently, so the summary of a very long text may repeat ideas or lose links between sections.

## Test Cases

All results below are the **actual output** of `app.py`. It was tested on Windows 11 with Python 3.14, transformers 4.57.6 and torch 2.14.1 (CPU). The paragraphs are in `sample_inputs.txt`.

### Test Case 1: Artificial Intelligence

- **Original Word Count:** 178
- **Generated Summary:**
  > Artificial Intelligence is a branch of computer science that focuses on building systems capable of performing tasks that normally require human intelligence. Modern AI is largely driven by machine learning, where computers learn patterns from large amounts of data. Deep learning has enabled major breakthroughs such as voice assistants, real-time language translation, self-driving car research, and medical image analysis.
- **Summary Word Count:** 59
- **Text Reduction:** 66.85%  (check: (178 − 59) / 178 × 100 = 66.85)
- **Key Points:**
  - Artificial Intelligence is a branch of computer science that focuses on building systems capable of performing tasks that normally require human intelligence.
  - Modern AI is largely driven by machine learning, where computers learn patterns from large amounts of data.
  - Deep learning has enabled major breakthroughs such as voice assistants, real-time language translation, self-driving car research, and medical image analysis.

### Test Case 2: Cloud Computing

- **Original Word Count:** 171
- **Generated Summary:**
  > Cloud computing is the delivery of computing services such as servers, storage, databases, networking, and software over the internet. One of the main advantages of the cloud is scalability, because resources can be increased or decreased quickly depending on demand. The cloud supports remote work, automatic backups, and disaster recovery.
- **Summary Word Count:** 50
- **Text Reduction:** 70.76%  (check: (171 − 50) / 171 × 100 = 70.76)
- **Key Points:**
  - Cloud computing is the delivery of computing services such as servers, storage, databases, networking, and software over the internet.
  - One of the main advantages of the cloud is scalability, because resources can be increased or decreased quickly depending on demand.
  - The cloud supports remote work, automatic backups, and disaster recovery.

### Test Case 3: Cybersecurity

- **Original Word Count:** 167
- **Generated Summary:**
  > Cybersecurity is the practice of protecting computers, networks, programs, and data from digital attacks, damage, or unauthorized access. Common attacks include phishing, where attackers send fake emails or messages to trick people into revealing passwords or bank details. Ransomware is a particularly dangerous form of malware that encrypts a victim's files and demands payment to unlock them.
- **Summary Word Count:** 57
- **Text Reduction:** 65.87%  (check: (167 − 57) / 167 × 100 = 65.87)
- **Key Points:**
  - Cybersecurity is the practice of protecting computers, networks, programs, and data from digital attacks, damage, or unauthorized access.
  - Common attacks include phishing, where attackers send fake emails or messages to trick people into revealing passwords or bank details.
  - Ransomware is a particularly dangerous form of malware that encrypts a victim's files and demands payment to unlock them.

### Error Handling Tests

| Input | Actual Output |
|---|---|
| Empty input | `Error: Input is empty. Please enter a paragraph of text.` |
| `Too short text here.` | `Error: Input is too short (4 words). Please enter at least 30 words so there is something to summarize.` |
| Only numbers and symbols | `Error: Input does not look like normal text. Please enter a paragraph of sentences.` |
| Long input (1548 words) | `Note: long input split into 3 parts to fit the model's limit.` Summary: 176 words, 88.63% reduction |
| transformers 5.x installed | `Error: this version of 'transformers' does not support the 'summarization' pipeline...` (no traceback) |

## Observation

- **Relevance:** The summaries pick up each paragraph's main topic. Each one opens with the paragraph's definition sentence, which is the most useful line for revision.
- **Coherence:** The output is grammatically correct and reads naturally, because BART copies and lightly compresses whole sentences from the source.
- **Conciseness:** The text was reduced by about 66–71% for the sample paragraphs (167–178 words down to 50–59 words).
- **Information retention:** The main definitions and the most prominent facts are kept. However, the model mostly keeps content from the **beginning** of the paragraph. For example, the AI summary leaves out the concerns about bias and responsible AI. The Cloud summary leaves out the IaaS/PaaS/SaaS models and the challenges. The Cybersecurity summary leaves out defenses such as firewalls and multi-factor authentication.
- **Limitations:**
  - BART-large-CNN was trained on news articles, so it tends to be *extractive* (it copies sentences) and favors the first sentences ("lead bias").
  - Key points are just the summary's sentences, so there are usually only 2–4 of them.
  - Input is limited to 1024 tokens per chunk, so very long text is summarized in separate pieces.
  - The model is large (about 1.6 GB). The first download takes time, and each summary takes several seconds on a CPU.
  - Like any generative model, BART may occasionally produce inaccurate statements, so the notes should be checked against the source.

## Result

The Smart Study Notes Generator met its objective. It uses the pre-trained BART model through Hugging Face `pipeline("summarization")` to summarize paragraphs, displays key points, and correctly reports the original and summary word counts and the reduction percentage. It also handles empty, short, invalid and long input without crashing. All three sample paragraphs were summarized successfully, with text reductions of 66.85%, 70.76% and 65.87%.

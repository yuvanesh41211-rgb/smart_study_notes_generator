"""
Smart Study Notes Generator
---------------------------
A simple command-line tool that takes a paragraph of text and uses the
pre-trained BART model (facebook/bart-large-cnn) from Hugging Face
Transformers to produce:

    * a short summary
    * key points (one bullet per summary sentence)
    * original and summary word counts
    * the percentage by which the text was reduced
"""

import re
import sys

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/bart-large-cnn"

# Inputs shorter than this are too small to summarize meaningfully.
MIN_WORDS = 30

# BART can read at most 1024 tokens at once. We keep each chunk a bit below
# that limit to leave room for the special start/end tokens.
MAX_TOKENS_PER_CHUNK = 900

LINE = "=" * 60


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load the BART summarization pipeline. Returns None if loading fails."""
    print("Loading the BART summarization model...")
    print("(The first run downloads about 1.6 GB, so it may take a while.)\n")

    try:
        # Imported here so that a missing library gives a friendly message.
        from transformers import pipeline
        from transformers.utils import logging as hf_logging

        # Hide non-essential warnings from the transformers library.
        hf_logging.set_verbosity_error()

        summarizer = pipeline("summarization", model=MODEL_NAME)
        print("Model loaded successfully.\n")
        return summarizer

    except ImportError:
        print("Error: 'transformers' or 'torch' is not installed.")
        print("Run:  pip install -r requirements.txt")
    except KeyError:
        # transformers 5.x removed the "summarization" pipeline task.
        print("Error: this version of 'transformers' does not support the")
        print("'summarization' pipeline. Install a 4.x version with:")
        print("  pip install -r requirements.txt")
    except OSError:
        print("Error: could not download or find the model files.")
        print("Check your internet connection and try again.")
    except Exception as error:
        print(f"Error: the model could not be loaded ({error}).")
    return None


# ---------------------------------------------------------------------------
# Input handling and validation
# ---------------------------------------------------------------------------
def read_paragraph():
    """
    Read a paragraph from the user. The paragraph may span several lines;
    an empty line marks the end. Returns None if input has ended (EOF).
    """
    print("Enter or paste your paragraph below.")
    print("Press Enter on an empty line when you are done:\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            # Input stream closed (e.g. Ctrl+Z / Ctrl+D or end of a file).
            if not lines:
                return None
            break
        if line.strip() == "":
            break
        lines.append(line.strip())

    return " ".join(lines)


def validate_input(text):
    """Return an error message if the text is not usable, otherwise None."""
    if not text or not text.strip():
        return "Input is empty. Please enter a paragraph of text."

    words = text.split()

    # Count words that actually contain letters (rejects input like "123 !!!").
    real_words = [w for w in words if re.search(r"[A-Za-z]", w)]
    if len(real_words) < len(words) / 2:
        return "Input does not look like normal text. Please enter a paragraph of sentences."

    if len(words) < MIN_WORDS:
        return (f"Input is too short ({len(words)} words). "
                f"Please enter at least {MIN_WORDS} words so there is something to summarize.")

    return None


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
def split_into_sentences(text):
    """Split text into sentences at '.', '!' or '?' followed by a space."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_into_chunks(text, tokenizer):
    """
    Long-input handling: group whole sentences into chunks that each fit
    within BART's input limit, so no part of the text is cut off.
    """
    chunks = []
    current = []
    current_tokens = 0

    for sentence in split_into_sentences(text):
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current and current_tokens + sentence_tokens > MAX_TOKENS_PER_CHUNK:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_chunk(summarizer, chunk):
    """Summarize one chunk, choosing a summary length based on input size."""
    token_count = len(summarizer.tokenizer.encode(chunk))

    # Aim for a summary of roughly 25-50% of the input, capped at 150 tokens.
    max_len = min(150, max(30, token_count // 2))
    min_len = min(60, max(10, token_count // 4))

    result = summarizer(
        chunk,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,      # deterministic output: same input -> same summary
        truncation=True,      # safety net in case a single sentence is too long
    )
    summary = result[0]["summary_text"].strip()

    # If the summary hit max_length it can stop mid-sentence. In that case,
    # drop the unfinished last sentence (as long as a complete one remains).
    last_end = max(summary.rfind("."), summary.rfind("!"), summary.rfind("?"))
    if last_end != -1 and last_end != len(summary) - 1:
        summary = summary[:last_end + 1]
    return summary


def generate_summary(summarizer, text):
    """Summarize the full text, chunk by chunk if it is long."""
    chunks = split_into_chunks(text, summarizer.tokenizer)
    if len(chunks) > 1:
        print(f"Note: long input split into {len(chunks)} parts to fit the model's limit.")

    summaries = [summarize_chunk(summarizer, chunk) for chunk in chunks]
    summary = " ".join(summaries)

    # Tidy up spacing such as "word ." that the model sometimes produces.
    summary = re.sub(r"\s+([.,!?;:])", r"\1", summary)
    return re.sub(r"\s+", " ", summary).strip()


# ---------------------------------------------------------------------------
# Statistics and key points
# ---------------------------------------------------------------------------
def count_words(text):
    """Count words separated by whitespace."""
    return len(text.split())


def reduction_percentage(original_count, summary_count):
    """((Original - Summary) / Original) * 100"""
    if original_count == 0:
        return 0.0
    return (original_count - summary_count) / original_count * 100


def extract_key_points(summary):
    """Each sentence of the summary becomes one key point."""
    return split_into_sentences(summary)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def display_results(original, summary):
    original_count = count_words(original)
    summary_count = count_words(summary)
    reduction = reduction_percentage(original_count, summary_count)

    print()
    print(LINE)
    print("SMART STUDY NOTES GENERATOR".center(60))
    print(LINE)
    print("\nOriginal Text:")
    print(original)
    print(f"\nOriginal Word Count: {original_count}")
    print("\nGenerated Summary:")
    print(summary)
    print(f"\nSummary Word Count: {summary_count}")
    print(f"\nText Reduction: {reduction:.2f}%")
    print("\nKey Points:")
    for point in extract_key_points(summary):
        print(f"• {point}")
    print()
    print(LINE)
    print()


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def ask_to_continue():
    """Ask whether the user wants to summarize another paragraph."""
    try:
        answer = input("Summarize another paragraph? (y/n): ").strip().lower()
    except EOFError:
        return False
    print()
    return answer in ("y", "yes")


def main():
    # Make sure the bullet character prints correctly on Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    print(LINE)
    print("SMART STUDY NOTES GENERATOR".center(60))
    print(LINE)
    print()

    summarizer = load_model()
    if summarizer is None:
        sys.exit(1)

    while True:
        text = read_paragraph()
        if text is None:
            print("No more input. Exiting.")
            break

        error = validate_input(text)
        if error:
            print(f"\nError: {error}\n")
            continue  # ask for a paragraph again

        print("\nGenerating summary, please wait...")
        try:
            summary = generate_summary(summarizer, text)
        except Exception as error:
            print(f"\nError: something went wrong while summarizing ({error}).")
            print("Please try again with a different paragraph.\n")
            continue

        if not summary:
            print("\nError: the model returned an empty summary. Please try different text.\n")
            continue

        display_results(text, summary)

        if not ask_to_continue():
            break

    print("Thank you for using Smart Study Notes Generator!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram stopped by user. Goodbye!")

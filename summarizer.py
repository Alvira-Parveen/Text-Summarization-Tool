import os
import logging
import warnings
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline

# ---------------------------
# Suppress logs & warnings
# ---------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ---------------------------
# Abstractive summarizer (Hugging Face)
# ---------------------------
abstractive_summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1  # CPU, change to 0 for GPU if available
)

# ---------------------------
# Functions
# ---------------------------
def extractive_summary(text, sentences_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

def abstractive_summary(text, max_length=300, min_length=50):
    summary = abstractive_summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    return summary[0]['summary_text']

# ---------------------------
# Main Program
# ---------------------------
if __name__ == "__main__":
    text = input("Paste your article here:\n")

    # Generate summaries
    ext_summary = extractive_summary(text)
    abs_summary = abstractive_summary(text)

    # Display summaries
    print("\n--- Extractive Summary ---\n")
    print(ext_summary)

    print("\n--- Abstractive Summary ---\n")
    print(abs_summary)

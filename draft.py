from transformers import AutoTokenizer

# The text for which to count tokens
text = "This is an example sentence to demonstrate BGE-M3 token counting."

# Load the BGE-M3 tokenizer
tokenizer = AutoTokenizer.from_pretrained("hiieu/halong_embedding")

# Tokenize the text, applying the same preprocessing as the model
# BGE-M3 performs lowercasing and stripping, and adds special tokens
tokens = tokenizer(
    text.lower().strip(),
    add_special_tokens=True,
    truncation=True,
    max_length=512  # Or the desired maximum length
)

# The number of tokens is the length of the 'input_ids' list
token_count = len(tokens["input_ids"])

print(f"The text has {token_count} tokens according to BGE-M3.")
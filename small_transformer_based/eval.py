import torch
import torch.nn as nn

from small_transformer_based.train import (
    CustomTokenizer,
    CustomTransformer,
    evaluate_true,
)


# Load the tokenizer and its vocabulary
tokenizer = CustomTokenizer()
tokenizer.load_vocab("vocab.json")  # Load the vocabulary

# Initialize the model with the correct vocab_size
model = CustomTransformer(vocab_size=len(tokenizer.vocab))

# Conditional DataParallel wrapping
if torch.cuda.device_count() >= 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

# Device selection: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU for evaluation")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal) GPU for evaluation on Apple Silicon")
else:
    device = torch.device("cpu")
    print("Using CPU for evaluation")

# Load the model checkpoint (adjust the path as needed)
print("##### LOADING THE MODEL... #####")
checkpoint = torch.load(
    "small_transformer_based/results/25.3M/checkpoint_epoch0_iter2.pth",
    map_location=device,
)

# Extract the model's state_dict
state_dict = checkpoint["model_state_dict"]

# # Load the state_dict into the model
model.load_state_dict(state_dict)
print("#####... MODEL LOADED #####")

model.eval()

model.to(device)


evaluate_true(model, tokenizer, device, tta=False)

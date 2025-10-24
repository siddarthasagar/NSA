import os
import re
import math
import json
import torch
import random
import hashlib
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from collections import Counter
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import train_test_split
from einops import repeat
from auxilaries.generate_transformation import generate_samples
from plots import return_task_grid
from llm.selector_prompt import generate_selector_prompt
from task import Task
from shutil import rmtree


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=6500):
        super(SinusoidalPositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        pe = torch.zeros(max_len, n_embd)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Helper Functions
def extract_first_from_logits(tokenizer, logits, index, number, already_added=[]):
    first_predictions = logits[0][index].topk(30, dim=-1)
    corresponding_tokens = [tokenizer.decode([int(token_id.cpu().numpy())]).strip() for token_id in first_predictions.indices]
    top_predictions = [x for x in corresponding_tokens if x not in already_added + ["no_trans"]]
    top_predictions = top_predictions[:number]
    return top_predictions

def evaluate_true(model, tokenizer, device, tta=True, tta_epochs=1):
    model.eval()
    dataset_splits = {
        "train": "dataset/training",
        "val": "dataset/validation"
    }
    results = {
        "train": {},
        "val": {}
    }
    proposed_transformations_dict = {
        "train": {},
        "val": {}
    }
    for split, directory in dataset_splits.items():
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Skipping {split} split.")
            continue
        task_files = [f for f in os.listdir(directory) if f.endswith(".json")]
        task_ids = [f for f in task_files]
        if split == "val" and tta:
            results_file = "arga_evaluation_tta.json"
        elif split == "train" and tta:
            results_file = f"arga_training_tta_epoch{tta_epochs}.json"
        elif split == "train" and not tta:
            results_file = "arga_training_no_tta.json"
        elif split == "val" and not tta:
            results_file = "arga_evaluation_no_tta.json"
        else:
            raise ValueError("Invalid dataset split encountered.")

        if os.path.exists(results_file):
            print(f"File {results_file} found and will be updated!")
            with open(results_file, 'r') as f:
                loaded_results = json.load(f)
                if split in loaded_results:
                    results[split].update(loaded_results[split])
        else:
            print(f"No existing {results_file} found. Creating a new one.")

        correct_solved = sum(1 for task_id in results[split] if results[split][task_id].get('solved'))
        total_tasks = len(task_ids)
        print(f"Processing {total_tasks} tasks in the '{split}' split. {correct_solved} already solved.")

        for task_id_json in tqdm(task_ids, total=total_tasks, desc=f"Evaluating {split.capitalize()} Tasks"):
            task_id = task_id_json
            task_key = task_id_json.replace('.json', '')
            if task_key in results[split]:
                if results[split][task_key].get('solved'):
                    print(f"Task {task_key} is already solved in results. Skipping.")
                    continue
            print(f"Processing task {task_id} in '{split}' split.")

            if tta:
                checkpoint_path = "small_transformer_based/results_mine/25.3M/checkpoint_epoch27_iter874.pth"
                if not os.path.exists(checkpoint_path):
                    print(f"Checkpoint not found at {checkpoint_path}. Skipping TTA for task {task_id}.")
                    model_to_use = model
                else:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    state_dict = checkpoint['model_state_dict']
                    model_tta = deepcopy(model)
                    model_tta.load_state_dict(state_dict)
                    model_tta = model_tta.to(device)
                    model_tta.train()
                    output_folder = f"tta/{task_key}"
                    all_transformations_path = f"tta/{task_key}.json"
                    if os.path.exists(all_transformations_path):
                        try:
                            with open(all_transformations_path, 'r') as f:
                                tta_data = json.load(f)
                        except Exception as e:
                            print(f"Error loading TTA data for {task_key}: {e}. Regenerating data.")
                            os.remove(all_transformations_path)
                            if os.path.exists(output_folder):
                                rmtree(output_folder)
                            tta_data = []
                    else:
                        tta_data = []

                    if os.path.exists(output_folder) and len(os.listdir(output_folder)) != len(tta_data):
                        if abs(len(os.listdir(output_folder)) - len(tta_data)) > 100 and len(tta_data) < 2000:
                            print(f"Inconsistent TTA data for {task_key}. Regenerating.")
                            if os.path.exists(all_transformations_path):
                                os.remove(all_transformations_path)
                            rmtree(output_folder)
                            tta_data = []

                    if not tta_data:
                        Path(output_folder).mkdir(exist_ok=True, parents=True)
                        print(f"--- Data Generation for {task_key} ---")
                        generate_samples(
                            number_of_samples=2500,
                            output_folder=output_folder,
                            all_transformations_path=all_transformations_path,
                            no_of_trans=3,
                            transformation_ops=None,
                            chosen_task=task_key
                        )
                        with open(all_transformations_path, 'r') as f:
                            tta_data = json.load(f)
                    tta_data = [x for x in tta_data if len(x["input"]) < 12000]
                    tta_dataset = CustomDataset(tta_data, tokenizer)
                    tta_loader = DataLoader(tta_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
                    tta_optimizer = optim.AdamW(model_tta.parameters(), lr=5e-5)
                    tta_scheduler = optim.lr_scheduler.StepLR(tta_optimizer, step_size=10, gamma=0.1)
                    print(f"--- Model Training for {tta_epochs} Epochs ---")
                    for epoch in range(tta_epochs):
                        for batch_idx, (input_ids_batch, output_ids_batch) in tqdm(enumerate(tta_loader), total=len(tta_loader), desc=f"TTA Epoch {epoch+1}"):
                            input_ids_batch = input_ids_batch.to(device)
                            output_ids_batch = output_ids_batch.to(device)
                            output_ids_batch = output_ids_batch[output_ids_batch != tokenizer.vocab.get("<BOS>", -100)]
                            output_ids_batch = output_ids_batch[output_ids_batch != tokenizer.vocab.get("<EOS>", -100)]
                            tta_optimizer.zero_grad()
                            attention_mask = (input_ids_batch != tokenizer.vocab.get("<PAD>", 0)).to(device)
                            logits_batch = model_tta(input_ids_batch, attention_mask=attention_mask)
                            logits_batch = logits_batch.view(-1, logits_batch.size(-1))
                            targets_batch = output_ids_batch.view(-1)
                            loss = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab.get("<PAD>", 0))(logits_batch, targets_batch)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model_tta.parameters(), 1)
                            tta_optimizer.step()
                        tta_scheduler.step()
                    model_to_use = model_tta
            else:
                model_to_use = model
            try:
                grid = return_task_grid(task_id)["train"]
            except Exception as e:
                print(f"Error retrieving grid for task {task_id}: {e}")
                continue
            prompt = generate_selector_prompt(grid)
            prompt = extract_input_output_pairs(prompt)
            input_ids = tokenizer.encode(prompt)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
            attention_mask = (input_ids != tokenizer.vocab.get("<PAD>", 0)).to(device)
            model_to_use.eval()
            with torch.no_grad():
                logits = model_to_use(input_ids, attention_mask=attention_mask)
                first_token = torch.argmax(logits[0][1]).item()
                second_token = torch.argmax(logits[0][2]).item()
                include_second = tokenizer.decode([second_token]) != "no_trans"
                include_third = include_second and (tokenizer.decode([torch.argmax(logits[0][3]).item()]) != "no_trans")
                top3_predictions = []
                if not include_second and not include_third:
                    to_consider = 5
                    top3_predictions = extract_first_from_logits(tokenizer=tokenizer, logits=logits, index=0, number=to_consider, already_added=[])
                if include_second:
                    to_consider = 4
                    preds_first = extract_first_from_logits(tokenizer=tokenizer, logits=logits, index=0, number=to_consider)
                    preds_second = extract_first_from_logits(tokenizer=tokenizer, logits=logits, index=1, number=to_consider, already_added=preds_first)
                    top3_predictions += preds_first + preds_second
                if include_third:
                    to_consider = 3
                    preds_third = extract_first_from_logits(tokenizer=tokenizer, logits=logits, index=2, number=to_consider, already_added=top3_predictions)
                    top3_predictions += preds_third
            top3_predictions = [pred for pred in top3_predictions if pred != "no_trans"]
            top3_predictions = list(dict.fromkeys(top3_predictions))
            proposed_transformations_dict[split][task_id] = top3_predictions
            data_path = dataset_splits[split]
            try:
                task = Task(os.path.join(data_path, task_id), proposed_transformations=top3_predictions)
                solved = task.solve()
                if solved:
                    print(f"Task {task_id} solved successfully with predicted transformations.")
                    correct_solved += 1
                else:
                    print(f"Task {task_id} could not be solved with predicted transformations.")
            except Exception as e:
                print(f"Error solving task {task_id} with predicted transformations: {e}")
                solved = False
            results[split][task_key] = {
                'solved': solved,
                'predictions': top3_predictions
            }
            with open(results_file, 'w') as f:
                json.dump({split: results[split]}, f, indent=4)
            print(f"Results saved to {results_file}")
        print(f"Number of '{split}' tasks correctly solved: {correct_solved} out of {total_tasks}")
    proposed_transformations_file = "proposed_transformations.txt"
    with open(proposed_transformations_file, 'w') as f:
        for split in proposed_transformations_dict:
            for task_id, transformations in proposed_transformations_dict[split].items():
                f.write(f"{split}/{task_id}: {transformations}\n")
    print(f"Proposed transformations saved to {proposed_transformations_file}")
    return results

def create_training_data_summary(train_data):
    training_data_summary = []
    print("GENERATING A TRAINING DATA SUMMARY...")
    for data in tqdm(train_data):
        x = decode2task(extract_input_output_pairs(data["input"]))
        task_dict = {
            "transformation": data["output"],
            "input_dims": [np.array(pair["input"]).shape for pair in x],
            "output_dims": [np.array(pair["output"]).shape for pair in x]
        }
        training_data_summary.append(task_dict)
    with open('training_data_summary.json', 'w') as json_file:
        json.dump(training_data_summary, json_file, indent=4)
    print("...CREATED AND SAVED DATA SUMMARY IN JSON")

def save_model_checkpoint(model, optimizer, epoch, iteration, save_path):
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved at {save_path}")

def balance_transformations(data):
    required_transformations = [
        "update_color", "move_node", "extend_node", "move_node_max",
        "rotate_node", "add_border", "fill_rectangle", "extract",
        "hollow_rectangle", "mirror",
        "flip", "remove_node", "insert",
        'duplicate', 'upscale_grid', "crop", "fill", "magnet", "beam", "shift",
        "arbitrary_duplicate", "rotate_duplicate",
        "rotate_grid",
        "connect", "recolor", "truncate"
    ]
    transformation_samples = {transformation: [] for transformation in required_transformations}
    less_than_two_no_trans_samples = []
    for item in data:
        output_text = item['output'].strip()
        transformations = output_text.split()
        no_trans_count = transformations.count('no_trans')
        if len(transformations) == 3 and no_trans_count == 2:
            first_transformation = transformations[0]
            if first_transformation in required_transformations:
                transformation_samples[first_transformation].append(item)
        elif no_trans_count < 2:
            less_than_two_no_trans_samples.append(item)
    non_zero_counts = [len(samples) for samples in transformation_samples.values() if len(samples) > 0]
    if not non_zero_counts:
        print("No samples found with two 'no_trans' transformations to balance.")
        balanced_data = less_than_two_no_trans_samples.copy()
        return balanced_data
    min_count = min(non_zero_counts)
    print(f"Balancing to {min_count} samples per transformation.")
    balanced_data = []
    for transformation in required_transformations:
        samples = transformation_samples[transformation]
        if len(samples) >= min_count:
            balanced_subset = samples[:min_count]
            balanced_data.extend(balanced_subset)
    balanced_data.extend(less_than_two_no_trans_samples)
    print(f"Total balanced samples: {len(balanced_data)}")
    return balanced_data

def decode2task(text):
    pattern = re.compile(r"Input\s*\n([\d\s\|\n]+)Output\s*\n([\d\s\|\n]+)", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        pattern = re.compile(r"Input:\s*\n([\d\s\|\n]+)Output:\s*\n([\d\s\|\n]+)", re.DOTALL)
        matches = pattern.findall(text)
    def parse_grid(grid_text):
        rows = grid_text.strip().split("\n")
        return [[int(cell.strip()) for cell in row.split('|')] for row in rows]
    input_output_pairs = []
    for input_grid, output_grid in matches:
        input_output_pairs.append({
            'input': parse_grid(input_grid),
            'output': parse_grid(output_grid)
        })
    return input_output_pairs

def extract_input_output_pairs(text, use_2dpe=False):
    pattern = re.compile(r"Input:\n([\d\|\n]+)\nOutput:\n([\d\|\n]+)", re.DOTALL)
    matches = pattern.findall(text)
    input_output_pairs = []
    for match in matches:
        input_grid_str = match[0].strip().split("\n")
        output_grid_str = match[1].strip().split("\n")
        input_grid = "\n".join(input_grid_str)
        output_grid = "\n".join(output_grid_str)
        input_output_pairs.append(f"Input:\n{input_grid}\nOutput:\n{output_grid}")
    return "\n".join(input_output_pairs)

# Custom Tokenizer
class CustomTokenizer:
    def __init__(self, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"], transformations=None):
        self.special_tokens = special_tokens
        self.transformations = transformations or [
            "update_color", "move_node", "extend_node", "move_node_max",
            "rotate_node", "add_border", "fill_rectangle", "extract",
            "hollow_rectangle", "mirror", "flip", "remove_node", "insert",
            'duplicate', 'upscale_grid', "crop", "fill", "magnet", "beam", "shift", "no_trans",
            "arbitrary_duplicate", "rotate_duplicate", "mirror_grid", 'rotate_grid', "connect", "recolor",
            "truncate",
        ]
        self.vocab = {}
        self.inv_vocab = {}
        self.token_count = Counter()
        self.build_initial_vocab()

    def build_initial_vocab(self):
        idx = 0
        for token in self.special_tokens:
            self.vocab[token] = idx
            idx += 1
        predefined_tokens = (
            ["Input", "Output", "\n", "|"] +
            [str(i) for i in range(10)] +
            self.transformations
        )
        for token in predefined_tokens:
            self.vocab[token] = idx
            idx += 1

    def build_vocab(self, data, min_freq=1):
        print("...BUILD THE VOCABULARY")
        for text in tqdm(data):
            tokens = self.tokenize(text)
            self.token_count.update(tokens)
        for token, count in self.token_count.items():
            if token not in self.vocab and count >= min_freq:
                self.vocab[token] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        tokens = []
        if isinstance(text, str):
            text = text.replace("Input:", " Input ").replace("Output:", " Output ")
        elif isinstance(text, list):
            text = str(text[0])
        else:
            raise Exception("Wrong type")
        combined_keywords = sorted(
            self.transformations,
            key=len, reverse=True
        )
        pattern = (
            r'\d+|Input|:|,|Output|\||\n|\{|\}|' +
            '|'.join(map(re.escape, combined_keywords))
        )
        grid_tokens = re.findall(pattern, text)
        for token in grid_tokens:
            if token in self.vocab:
                tokens.append(token)
            else:
                tokens.append("<UNK>")
        return tokens

    def encode(self, text, max_length=None):
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        token_ids = [self.vocab["<BOS>"]] + token_ids + [self.vocab["<EOS>"]]
        if max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def decode(self, token_ids):
        tokens = [self.inv_vocab.get(token_id, "<UNK>") for token_id in token_ids]
        decoded_text = " ".join(tokens)
        return decoded_text.replace("<BOS>", "").replace("<EOS>", "").strip()

    def save_vocab(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.vocab, f)

    def load_vocab(self, file_path):
        with open(file_path, 'r') as f:
            self.vocab = json.load(f)
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

# Transformer Model
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, n_positions=512, n_embd=512, n_layer=8, n_head=8, dropout=0., use_2dpe=False, num_cls_tokens=3):
        super(CustomTransformer, self).__init__()
        n_inner = 4 * n_embd
        self.num_cls_tokens = num_cls_tokens
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.cls_tokens = nn.Parameter(torch.randn(1, self.num_cls_tokens, n_embd))
        if use_2dpe:
            self.pos_encoding = SinusoidalPositionalEncoding2D(n_embd)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(n_embd)
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=n_inner, dropout=dropout,
                                       batch_first=True)
            for _ in range(n_layer)
        ])
        self.fc_out = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        b, n, _ = x.shape
        x = self.pos_encoding(x)
        x = self.dropout(x)
        cls_tokens = repeat(self.cls_tokens, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        if attention_mask is not None:
            cls_attention_mask = torch.ones((b, self.num_cls_tokens), device=attention_mask.device).type_as(attention_mask)
            attention_mask = torch.cat((cls_attention_mask, attention_mask), dim=1)
        for block in self.transformer_blocks:
            x = block(x, src_key_padding_mask=(~attention_mask.bool()) if attention_mask is not None else None)
        cls_outputs = x[:, :self.num_cls_tokens, :]
        logits = self.fc_out(cls_outputs)
        return logits

# Dataset Implementation
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=25600):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        output_text = str(item["output"])
        combined_input = extract_input_output_pairs(input_text)
        input_ids = self.tokenizer.encode(combined_input, max_length=self.max_length)
        output_ids = self.tokenizer.encode(output_text, max_length=self.max_length)
        return torch.tensor(input_ids), torch.tensor(output_ids)

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    output_ids = [item[1] for item in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_output_ids = pad_sequence(output_ids, batch_first=True, padding_value=0)
    return padded_input_ids, padded_output_ids

# Helper Function to calculate file hash
def calculate_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Training Function
def train(model, train_loader, val_loader, train_eval_loader, optimizer, scheduler, tokenizer, device, epoch, save_iterations, total_params_millions, plot_dir):
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    batch_losses = 0
    for batch_idx, (input_ids, output_ids) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        output_ids = output_ids.to(device)
        optimizer.zero_grad()
        attention_mask = (input_ids != tokenizer.vocab["<PAD>"]).to(device)
        logits = model(input_ids, attention_mask=attention_mask)
        # logits shape: [batch, num_cls_tokens, vocab]
        # We must align targets to exactly num_cls_tokens per sample.
        num_cls_tokens = getattr(model, 'module', model).num_cls_tokens if hasattr(model, 'module') else model.num_cls_tokens
        pad_id = tokenizer.vocab["<PAD>"]
        bos_id = tokenizer.vocab["<BOS>"]
        eos_id = tokenizer.vocab["<EOS>"]

        # Build targets of shape [batch, num_cls_tokens]
        targets_batch = []
        for row in output_ids:  # row shape: [seq_len]
            # Remove BOS tokens
            row_wo_bos = row[row != bos_id]
            # Stop at EOS if present
            eos_positions = (row_wo_bos == eos_id).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                row_wo_bos = row_wo_bos[:eos_positions[0]]
            # Remove PADs
            row_clean = row_wo_bos[row_wo_bos != pad_id]
            # Take first K label tokens; pad if fewer than K
            k = num_cls_tokens
            if row_clean.numel() >= k:
                take = row_clean[:k]
            else:
                pad_needed = k - row_clean.numel()
                take = torch.cat([row_clean, torch.full((pad_needed,), pad_id, dtype=row_clean.dtype, device=row_clean.device)])
            targets_batch.append(take)
        targets = torch.stack(targets_batch, dim=0)  # [batch, num_cls_tokens]

        # Compute loss
        logits = logits.view(-1, logits.size(-1))                 # [(batch*num_cls_tokens), vocab]
        targets = targets.view(-1)                                 # [(batch*num_cls_tokens)]
        loss = nn.CrossEntropyLoss(ignore_index=pad_id)(logits, targets)
        progress_bar.set_postfix(loss=loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        batch_losses += loss.item() / save_iterations
        if (batch_idx + 1) % args.print_iterations == 0:
            random_idx = random.randint(0, len(logits) - 1)
            top3_logits = logits[random_idx].topk(3, dim=-1)
            top3_predictions = [tokenizer.decode([int(token_id)]) for token_id in top3_logits.indices.cpu().numpy()]
            true_output = tokenizer.decode([int(targets[random_idx].cpu().numpy())])
            print(f"Iteration {batch_idx + 1}:")
            print(f"Top 3 Predictions: {top3_predictions}")
            print(f"True Output: {true_output}")
            print(f"Loss: {loss.item()}")
            checkpoint_path = os.path.join(
                plot_dir, f"checkpoint_epoch{epoch}_iter{batch_idx + 1}.pth"
            )
            save_model_checkpoint(model, optimizer, epoch, batch_idx + 1, checkpoint_path)
        if (batch_idx + 1) % save_iterations == 0:
            batch_losses = 0
    scheduler.step()

# Main Function
def main(data_path, epochs=1, batch_size=32, save_iterations=100):
    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU acceleration")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) GPU acceleration on Apple Silicon")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU acceleration available)")
    
    tokenizer = CustomTokenizer()
    cache_file = "dataset_cache.txt"
    vocab_file = "vocab.json"
    current_hash = calculate_file_hash(data_path)
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_hash = f.read().strip()
    else:
        cached_hash = None
    with open(data_path, 'r') as f:
        data = json.load(f)
    if not len(data):
        raise Exception("No data loaded!")
    data = [x for x in data if len(x["input"]) < 12000]
    print("Balancing transformations in the dataset...")
    balanced_data = balance_transformations(data)
    print(f"Balanced dataset contains {len(balanced_data)} samples")
    if cached_hash != current_hash or not os.path.exists("vocab.json"):
        print("Dataset changed or vocabulary does not exist yet. Creating a new vocabulary.")
        all_texts = [f"{extract_input_output_pairs(item['input'])} {item['output']}" for item in tqdm(balanced_data)]
        tokenizer.build_vocab(all_texts)
        tokenizer.save_vocab(vocab_file)
        with open(cache_file, 'w') as f:
            f.write(current_hash)
    else:
        print("Detected same dataset as before. Loading the vocabulary.")
        tokenizer.load_vocab(vocab_file)
    if args.overfit_single_example:
        print("Overfitting on a single example")
        data = [data[0]]
        train_data = data
        val_data = data
    else:
        train_data, val_data = train_test_split(balanced_data, test_size=0.1, random_state=42)
        _, train_eval_data = train_test_split(train_data, test_size=0.1, random_state=42)
    if cached_hash != current_hash or not os.path.exists("vocab.json"):
        create_training_data_summary(train_data)
    train_dataset = CustomDataset(train_data, tokenizer)
    val_dataset = CustomDataset(val_data, tokenizer)
    train_eval_dataset = CustomDataset(train_eval_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    model = CustomTransformer(vocab_size=len(tokenizer.vocab), use_2dpe=args.use_2dpe)
    print(f"LENGTH OF VOCABULARY: {len(tokenizer.vocab)}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_millions = total_params / 1_000_000
    print(f"Total number of parameters: {total_params_millions:.1f}M")
    plot_dir = f"small_transformer_based/results/{total_params_millions:.1f}M"
    os.makedirs(plot_dir, exist_ok=True)
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        train(model, train_loader, val_loader, train_eval_loader, optimizer, scheduler, tokenizer, device, epoch, save_iterations, total_params_millions, plot_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Custom Transformer Model")
    parser.add_argument('--data_path', type=str, default="full_trans.json")
    parser.add_argument('--save_iterations', type=int, default=1)
    parser.add_argument('--print_iterations', type=int, default=1)
    parser.add_argument("--use_2dpe", action="store_true", help="Enable 2D positional encoding.")
    parser.add_argument('--overfit_single_example', action='store_true', help="If set, the model will train on a single example.")
    args = parser.parse_args()
    batch_size = 32
    print(f"RUNNING CODE WITH BATCH_SIZE {batch_size}")
    main(args.data_path, epochs=100, batch_size=batch_size, save_iterations=args.save_iterations)

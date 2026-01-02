import argparse
import os

# --- 1. Argument Parsing ---
# Setup argument parser to accept command-line arguments for script configuration.
parser = argparse.ArgumentParser(description="Simple finetune a model from Hugging Face on a standard dataset")

# Define command-line arguments.
parser.add_argument("--use_lora", action="store_true", help="Use LoRA weights.")
parser.add_argument("--model", type=str, default='google/gemma-3-1b-pt', help="Model ID in Hugging Face.")
parser.add_argument("--training_samples", type=int, default=1000, help="Number of data points to train with.")
parser.add_argument("--eval_samples", type=int, default=100, help="Number of data points to evaluate with.")

# Parse the arguments provided at runtime.
args = parser.parse_args()


# --- 2. Configuration and Initialization ---
# Assign parsed arguments and other configurations to variables.
model_name = args.model
use_lora = args.use_lora
dataset_name = 'philschmid/gretel-synthetic-text-to-sql'
n_training_samples = args.training_samples
n_eval_samples = args.eval_samples

log_dir        = "logs"
# Retrieve Hugging Face token from environment variables for authentication.
hf_token = os.environ['HF_TOKEN']

# Training hyperparameters.
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
max_steps = int(n_training_samples*1.5)

# This is ignored because max_steps is set, which takes precedence.
epochs = 1000


print ('------ finetuning simple -------')
# Print a summary of the configuration for the current run.
print ('model           ', model_name)
print ('dataset         ', dataset_name)
print ('training samples', n_training_samples)
print ('eval samples    ', n_eval_samples)
print ('max steps       ', max_steps)
print ('using lora      ', use_lora)
print ('per_device_train_batch_size', per_device_train_batch_size)
print ('gradient_accumulation_steps', gradient_accumulation_steps)


# --------------------------------------------------------------------------------------------------------
# --- 3. Import Libraries ---
print ('\nloading libraries ...', flush=True)

from huggingface_hub import login
import os
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig
from trl import SFTTrainer
import pandas as pd
import numpy as np


# --------------------------------------------------------------------------------------------------------
# --- 4. Authenticate with Hugging Face ---
print ('authenticating to HF ... ', flush=True)
login(hf_token)


# --------------------------------------------------------------------------------------------------------
# --- 5. Load and Prepare Dataset ---
print ('\nloading and preparing dataset ...', flush=True)


# System message for the assistant
system_message = """You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""

# User prompt that combines the user query and the schema
user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""
def create_conversation(sample):
  return {
    "messages": [
      # {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(question=sample["sql_prompt"], context=sample["sql_context"])},
      {"role": "assistant", "content": sample["sql"]}
    ]
  }


# Load the dataset from the Hugging Face Hub.
dataset = load_dataset(dataset_name)

# Create training and evaluation subsets by shuffling and selecting a specific number of samples.
dataset_train = dataset['train'].shuffle().select(range(n_training_samples))
dataset_eval  = dataset['test'].shuffle().select(range(n_eval_samples))

# Apply the `create_conversation` function to format each sample into the required message structure for the trainer.
# The original columns are removed as they are now part of the 'messages' column.
dataset_train = dataset_train.map(create_conversation, remove_columns=dataset_train.features,batched=False)
dataset_eval  = dataset_eval.map(create_conversation, remove_columns=dataset_eval.features,batched=False)

# Print the size of the prepared datasets.
print ('train',len(dataset_train))
print ('eval', len(dataset_eval))

# --------------------------------------------------------------------------------------------------------
# --- 6. Load Model and Tokenizer ---
print ('\nloading model and tokenizer ...', flush=True)

# Select model class based on id
if model_name == "google/gemma-3-1b-pt":
    model_class = AutoModelForCausalLM
else:
    model_class = AutoModelForImageTextToText

# This line overrides the previous conditional, forcing the use of AutoModelForCausalLM.
model_class = AutoModelForCausalLM

# Check GPU capability to determine the optimal torch data type (bfloat16 for Ampere or newer).
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
# model_kwargs["quantization_config"] = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_quant_type='nf4',
#    bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
#    bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
#)

# Load model and tokenizer
model = model_class.from_pretrained(model_name, **model_kwargs)
# It's important to use the instruction-tuned (IT) version of the tokenizer for proper chat templating.
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it") # Load the Instruction Tokenizer to use the official Gemma template

# Add a padding token to the tokenizer and resize the model's token embeddings to include it.
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model.resize_token_embeddings(len(tokenizer))

# --------------------------------------------------------------------------------------------------------
# --- 7. Configure Training ---
print ('\nconfiguring training ...', flush=True)

# Configure PEFT (Parameter-Efficient Fine-Tuning) with LoRA if `use_lora` is True.
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
) if use_lora else None

# Configure the training arguments using SFTConfig from the TRL library.
args = SFTConfig(
    output_dir="gemma-text-to-sql",         # directory to save and repository id
    max_length=512,                         # max sequence length for model and packing of the dataset
    packing=True,                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=epochs,                                    # number of training epochs
    per_device_train_batch_size=per_device_train_batch_size,    # batch size per device during training
    gradient_accumulation_steps=gradient_accumulation_steps,    # number of steps before performing a backward/update pass
    max_steps = max_steps,
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_dir=log_dir,
    logging_steps=5,                       # log every 10 steps
    save_strategy="no",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                      # push model to hub
    report_to="none",                # report metrics
    do_eval=True,
    eval_steps=60,
    eval_strategy='steps',
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    },
)

# Initialize the SFTTrainer with the model, training arguments, datasets, and PEFT config.
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    peft_config=peft_config if use_lora else None,
    processing_class=tokenizer
)

# --------------------------------------------------------------------------------------------------------
# --- 8. Start Training ---
print ('\ntraining ...', flush=True)
train_output = trainer.train()


# --------------------------------------------------------------------------------------------------------
# --- 9. Save Training Metrics ---
print ('\nsaving training traces ...', flush=True)
# Create a unique filename based on the model and training parameters.
model_str = model_name.replace('/','_')
filename = f'results/{model_str}--{"withlora" if use_lora else "nolora"}--n_training_samples_{n_training_samples}--n_eval_samples_{n_eval_samples}--max_steps_{max_steps}.parquet'
# Convert the training history (logs) into a pandas DataFrame.
s = pd.DataFrame(trainer.state.log_history)
# Save the DataFrame to a Parquet file for later analysis.
s.to_parquet(filename)
print ('saved to', filename)

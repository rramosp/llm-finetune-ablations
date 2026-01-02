#!/bin/bash

#MODEL=google/gemma-3-1b-pt
MODEL=Qwen/Qwen3-0.6B

#python finetune-simple.py --training_samples 100 --model $MODEL --use_lora
#python finetune-simple.py --training_samples 100 --model $MODEL

#python finetune-simple.py --training_samples 500 --model $MODEL --use_lora
#python finetune-simple.py --training_samples 500 --model $MODEL

python finetune-simple.py --training_samples 1000 --model $MODEL --use_lora
python finetune-simple.py --training_samples 1000 --model $MODEL 

python finetune-simple.py --training_samples 5000 --model $MODEL --use_lora
python finetune-simple.py --training_samples 5000 --model $MODEL 

python finetune-simple.py --training_samples 10000 --model $MODEL --use_lora
python finetune-simple.py --training_samples 10000 --model $MODEL 

python finetune-simple.py --training_samples 20000 --model $MODEL  --use_lora
python finetune-simple.py --training_samples 20000 --model $MODEL 

python finetune-simple.py --training_samples 50000 --model $MODEL --use_lora
python finetune-simple.py --training_samples 50000 --model $MODEL 








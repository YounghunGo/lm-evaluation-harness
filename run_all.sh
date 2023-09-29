#!/bin/bash

#for batch in 1 2 4 8 16 32 64 128
for batch in 1 2 4 8 16 32 64
#for batch in 16
#for batch in 16 32 64
do
    #echo $batch
    #python main.py --model hf-causal-experimental --model_args pretrained=/home/work/llama-30b-hf,use_accelerate=True --batch_size $batch --tasks hellaswag --no_cache --device cuda > log/$batch.log
    deepspeed --num_gpus 4 main.py --model hf-causal-experimental --model_args pretrained=/home/work/llama-30b-hf,use_accelerate=False --batch_size $batch --tasks hellaswag --no_cache --device cuda > log/tp/$batch.log
done

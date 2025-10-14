#!/bin/bash

# 检查 HF_TOKEN 环境变量是否存在，如果存在则登录
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to Hugging Face with provided token..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "HF_TOKEN environment variable not set. Skipping Hugging Face login."
    echo "You might not be able to download private models."
fi


echo "Downloading Wan-AI/Wan2.2-Animate-14B..."
huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B


echo "Downloading black-forest-labs/FLUX.1-Kontext-dev..."
huggingface-cli download black-forest-labs/FLUX.1-Kontext-dev --local-dir ./FLUX.1-Kontext-dev

echo "Moving FLUX.1-Kontext-dev to Wan2.2-Animate-14B/process_checkpoint..."
mv ./FLUX.1-Kontext-dev ./Wan2.2-Animate-14B/process_checkpoint

exec "$@"

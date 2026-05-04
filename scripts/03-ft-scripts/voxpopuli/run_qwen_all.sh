#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/01_run_qwen_ft_eval_data.sh"
bash "$SCRIPT_DIR/01_run_qwen_ft_combined.sh"
bash "$SCRIPT_DIR/02_merge_lora_qwen.sh"

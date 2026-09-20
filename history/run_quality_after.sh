#!/usr/bin/env bash
while kill -0 27330 2>/dev/null; do sleep 5; done
/home/ben/qwen-lab/venv/bin/python /home/ben/qwen-lab/scripts/quality_screen.py --model qwen-lab --long-context --out /home/ben/qwen-lab/results/quality-fp8-mtp3.json > /home/ben/qwen-lab/logs/quality-fp8-mtp3.log 2>&1

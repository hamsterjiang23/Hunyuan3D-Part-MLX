@echo off
setlocal
title Hunyuan3D-Part CUDA benchmark - keep open
cd /d "%~dp0\.."
set "OUTPUT=artifacts\official_paper_protocol_cuda_200_float64_bs8"
if not exist "%OUTPUT%" mkdir "%OUTPUT%"

set "PYTHONUNBUFFERED=1"
".venv\Scripts\python.exe" scripts\benchmark_p3sam_partobjaverse.py ^
  --backend cuda ^
  --dataset models\PartObjaverse-Tiny ^
  --weights models\p3sam.safetensors ^
  --output "%OUTPUT%" ^
  --upstream .upstream\hunyuan3d-part ^
  --points 100000 ^
  --prompts 400 ^
  --prompt-batch-size 4 ^
  --seed 42 ^
  --official-fps-start ^
  --no-clean-mesh ^
  --postprocess ^
  --postprocess-threshold 0.95 >> "%OUTPUT%\benchmark.log" 2>&1
if errorlevel 1 exit /b %errorlevel%

".venv\Scripts\python.exe" scripts\recompute_p3sam_benchmark_metrics.py ^
  --benchmark "%OUTPUT%" ^
  --dataset models\PartObjaverse-Tiny >> "%OUTPUT%\postprocess.log" 2>&1
if errorlevel 1 exit /b %errorlevel%

".venv\Scripts\python.exe" scripts\summarize_p3sam_benchmark.py ^
  --records "%OUTPUT%\records.jsonl" ^
  --metadata models\PartObjaverse-Tiny\PartObjaverse-Tiny_semantic.json ^
  --output "%OUTPUT%\paper_summary.json" >> "%OUTPUT%\postprocess.log" 2>&1
if errorlevel 1 exit /b %errorlevel%

".venv\Scripts\python.exe" scripts\audit_p3sam_benchmark.py ^
  --benchmark "%OUTPUT%" ^
  --dataset models\PartObjaverse-Tiny ^
  --metadata models\PartObjaverse-Tiny\PartObjaverse-Tiny_semantic.json ^
  --backend cuda ^
  --output "%OUTPUT%\audit.json" >> "%OUTPUT%\postprocess.log" 2>&1
exit /b %errorlevel%

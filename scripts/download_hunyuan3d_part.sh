#!/usr/bin/env bash
set -euo pipefail

endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
repo="${HUNYUAN_REPO:-tencent/Hunyuan3D-Part}"
output_dir="${HUNYUAN_WEIGHTS_DIR:-/Users/mt/hamster/models/Hunyuan3D-Part}"
base_url="${endpoint}/${repo}/resolve/main"

files=(
  config.json
  conditioner/config.json
  conditioner/conditioner.safetensors
  model/config.json
  model/model.safetensors
  p3sam/config.json
  p3sam/p3sam.safetensors
  scheduler/config.json
  shapevae/config.json
  shapevae/shapevae.safetensors
)

mkdir -p "${output_dir}"

for relative_path in "${files[@]}"; do
  destination="${output_dir}/${relative_path}"
  mkdir -p "$(dirname "${destination}")"
  printf 'downloading %s\n' "${relative_path}"
  curl \
    --location \
    --fail \
    --retry 5 \
    --retry-delay 3 \
    --connect-timeout 20 \
    --continue-at - \
    --output "${destination}" \
    "${base_url}/${relative_path}"
done

printf 'download complete\n'
find "${output_dir}" -maxdepth 3 -type f -exec stat -f '%N %z' {} \;
shasum -a 256 \
  "${output_dir}/conditioner/conditioner.safetensors" \
  "${output_dir}/model/model.safetensors" \
  "${output_dir}/p3sam/p3sam.safetensors" \
  "${output_dir}/shapevae/shapevae.safetensors"

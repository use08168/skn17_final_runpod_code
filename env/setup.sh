cd /workspace/skn17_final_runpod_code
cat << 'EOF' > env/setup.sh
#!/usr/bin/env bash
set -e

PROJECT_ENV_NAME="skn17_final_env"
KERNEL_DISPLAY_NAME="skn17_final_env (Runpod)"

echo "[setup] 프로젝트 환경: ${PROJECT_ENV_NAME}"

MINICONDA_DIR="$HOME/miniconda3"
CONDA_BIN="${MINICONDA_DIR}/bin/conda"

# -----------------------------
# 1) conda 존재 여부 확인
# -----------------------------
if [ ! -x "${CONDA_BIN}" ]; then
  echo "[ERROR] ${CONDA_BIN} 을(를) 찾을 수 없습니다."
  echo "        먼저 Miniconda를 설치해야 합니다."
  echo "        예시:"
  echo "          cd /workspace"
  echo "          curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh"
  echo "          bash miniconda.sh -b -p \$HOME/miniconda3"
  exit 1
fi

# -----------------------------
# 2) env 존재 여부 확인 후 생성/재사용
# -----------------------------
if "${CONDA_BIN}" env list | grep -q "^${PROJECT_ENV_NAME} "; then
  echo "[setup] 기존 env 발견 → ${PROJECT_ENV_NAME} 재사용"
else
  echo "[setup] 새 env 생성 → ${PROJECT_ENV_NAME}"
  "${CONDA_BIN}" create -n "${PROJECT_ENV_NAME}" python=3.10 -y
fi

# -----------------------------
# 3) env 안에서 pip 업그레이드
# -----------------------------
echo "[setup] env 내부에서 pip 업그레이드"
"${CONDA_BIN}" run -n "${PROJECT_ENV_NAME}" python -m pip install --upgrade pip

# -----------------------------
# 4) PyTorch(CUDA 12.1) 설치
#    (requirements.txt 에서 torch/torchvision/torchaudio 는 제거된 상태여야 함)
# -----------------------------
echo "[setup] PyTorch (CUDA 12.1) 설치"
"${CONDA_BIN}" run -n "${PROJECT_ENV_NAME}" python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.5.1" torchvision torchaudio

# -----------------------------
# 5) 나머지 requirements 설치
# -----------------------------
if [ -f "env/requirements.txt" ]; then
  echo "[setup] env/requirements.txt 기반 패키지 설치"
  "${CONDA_BIN}" run -n "${PROJECT_ENV_NAME}" python -m pip install -r env/requirements.txt
else
  echo "[WARN] env/requirements.txt 를 찾을 수 없습니다."
fi

# -----------------------------
# 6) 추가 패키지 설치
# -----------------------------
echo "[setup] audio_separator 설치"
"${CONDA_BIN}" run -n "${PROJECT_ENV_NAME}" python -m pip install audio_separator

echo "[setup] bitsandbytes 업그레이드"
"${CONDA_BIN}" run -n "${PROJECT_ENV_NAME}" python -m pip install -U bitsandbytes

# -----------------------------
# 7) Jupyter 커널 등록 (env 안에서 실행)
# -----------------------------
echo "[setup] Jupyter 커널 등록"
"${CONDA_BIN}" run -n "${PROJECT_ENV_NAME}" python -m ipykernel install \
  --user --name "${PROJECT_ENV_NAME}" --display-name "${KERNEL_DISPLAY_NAME}"

# -----------------------------
# 8) 완료 메시지
# -----------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
PIPELINE_DIR="${REPO_ROOT}"

echo
echo "[DONE] 기본 환경 설정 완료."
echo " - conda env      : ${PROJECT_ENV_NAME}"
echo " - 커널 이름      : ${KERNEL_DISPLAY_NAME}"
echo " - pipeline       : ${PIPELINE_DIR}"
echo " - 추가 패키지    : audio_separator, bitsandbytes"
echo
echo "⚠️  중요: 실제 경로는 /workspace/skn17_final_runpod_code/baseball_pipeline 입니다."
echo "   심볼릭 링크를 사용하지 않으므로 모든 코드에서 실제 경로를 사용하세요."
EOF

chmod +x env/setup.sh

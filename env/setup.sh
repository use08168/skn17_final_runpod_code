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
  echo "        (cd /workspace && curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && bash miniconda.sh -b -p \$HOME/miniconda3)"
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
# 3) env 안에서 pip 설치 (conda run 사용)
# -----------------------------
echo "[setup] env 내부에서 pip 업그레이드"
"${CONDA_BIN}" run -n "${PROJECT_ENV_NAME}" python -m pip install --upgrade pip

if [ -f "env/requirements.txt" ]; then
  echo "[setup] env/requirements.txt 기반 패키지 설치"
  "${CONDA_BIN}" run -n "${PROJECT_ENV_NAME}" python -m pip install -r env/requirements.txt
else
  echo "[WARN] env/requirements.txt 를 찾을 수 없습니다."
fi

# -----------------------------
# 4) Jupyter 커널 등록 (역시 env 안에서 실행)
# -----------------------------
echo "[setup] Jupyter 커널 등록"
"${CONDA_BIN}" run -n "${PROJECT_ENV_NAME}" python -m ipykernel install --user --name "${PROJECT_ENV_NAME}" --display-name "${KERNEL_DISPLAY_NAME}"

# -----------------------------
# 5) /workspace/baseball_pipeline 심볼릭 링크 생성
# -----------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
PIPELINE_DIR="${REPO_ROOT}/baseball_pipeline"
LINK_TARGET="/workspace/baseball_pipeline"

if [ -d "${PIPELINE_DIR}" ]; then
  if [ ! -e "${LINK_TARGET}" ]; then
    echo "[setup] ${LINK_TARGET} → ${PIPELINE_DIR} 심볼릭 링크 생성"
    ln -s "${PIPELINE_DIR}" "${LINK_TARGET}"
  else
    echo "[setup] ${LINK_TARGET} 이미 존재 → 링크 생성 스킵"
  fi
else
  echo "[WARN] 레포 안에 baseball_pipeline 디렉토리가 없습니다."
fi

echo
echo "[DONE] 기본 환경 설정 완료."
echo " - conda env : ${PROJECT_ENV_NAME}"
echo " - 커널 이름 : ${KERNEL_DISPLAY_NAME}"
echo " - pipeline  : ${PIPELINE_DIR} (→ ${LINK_TARGET})"
EOF

chmod +x env/setup.sh

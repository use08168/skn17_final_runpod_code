# fish-speech/tools/vqgan/extract_vq.py
# 아래 전체 코드로 변경

import os
import subprocess as sp
import sys
import time
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import click
import numpy as np
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

# ==============================
# fish_speech.utils.file 의존 제거 버전
# 여기서 직접 AUDIO_EXTENSIONS / list_files / load_filelist 정의
# ==============================

AUDIO_EXTENSIONS = [
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".wma",
    ".aac",
]

def list_files(root: str | Path, extensions, recursive: bool = True, sort: bool = False):
    root = Path(root)
    exts = {e.lower() for e in extensions}
    files: list[str] = []

    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(str(p))
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                files.append(str(p))

    return sorted(files) if sort else files


def load_filelist(path: Path):
    """
    filelist.txt 형식:
        /path/to/audio_1.wav|meta1|meta2...
        /path/to/audio_2.wav
    처럼 되어 있을 때, [ [audio_path, ...], ... ] 리스트를 리턴.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            rows.append(parts)
    return rows


# register eval resolver
OmegaConf.register_new_resolver("eval", eval)

backends = torchaudio.list_audio_backends()
if "ffmpeg" in backends:
    backend = "ffmpeg"
else:
    backend = "soundfile"

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{extra[rank]} - <level>{message}</level>"
)
logger.configure(extra={"rank": f"RANK: {RANK} / {WORLD_SIZE}"})
logger.remove()
logger.add(sys.stderr, format=logger_format)


# ==============================
# VQ 모델 로더
# ==============================
@lru_cache(maxsize=1)
def get_model(
    config_name: str = "modded_dac_vq",
    checkpoint_path: str = "checkpoints/openaudio-s1-mini/codec.pth",
    device: str | torch.device = "cuda",
):
    # Hydra 설정 로드
    with initialize(version_base="1.3", config_path="../../fish_speech/configs"):
        cfg = compose(config_name=config_name)

    # 모델 인스턴스 생성
    model = instantiate(cfg)

    # ---- device / map_location 정리 ----
    if isinstance(device, str):
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA 디바이스가 요청되었지만 사용 불가입니다. "
                "VQ 인코더를 CPU로 로드합니다."
            )
            model_device = torch.device("cpu")
        else:
            model_device = torch.device(device)
    elif isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA 디바이스가 요청되었지만 사용 불가입니다. "
                "VQ 인코더를 CPU로 로드합니다."
            )
            model_device = torch.device("cpu")
        else:
            model_device = device
    else:
        logger.warning("알 수 없는 device 인자입니다. CPU로 강제합니다.")
        model_device = torch.device("cpu")

    map_location = model_device

    # ---- 체크포인트 로드 ----
    state_dict = torch.load(
        checkpoint_path,
        map_location=map_location,
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(model_device)

    # model에 device 속성이 없을 수 있어서 붙여 둠
    if not hasattr(model, "device"):
        model.device = model_device

    logger.info(f"Loaded model on {model_device}")
    return model


# ==============================
# 배치 처리
# ==============================
@torch.inference_mode()
def process_batch(files: list[Path], model) -> float:
    wavs = []
    audio_lengths = []
    new_files = []
    max_length = 0
    total_time = 0.0

    device = getattr(model, "device", next(model.parameters()).device)

    for file in files:
        try:
            wav, sr = torchaudio.load(str(file), backend=backend)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            continue

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # 모델 디바이스로 이동 후 리샘플
        wav = wav.to(device)
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)[0]

        total_time += len(wav) / model.sample_rate
        max_length = max(max_length, len(wav))

        wavs.append(wav)
        audio_lengths.append(len(wav))
        new_files.append(file)

    files = new_files

    if not wavs:
        return 0.0

    # Pad to max length
    for i, w in enumerate(wavs):
        if len(w) < max_length:
            wavs[i] = torch.nn.functional.pad(w, (0, max_length - len(w)), "constant")

    audios = torch.stack(wavs, dim=0)[:, None]  # [B, 1, T]
    audio_lengths = torch.tensor(audio_lengths, device=device, dtype=torch.long)

    # Encode
    indices, feature_lengths = model.encode(audios, audio_lengths)
    outputs = indices.detach().cpu().numpy()

    for file, length, feature in zip(files, feature_lengths, outputs):
        feature = feature[:, : int(length)]
        with open(file.with_suffix(".npy"), "wb") as f:
            np.save(f, feature)

    return total_time


# ==============================
# CLI main
# ==============================
@click.command()
@click.argument("folder")
@click.option("--num-workers", default=1)
@click.option("--config-name", default="modded_dac_vq")
@click.option(
    "--checkpoint-path",
    default="checkpoints/openaudio-s1-mini/codec.pth",
)
@click.option("--batch-size", default=64)
@click.option("--filelist", default=None, type=Path)
def main(
    folder: str,
    num_workers: int,
    config_name: str,
    checkpoint_path: str,
    batch_size: int,
    filelist: Path,
):
    if num_workers > 1 and WORLD_SIZE != num_workers:
        assert WORLD_SIZE == 1, "You should either use SLURM or this launcher, not both"

        logger.info(f"Spawning {num_workers} workers")

        if torch.cuda.is_available():
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible_devices is None:
                visible_devices = list(range(torch.cuda.device_count()))
            else:
                visible_devices = visible_devices.split(",")
        else:
            visible_devices = [""]

        processes = []
        for i in range(num_workers):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(visible_devices[i % len(visible_devices)])
            env["SLURM_PROCID"] = str(i)
            env["SLURM_NTASKS"] = str(num_workers)

            processes.append(
                sp.Popen(
                    [sys.executable] + sys.argv.copy(),
                    env=env,
                )
            )

        for p in processes:
            p.wait()

        logger.info(f"All workers finished")
        return

    # Worker
    logger.info(f"Starting worker")
    if filelist:
        files = [i[0] for i in load_filelist(filelist)]
    else:
        files = list_files(folder, AUDIO_EXTENSIONS, recursive=True, sort=False)

    print(f"Found {len(files)} files")
    files = [Path(f) for f in files if not Path(f).with_suffix(".npy").exists()]

    total_files = len(files)
    files = files[RANK::WORLD_SIZE]
    logger.info(f"Processing {len(files)}/{total_files} files")

    total_time = 0.0
    begin_time = time.time()
    processed_files = 0
    model = get_model(config_name, checkpoint_path)

    for n_batch, idx in enumerate(range(0, len(files), batch_size)):
        batch = files[idx : idx + batch_size]
        batch_time = process_batch(batch, model)

        total_time += batch_time
        processed_files += len(batch)

        if (n_batch + 1) % 10 == 0:
            eta = (
                (time.time() - begin_time)
                / max(processed_files, 1)
                * (len(files) - processed_files)
            )
            logger.info(
                f"Processed {processed_files} files, {total_time / 3600:.2f} hours of audio, "
                + f"ETA: {timedelta(seconds=round(eta))}s"
            )

    logger.info(
        f"Finished processing {len(files)} files, {total_time / 3600:.2f} hours of audio"
    )


if __name__ == "__main__":
    main()


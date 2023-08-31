# coding: utf-8

#환경세팅
import os
import shutil
import sys
import time    
now_dir = os.getcwd()
sys.path.append(now_dir)
import traceback, pdb
import warnings

import numpy as np
import torch
import torchaudio

import logging
import threading
from random import shuffle
from subprocess import Popen
from time import sleep
from pydub import AudioSegment

import ffmpeg
import soundfile as sf
from config import Config
from fairseq import checkpoint_utils

logging.getLogger("numba").setLevel(logging.WARNING)


tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('./waveglow/')
sys.path.append('./tacotron2/')
import numpy as np
import argparse
import librosa

from tacotron2.hparams import create_hparams
from tacotron2.model import Tacotron2
from tacotron2.layers import TacotronSTFT, STFT
from tacotron2.audio_processing import griffin_lim
from tacotron2.text import text_to_sequence
from tacotron2.utils import load_filepaths_and_text
from waveglow.denoiser import Denoiser

# 현재 시스템에서 사용 가능한 GPU를 확인하고, 해당 GPU에 대한 정보를 수집
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True 
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = ("gpu가 필요합니다")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])

hubert_model = None

# 다양한 모델과 가중치 파일을 로드하는 작업을 수행
def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

def uvr_convert_format(inp_root, format0):
    infos = []
    
    try:
        inp_root = os.path.abspath(os.path.expanduser(inp_root))

        # 입력 경로가 비어있지 않은 경우, 해당 경로 내의 모든 파일의 경로를 생성하여 paths 리스트에 저장
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)] 
            print(paths)
        # 입력 경로가 비어있는 경우, paths 리스트 내의 각 요소의 이름만 저장
        else:
            paths = [path.name for path in paths]  
        
        for path in paths:  # paths 리스트의 각 경로에 대해 반복
            inp_path = os.path.join(inp_root, path)  # 입력 경로와 파일 이름을 연결하여 입력 파일 경로 생성
            need_reformat = 1  # 포맷 변경이 필요한지 여부를 나타내는 플래그
            done = 0  # 처리 완료 여부를 나타내는 플래그
            
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")  # 입력 파일 정보를 가져옴
                # 입력 파일의 채널 수가 2이고 샘플 레이트가 44100인 경우
                if info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100":  
                    need_reformat = 0  # 포맷 변경이 필요하지 않음
                    done = 1  # 처리 완료 플래그를 설정
            except:
                need_reformat = 1  # 포맷 변경이 필요함
                traceback.print_exc() 
            
            if need_reformat == 1:
                tmp_path = "%s/%s" % (tmp, os.path.basename(inp_path))
                os.system(
                    "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
    except:
        traceback.print_exc() 
        
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
        
    print("clean_empty_cache")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return infos


# 외부 프로세스가 실행을 완료했는지 여부를 확인하는 역할
def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True

# 여러 개의 외부 프로세스가 실행을 완료했는지 여부를 확인하는 역할
def if_done_multi(done, ps):
    while 1:

        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True

# 데이터셋의 전처리를 수행하는 작업을 처리
#함수는 주어진 데이터셋 디렉토리 trainset_dir을 전처리하여 결과를 exp_dir 디렉토리에 저장
def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,}

    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    cmd = (
        config.python_cmd
        + " trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s "
        % (trainset_dir, sr, n_p, now_dir, exp_dir)
        + str(config.noparallel)
    )
    print(cmd)
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            print(f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)


# F0 피처를 추출하는 작업을 처리, 함수는 주어진 입력 인수들을 기반으로 F0 추출을 수행하고 결과를 로그에 저장
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        cmd = config.python_cmd + " extract_f0_print.py %s/logs/%s %s %s" % (
            now_dir,
            exp_dir,
            n_p,
            f0method,
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
        done = [False]
        threading.Thread(
            target=if_done,
            args=(
                done,
                p,
            ),
        ).start()
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                print(f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        print(log)
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            config.python_cmd
            + " extract_feature_print.py %s %s %s %s %s/logs/%s %s"
            % (
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        print(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            print(f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)

# 모델 훈련을 수행하는 역할을 합니다. 주어진 입력 인수들을 기반으로 파일 리스트를 생성하고, 명령어를 구성하여 모델 훈련을 실행
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # 생성 filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:", gpus16)
    if pretrained_G14 == "":
        print("no pretrained Generator")
    if pretrained_D15 == "":
        print("no pretrained Discriminator")
    if gpus16:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == ("확인") else 0,
                1 if if_cache_gpu17 == ("확인") else 0,
                1 if if_save_every_weights18 == ("확인") else 0,
                version19,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "\b",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "\b",
                1 if if_save_latest13 == ("확인") else 0,
                1 if if_cache_gpu17 == ("확인") else 0,
                1 if if_save_every_weights18 == ("확인") else 0,
                version19,
            )
        )
    try:  
        process = Popen(cmd, shell=True, cwd=now_dir)
        process.communicate()  # 명령어 실행이 완료될 때까지 대기
        if process.returncode != 0:
            raise CalledProcessError(process.returncode, cmd)
    except CalledProcessError:
        print("Error occurred while running the command:", cmd)  

def merge_audio_files(folder_path, output_file):
    combined_audio = None

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Load audio file
            audio = AudioSegment.from_file(file_path)

            # Combine audio files
            if combined_audio is None:
                combined_audio = audio
            else:
                combined_audio += audio

    # Normalize the merged audio
    combined_audio = combined_audio.normalize()

    # Export the merged audio
    combined_audio.export(output_file, format="wav")
    print(f"Merged audio saved to: {output_file}")

    # Remove source files except for merged_audio.wav
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path != output_file:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

def delete_files_in_folders(folder_paths):
    for folder_path in folder_paths:
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)

                # Check if the file path contains "/logs/mute/"
                if "/logs/mute/" in file_path:
                    continue  # Skip the file deletion

                os.remove(file_path)
                print(f"Deleted file: {file_path}")

            for dir in dirs:
                dir_path = os.path.join(root, dir)

                # Check if the directory path is "/logs/mute/"
                if dir_path == "./logs/mute":
                    continue  # Skip the directory deletion

                # Check if the directory is empty
                if not os.listdir(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"Deleted empty folder: {dir_path}")
   
# Train Preprocess
merge_audio_files(folder_path = "./test_raw/", 
                  output_file = "./test_raw/merged_audio.wav")

uvr_convert_format(inp_root = "./test_raw",
                  format0="wav")


preprocess_dataset(trainset_dir = "./test_raw", 
                  exp_dir = "test", 
                  sr = "48k", 
                  n_p = int(np.ceil(config.n_cpu / 1.5)))

extract_f0_feature(gpus = gpus,
                  n_p = int(np.ceil(config.n_cpu / 1.5)), 
                  f0method = "harvest", 
                  if_f0 = False, 
                  exp_dir = "test", 
                  version19 = "v2")

# Train 
click_train(
    exp_dir1 = "test", # 이걸 나중에 고객 ID로 하면 될듯
    sr2 = "48k",
    if_f0_3 = False,
    spk_id5 = 0,
    save_epoch10 = 200,
    total_epoch11 = 200,
    batch_size12 = 12,
    if_save_latest13 = False,
    pretrained_G14 = "./pretrained_v2/G48k.pth",
    pretrained_D15 = "./pretrained_v2/D48k.pth",
    gpus16 = 0,
    if_cache_gpu17 = False,
    if_save_every_weights18 = False,
    version19 = "v2")


# 잔여파일 제거
delete_files_in_folders([
    "./test_raw/",
    "./logs/test"
    
])

   
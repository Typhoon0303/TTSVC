# coding: utf-8

#환경세팅
import os
import shutil
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
import traceback
import warnings

import numpy as np
import torch
import torchaudio

import logging
import threading

from pydub import AudioSegment

import ffmpeg
import soundfile as sf
from config import Config
from fairseq import checkpoint_utils
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

from infer_uvr5 import _audio_pre_, _audio_pre_new
from my_utils import load_audio
from vc_infer_pipeline import VC

logging.getLogger("numba").setLevel(logging.WARNING)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()

sys.path.append('./waveglow/')
sys.path.append('./tacotron2/')

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
    gpu_info = ("no gpu")
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

# 단일 입력 오디오에 대한 음성 변환을 수행합니다. 
# 입력 오디오와 관련된 속성 및 변환 옵션을 사용하여 변환 작업을 수행하고 결과를 반환합니다.
def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
           else file_index2
        )
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=f0_file,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        
        # Save output audio to WAV file
        output_file_path = "./test_output/output.wav" 
        sf.write(output_file_path, audio_opt, tgt_sr)
        
        return (
            "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
                index_info,
                times[0],
                times[1],
                times[2],
            ),
            (tgt_sr, audio_opt),
            output_file_path,
        )
    
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

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


# VC(음성 변환) 모델을 초기화하고 설정하는 역할
def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    weight_root = "weights"
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )

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

def TTS_inference_male(text):
    
    # 모델의 경로지정
    
    tacotron2_ckpt_path = os.path.join("./male_tacotron/tc2_100000.ckpt")
    waveglow_ckpt_path =  os.path.join("./male_waveglow/wg_350000.ckpt")
    # 모델 구성
    hparams = create_hparams()

    checkpoint_path = tacotron2_ckpt_path
    model = Tacotron2(hparams).cuda()
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()

    waveglow_path = waveglow_ckpt_path
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    abs_tc2_path = os.path.abspath(checkpoint_path)
    abs_wg_path = os.path.abspath(waveglow_path)
    tc2_num = abs_tc2_path.split('_')[-1]
    wg_num = abs_wg_path.split('_')[-1]
    audio_prefix = tc2_num + "_" + wg_num
    
    # 입력 데이터 처리
    
    texts = [f"{text}"]

    dir_name = os.path.join("./test_inference_raw/")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for i in range(len(texts)):
        sequence = np.array(text_to_sequence(texts[i],
                                             ['hangul_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()

        mel, mel_postnet, _, alignment = model.inference(sequence)

        with torch.no_grad():
            audio = waveglow.infer(mel_postnet, sigma=0.666)

        audio_denoised = denoiser(audio, strength=0.01)[:, 0]

        sf.write(
            '{}/instrument.wav'.format(dir_name, audio_prefix, i),
            audio_denoised.cpu().numpy().T,
            hparams.sampling_rate
        )
    # 함수 반환
    return print('Success')

def TTS_inference_female(text):
    
    # 모델의 경로지정
    
    tacotron2_ckpt_path = os.path.join("./female_tacotron/tc2_130000.ckpt")
    waveglow_ckpt_path =  os.path.join("./female_waveglow/wg_390000.ckpt")
    
    # 모델 구성
    
    hparams = create_hparams()

    checkpoint_path = tacotron2_ckpt_path
    model = Tacotron2(hparams).cuda()
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()

    waveglow_path = waveglow_ckpt_path
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    abs_tc2_path = os.path.abspath(checkpoint_path)
    abs_wg_path = os.path.abspath(waveglow_path)
    tc2_num = abs_tc2_path.split('_')[-1]
    wg_num = abs_wg_path.split('_')[-1]
    audio_prefix = tc2_num + "_" + wg_num
    
    # 입력 데이터 처리
    
    texts = [f"{text}"]

    dir_name = os.path.join("./test_inference_raw/")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for i in range(len(texts)):
        sequence = np.array(text_to_sequence(texts[i],
                                             ['hangul_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()

        mel, mel_postnet, _, alignment = model.inference(sequence)

        with torch.no_grad():
            audio = waveglow.infer(mel_postnet, sigma=0.666)

        audio_denoised = denoiser(audio, strength=0.01)[:, 0]

        sf.write(
            '{}/instrument.wav'.format(dir_name, audio_prefix, i),
            audio_denoised.cpu().numpy().T,
            hparams.sampling_rate
        )
    # 함수 반환
    return print('Success')



#Inference 할 문장 적기

#TTS_inference_male("어머니 식사는 잘 잡수셨어요?.")
TTS_inference_female("옛날로 돌아가보면 사실 칠사 월드컵 네덜란드는 지금의 시선으로 보면 되게 형편없습니다. 보다가 계속 꺼버리고 결국엔 대부분의") # 최대 글자 수

uvr_convert_format(inp_root = "./test_inference_raw",
                  format0="wav")

# Inference
get_vc(sid = "test.pth", 
       to_return_protect0 = 0.5, 
       to_return_protect1 = 0.5)

vc_single(
    sid = 0,
    input_audio_path = "./test_inference_raw/instrument.wav", # inference 대상
    f0_up_key = 0, 
    f0_file = False, 
    f0_method = "crepe",
    file_index = "",
    file_index2 = "",
    index_rate = 0.75,
    filter_radius = 3,
    resample_sr = 0,
    rms_mix_rate = 1,
    protect = 0.33,
)
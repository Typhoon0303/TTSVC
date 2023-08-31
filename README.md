# my_childrens_voice_AI
--------------------
+ 프로젝트 목표

  단일화자의 50 문장 데이터로 TTS 모델 구현 

+ 프로젝트 설명

  TTS (Text To Speech) + SVC (Singing Voice Conversion) 모델을 접목

  TTS model의 Output TTS 문장 -> SVC model로 Voice Conversion 된 Output TTS 문장

  ex) (female) 어머니 진지는 잡수셨어요? -> (단일화자) 어머니 진지는 잡수셨어요?
--------------------
## 실행

### 환경 세팅
  + Linux - RTX 3090 (Need GPU)
    
        
        pip install -r requirement.txt
        
### Pretrained Model 다운로드
  https://drive.google.com/drive/folders/1j8GGrlrfNsXrIE-BYqFxSC1i8mSO2vWX?usp=drive_link
    
  + hubert_base.pt
  + ./pretrained_v2
  + ./male_tacotron
  + ./male_waveglow
  + ./female_tacotron
  + ./female_waveglow

### Dataset
  + 단일 화자 문장녹음 50 문장 (./test_raw)

        ./test_raw/001.wav
    
        ./test_raw/002.wav
    
        ./test_raw/003.wav
    
        .
        .
        .
    
        ./test_raw/050.wav


### Train
  + ./test_raw 안의 데이터로 학습 -> ./weight에 모델 저장 (./weight/test.pth) 
  
        
        python main_train.py
        
  
### Inference
  + male/female_tacotron + male/female_waveglow로 TTS 문장 생성
  
    -> weight/test.pth로 Voice Conversion (./test_output/output.wav)
        
        python inference.py
        
 
## Reference
+ [Tacotron2](https://github.com/NVIDIA/tacotron2)
+ [Waveglow](https://github.com/NVIDIA/waveglow)
+ [Retrieval-based-Voice-Conversion-webUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
+ [AIHUB - 감성 및 발화 스타일별 음성합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=466)

# DINOv3 × MOT17: 같은 보행자 내부의 지점 매칭

![Status: code and CPU tests complete](https://img.shields.io/badge/status-code%20ready%20%7C%20GPU%20evaluation%20pending-yellow)

## 목표와 현재 상태

같은 사람이 찍힌 두 프레임에서, 첫 프레임의 옷이나 가방에 고른 물리적 지점을 두 번째 프레임에서도 찾을 수 있을까? 사전 학습된 **DINOv3 ViT-S/16**의 패치 특징을 추출해 코사인 유사도가 가장 높은 패치를 예측점으로 사용한다. 인물 ROI와 정답점은 **사람이 예측을 보기 전에 표시**한다. 이 실험은 사람 검출·ID 추적·학습·가상 피팅이 아니다.

코드와 CPU 단위 테스트는 준비됐다. **실제 MOT17 영상·DINOv3 가중치를 이용한 추론, 성능 숫자와 데모는 아직 생성되지 않았다.** 재현 절차는 아래 순서대로 진행한다.

## 입력 영상과 출처

[MOTChallenge의 MOT17 training set](https://motchallenge.net/data/MOT17/) 중 `MOT17-09-FRCNN`을 이용한다. 공식 표의 시퀀스 사양은 1920×1080, 30fps, 525프레임이다. **실제로 제공받아 검사한 `MOT17-09-FRCNN-raw.mp4`의 비디오 스트림은 960×540, 30fps, 525프레임(17.5초)**이다. 따라서 이 MP4에 원본 1920×1080의 좌표를 그대로 적용하면 안 된다. `FRCNN`은 MOT17의 공개 검출 결과 종류를 구별하는 이름이고 **이 코드가 Faster R-CNN을 실행한다는 뜻은 아니다**. 같은 사람의 박스/ID 주석(`gt.txt`)도 이용할 수 있지만 **옷의 같은 한 점에 대한 주석은 없다**. 따라서 점 정답은 직접 표시한다. 동영상 MP4와 원본 `img1/*.jpg`의 프레임 대응이 완전히 동일하다고 임의로 가정하지 않는다.

## 독립 프로젝트 구조

```text
dino-mot17/
  mot17_dinov3_point_match.py   단일 점 눈으로 확인
  mot17_annotate_pairs.py       점 정답 선표시
  mot17_dinov3_batch.py         모델 한 번 로드, 전 점 평가
  tests/                       CPU 단위 테스트
  results/                      실행 뒤에만 생성, Git에 올리지 않음
```

`deep-skin` 저장소와 데이터를 공유하지 않는다. 저장소에는 영상, 모델 가중치, 접근 토큰, 사용자가 표시한 `pairs.json`, 실행 결과를 포함하지 않는다.

## Windows PowerShell: 첫 실행

PowerShell에서 새 폴더에 클론하거나 이 저장소의 파일을 압축 해제한 뒤 진행한다. `>` 앞에 `PS`가 보이는지 확인한다.

```powershell
cd "$HOME\Documents\GitHub\dino-mot17"
py -3.11 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m ensurepip --upgrade
& $py -m pip install --upgrade pip
```

**CUDA 지원 PyTorch**는 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 Windows·Pip·Python·현재 CUDA 빌드를 골라 표시되는 명령으로 설치한다. 명령을 실행한 후:

```powershell
& $py -m pip install -r requirements.txt
& $py -c "import cv2, numpy, torch, transformers; assert torch.cuda.is_available(), 'CUDA 없음'; print(torch.cuda.get_device_name(0))"
& $py -m unittest discover -s tests -p 'test_*.py' -v
```

기존에 로그인한 Hugging Face 계정에서 [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) 접근 승인이 필요하다. 새로운 가상환경의 `hf` 명령으로 필요할 때 로그인한다. **토큰 문자열을 코드·Git·콘솔 캡처에 넣지 않는다.**

```powershell
& .\.venv\Scripts\hf.exe auth whoami
# 로그인이 안 되어 있으면: & .\.venv\Scripts\hf.exe auth login
```

## 데이터 배치와 파일 검사

이 저장소의 `data/` 아래에 MP4를 복사하거나, 다운로드 경로를 변수에 지정한다. 원본 프레임이 있다면 공식 MOT17 training ZIP을 풀고 `MOT17-09-FRCNN\img1\000001.jpg` 경로를 확인한다.

```powershell
$video = "$HOME\Downloads\MOT17-09-FRCNN-raw.mp4"
Test-Path $video
```

`False`라면 실제 파일 이름/위치를 확인한다. 아래 `61:76`은 **예시 번호**이므로 그 두 프레임에 동일한 인물이 보이는지 영상을 먼저 보고 조정한다. MOT 파일의 프레임 번호는 **1부터 시작**한다.
첨부된 MP4를 직접 살펴본 결과 `61:76`에는 검은 옷과 가방을 든 보행자가 두 프레임 모두 보인다. 인물이 가까워지며 가방 위치가 변하므로 첫 시연 후보로 적절하다. 실제 점 정답은 확대 화면에서 직접 확인해 표시한다. `91:106`도 인물이 지속해서 보이지만 가림과 시점 변화가 커서 별도 난이도 표본으로 취급한다.

## 1. 한 쌍 눈으로 확인하기

```powershell
& $py mot17_dinov3_point_match.py --video $video --t0 2 --t1 2.5 --output results\single_demo
```

두 프레임에서 같은 사람의 상체/가방 ROI를 직접 드래그한다. A에서 특징 있는 점을 클릭하고 B에서 같은 물리적 점을 클릭한다. 예측을 보기 전에 B의 정답을 고른다. `results\single_demo\result.json`, `match.jpg`, `similarity_heatmap.jpg`를 열어 점이 잘못 지정되지 않았는지 본다. 단일 점은 *데모*이고 실험 성능으로 일반화하지 않는다.

## 2. 여러 점 정답을 먼저 표시하기

```powershell
& $py mot17_annotate_pairs.py --video $video --frames 61:76 91:106 --points-per-pair 3 --output pairs.json
```

장면이 맞지 않으면 번호를 바꾼다. 같은 사람의 ROI를 양쪽에서 고른 다음 식별 가능한 옷 무늬·가방 로고의 **같은 지점**을 고른다. 가려져서 보이지 않는 지점은 평가 표본에서 제외하고 이유를 기록한다. 이 작업은 모델의 예측 결과를 보기 전에 끝내야 선택 편향을 줄일 수 있다.
기존 `pairs.json`이 있으면 실수로 중복 주석을 만들지 않도록 실행을 멈춘다. 추가 표시를 명시적으로 원할 때만 `--append`를 붙인다.

원본 JPG 프레임으로 평가할 경우, MP4에서 만들었던 점 좌표를 재사용하지 말고 `--sequence-dir`로 **다시 표시**한다.

```powershell
$seq = "$HOME\Downloads\MOT17\train\MOT17-09-FRCNN"
Test-Path "$seq\img1\000061.jpg"
& $py mot17_annotate_pairs.py --sequence-dir $seq --frames 61:76 91:106 --points-per-pair 3 --output pairs_original.json
```

## 3. 배치 평가

```powershell
& $py mot17_dinov3_batch.py --video $video --pairs-json pairs.json --dry-run
& $py mot17_dinov3_batch.py --video $video --pairs-json pairs.json --output results\mot17_vits16_448
Get-Content results\mot17_vits16_448\summary.json
```

원본 프레임을 쓸 때는 위 명령의 `--video $video --pairs-json pairs.json`을 `--sequence-dir $seq --pairs-json pairs_original.json`으로 **둘 다** 바꾼다. 모델은 한 번만 로드하고 중복 ROI 특징은 캐시한다. 그림은 `match_0000.jpg`(A의 빨간 시작점, B의 초록 예측점과 파란 정답 원)와 `heatmap_0000.jpg`(같은 레터박스 좌표의 상대 유사도) 형태로 나온다.

## 4. 수치와 결론

- `summary.json`: 점 개수, 프레임 쌍 수, PCK@0.05/0.10, 원본 픽셀 오차 중앙값, 왕복 일치율, 단순 기준선 성능.
- `points.csv`: 각 점의 두 프레임·ROI·정답·예측·오차·기준선 결과.
- **PCK@0.10**: 예측과 정답의 거리 ÷ 대상 ROI의 긴 변이 0.10 이하인 점의 비율. 계산상 0.10이므로 10% 허용 범위이며 정답 확률이 아니다.
- `baseline_static_pck_at_010`: ROI 안의 상대 위치가 그대로라고 가정한 기준선. 짧은 간격에서 강할 수 있다.
- `cycle_consistent_strict`와 `cycle_consistent_within_one_patch`: A→B→A의 엄격/완화 판정. 완화 기준을 기존 수치와 섞지 않는다.

발표에는 성능 숫자와 함께 성공 그림 1~2장, 실패 그림 1~2장, 표본 선택법과 한계를 넣는다. 목표 표본 30쌍 × 5점 = 150점은 **권장 확장 계획이지 현재 확보한 실험 결과가 아니다.** 비교 모델을 추가하면 같은 점 주석·입력 크기·평가식으로 재계산한다. DINOv2는 토큰 구조/전처리가 다를 수 있으므로 이 코드의 모델 이름만 바꿔 비교하지 않는다.

## 재현 상태

- [x] 독립 저장소 구조, 지표 보정, 여백 검사, 다점 평가, 간단한 기준선, CPU 테스트
- [ ] 실제 MOT17에서 동일 인물·점 주석을 확보
- [ ] RTX 4080에서 DINOv3 추론 후 `summary.json`과 그림 검수
- [ ] 동일한 점으로 후속 비교 모델 평가 및 발표 자료 정리

실제 파일이 없는 환경에서 결과 숫자를 추측하거나 완료로 표시하지 않는다.

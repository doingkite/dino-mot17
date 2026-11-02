# 공간비전 후속 실험: DINOv3 × SIFT 평면 정합

## 질문과 실험 범위

**카메라 시점이 바뀌었을 때, 사전학습된 DINOv3 ViT-S/16의 패치 특징으로 추정한 평면 변환은 SIFT 기준선과 비교해 얼마나 정확한가?** MOT17 실험은 같은 사람의 수동 ROI 안에서 0.5초 뒤 한 지점을 찾았다. 이번에는 서로 다른 시점의 *장면 전체*를 정합하고 기하학적 오차를 평가한다. 이것은 **평면 호모그래피** 실험이며 카메라의 미터 단위 3D 위치·깊이·SLAM 성공률을 측정하지 않는다.

데이터: [HPatches 공식 full image sequences](https://github.com/hpatches/hpatches-dataset#full-image-sequences), [직접 ZIP 링크](https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip). 공식 설명의 압축 파일 크기는 약 **1.3GB**이다. `hpatches-release`의 65×65 패치 데이터와 혼동하지 않는다. 이미지는 같은 장면 1번부터 6번까지, `H_1_2` 등의 파일은 원본 이미지 1→상대 이미지의 정답 호모그래피이다. `v_*`는 시점 변화, `i_*`는 조명 변화 시퀀스다. 데이터 사용 시 원본 데이터 출처 및 인용 정보를 함께 표기한다.

## Windows PowerShell에서 데이터 받기

기존 `dino-mot17`의 `.venv`, CUDA PyTorch, Hugging Face DINOv3 접근 권한을 재사용한다. `data/`와 `results/`는 Git에서 제외되어 있다.

```powershell
cd "$HOME\Documents\GitHub\dino-mot17"
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& .\.venv\Scripts\hf.exe download vbalnt/hpatches hpatches-sequences-release.zip --repo-type dataset --local-dir data\hpatches-download
Expand-Archive -Path data\hpatches-download\hpatches-sequences-release.zip -DestinationPath data\hpatches -Force
Get-ChildItem data\hpatches -Directory | Select-Object -First 10 Name
```

압축 해제 결과에 `v_*` 폴더들이 직접 들어 있는 디렉터리를 `$dataset`으로 지정한다. ZIP이 중첩 디렉터리를 만들었으면 다음 후보를 확인한다.

```powershell
$dataset = "data\hpatches\hpatches-sequences-release"
Test-Path "$dataset\v_*\1.ppm"
```

`Test-Path` 결과가 `False`라면 다음으로 실제 `v_*` 위치를 찾고 `$dataset`을 그 **상위 폴더**로 바꾼다.

```powershell
Get-ChildItem data\hpatches -Directory -Recurse -Filter 'v_*' | Select-Object -First 3 FullName
```

## 실행 순서

먼저 설치·파일 배치·호모그래피를 읽을 수 있는지 확인한다. `--dry-run`은 DINOv3를 로드하지 않는다.

```powershell
& $py -m unittest discover -s tests -p 'test_*.py' -v
& $py hpatches_spatial_eval.py --data $dataset --categories v --max-sequences 3 --dry-run
```

작은 실제 데이터 파일럿(정렬된 시점 변화 시퀀스 3개 × 대상 이미지 5장 = 15쌍)을 실행한다. *보고할 결과*로 쓰기 전에 뒤의 전체 시점 변화 평가를 수행한다.

```powershell
& $py hpatches_spatial_eval.py --data $dataset --categories v --max-sequences 3 --output results\hpatches_pilot
Get-Content results\hpatches_pilot\summary.json
Get-ChildItem results\hpatches_pilot\*.jpg | Select-Object -First 6 Name
```

파일럿이 정상이라면 사전 지정된 **전체 `v_*` 시퀀스**로 실행한다. `--max-sequences`를 주지 않는다. 조명 변화 `i_*`를 함께 보고 싶으면 `--categories v i`로 별도 출력 폴더를 쓴다.

```powershell
& $py hpatches_spatial_eval.py --data $dataset --categories v --output results\hpatches_viewpoint_full --visualize 10
Get-Content results\hpatches_viewpoint_full\summary.json
Import-Csv results\hpatches_viewpoint_full\pairs.csv | Sort-Object sequence, target, method | Select-Object -First 12
```

SIFT만 먼저 설치·데이터·기하학을 검사할 수도 있다. 이는 **DINOv3 비교 결과가 아니다**.

```powershell
& $py hpatches_spatial_eval.py --data $dataset --categories v --max-sequences 3 --methods SIFT --output results\hpatches_sift_smoke
```

### 추가: 제안한 대응점 자체의 정확도

업데이트된 코드는 기존 추정과 별개로 각 방법이 제안한 **모든 대응점**을 HPatches 정답 호모그래피로 채점한다. 먼저 DINOv3/SIFT가 대응점과 호모그래피를 **정답 없이** 구한 뒤, 각 소스점의 정답 목표 위치를 투영해 제안한 목표점까지의 거리를 잰다. 정답 위치가 대상 사진 밖이면 잘못된 대응으로 센다. `pairs.csv`에는 `correct_matches_5px`, `correct_matches_10px`, `correct_matches_16px`와 각 `match_precision_*`가 추가된다. `summary.json`의 `match_quality`에는 전체 대응점으로 가중한 micro precision과 이미지 쌍별 precision을 동일 가중 평균한 수치가 함께 나온다. 이 수치는 **정확도(precision)이지 재현율(recall)이 아니다**. DINOv3 패치 중심과 SIFT의 소수점 좌표 정밀도 차이를 해석할 때 5/10/16px을 모두 보고, 16px만 골라 강조하지 않는다.

옛 결과 파일을 덮어쓰지 않도록 새 경로로 전체 `v_*` 295쌍을 재실행한다. GPU에 저장된 사전학습 모델을 다시 불러 추론하지만 **추가 학습은 하지 않는다**.

```powershell
cd "$HOME\Documents\GitHub\dino-mot17"
$py = ".\.venv\Scripts\python.exe"
$dataset = "data\hpatches\hpatches-sequences-release"
& $py -m unittest discover -s tests -p 'test_*.py' -v
& $py hpatches_spatial_eval.py --data $dataset --categories v --output results\hpatches_match_quality_v --visualize 10
$r = Get-Content results\hpatches_match_quality_v\summary.json -Raw | ConvertFrom-Json
$r.match_quality
$r.summary
```

업데이트된 예제 그림의 가는 연결선은 **초록=정답 투영과 5px 이내, 빨강=5px 초과**이며 시각화를 위해 처음 30개만 보여준다. 큰 윤곽은 종전대로 파랑=정답 호모그래피, 초록=추정 호모그래피다. 빨강 선이 많이 보이면서 호모그래피가 어긋나면 초기 대응점 오류 가능성이 높다. 선의 5px 정오와 최종 사각형 오차는 서로 다른 지표다.

기존 실험에서 관찰된 시점 변화 59개 장면·295쌍의 중앙값 오차는 DINOv3 12.45px, SIFT 0.29px(448 좌표계)였다. **추가 대응점 지표의 실제 값은 새 실행 결과를 확인하기 전까지 빈칸으로 둔다.** 조명 변화 57개 장면·285쌍은 항등변환 기준선도 투영오차 중앙값 0px이므로, 조명 불변성을 이 호모그래피 오차만으로 주장하지 않는다.

## 코드의 실제 파이프라인

1. 동일한 두 원본 이미지를 가로·세로 비율을 유지한 채 **448×448 레터박스**로 변환하고, 여백·좌표 변환 행렬을 보관한다. SIFT와 DINOv3 모두 이 이미지를 입력받는다.
2. DINOv3는 가중치를 바꾸지 않고 패치마다 특징 벡터를 추출한다. 여백 패치를 제외하고 **상호 최근접 코사인 매칭**, 영역당 최대 8개로 공간 분산을 확보한다. SIFT는 동일한 입력 이미지에서 키포인트·기술자를 추출하고 Lowe ratio 0.8로 매칭한다.
3. 각 방법이 선택한 대응점으로 **동일한 3px RANSAC 임계값**의 호모그래피를 따로 추정한다. 네 점 미만이거나 행렬 추정 실패 시 해당 쌍을 실패로 남긴다.
4. **두 방법의 추정이 끝난 뒤** HPatches 정답 `H_1_X`를 448 좌표계로 변환한다. 실제로 두 이미지에서 보이는 격자 지점들에 정답·추정 변환을 적용하고 **투영 위치 차이 중앙값(px)**을 구한다.
5. `summary.json`: 방법별 전체 쌍 수·정합 성공 수·정합된 쌍의 오차 중앙값·**전체 쌍 분모**의 success@5px/10px, 대상 이미지 번호 2..6별 분해, 동일 이미지 쌍의 양 방법 승패. `pairs.csv`: 각 이미지 쌍과 방법별 대응점 수·RANSAC 내점 수·오차. `*.jpg`: 왼쪽 원본, 오른쪽 대상, 대응선 30개 이내, 파랑 정답 윤곽·초록 추정 윤곽.

**주의:** `success@5px`는 이 프로젝트에서 정의한 *가시 공통 영역 격자 투영 오차 중앙값 ≤5 모델 픽셀*의 쌍 비율이다. 원 논문의 HPatches 공식 MMA, corner error, 3D 거리 지표와 동일하다고 주장하지 않는다. 16픽셀 패치 중심의 DINOv3와 SIFT 키포인트의 정밀도가 다르므로 SIFT가 소규모 평면 정합에서 유리할 수도 있다. 결과가 그럴 경우에도 실패 조건과 정밀도 차이를 그대로 보고한다.

## 발표 구성과 검수 체크

1. 문제: MOT17은 짧은 간격에서 DINOv3와 단순 위치 기준선 모두 PCK@0.10 3/3이었다. 이 사례는 공간 정합을 검증하지 않는다.
2. 질문·데이터: 시점 변화가 있는 평면을 이미지 두 장으로 맞출 수 있는가? HPatches `v_*`, 사전에 고정한 전체 시퀀스와 모든 대상 이미지 2..6.
3. 방법: 입력·대응점·RANSAC·정답 호모그래피 평가. 코드에서 정답을 방법의 입력으로 주지 않음을 보여준다.
4. 정량: DINOv3/SIFT의 **쌍 수, 성공률, 실패 포함 정책, 정합된 쌍 오차**를 한 표로 비교한다.
5. 정성: 같은 쌍의 DINOv3/SIFT 윤곽을 나란히, 성공과 실패 각각 보여준다. 반복 무늬·시점 변화·패치 해상도를 실제 그림과 연결해 분석한다.
6. 결론: 관찰 범위를 정확히 제한한다. Unity 평면 고정 데모는 이 결과를 시각화한 후속 단계이며, 현실 세계 3D 위치를 복원한 것처럼 설명하지 않는다.

**검수 상태:** 합성 시퀀스에서 SIFT 전 과정, 모의 DINOv3 경로, 대응점 정오·집계 테스트가 통과했다. 위 기존 295쌍 수치는 사용자 RTX 4080에서 생성된 `summary.json`과 이미지로 확인했다. 이 실행 환경에는 HPatches 원본 이미지·NVIDIA GPU·DINOv3 모델 파일이 없으므로 **추가 대응점 지표의 실제 수치는 아직 측정하지 않았다.**

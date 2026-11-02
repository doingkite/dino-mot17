# DINOv3 공간 대응 실험 v2: 수정 사항과 실행 가이드

2026-11-02 검토에서 나온 수정·추가 실험을 코드로 옮긴 결과입니다. **기존 v1 명령은 그대로 동작하며 같은 수치를 냅니다**(SIFT 295쌍 재실행으로 확인: 오차 중앙값 0.293px, 5px 성공률 95.9%). 새 기능은 모두 옵션으로만 켜집니다.

- 덮어쓰기 전의 원본 파일은 `_backup_before_v2/`에 있습니다.
- CPU 테스트: `python -m unittest discover -s tests -p "test_*.py"` (27개, 모델 가중치 없이 실행)

## 1. 이미 확인한 사실 (GPU 없이 이 PC의 데이터로 확인)

### 1-1. 조명 변화(i_*) 결과는 착시

| 확인 항목 | 결과 |
|---|---|
| i_* 정답 호모그래피 `H_1_X` 중 정확히 단위행렬인 것 | **285개 중 280개** (v_*는 0개) |
| "움직임 없음"만 예측하는 기준선 (`hpatches_identity_baseline.py`) | 오차 중앙값 0.0px, 5px 이내 281/285 |
| 기존 DINOv3 i_* 결과 | 오차 ~1e-14px, 5px 성공 98.9% |

같은 크기 두 이미지의 16px 격자가 정확히 겹치기 때문에, 같은 위치의 패치끼리만 이어도 만점이 나옵니다. **i_* 결과는 발표에 쓰지 마세요.** 공정하게 평가하려면 `--target-perturb`로 다시 실행하면 됩니다(아래 3-1).

SIFT는 대상 이미지를 이동·변형해도 성능이 유지되어, 정답 합성 코드가 올바르게 동작함을 확인했습니다(`results/v2_sift_check/`).

| SIFT, i_* | 5px 성공 | 호모그래피 정확도@3px (원본 해상도) |
|---|---:|---:|
| 원래 조건 | 94.4% | 76.5% |
| 7px 이동 | 94.0% | 73.7% |
| 무작위 변형 40px | 94.7% | 75.4% |

### 1-2. 기존 v_* 결과의 통계 (`hpatches_stats.py`)

| | 오차 중앙값 [95% CI] | 5px 성공 [95% CI] | 대응점 정확도@5px |
|---|---:|---:|---:|
| DINOv3 | 12.45px [9.82, 15.20] | 18.0% [12.5, 24.4] | 25.6% |
| SIFT | 0.29px [0.23, 0.37] | 95.9% [92.5, 98.6] | 89.2% |

- 신뢰구간은 쌍이 아니라 **59개 장면 단위로 재표집**했습니다. 같은 장면의 쌍은 서로 독립이 아니기 때문입니다.
- 짝지은 검정: SIFT 우세 284쌍 / DINOv3 우세 11쌍. Wilcoxon p = 1.4e-37, 부호 검정 p = 1.0e-69.
- **DINOv3 오류의 63%는 5~16px 근처의 "근접 실수"**입니다(SIFT는 7.5%). DINOv3는 위치를 대략은 알지만 정밀하지 않다는 근거입니다.
- **DINOv3가 이긴 11쌍은 모두 SIFT가 100px 이상 완전히 실패한 쌍**입니다(v_there, v_colors, v_graffiti, v_wounded 등의 대상 4~6번). 결합 방식을 시도할 근거입니다.
- DINOv3가 가장 크게 실패한 장면: v_bark 146.9px, v_tabletop 119.2px.
- 원본 해상도 기준 SIFT 호모그래피 정확도는 @1/3/5/10px = 29.8 / 59.3 / 71.2 / 84.1%입니다(448px 화면에서 매칭했기 때문에 원본 해상도에서는 더 엄격해짐).

## 2. 바뀐 파일

| 파일 | 내용 |
|---|---|
| `hpatches_spatial_eval.py` | v2. 아래 옵션 추가. 기존 출력에 원본 해상도 지표(모서리 오차, 1~10px 평균 매칭 정확도)도 추가 |
| `hpatches_stats.py` | 신규. 장면 단위 bootstrap 신뢰구간, Wilcoxon·부호 검정, 오류 분해, 장면별·대상별 표 |
| `hpatches_sweep.py` | 신규. 한 번에 한 요인만 바꾸는 실험 묶음 실행 + 비교표(중단 후 이어서 실행 가능) |
| `dense_correspondence.py` | 신규. SPair·TAP-Vid 공용(DINOv3 조밀 특징, 조밀 SIFT, 좌표 변환) |
| `spair_eval.py` | 신규. SPair-71k 의미 대응(PCK@0.05/0.10/0.15), DINOv3 · SIFT · 박스 위치 기준선 |
| `tapvid_davis_eval.py` | 신규. TAP-Vid-DAVIS 점 대응(1~16px 정확도), DINOv3 · SIFT · 정지 기준선 |
| `mot17_gt_tools.py` | 신규. MOT17 정답 박스로 ROI 쌍 자동 생성 / 같은 사람 찾기(rank-1) 평가 |
| `mot17_annotate_pairs.py` | `--roi-pairs` 추가: 박스는 정답에서 가져오고 점만 클릭 |
| `mot17_dinov3_batch.py` | 정지 기준선 PCK@0.05, 기준선 오차 중앙값, 프레임 간격별 요약 추가 |
| `tests/test_hpatches_v2.py`, `tests/test_benchmarks.py` | 신규 테스트 |

`hpatches_spatial_eval.py`의 새 옵션:

| 옵션 | 의미 |
|---|---|
| `--dino-ransac-threshold 8` | DINOv3 매칭에만 다른 RANSAC 기준 적용(SIFT는 `--ransac-threshold` 유지) |
| `--refine quadratic` / `soft` | 매칭된 패치 주변 3×3 유사도로 패치 안 위치 보정 |
| `--target-perturb shift --perturb-px 7` | 대상 이미지를 장면별 고정 난수로 이동, 정답에 합성 (i_* 공정 평가용) |
| `--target-perturb homography --perturb-px 40` | 네 모서리를 최대 40px 무작위로 움직인 변형 |
| `--methods DINOguidedSIFT` | DINOv3 호모그래피 주변 24px(`--guide-radius`) 안에서만 SIFT 매칭 |
| `--methods SIFTdinoFilter` | SIFT 매칭 중 DINOv3 유사도 상위 5 패치와 맞는 것만 남김 |
| `--methods LoFTR LightGlue` | 학습형 매칭 기준선 (`pip install kornia` 필요, 첫 실행 때 가중치 다운로드) |
| `--visualize-pairs v_bark:3 v_apprentices:6` | 원하는 쌍 시각화 (PPT 실패 사례용) |

## 3. 실행 순서 (Windows PowerShell, RTX 4080 기준)

```powershell
cd "$HOME\Documents\GitHub\dino-mot17"
$py = ".\.venv\Scripts\python.exe"
$dataset = "data\hpatches\hpatches-sequences-release"
& $py -m unittest discover -s tests -p "test_*.py"
```

### 3-1. 원인 분리 묶음 (가장 먼저, 약 30~60분)

```powershell
& $py hpatches_sweep.py --data $dataset --preset core --output results\sweep
Get-Content results\sweep\sweep_summary.md
```

`core`에는 다음 실험이 들어 있습니다.
- SIFT 기준 1회
- DINOv3 RANSAC 기준 3/6/8/12/16px
- 입력 해상도 448/672/896px
- 위치 보정 없음/quadratic/soft
- 결합 방식 2종
- i_* 세 조건(원래, 7px 이동, 무작위 변형)

다음 단계에 필요하면 `--preset models`(ViT-B/L, Hugging Face에서 각 모델 접근 승인 필요)와 `--preset learned`(kornia 필요)를 실행합니다.

결과 해석에는 통계 스크립트를 씁니다.

```powershell
& $py hpatches_stats.py results\sweep\v_SIFT_448\pairs.csv R3=results\sweep\v_DINO_r3\pairs.csv R12=results\sweep\v_DINO_r12\pairs.csv --reference SIFT --output results\sweep\stats_ransac
& $py hpatches_stats.py results\sweep\v_hybrid\pairs.csv --reference SIFT --output results\sweep\stats_hybrid
```

PPT 실패 사례 그림을 다시 뽑으려면 다음을 실행합니다.

```powershell
& $py hpatches_spatial_eval.py --data $dataset --categories v --output results\figures --visualize 0 --visualize-pairs v_bark:3 v_apprentices:6 v_tabletop:5 v_there:4
```

### 3-2. SPair-71k: DINOv3가 유리할 것으로 예상되는 무대

1. 데이터를 받습니다: <https://cvlab.postech.ac.kr/project/SPair-71k/data/SPair-71k.tar.gz> (약 231MB).
   - 주석 파일 이름에 `:`가 들어 있어 Windows 기본 압축 해제가 실패할 수 있습니다. **7-Zip**으로 풀면 `_`로 바뀌어 저장되며, 코드는 파일 이름에 의존하지 않으므로 그대로 동작합니다.
2. `data\SPair-71k\` 아래에 `JPEGImages\`, `PairAnnotation\test\`가 있으면 됩니다.

```powershell
& $py spair_eval.py --data data\SPair-71k --split test --dry-run
& $py spair_eval.py --data data\SPair-71k --split test --max-pairs 300 --output results\spair_pilot
& $py spair_eval.py --data data\SPair-71k --split test --output results\spair_test
& $py spair_eval.py --data data\SPair-71k --split test --refine quadratic --output results\spair_test_refine
```

- 결과는 `summary.json`의 `overall`(전체), `by_category`(범주별), `by_viewpoint_variation`(시점 차이별)에 있습니다.
- `bboxprior`는 이미지를 보지 않고 박스 안 상대 위치만 옮기는 기준선입니다. DINOv3가 이 기준선을 넘는지가 핵심입니다.

### 3-3. TAP-Vid-DAVIS: MOT17 수동 3점 대체

데이터: `tapvid_davis.zip` ([tapnet 저장소](https://github.com/google-deepmind/tapnet) 안내에 따라 <https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip>). 압축을 풀면 `tapvid_davis.pkl`이 나옵니다.

```powershell
& $py tapvid_davis_eval.py --pkl data\tapvid_davis\tapvid_davis.pkl --dry-run
& $py tapvid_davis_eval.py --pkl data\tapvid_davis\tapvid_davis.pkl --max-videos 3 --max-tracks 50 --output results\tapvid_pilot
& $py tapvid_davis_eval.py --pkl data\tapvid_davis\tapvid_davis.pkl --output results\tapvid_davis
```

- `by_frame_gap`에서 프레임 간격이 커질수록 정지 기준선이 무너지고 DINOv3가 버티는지를 봅니다.
- 가림 여부(occlusion)는 평가하지 않으므로, 논문의 AJ(Average Jaccard)와 직접 비교하면 안 됩니다.

### 3-4. MOT17 확장 (선택)

정답 박스(`gt.txt`)는 MOT17 **train** 폴더에만 있습니다. <https://motchallenge.net/data/MOT17/>에서 받아 `data\MOT17\train\MOT17-09-FRCNN\`(`img1\`, `gt\gt.txt`)만 있으면 됩니다.

```powershell
$seq = "data\MOT17\train\MOT17-09-FRCNN"
& $py mot17_gt_tools.py reid --sequence-dir $seq --gaps 15 30 60 120 --output results\mot17_reid
& $py mot17_gt_tools.py roi-pairs --sequence-dir $seq --gaps 15 30 60 --per-gap 10 --output roi_pairs.json
& $py mot17_annotate_pairs.py --sequence-dir $seq --roi-pairs roi_pairs.json --points-per-pair 3 --output pairs_gt_rois.json
& $py mot17_dinov3_batch.py --sequence-dir $seq --pairs-json pairs_gt_rois.json --output results\mot17_gt_rois
```

- `reid`는 같은 사람 찾기 rank-1 정확도입니다. DINOv3를 움직임 기준선(IoU), 색 히스토그램, 무작위 기대값과 비교합니다.
- 점 평가는 30쌍 × 3점 = 90점이 되고, `by_gap`에서 PCK@0.05와 정지 기준선을 간격별로 비교할 수 있습니다.

## 4. 피부 실험 (deep-skin 저장소)

바뀐 파일은 `notebooks/dy/` 아래 네 개입니다.
- `train_dinov3.py`: `--pooling`, `--embedding-cache-dir`, `--test-csv` 추가. 기본값은 v1과 같습니다.
- `train_dinov3_finetune.py`: 신규. 마지막 N개 블록 미세조정 + 증강 + 가중 샘플러
- `make_subject_test_split.py`: 신규
- `run_dinov3_seeds.py`: 신규

**테스트 분할은 이미 만들어 두었습니다.** 기존 검증셋 107명을 주름 등급으로 층화해 피험자 단위로 반씩 나눴고, 학습셋·서로 간 피험자 겹침이 없음을 확인했습니다.
- `data/processed/dy_fg_val2_metadata.csv`: 53명 530장. 조기 종료용입니다.
- `data/processed/dy_fg_test_metadata.csv`: 54명 540장. 마지막에 한 번만 채점합니다.

```powershell
cd "$HOME\Documents\GitHub\deep-skin"
$py = ".\.venv\Scripts\python.exe"
$val2 = "data\processed\dy_fg_val2_metadata.csv"; $test = "data\processed\dy_fg_test_metadata.csv"
# 고정 특징 3종 x 시드 3개 (임베딩은 풀링별 1회만 추출)
& $py notebooks\dy\run_dinov3_seeds.py --configs frozen_pooler frozen_mean frozen_cls_mean --val-csv $val2 --test-csv $test --model-revision 114c1379950215c8b35dfcd4e90a5c251dde0d32
# ResNet 조건에 맞춘 비교: 증강+샘플러(고정), 마지막 2/4블록 미세조정
& $py notebooks\dy\run_dinov3_seeds.py --configs frozen_aug_cls_mean finetune_last2 finetune_last4 --val-csv $val2 --test-csv $test --model-revision 114c1379950215c8b35dfcd4e90a5c251dde0d32
Get-Content checkpoints\trained\dy_forehead_glabella\seed_summary\seed_summary.md
```

- `frozen_pooler`는 0.7067을 낸 v1 설정과 같습니다(가중 샘플러 없음).
- 공정한 비교를 위해 ResNet-50도 같은 val2/test, 같은 시드 3개로 다시 돌리세요(`train.py`의 `--val-csv` 사용).
- `train_dinov3_finetune.py`와 `run_dinov3_seeds.py`는 이 환경에 torch가 없어 **문법 검사와 코드 검토만 했습니다.** 처음에는 `--max-train-rows 200 --max-val-rows 100 --epochs 1`로 짧게 돌려 보세요.

## 5. 아직 검증하지 못한 것

- DINOv3가 실제로 들어가는 실행 경로 전부: 가중치에 접근할 수 없어서입니다. 가짜 인코더로 데이터 흐름은 테스트했지만, 실제 수치는 위 명령을 실행해야 나옵니다.
- LoFTR / LightGlue 경로: kornia API 기준으로 작성했고 실행 테스트는 하지 못했습니다.
- SPair·TAP-Vid는 합성 데이터로만 끝까지 실행해 봤습니다. 실제 파일 형식이 다르면 `--dry-run` 단계에서 오류가 납니다.

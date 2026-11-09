# Pre-registration — verbatim text (Korean)

The three stages below are copied verbatim from the author's internal research memo (`ideas/2026-11-01_benchmark_resolution.md`), in which each stage was written, with its date, before the corresponding experiment was launched. The memo was not under version control, so there is no commit history; the launch records in `TIMELINE.md` date the start of each experiment independently. `PREREGISTRATION_en.md` is an English translation.


---

# Stage: Killing experiment — written 2026-11-01

## 죽이기 실험 (사전 등록, 2026-11-01)
### 설계
- 데이터: MPI Sintel training 23장면·1,064프레임(기존 캐시). 전처리·해상도(256)·학습 예산(60 epoch)·옵티마이저는 JOSA A와 동일.
- **fold 설계**: 5-fold 장면 CV를 서로 독립인 fold 배정으로 R=5회 반복(배정 시드 201..205). 각 반복에서 23장면이 테스트에 정확히 한 번씩 들어가
  (fold 크기 5,5,5,4,4), 학습셋은 18~19장면. 23장면 × 5반복 = 장면당 5개의 서로 다른 학습 구성에서 점수 확보.
  (기존 5개 파티션은 테스트에 16/23 장면만 등장 — `SUMMARY.json` 확인 — 하므로 재사용 불가. 전 장면 커버가 필요한 이유.)
- 아키텍처: **H1 주 실험 DirectCNN**(런당 14.7분, T4 실측), **H2 보조 UNet**(24.0분). 시드 S=2 (42, 43).
- 런 수·비용: 5×5×2 = 50런/아키텍처. DirectCNN 12.2 GPU-h + UNet 20.0 GPU-h = 32.3 GPU-h; 6 T4 샤드면 벽시계 ≈ 5.5h.
- **구현 변경(필수)**: ① `make_loaders`에 `protocol='cvfold'`(테스트 장면 목록 명시) 추가, ② 평가를 **프레임별** 저장으로 변경
  (`reduction='none'`, 테스트 로더 `shuffle=False` 확인 후 `dataset.meta`로 장면·프레임 매핑), ③ 프레임별 R_MSE·R_PSNR 외에
  Sintel intrinsic 문헌의 표준 지표 **si-MSE·LMSE·DSSIM**을 함께 저장(정의는 Grosse 2009·Narihira 2015 원문을 읽고 구현 — 기억으로 쓰지 않음),
  ④ 결과 파일명에 버전 태그(`cv_<arch>_r<rep>_f<fold>_s<seed>_v1.json`; probe_B 덮어쓰기 사고 재발 방지), ⑤ 셋업 스크립트 `set -e`.
- 1차 추정량: 장면 u 점수 y_u = PSNR(장면 내 프레임 MSE 평균) (JOSA A 집계 관례를 장면 단위로). 2차: 프레임 PSNR 평균, si-MSE 등.

### 가설과 판정 규칙
- **H1 (분해)**: 교차 랜덤효과 적합 후 테스트 쪽 예측 분산 V_test = (1−5/23)·Ŝ²_γ/5 를 JOSA A 관측 σ²_part(DirectCNN 0.927)와 대조.
  점유율 s = V_test/0.927 (부트스트랩 CI 포함).
  - s ≥ 0.7 → **테스트 쪽 지배**: 기여의 무게 = 설계(전 장면 K-fold로 소거·층화). 가설지 진입.
  - s ≤ 0.3 → **학습 쪽/상호작용 지배**: 기여의 무게 = "일반화는 어느 장면을 봤는가에 달렸다"의 정량화. 가설지 진입(프레이밍 전환).
  - 0.3 < s < 0.7 → 혼합: 두 성분을 함께 보고하는 해상도 공식. 가설지 진입.
- **사망 조건**: 장면 주효과가 불안정 — σ̂²_scene/(σ̂²_scene+σ̂²_{γδ}+σ̂²_ε) < 0.5 (장면 난이도가 학습 구성에 따라 뒤바뀜).
  이때는 "단위 난이도로부터의 해상도"에 안정된 추정 대상이 없다 → 경험적 재표집만 남는 Level 2 → 부정 결과 기록 후
  프레이밍 재검토(구변). 숫자를 맞추기 위한 사후 모형 변경 금지.
- **H2 (상호작용, UNet−DirectCNN)**: S²_{Δγ} 추정 → 짝지은 비교 MDE 계산. 예측: JOSA A가 본 "6쌍 중 3쌍 뒤집힘"이 성립하려면
  S²_{Δγ}가 시드 분산(0.06~0.07)보다 커야 한다. 작으면 뒤집힘의 원인은 학습 쪽 → 보고.
- **H3 (사전 예측 대조)**: Ŝ²_γ의 95% CI가 위 표의 함의값(DirectCNN 5.92, UNet 3.35)을 포함하는지 기록. 포함 여부 자체가 결과.

### 정직성 규칙
- null·사망 결과는 결과 파일과 세션 로그에 그대로 남기고 논문에 보고한다.
- 집계 수치는 전부 `results/benchmark_resolution/*.json`에서 스크립트로 생성(손 입력 금지). 재현 PASS는 정의 검증이 아니다.
- 기존 325런과 새 런은 하드웨어(T4)·예산이 같아야 대조 가능 — 바뀌면 새 런에서 기존 5파티션을 1개 이상 재현해 보정한다.


---

# Stage: Main experiment (update: primary estimand redefined; H4–H6 added) — written 2026-11-02

### 본실험 설계 (사전 등록 갱신, 2026-11-02)
- 4아키텍처(direct·unet·transformer·proposed) × R=5 × 5fold × S=3 = 300런(≈124 GPU-h, 12 T4 ≈ 10.4h). 1차 추정 대상 si-MSE·LMSE·DSSIM(장면 등가중), 2차 PSNR.
- 추가 가설: H4 게이지 — 예측 albedo를 장면별 최적 게인으로 정렬한 뒤의 PSNR(= si-PSNR)에서 상호작용이 PSNR 대비 ≥50% 감소하면 "게인 효과" 해석 확정.
  H5 — TransUNet의 시드 분산(JOSA A ρ=1.3 예외)이 장면 수준에서도 큰지. H6 — 층화 파티션(장면별 si-MSE 난이도 3층)이 무작위 대비 V_test를 얼마나 줄이는지(시뮬레이션 + 실측 런).
- 실행 시점: 사용자 승인 후. 2편 CVPR 제출(11월 중순)·JOSA A 결정과 충돌하지 않는 날.

### 본실험 v2 실행 기록 (2026-11-02 09:10 KST 가동, 사용자 승인)
- 그리드 340런(무작위 300 + 층화 40), 예상 137.4 GPU-h, 16× T4 LPT 균형 샤딩(최대 8.75h). 코드 `experiment_cv2.py`, 층화표 `strata_v2.json`
  (v1 DirectCNN si-MSE 난이도, 3층 8/8/7 — 척도 불변 지표 사용 이유: v1 ICC 0.95 vs PSNR 0.65), 분석 `analyze_cv2.py`(합성 데이터로 전 경로 검증).
- 추가 통제: direct/unet seed 42·43은 v1과 동일 구성 재실행 → GPU 비결정성 바닥(같은 시드, 독립 런의 차) 측정.
- H4 구현: 프레임별 게인 충분통계(Σx·x̂, Σx̂², Σx², n)를 저장해 장면 풀링 최적 게인 PSNR을 사후 계산 → PSNR 대비 상호작용 감소율.
- H3 확장: 6개 아키텍처 쌍 × JOSA A 5파티션 = 30점 표본 외 예측(v1은 1쌍 5점, r=0.958).


---

# Stage: Optical-flow arm — written 2026-11-03

## 두 번째 벤치마크 arm — Sintel optical flow (사전 등록 2026-11-03, 사용자 결정: IJCV 타깃·flow 선택)
목적: "같은 23개 단위(장면)에서 다른 과제도 파티션이 지배하는가"를 가장 통제된 형태로 묻는다. 데이터 단위·fold 배정(seed 201–205)이
intrinsic arm과 **동일**하므로 파티션 수준 점수를 과제 간에 직접 상관시킬 수 있다.

### 설계
- 데이터: MPI-Sintel training, clean pass, 23장면 1,041쌍. GT flow·invalid·occlusions = `MPI-Sintel-training_extras.zip`(3.1GB, UW 미러
  `sintel.cs.washington.edu`; 공식 downloads 페이지에서 URL 확인). .flo 헤더(PIEH 태그·W·H·float32 u,v 교차)는 동봉 `flow_code`로 인스턴스에서 확인 후 사용.
  invalid 마스크 픽셀 제외, |gt|<400 유효(torchvision 참조 학습 스크립트 관례).
- 모델: torchvision `raft_small`(0.99M) 1차, `raft_large`(5.3M) 2차 — 둘 다 **C+T 사전학습 가중치(Raft_*_Weights.C_T_V2; FlyingChairs·FlyingThings, Sintel 미포함 → 누출 없음)에서 미세조정**.
  Sintel 표준 관행(C+T→Sintel 미세조정)을 따르되 예산을 줄인다. 시드는 데이터 순서·크롭·플립에만 작용(초기화는 공유 체크포인트) — intrinsic arm(from scratch)과 시드 요인의 성격이 다름을 명시.
- 학습 예산(고정): 3,000 iteration, batch 4, 무작위 크롭 320×448 + 수평 플립 0.5(최소 증강 — 표준 레시피의 광도·스케일 증강 생략, 절대 EPE에 영향·분산 구조 질문에는 무관함을 서술),
  AdamW lr 1e-4·wd 1e-5·OneCycle, grad clip 1.0, 시퀀스 손실 γ=0.8(참조 구현 관례), fp32, 학습 12 iteration·평가 12 iteration.
- 평가: 테스트 fold 전 쌍, 전해상도(436×1024, 하단 4px 패딩 후 크롭). 쌍별: EPE(1차), 1/3/5px 이상치율, Fl-all(EPE>3 & >5%|gt|), occluded/unoccluded EPE, GT 평균 크기(난이도 공변량), 유효 픽셀 비율; 장면별 집계는 쌍 등가중과 픽셀 가중 둘 다.
- zero-shot 기준선: C+T 가중치로 1,041쌍 전부 1회 평가 → "Sintel을 본 적 없는 모델"의 장면 난이도표(순수 테스트 쪽 공변량).
- 그리드: R=5(fold seed 201–205, intrinsic과 동일) × 5fold × S=3(42,43,44) × {small, large} = 150런 + zero-shot 2런. 층화 arm 없음(H6 결론 반영).
  런타임은 보정 인스턴스 1대로 실측 후 LPT 샤딩.

### 가설 (사전 등록)
- **F-H1**: EPE(장면 등가중)의 분산 분해에서 테스트 쪽 점유율 ≥ 0.7이면 "파티션 지배는 과제 일반적", ≤ 0.3이면 "intrinsic 특이", 사이면 혼합. 사망 조건 동일(장면 ICC < 0.5 → 장면 난이도가 안정 속성이 아님).
- **F-H2**: raft_small vs raft_large 장면×방법 상호작용 점유율. 같은 계열이므로 intrinsic의 동계열 쌍(0.00–0.20)과 같은 수준이면 "상호작용 ∝ 아키텍처 비유사성" 가설과 정합. 큰 값이면 그 가설 기각(정직 보고).
- **F-H3 (교차 과제)**: 같은 fold(rep, fold)의 intrinsic 점수(si-MSE·PSNR, 4모델 평균)와 flow EPE의 fold 수준 상관. 양의 상관(어려운 장면이 양쪽에서 어려움)이면 "장면 난이도의 공통 축"; 0 근처면 과제별 독립 난이도 → 어느 쪽이든 IJCV의 일반성 절 재료.
- **F-H4**: zero-shot 장면 난이도표가 미세조정 후 장면 난이도(fold 평균 EPE의 장면 성분)를 얼마나 설명하는가(Spearman). 높으면 "난이도는 데이터 속성", 낮으면 "학습이 난이도 순서를 재배열".
- 지표 게이지 질문은 flow에는 해당 없음(EPE는 절대 척도; 대신 1/3/5px 이상치율·Fl-all로 지표 관례 강건성 점검).

### 정직성·재현 규칙
- 수치는 `results/benchmark_resolution/results_flow_v1/*.json` → `analyze_flow.py` 생성. 손 입력 금지. 실패 런·중단은 로그와 함께 보고.
- 이 arm은 IJCV 원고의 "일반성" 절 전용. 결과가 intrinsic과 다르면(예: 시드 지배) 그대로 쓴다 — "파티션 지배는 과제·훈련 방식에 의존한다"도 발견이다.

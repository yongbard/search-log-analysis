# 검색 분석 프로젝트

## 개요
이 프로젝트는 사용자 검색 행동을 분석하고 클릭률을 예측하기 위한 다양한 머신러닝 모델을 활용한 종합적인 검색 분석 시스템입니다. LSTM, XGBoost, LightGBM, BERT 등 다양한 모델 구현과 데이터 생성, 특성 엔지니어링을 포함합니다.

## 주요 기능
- 실제와 유사한 검색 로그 데이터 생성
- 검색 분석을 위한 고급 특성 엔지니어링
- 다양한 모델 구현:
  - 시퀀스 예측을 위한 LSTM
  - 클릭 예측을 위한 XGBoost
  - 비교 분석용 LightGBM
  - 텍스트 기반 분석을 위한 BERT
- 최적의 모델 성능을 위한 하이퍼파라미터 튜닝
- A/B 테스트 시뮬레이션 기능

## 기술 스택
- Python 3.x
- PyTorch
- XGBoost
- LightGBM
- Transformers (BERT)
- pandas
- numpy
- scikit-learn

## 프로젝트 구조
```
├── main.py            # 모든 구현이 포함된 메인 스크립트
├── components/
    ├── data_generation.py    # 검색 로그 데이터 생성
    ├── feature_engineering.py # 데이터 준비 및 특성 엔지니어링
    ├── models/
        ├── lstm.py           # LSTM 모델 구현
        ├── xgboost_model.py  # XGBoost 구현
        └── bert_model.py     # BERT 모델 구현
```

## 주요 기능 설명

### 1. 데이터 생성
- `generate_search_logs()`: 다음 요소를 포함한 가상 검색 로그 데이터 생성:
  - 사용자 ID
  - 검색어
  - 카테고리
  - 타임스탬프
  - 세션 정보
  - 클릭 데이터

### 2. 데이터 준비
- `prepare_data()`: 다음을 포함한 특성 엔지니어링 수행:
  - 시간 관련 특성 (시간, 요일)
  - 텍스트 특성 (검색어 길이)
  - 순차적 특성 (카테고리 유사도)
  - 범주형 변수 인코딩

### 3. 모델 구현
- LSTM 모델 (`SearchLSTM`):
  - 사용자 검색 패턴의 시퀀스 예측
  - 커스터마이징 가능한 아키텍처 (임베딩 차원, 은닉 차원, 레이어)
  - 정규화를 위한 드롭아웃

- XGBoost 구현:
  - 클릭률 예측
  - RandomizedSearchCV를 통한 하이퍼파라미터 튜닝
  - 클릭 예측을 위한 이진 분류

### 4. A/B 테스트
- `ab_test_simulation()`: A/B 테스트 시나리오 시뮬레이션:
  - 대조군 vs 실험군
  - CTR 계산 및 비교
  - 성능 향상 측정

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

```python
# 가상 데이터 생성
df = generate_search_logs()

# 특성 준비
df = prepare_data(df)

# LSTM 모델 학습
best_lstm_params = hyperparameter_tuning_lstm(df)

# XGBoost 모델 학습
X = df[features]
y = df['clicked']
best_xgb_model = hyperparameter_tuning_xgboost(X, y)

# A/B 테스트 시뮬레이션 실행
ab_test_simulation(best_xgb_model, X, y)
```

## 필요 라이브러리
```
pandas
numpy
torch
scikit-learn
xgboost
lightgbm
transformers
```

## 참고사항
현재 프로젝트는 시연 목적으로 가상 데이터를 사용합니다. 실제 운영 환경에서는 실제 데이터 파이프라인으로 데이터 생성 함수를 대체해야 합니다.

## 향후 개선 계획
1. 실시간 예측 기능 추가
2. 더 정교한 세션 처리 구현
3. 다국어 검색 쿼리 지원 추가
4. 더 다양한 A/B 테스트 시나리오 구현
5. 모니터링 및 로깅 기능 추가

## 라이센스
MIT 라이센스
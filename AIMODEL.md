# 데이터 기반 모델 학습 과정

## 1. 이미지 분석 (Image Classification / Object Detection)
- **데이터:** 이미지 + 라벨
- **학습 과정:**
  1. 이미지 → 픽셀 텐서
  2. CNN / ViT 모델 입력
  3. 특징 추출 → Dense로 분류
  4. 예측 vs 정답 → Loss 계산 → 역전파
  5. 반복 학습

## 2. 감성 분석 (Sentiment Analysis)
- **데이터:** 문장 + 감정 라벨
- **학습 과정:**
  1. 텍스트 → 토큰화 → 임베딩
  2. RNN / LSTM / Transformer 통과
  3. 문장 벡터 → Dense로 감정 예측
  4. 예측 vs 정답 → Loss 계산 → 역전파
  5. 반복 학습

## 3. 번역 (Neural Machine Translation)
- **데이터:** 원문 문장 + 번역문
- **학습 과정:**
  1. 문장 토큰화
  2. 인코더 → 문장 의미 벡터
  3. 디코더 → 한 단어씩 번역 생성
  4. 예측 단어 vs 정답 단어 → Loss 계산 → 역전파
  5. 반복 학습

## 🔹 다른 학습 방법
- **Transfer Learning:** 사전학습 모델 활용, 마지막 레이어만 재학습  
- **Fine-tuning:** 모델 전체 또는 일부 레이어 재학습  
- **Zero-shot / Few-shot:** 예시만으로 문제 해결  
- **Multi-task Learning:** 한 모델 여러 과제 학습  
- **Self-supervised Learning:** 라벨 없이 데이터 구조 학습

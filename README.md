### 머신러닝 프로젝트: 개선된 코드와 실험 보고서

---

### **1️⃣ 개요 & EDA (탐색적 데이터 분석)**

#### 문제 정의  
이미지 분류 문제를 해결하기 위해 새로운 접근 방식을 시도하며, 주어진 데이터를 활용해 각 이미지를 올바르게 분류하는 것이 목표다. 데이터 불균형과 다양한 이미지 품질이 주요 문제로, 이를 해결하기 위한 전처리와 학습 설계가 핵심이었다.

#### 데이터셋 특성 분석  
- 데이터는 이미지 경로(`img_path`)와 라벨(`label`)로 구성된 CSV 파일.  
- 클래스는 불균형 상태이며, 일부 클래스는 샘플 수가 적음.  
- 다양한 해상도의 이미지를 포함하고 있어 전처리가 필수.  

#### 기초 통계 분석  
- 클래스별 샘플 분포를 분석해 데이터 불균형 문제 확인.  
- 데이터의 다양성을 고려해 증강 기법 설계.

---

### **2️⃣ 실험 방법**

#### **1. 데이터 전처리 과정**  
**개선된 코드**:
```python
train_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.OneOf([
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
    ], p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

#### **2. 하이퍼파라미터 설정**  
**설정 값**:
```python
CFG = {
    'IMG_SIZE': 224,            # 이미지 크기
    'EPOCHS': 10,               # 학습 에포크 수
    'LEARNING_RATE': 3e-4,      # 학습률
    'BATCH_SIZE': 32,           # 배치 크기
    'SEED': 42                  # 랜덤 시드 고정
}
```

#### 설정 설명
- **IMG_SIZE**: 모든 이미지를 224x224로 크기 조정해 입력.  
- **EPOCHS**: 10번 반복 학습을 진행.  
- **LEARNING_RATE**: 3e-4로 설정해 안정적인 학습.  
- **BATCH_SIZE**: 메모리 사용량과 학습 속도 균형을 고려해 32로 설정.  
- **SEED**: 결과 재현성을 위해 랜덤 시드 고정.

---

#### **3. 사용한 모델 설명**  
**개선된 코드**:
```python
def build_model(model_name, num_classes):
    model = create_model(model_name, pretrained=True, num_classes=num_classes)
    return model.to(device)

model_names = ['convnext_large', 'swin_large_patch4_window7_224']
models_to_train = [build_model(name, num_classes=len(le.classes_)) for name in model_names]
```

**설명**:
- 최신 아키텍처(`convnext_large`, `swin_large_patch4_window7_224`) 사용.  
- Timm 라이브러리를 활용해 모델 생성 과정을 단순화.  

---

#### **4. 손실 함수 및 학습 최적화**  

**개선된 손실 함수**:
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return focal_loss
```

**학습 최적화 코드**:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
```

#### 설정 설명
- **Focal Loss**: 클래스 불균형 문제를 해결하기 위해 도입.  
- **AdamW**: 가중치 감소를 효과적으로 처리하는 최적화 알고리즘.  
- **CosineAnnealingWarmRestarts**: 학습률을 주기적으로 리셋해 지역 최적값 탈출 유도.  

---

#### **5. 앙상블 적용**  
**개선된 코드**:
```python
def weighted_voting(models, weights, loader, device):
    ensemble_preds = []
    for model, weight in zip(models, weights):
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs in tqdm(loader):
                imgs = imgs.to(device)
                outputs = model(imgs).softmax(dim=1) * weight
                preds.append(outputs.cpu().numpy())
        ensemble_preds.append(np.vstack(preds))
    return np.sum(ensemble_preds, axis=0)

weights = [0.4, 0.3, 0.3]
predictions = weighted_voting(trained_models, weights, test_loader, device)
final_preds = predictions.argmax(axis=1)
final_preds = le.inverse_transform(final_preds)
```

**설명**:
- 두 모델(`convnext_large`와 `swin_large_patch4_window7_224`)의 예측 결과를 가중치 기반으로 통합.  
- 모델별 가중치는 각각 0.4, 0.3으로 설정.  

---

### **3️⃣ 실험 결과**

#### Epoch 1 결과  
- **F1-score**: 0.9319로 매우 높은 초기 성능 달성.  
- 데이터 증강 및 Focal Loss가 초기 학습 성능에 크게 기여.

#### 성능 평가 지표  
- **F1-score (Macro)**: 클래스 불균형 데이터를 공정하게 평가.  
- **Validation Loss**: 학습 중 과적합 여부를 모니터링.  

---

### **4️⃣ 결론**

#### 최종 결과 요약    
- Epoch 1에서 F1-score 0.9319로 뛰어난 초기 성능 기록.

#### 한계점  
- 일부 소수 클래스에서 성능이 낮음.  
- 학습 시간이 길어 실시간 응용에는 최적화 필요.  

#### 개선 방안  
- SMOTE, CutMix 등 데이터 증강 기법 추가.  
- 경량화된 모델로 학습 속도 최적화.  

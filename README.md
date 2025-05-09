# 2023 DATA CREATOR CAMP
Team DASH (KFood health Image Classification)

<br>

### 데이터셋 구성  <br>
* 음식 이미지
  * train: 33,048개  <br>
  * val: 4,210개  <br>

* 건강관리를 위한 음식 이미지
  * kfood_health_train: 14,026개  <br>
  * kfood_health_val: 1,777개  <br>

### 1-1. 각 클래스로 하는 분류 데이터셋과 데이터로더 준비
```python
# 이미지 전처리 및 데이터셋 
transform = transform.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_dataset = ImageFolder(train_dir, transform = transform)
val_dataset = ImageFolder(val_dir, transform=transform)

# 데이터로더 
batch_size = 16
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```        
<br>

### 1-1. 42개 클래스 시각화
```python
menu_folders = os.listdir(train_dir)
menu_folders = natsort.natsorted(menu_folders)
plt.figure(figsize=(15,15))

for i, trainfolder in enumerate(menu_folders):
    trainmenu_path = os.path.join(train_dir, trainfolder)
    imgfiles = os.listdir(trianmenu_path)

    for j, imgfile in enumerate(imgfiles[:1]):
        imgpath = os.path.join(trainmenu_path, imgfile)
        img = mpimg.imread(imgpath)

        name = unicodedata.normalize('NFC', trainfolder)
        plt.subplot(7, 7, i+1)
        plt.imshow(img)
        plt.title(name)
        plt.axis('off')
plt.show()
```
<img width="777" alt="image" src="https://github.com/user-attachments/assets/2227aeac-58db-404b-bc68-5c8c600f2617" />


<br> 


### 1-2. 모델 
```python
# 모델 정의
resnet18 = models.resnet18(pretrained=False)
num_classes = 42

# 마지막 Fully Connected Layer 변경
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
```
```python
# 모델 학습
num_epochs = 50
save_epochs = 1

for epoch in range(num_epochs):
    resnet18.train()
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 499:
            print(f'Epoch {epoch +1}, Batch {i+1}, Loss: {running_loss / 100: .3f}')
            running_loss = 0.0
```
```python
# Validation 데이터셋에 대한 모델 검증
if epoch % validation_epochs == 0:
   resnet18.eval()
   correct = 0
   total = 0

   with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = resnet18(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 잘못 분류된 경우 확인
            misclassified_mask = predicted != labels
            misclassified_examples.extend([(inputs[i], predicted[i].item[(), labels[i].item()) for i, is_misclassified in enumerate(misclassified_mask) if is_misclssified])

   accuracy = 100 * correct / total
   accuracy_list.append(accuracy) # accuracy 리스트에 추가
   print(f'Validation Accuracy after {epoch+1} epochs: {accuracy: .2f}%')
```

<img width="345" alt="image" src="https://github.com/user-attachments/assets/715e04ca-4f1c-45e9-ac68-1c717290995f" />

<br>

<br>

### 2-1. ResNet18 결과 비교
classification accuracy<br>
* 상위 7개 음식
1. 미역국 - 89.80%
2. 육개장 - 88.52%
3. 알밥 - 87.10%
4. 잡곡밥 - 86.41%
5. 꿀떡 - 85.58%
6. 시래기국 - 85.19%
7. 계란국 - 83.51%   
* 하위 7개 음식
1. 장어구이 - 48.89%
2. 갈비구이 - 50.59%
3. 생선전 - 51.33%
4. 고등어구이 - 51.61%
5. 감자전 - 54.55%
6. 더덕구이 - 54.74%
7. 주먹밥 - 55.75%

<br>

### 2-1. 결과 분석
**1. '구이'류 음식, 비교적 분류 정확도가 낮은 편에 속함**
<img width="1198" alt="스크린샷 2025-04-06 오후 7 47 49" src="https://github.com/user-attachments/assets/5144664a-907a-4470-89c2-4f3b0243e6b1" />

**2. 음식 형태(모양)가 비슷한 경우, 오인하는 경향 있음**
<img width="1195" alt="스크린샷 2025-04-06 오후 7 47 57" src="https://github.com/user-attachments/assets/4ad3ee1d-d191-48b7-a6a0-04faae272fd3" />

**3. 음식 색상이 비슷한 경우, 오인하는 경향 있음**
<img width="1194" alt="스크린샷 2025-04-06 오후 7 48 18" src="https://github.com/user-attachments/assets/cd0e3467-8822-48c8-b752-f252d60c6e7b" />

**4. 다양한 음식이 있는 경우, 오인하는 경향 있음**
<img width="1197" alt="스크린샷 2025-04-06 오후 7 48 30" src="https://github.com/user-attachments/assets/ef1fb17a-5016-43ba-9a43-f8770a3d3ae8" />

**5. 오인한 경우, 대개 색조와 명도 등 음식 고유 색상이 소실됨 확인**
<img width="1194" alt="스크린샷 2025-04-06 오후 7 54 28" src="https://github.com/user-attachments/assets/30d432dd-68cd-4e04-b3f2-6d5995900f64" />

<br>

### 2-2. ① 2-1의 분석을 기반으로 성능 향상을 위한 작업 수행(Augmentation 적용)
**point1)** <br>
이미지 전처리 과정 중 **<mark>밝기, 대비, 채도 등 조절</mark>** <br>

**기대효과)** <br>
데이터의  **<mark>일관성을 높여</mark>** 시각적 특징을 파악하고 <br>
더 나아가, **<mark>노이즈 감소</mark>** 를 통해 성능을 향상시키고자 함 <br>

```python
transform.RandomHorizontalFlip(), # 좌우 반전
transform.RandomVerticalFlip(),   # 상하 반전
transform.RandomRotation(90),     # 90도 회전
transform.ColorJitter(
      brightness=(0.5, 2),    # 밝기
      contrast=(0.5, 1.5),    # 대비
      seturation=(0.8, 1.5))  # 채도
transform.RandomResizedCrop(size=(240, 240),    # 잘라내고 조절할 크기
                            scale(0.8, 1.2))    # 스케일 범위(줌인 및 줌아웃 효과)
```
<br>

### 2-2. ② 2-1의 분석을 기반으로 성능 향상을 위한 작업 수행(Batch Size, ResNet 수정)
**point1)** <br>
데이터로더 설정 시,  <br>
Batch Size를 16 -> **<mark>32로 수정</mark>**  <br>

**기대효과)** <br>
기존보다 더 많은 데이터를 한 번에 처리하여 <br>
**<mark>학습 시간을 단축</mark>** 하고자 함

```python
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
```
<br>

**point2)** <br>
ResNet18 -> **<mark>ResNet50으로 수정</mark>**  <br>

**기대효과)** <br>
기존보다 **<mark>더 깊은 신경망을 통해</mark>** 복잡한 특징을 추출하고 <br>
더욱 **<mark>다양한 정보를 학습</mark>** 시키고자 함 <br>

```python
# 모델 정의
resnet50 = models.resnet50(pretrained=False)

# Reset Parameters (가중치 초기화)
def reset_parameters(module):
    if hasattr(module, 'reset_parameters'):
       module.reset_parameters()

resnet50.apply(reset_parameters)

# 마지막 Fully Connected Layer 변경
num_classes = 512 # 클래스 수
resnet50.fc = nn.Linear(2048, num_classes)
```
<img width="505" alt="image" src="https://github.com/user-attachments/assets/e656e59d-1b53-4f11-952f-26b876c91dc3" />


<br>

### 2-3. ①, ② 의 성능 경향 파악

<img width="923" alt="스크린샷 2025-04-06 오후 9 14 30" src="https://github.com/user-attachments/assets/10c6022b-bcb5-4fd2-80bc-b77520ff0633" />

----
<br>

### 3-1. 건강관리를 위한 음식 이미지 데이터셋과 데이터로더 준비
```python
# 데이터셋 경로(kfood_health)
data_root = "/content/drive/MyDrive"
train_dir = os.path.join(data_root, "kfood_health_train")
val_dir = os.path.join(data_root, "kfood_health_val")

transform = transform.Compose([
    transform.Resize((224, 224)),
    transform.ToTensor(),
    transform.Normalize((0.485, 0.4546, 0.496), (0.229, 0.224, 0.225))]) # ImageNet mean/std 사용

train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset = ImageFolder(val_dir, transform=transform)

# 데이터로더 
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle =True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
```

<br>

### 3-1. 건강관리를 위한 음식 이미지 데이터 각 클래스별 하나씩 시각화
<img width="777" alt="image" src="https://github.com/user-attachments/assets/f990e6f5-bcda-42c7-a7ee-2cb88c8eb40c" />

### 3-1. 모델
```python
resnet18 = models.resnet18(pretrained=False)

num_classes = 42
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
```
<br>

### 3-1. Task에 따른 Transfer Learning 전략
<img width="737" alt="스크린샷 2025-04-06 오후 9 46 19" src="https://github.com/user-attachments/assets/2612ebc6-4c64-4821-8412-d0dc7b28c7c4" />

<br>

### 3-1. 학습이 안된 모델의 첫번째 layer의 filter 확인
```python
for w in resnet18.parameters():
    w = w.data.cpu()
    print(w.shape)
    break
# 가중치 renormalization
min_w = torch.min(w)
w1 = (-1/(2 * min_w)) * w + 0.5

# grid
grid_size = len(w1)
x_grid = [w1[i] for i in range(grid_size)]
x_grid = torchivision.utils.make_grid(x_grid, nrow=8, padding=1)

plt.figure(figsize=(10, 10))
imshow(x_grid)
```
* 초기화된 모델의 첫번째 layer의 filter
<img width="405" alt="image" src="https://github.com/user-attachments/assets/448e8666-d94c-409e-8dad-586cc3069afe" />

<br>

### 3-1. 학습된 모델의 파라미터 적용 및 linear probing 진행
Resnet 18-50epochs에서의 accuracy  시각화 - 상승과 하락을 반복하며 **<mark>93~4%**</mark> 에서 수렴
  
<img width="666" alt="image" src="https://github.com/user-attachments/assets/3e33623c-f877-4920-b699-abc7b2075958" />

<br>
<br>

## 결과 분석
모델 평가 및 결과 분석

<img width="498" alt="image" src="https://github.com/user-attachments/assets/b03c34bb-73d3-4c9d-aecf-c984f1df1ef9" />
<br>

**Validation Accuracy :** 
**<mark>94.16%**</mark>  
<br>
13개 중 **11개 음식 분류 정확도** **<mark>95% 이상**</mark>  <br>

**모든 클래스**의 **분류 정확도** **<mark>70% 이상**</mark>  <br>



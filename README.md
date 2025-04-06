# 2023 DATA CREATOR CAMP
Team DASH (Food Image Classification)

한국 음식 이미지 분류 모델  
<br>

### 데이터셋 구성  <br>
* 음식 이미지
  * train: 33,048개  <br>
  * val: 4,210개  <br>

* 건강관리를 위한 음식 이미지
  * kfood_health_train: 14,026개  <br>
  * kfood_health_val: 1,777개  <br>


* 42개 클래스 시각화
```python
menu_folders = os.listdir(train_dir)
menu_folders = natsort.natsorted(menu_folders)
plt.figure(figsize=(15,15))

for i, trainfolder in enumerate(menu_folders):
    trainmenu_path= os.path.join(train_dir, trainfolder)
    imgfiles = os.listdir(trianmenu_path)

    for j, imgfile in enumerate(imgfiles[:1]):
        imgpath = os.path.join(trainmenu_path, imgfile)
        img= mpimg.imread(imgpath)

        name = unicodedata.normalize('NFC', trainfolder)
        plt.subplot(7, 7, i+1)
        plt.imshow(img)
        plt.title(name)
        plt.axis('off')
plt.show()
```

<img width="359" alt="image" src="https://github.com/user-attachments/assets/89d64164-29f2-475a-966f-17c9735f7821" />


* 모델 정의 
ResNet18 <br>

#### 결과 비교
* classification accuracy
<br>

상위 7개 음식                                    
1. 미역국 - 89.80%
2. 육개장 - 88.52%
3. 알밥 - 87.10%
4. 잡곡밥 - 86.41%
5. 꿀떡 - 85.58%
6. 시래기국 - 85.19%
7. 계란국 - 83.51%
   
하위 7개 음식
1. 장어구이 - 48.89%
2. 갈비구이 - 50.59%
3. 생선전 - 51.33%
4. 고등어구이 - 51.61%
5. 감자전 - 54.55%
6. 더덕구이 - 54.74%
7. 주먹밥 - 55.75%

<br>

#### 결과 분석
* '구이'류 음식, 비교적 분류 정확도 낮은 편 속함
  

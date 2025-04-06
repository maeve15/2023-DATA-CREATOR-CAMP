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

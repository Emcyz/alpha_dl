# alpha_dl
알파프로젝트

사용 데이터
original image
![original_img](https://user-images.githubusercontent.com/11612272/170933049-7af0a6c9-6f8a-4538-9611-9b333095183e.png)
heatmap
![heatmap](https://user-images.githubusercontent.com/11612272/170933043-5ea4759a-69fc-49b8-bc34-5e5cb39b7933.png)
original image에서 heatmap 부분을 추출한 masked img
![masked_img](https://user-images.githubusercontent.com/11612272/170933048-525e1ef5-d192-41c0-bd3c-845b90216395.png)


# 모델 구조
## Image -> Heatmap (img2heatmap model)
![img2heatmap](https://user-images.githubusercontent.com/11612272/170927646-0d2d5fa2-ccf6-465d-abf1-5a8bf2a3b67f.png)
original image를 사용하여 차량 heatmap을 추출한다.



## 
## 
## Heatmap(masked img) -> Center (hm2center model)
![hm2center](https://user-images.githubusercontent.com/11612272/170927347-bd5bfb91-ed19-4b1d-9c9e-bd8bea669f04.png)

# alpha_dl
알파프로젝트

사용 데이터
original image
![original_img](https://user-images.githubusercontent.com/11612272/170933049-7af0a6c9-6f8a-4538-9611-9b333095183e.png)
heatmap
![heatmap](https://user-images.githubusercontent.com/11612272/170933043-5ea4759a-69fc-49b8-bc34-5e5cb39b7933.png)
original image에서 heatmap 부분을 추출한 masked img, 차량에 해당되는 부분이다.
![masked_img](https://user-images.githubusercontent.com/11612272/170933048-525e1ef5-d192-41c0-bd3c-845b90216395.png)
각 차량의 중심을 나타내는 heatmap center이다 
![hm_center](https://user-images.githubusercontent.com/11612272/170957872-42e8c8dd-495b-4fb1-b432-a0d59cd2c85d.png)


# 모델 구조
## Image -> Heatmap (img2heatmap model)
![img2heatmap](https://user-images.githubusercontent.com/11612272/170927646-0d2d5fa2-ccf6-465d-abf1-5a8bf2a3b67f.png)
original image를 사용하여 차량 heatmap을 추출한다.



## Heatmap(masked img) -> Center (hm2center model)
![hm2center](https://user-images.githubusercontent.com/11612272/170927347-bd5bfb91-ed19-4b1d-9c9e-bd8bea669f04.png)
추출한 heatmap로 masked_img를 만듦, masked_img를 입력으로 하여 각 차량의 중심점을 추출함

* NMS (Non Max Supression)
```python
output = self.sigmoid(output)
hmax = F.max_pool2d(output, (3, 3), stride=1, padding=1)
output = output * (hmax == output).float()
```
> 특징점 주변 낮은 값의 픽셀은 걸러지고 가장 큰 값의 픽셀만 남는다.

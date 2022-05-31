# alpha_dl
알파프로젝트

## 사용 데이터

original image
![original_img](https://user-images.githubusercontent.com/11612272/170933049-7af0a6c9-6f8a-4538-9611-9b333095183e.png)
heatmap
![heatmap](https://user-images.githubusercontent.com/11612272/170933043-5ea4759a-69fc-49b8-bc34-5e5cb39b7933.png)
original image에서 heatmap 부분을 추출한 masked img, 차량에 해당되는 부분이다.
![masked_img](https://user-images.githubusercontent.com/11612272/170933048-525e1ef5-d192-41c0-bd3c-845b90216395.png)
각 차량의 중심을 나타내는 heatmap center이다. gaussian 식을 사용하여 처리했다.
![hm_center](https://user-images.githubusercontent.com/11612272/170957872-42e8c8dd-495b-4fb1-b432-a0d59cd2c85d.png)


# 모델 구조
## Image -> Heatmap (img2heatmap model)
![img2heatmap](https://user-images.githubusercontent.com/11612272/170927646-0d2d5fa2-ccf6-465d-abf1-5a8bf2a3b67f.png)
* original image를 사용하여 차량 heatmap을 추출한다.
* 입력과 같은 크기의 heatmap을 출력한다.
* BasicBlock64부터 tanh 직전까지는 Feature Pyramid Network 구조이다.
### 전형적인 FPN의 구조
<img src="https://user-images.githubusercontent.com/11612272/170960130-62a53278-4430-4de0-86e8-682d07f24cf4.png"  width="600" height="450">



## Heatmap(masked img) -> Center (hm2center model)
![hm2center](https://user-images.githubusercontent.com/11612272/170927347-bd5bfb91-ed19-4b1d-9c9e-bd8bea669f04.png)


* img2heatmap model로 추출한 heatmap로 masked_img를 만들고 이를 입력으로 하여 각 차량의 중심점을 추출함
* 원본 이미지의 1/4의 크기의 heatmap_center를 출력한다.
* hm_center는 각 차량의 중심만 1이고 이외엔 0이다.
* NMS (Non Max Supression)
```python
output = self.sigmoid(output)
hmax = F.max_pool2d(output, (3, 3), stride=1, padding=1)
output = output * (hmax == output).float()
```
> 특징점 주변 낮은 값의 픽셀은 걸러지고 가장 큰 값의 픽셀만 남는다.
* Focal Loss 사용
```python
def FL_of_CornerNet(X, y, alpha=2, beta=4):
  p_inds = y.eq(1).float() # eq(1) : 1이면 True, 아니면 False.
  n_inds = (-p_inds + 1.)

  p_loss = (torch.log(X) * torch.pow(1 - X, alpha) * p_inds).sum()
  n_loss = (torch.log(1 - X) * torch.pow(X, alpha) * torch.pow(1 - y, beta) * n_inds).sum()

  p_num = p_inds.sum()

  return -(p_loss + n_loss) / p_num
```
> 배경 즉 특징점이 아닌 영역에서의 loss 값을 상대적으로 줄인다.

## 결과
![test_result](https://user-images.githubusercontent.com/11612272/171121428-b244acc4-b4ba-4581-a84b-78d94f0350af.png)
* 회색은 heatmap > 0.7, 파란색 원은 hm_center > 0.7 인 영역
* train set에선 괜찮치만 test set에선 성능이 안 좋다.
* 용량 문제로 데이터를 10000개에서 3000개로 줄인 것이 원인으로 보임

# Data-Quantization

This project contains PyTorch verison of data quantization method based on Darknet-19, Yolo V2 backbone model.
![image](https://github.com/dylee0907/Data-Quantization/assets/79738681/afafcb0e-2418-474d-820d-b72208726b48)

### **Quantization** project contains info reqarding Per-Tensor quantiztaoin using Asymmetric Method, Per-Tensor Quantization using Symmetric Method and Per-Channel Quantization.
-매핑 함수(Mapping function)는 부동소수점에서 정수 공간으로 값을 매핑한다. 

일반적으로 사용되는 매핑 함수는 다음과 같은 선형변환(Linear Transformation)이다.

### **Q(r) = round(r/S+Z)**

여기에서 **r은 입력값(FP), S는 Scaling Factor, Z는 offset**이다.

양자화된 값을 부동소수점으로 다시 복귀하기 위한 역함수(Inverse Function)는 다음과 같다.

### **r'=(Q(r)-Z)S**

r'!=r이고, 이들의 차이값을 양자화 에러(error) 또는 손실(loss)이라고 부른다. 

매핑 함수에 따라서 양자화 손실이 다를 수 있고, 이러한 손실을 최소화하는 것이 관건이다. 

딥러닝에서 양자화는 보통 정수 8비트(bitwidth)로 하는 것이 가장 보편적인데, 실제로는 정수 8비트가 아닌 고정소수점 8비트로 전환(매핑)하는 것이다. 

즉, 256개 이내의 서로 다른 소수점을 갖는 숫자로 전환(매핑)하는 것이다. 즉, 256개 이내의 서로 다른 소수점을 갖는 숫자로 전환하는 것이다.


### **Quantiztaion2** project contains info regarding Weight-only quantization method
-딥러닝 모델은 신경망 아키텍처에 가중치(weights, layer parameters)를 포함하고 있다.

딥러닝 모델에서 양자화는 가중치(weights, layer parameters)와 활성화(activations, layer output)에 대해서 가능하다.

Weights를 양자화하면 모델의 크기를 줄일 수 있다.

Activations를 양자화하면 모델 수행 속도를 향상시킬 수 있다.


### **Quantiztaion3** project contains info regarding Post-Training Dynamic Quantization
-딥러닝 모델의 각 층(layer)의 Activation을 모델의 inference 중에 quantization시키는 것을 dynamic quantization이라고 부른다.

딥러닝의 입력 Tensor value에 따라 quantization parameter(clipping range)가 달라질 수 있고, 따라서 입력 텐서값에 따라 clippling range를 보정해주는 것이 dynamic quantization이다.

각 층의 activation후에 quantization layer를 추가하면 된다.

Activation 전에 quantization을 할 수도 있지만, 이 경우 quantization threshold가 증가하고, 따라서 quantization step이 증가되어 quantization loss가 커질 수 있다.


### **Quantiztaion4** project contains info regarding Post-Training Static Quantization
-Static quantization은 각 층의 Activation에 대한 clipping range가 input tensor값에 상관없이 고정(fixed, static)되어 있는 경우이다.

입력값(예를 들어, 입력 이미지)에 따라서 각 층의 activation값이 변화하므로 sampling data를 이용해서 각 층의 activation에 대한 clipping range에 대한 수치를 수집한 후에, 최빈값(가장 빈도수가 많은 값)으로 quantization parameter를 정하게 된다.

PyTorch의 경우, 딥러닝 모델에서 각 층의 activation값을 추출하는 것은 register_forward_hook이라는 방법을 이용한다.

예를 들어, 100개의 서로 다른 입력값에 대해서 모델의 각 층의 activation값과 이에 따른 quantization parameter(scaling factor)를 구하고, quantization parameter 중에서 최빈도값을 최종 quantization parameter로 결정한 후에, 이 값을 이용해서 모델의 각 층의 activation값을 양자화하게 된다.

이 경우에 입력값에 따라 clipping range를 보정하는 과정이 딥러닝 모델의 수행 과정 중에 포함되지 않는 장점이 있지만, 입력값에 따라 정해진 clipping range가 맞지 않을 경우에는 모델의 예측값이 맞지 않을 가능성이 있다.


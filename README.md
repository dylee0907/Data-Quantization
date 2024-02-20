# Data-Quantization

This project contains PyTorch verison of data quantization method based on Darknet-19, Yolo V2 backbone model.

Quantization project contains info reqarding Per-Tensor quantiztaoin using Asymmetric Method, Per-Tensor Quantization using Symmetric Method and Per-Channel Quantization.
-매핑 함수(Mapping function)는 부동소수점에서 정수 공간으로 값을 매핑한다. 

일반적으로 사용되는 매핑 함수는 다음과 같은 선형변환(Linear Transformation)이다.

###**Q(r) = round(r/S+Z)**

여기에서 **r은 입력값(FP), S는 Scaling Factor, Z는 offset**이다.

양자화된 값을 부동소수점으로 다시 복귀하기 위한 역함수(Inverse Function)는 다음과 같다.

### **r'=(Q(r)-Z)S**

r'!=r이고, 이들의 차이값을 양자화 에러(error) 또는 손실(loss)이라고 부른다. 

매핑 함수에 따라서 양자화 손실이 다를 수 있고, 이러한 손실을 최소화하는 것이 관건이다. 

딥러닝에서 양자화는 보통 정수 8비트(bitwidth)로 하는 것이 가장 보편적인데, 실제로는 정수 8비트가 아닌 고정소수점 8비트로 전환(매핑)하는 것이다. 

즉, 256개 이내의 서로 다른 소수점을 갖는 숫자로 전환(매핑)하는 것이다. 즉, 256개 이내의 서로 다른 소수점을 갖는 숫자로 전환하는 것이다.


Quantiztaion2 project contains info regarding Weight-only quantization method
-딥러닝 모델은 신경망 아키텍처에 가중치(weights, layer parameters)를 포함하고 있다.

딥러닝 모델에서 양자화는 가중치(weights, layer parameters)와 활성화(activations, layer output)에 대해서 가능하다.

Weights를 양자화하면 모델의 크기를 줄일 수 있다.

Activations를 양자화하면 모델 수행 속도를 향상시킬 수 있다.


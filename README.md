# Data-Quantization

This project contains PyTorch verison of data quantization method based on Darknet-19, Yolo V2 backbone model.
![image](https://github.com/dylee0907/Data-Quantization/assets/79738681/afafcb0e-2418-474d-820d-b72208726b48)

### **Quantization** project contains info reqarding Per-Tensor quantiztaoin using Asymmetric Method, Per-Tensor Quantization using Symmetric Method and Per-Channel Quantization.
-A mapping function maps values from a floating-point space to an integer space.

A commonly used mapping function is the following linear transformation:

### **Q(r) = round(r/S+Z)**

**input value=r(FP), Scaling Factor=S, offset=Z**

Inverse Function for dequantizating quantized FP is same as the following equzation.

### **r'=(Q(r)-Z)S**

r'!=r, The difference between these two values is called quantization error or loss. 

Depending on the Mapping function, quantization loss could be different and it is priority to decrease the loss.

It is general to use INT 8bit for quantization in Deep learning, however it is actually mapped to 8bit fixed point format. 



### **Quantiztaion2** project contains info regarding Weight-only quantization method
-Deep learning models include weights and layer parameters within the neural network architecture.

In deep learning models, quantization is possible for both weights (layer parameters) and activations (layer outputs).

Quantizing the weights can reduce the size of the model.

Quantizing the activations can improve the execution speed of the model.


### **Quantiztaion3** project contains info regarding Post-Training Dynamic Quantization
-The process of quantizing the activation of each layer during the inference of a deep learning model is referred to as dynamic quantization.

In deep learning, the quantization parameter (clipping range) can vary depending on the input tensor value, hence the clipping range is adjusted based on the input tensor value in dynamic quantization.

A quantization layer can be added after the activation of each layer.

It is also possible to quantize before activation, but in this case, the quantization threshold may increase, resulting in a larger quantization step and potentially greater quantization loss.


### **Quantiztaion4** project contains info regarding Post-Training Static Quantization
-Static quantization is the case where the clipping range for the activation of each layer is fixed and does not depend on the input tensor values.

Since the activation values for each layer can change depending on the input (for example, an input image), numerical values for the clipping range for each layer's activation are collected using sampling data, and then the quantization parameter is determined using the mode (the value with the highest frequency).

In PyTorch, the method to extract the activation values for each layer from a deep learning model is to use register_forward_hook.

For example, one would calculate the activation values for each layer of the model and the corresponding quantization parameters (scaling factors) for 100 different input values. Then, the most frequent value among the quantization parameters is chosen as the final quantization parameter. This value is used to quantize the activation values for each layer of the model.

In this case, there is the advantage that the process of adjusting the clipping range according to the input values is not included in the execution process of the deep learning model. However, if the predetermined clipping range does not match the input values, there is a possibility that the model's predictions may be inaccurate.


# Train_small_GPT
computation:
- resource: A100 80G with 35H
- A100: 312 TFLOPS, MFU:0.5 -> 140TFLOPS
- total flops: 1.4*10^14 * 3600 * 35 = 1.76* 10^19 FLOPs

- Chinchilla Scalling law: T ~ 20*N
- flops need = 6*T*N -> N: 1B, T: 20B

memory:
- Model Weights(bf16 2 bytes): 2GB
- Gradient: 2GB
- AdamW(weight: fp32, 4bytes, Momentum: fp32, 4bytes, var: fp32, 4bytes - 12bytes):12GB 

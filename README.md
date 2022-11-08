# GPU-Accelerated zk-SNARKs

Author: Uroš Tešić\
Advisors: Karl Wüst and Moritz Schnedier\
Supervisor: Prof. Dr. Srđan Čapkun\
Degree: MSc in Computer Science\
Institution: ETH Zürich

Thesis: [PDF](https://github.com/UrosTesic/zcash-gpu-thesis/blob/299015c063bc2af757e52127605000dd743ef602/tex/main.pdf)

## Abstract

Cryptocurrencies promise a new age of money and payment systems. By distributing trust they prevent any party from controlling the flow of resources. However, this comes at a price -- most cryptocurrencies rely on a public ledger. This puts user's privacy at risk because anyone can monitor and track transactions.

The cryptocurrency Zcash provides completely private transactions by using zero-knowledge proofs (zk-SNARKs) to validate them. Unfortunately, zk-SNARKs are expensive to compute limiting their widespread adoption on computationally limited devices such as mobile phones. They are also complex to implement, preventing the development of hardware wallets for Zcash.

In this thesis, we take a look at porting computationally expensive part of zk-SNARKs to a GPU to take full advantage of the available processing power. We also explain the difficulties involved in developing OpenCL code meant to be executed on GPUs from multiple vendors. Finally, we identify the hardware constrains that need to be satisfied to achieve a significant speedup.

### Keywords:
zk-SNARKs, Zcash, OpenCL, GPU, multiexponentiation, elliptic curves, scalar multiplication

import numpy
import numpy as np
import matplotlib.pyplot as plt


class DataPlot:
    def __init__(self, data, N, lab):
        self.data = data
        self.N = N
        self.array = np.array(self.data)
        self.lnT = np.log2(self.array)
        self.gflops = 2.0e-6 * N * N * N / self.array
        self.label = lab


N = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]
simd3 = [14.67, 132.86, 472.17, 1218.92, 3025.64, 5324.17, 8301.51, 12640.88, 17698.95, 24604.94, 33652.81, 43477.43,
         53691.97, 67117.12, 82188.73, 99562.86]
parallel3 = [4.59, 33.43, 112.84, 272.09, 546.39, 926.01, 1468.20, 2203.74, 3105.16, 4324.21, 5682.25, 7473.79, 9409.63,
             11891.72, 14576.17, 17930.70]
openmp3 = [1.75, 9.09, 30.01, 82.15, 363.12, 613.26, 832.56, 1365.09, 2344.19, 2847.73, 3329.37, 4258.48, 5348.62,
           6942.50, 8269.04, 10904.43]
openmp_reorder3 = [0.94, 5.94, 18.26, 74.49, 338.61, 645.89, 926.88, 1305.41, 1771.96, 2411.01, 3302.72, 4072.99,
                   5743.55, 7873.67, 8254.11, 9472.46]
block3 = [1.32, 7.03, 15.95, 43.38, 81.54, 119.08, 198.20, 339.10, 394.28, 567.92, 782.52, 941.84, 1224.14, 1597.31,
          1860.53, 2480.17]
parallel2=[7.47,55.45,188.10,449.68,874.48,1510.97,2399.54,3652.48,5090.91,7076.93,9455.30,12377.19,16342.58,19904.52,24578.28,29496.09]
openmp2=[1.73,9.15,29.75,83.26,363.40,621.20,827.29,1260.32,1814.69,2440.89,3931.40,4328.33,5554.54,7060.46,9319.13,10851.74]
openmp_reorder2=[1.09,5.82,18.09,75.13,340.33,547.02,814.80,1214.20,1748.81,2698.32,3599.85,4290.90,5672.00,7051.52,9026.72,10304.89]
block2=[3.06,20.49,54.50,143.48,284.18,437.51,725.12,1104.94,1459.41,2084.45,2816.66,3493.08,4469.77,5687.71,6802.71,8286.06]

npN = np.array(N)
lnN = np.log2(N)

p_simd = DataPlot(simd3, npN, 'simd')
p_openmp = DataPlot(openmp3, npN, 'openmp')
p_block = DataPlot(block3, npN, 'block')
p_parallel = DataPlot(parallel3, npN, 'parallel')
p_reorder = DataPlot(openmp_reorder3, npN, 'openmp_reorder')

lst = [p_simd, p_openmp, p_block, p_parallel, p_reorder]

for item in lst:
    plt.plot(N, item.gflops, label=item.label)
    plt.scatter(N, item.gflops, marker='x', s=10)

plt.xlabel(r'N')
plt.ylabel(r'GFLOPS')

plt.legend()
plt.tight_layout()

plt.show()

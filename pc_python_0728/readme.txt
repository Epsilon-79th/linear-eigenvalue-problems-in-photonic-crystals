
dielectric_examples 保存介电材料空间位置对应的index，output 保存实验输出结果。
multi_gpu 为调用多个 gpu 程序，尚未完成。

此前的工作均仅用单个 gpu，脚本见 single_gpu 文件夹。

single_gpu 目录下各 .py 模块功能：

1. discretization：搭建全局稀疏矩阵块，返回FFT对角化后的矩阵块（Matrix-free）。
2. pcfft：旋度算子、厄密3*3对角块算子、全局算子 AMA'+\gamma B'B ，三种算子的简化 FFT matrix-free 操作。
3. dielectric：加载、生成介电材料空间位置对应的index（整数数组存储）。
4. lobpcg、davidson、gcg：大规模稀疏系统特征值求解器。
5. eigensolver：大规模 eigensolver 算法中涉及的局部小规模稠密问题求解，依赖 scipy（复杂问题需用到petsc4py、slepc4py）。
6. numerical_experiment、paper_1_test、paper_2_test：数值实验。
7. output：数值实验输出，包括绘制能带图、生成runtime表格的latex代码、估算收敛阶等数据处理功能。

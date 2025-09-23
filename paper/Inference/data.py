import sys
sys.path.append(r'I:\Code for Papers\2025_papers\Flow-Disentangled Feature Importance&ICLR\Flow-Disentangled-Feature-Importance')


from SHAP_Computation.utils import *



def generate_cov_matrix(d: int, rho: float) -> np.ndarray:
    """Generate a covariance matrix with the specified correlation structure."""
    cov_matrix = np.fromiter((rho**abs(i-j) for i in range(d) for j in range(d)), dtype=float).reshape(d, d)
    np.fill_diagonal(cov_matrix, 1.0)  # Set diagonal elements to 1

    return cov_matrix

'''
def generate_cov_matrix_diag(d: int, rho: float, split_at: int) -> np.ndarray:
    """Generate a block-diagonal covariance matrix with the specified correlation structure."""
    # 创建一个 d x d 的协方差矩阵，元素由 rho^abs(i-j) 计算得到
    cov_matrix_full = np.fromiter((rho**abs(i-j) for i in range(d) for j in range(d)), dtype=float).reshape(d, d)
    np.fill_diagonal(cov_matrix_full, 1.0)  # 将对角线元素设为 1
    
    # 根据 split_at 来分割矩阵为两个子矩阵
    # 子矩阵 1：cov_matrix_1
    cov_matrix_1 = cov_matrix_full[:split_at, :split_at]
    
    # 子矩阵 2：cov_matrix_2
    cov_matrix_2 = cov_matrix_full[split_at:, split_at:]
    
    # 创建一个 block-diagonal 矩阵，将 cov_matrix_1 和 cov_matrix_2 放在对角线
    block_matrix = np.block([
        [cov_matrix_1, np.zeros_like(cov_matrix_1)], 
        [np.zeros_like(cov_matrix_2), cov_matrix_2]
    ])
    
    return block_matrix
'''

def generate_cov_matrix_diag(d: int, rho: float, split_at: int) -> np.ndarray:
    """Generate a block-diagonal covariance matrix with the specified correlation structure."""
    cov_matrix_full = np.fromiter((rho**abs(i-j) for i in range(d) for j in range(d)), dtype=float).reshape(d, d)
    np.fill_diagonal(cov_matrix_full, 1.0)

    cov_matrix_1 = cov_matrix_full[:split_at, :split_at]
    cov_matrix_2 = cov_matrix_full[split_at:, split_at:]

    # 用 np.zeros 创建空矩阵，并将两块填入主对角线位置
    block_matrix = np.zeros((d, d))
    block_matrix[:split_at, :split_at] = cov_matrix_1
    block_matrix[split_at:, split_at:] = cov_matrix_2

    return block_matrix

def generate_cov_matrix_diag_sec_all_rho(d: int, rho: float, split_at: int) -> np.ndarray:
    """Generate a block-diagonal covariance matrix.
    
    - The top-left block is AR(1)-like: cov[i,j] = rho^|i-j|
    - The bottom-right block is a constant correlation matrix: diag=1, off-diag=rho
    """
    # ---- 构造 cov_matrix_1 (AR-like block) ----
    cov_matrix_1 = np.fromiter(
        (rho**abs(i - j) for i in range(split_at) for j in range(split_at)),
        dtype=float
    ).reshape(split_at, split_at)
    np.fill_diagonal(cov_matrix_1, 1.0)

    # ---- 构造 cov_matrix_2 (常值相关 block) ----
    d2 = d - split_at
    cov_matrix_2 = np.full((d2, d2), rho)
    np.fill_diagonal(cov_matrix_2, 1.0)

    # ---- 拼接为块对角矩阵 ----
    block_matrix = np.zeros((d, d))
    block_matrix[:split_at, :split_at] = cov_matrix_1
    block_matrix[split_at:, split_at:] = cov_matrix_2

    return block_matrix

def generate_cov_matrix_diag_all_rho(d: int, rho: float, split_at: int) -> np.ndarray:
    """生成块对角协方差矩阵:
    
    - 左上块: diag=1, off-diag=rho
    - 右下块: 常值相关矩阵 diag=1, off-diag=rho
    """
    # ---- 构造 cov_matrix_1 (diag=1, off=ρ) ----
    cov_matrix_1 = np.full((split_at, split_at), rho)
    np.fill_diagonal(cov_matrix_1, 1.0)

    # ---- 构造 cov_matrix_2 (常值相关 block) ----
    d2 = d - split_at
    cov_matrix_2 = np.full((d2, d2), rho)
    np.fill_diagonal(cov_matrix_2, 1.0)

    # ---- 拼接为块对角矩阵 ----
    block_matrix = np.zeros((d, d))
    block_matrix[:split_at, :split_at] = cov_matrix_1
    block_matrix[split_at:, split_at:] = cov_matrix_2

    return block_matrix

import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass
from typing import Tuple

def generate_cov_matrix_blocked(d: int, rho: float, split_at: int, tail_block_size: int = 10) -> np.ndarray:
    """
    生成块对角协方差矩阵：
    - 第一个块大小为 split_at，diag=1, off-diag=rho
    - 剩余 d - split_at 维按 tail_block_size 为一组继续分块（最后一块可为不足 tail_block_size 的余数），
      每个块都是等相关结构（compound symmetry）：diag=1, off-diag=rho
    """
    if split_at <= 0 or split_at > d:
        raise ValueError("split_at must be in (0, d].")
    if not ( -1.0/(d-1) < rho < 1.0 ):
        # 对等相关矩阵正定的常见充要条件之一（整体更严格的是每个子块的维度也要满足）
        pass

    # 计算尾部块的尺寸列表
    tail_dim = d - split_at
    blocks = [split_at]
    if tail_dim > 0:
        k, rem = divmod(tail_dim, tail_block_size)
        blocks.extend([tail_block_size] * k)
        if rem > 0:
            blocks.append(rem)

    # 组装块对角矩阵
    Sigma = np.zeros((d, d), dtype=float)
    start = 0
    for bsz in blocks:
        B = np.full((bsz, bsz), rho, dtype=float)
        np.fill_diagonal(B, 1.0)
        Sigma[start:start+bsz, start:start+bsz] = B
        start += bsz

    return Sigma

def generate_cov_matrix_blocked_exp_structure(
    d: int,
    rho: float,
    block_size: int = 5
) -> np.ndarray:
    """
    按 block_size(=5) 分块的块对角协方差矩阵：
    - 第一个 block 的 5 维按两组细分：
        * 组A: 索引 [0,1]  (X0, X1)  组内等相关 = rho
        * 组B: 索引 [2,3,4](X2, X3, X4) 组内等相关 = rho
        * 组间零相关
    - 第2个及之后的每个 block 为等相关（compound symmetry）：diag=1, off-diag=rho
    - 不同 block 间零相关
    正定性的常见条件（充分必要，用于等相关块）：-1/(m-1) < rho < 1。
    这里取最严格者 m = max(5, 3, 2) = 5 ⇒ 要求 -1/4 < rho < 1。
    """
    if d <= 0:
        raise ValueError("d must be positive.")
    # 取最严格块大小=5来做rho的快速检查（更细的检查见下）
    if not (-1.0/4 < rho < 1.0):
        raise ValueError("rho must satisfy -1/4 < rho < 1 to ensure PD for 5x5 compound-symmetry blocks.")

    Sigma = np.zeros((d, d), dtype=float)

    # ---- 先处理第一个 block 的 5 维（特殊细分） ----
    if d >= block_size:
        # 先清零该 5x5，再分别填A/B两个子块
        b0 = slice(0, block_size)
        # 组A与组B索引（相对于全局）
        idxA = np.array([0, 1])
        idxB = np.array([2, 3, 4])

        # A 组：2x2 等相关
        Sigma[np.ix_(idxA, idxA)] = rho
        # B 组：3x3 等相关
        Sigma[np.ix_(idxB, idxB)] = rho
        # 组间零相关
        Sigma[np.ix_(idxA, idxB)] = 0.0
        Sigma[np.ix_(idxB, idxA)] = 0.0
        # 对角线设回 1
        Sigma[idxA, idxA] = 1.0
        Sigma[idxB, idxB] = 1.0

        start = block_size
    else:
        # d < 5 的退化情形：仅一个不足 5 的块，直接按等相关填充
        B = np.full((d, d), rho, dtype=float)
        np.fill_diagonal(B, 1.0)
        Sigma[:d, :d] = B
        return Sigma

    # ---- 之后的各个 5 维 block（以及最后不足 5 的余块）按等相关填充 ----
    while start < d:
        end = min(start + block_size, d)
        bsz = end - start
        B = np.full((bsz, bsz), rho, dtype=float)
        np.fill_diagonal(B, 1.0)
        Sigma[start:end, start:end] = B
        start = end

    return Sigma






class DGP(abc.ABC):
    """Abstract base: needs `.generate(n, rho, seed)` returning X (n×d), y (n)."""
    d: int  # total feature dimension (W + Z)

    @abc.abstractmethod
    def generate(self, n: int, rho: float, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        ...

# -- Example 1: Linear --------------------------------------------------------
from scipy.linalg import sqrtm
@dataclass
class Example1Linear(DGP):
    β: float = 5.0
    σ_eps: float = 1.0
    d: int = 10  # W + 9 Z’s  (but only Z1 is correlated)

    def generate(self, n: int, rho: float, seed: int | None = None):
        rng = default_rng(seed)
        Σ = make_block_cov(self.d, rho, blocks=[(0, 1)])

        Sigma = Σ
        Sigma_sqrt = sqrtm(Sigma).real
        Sigma_sqrt_sq = Sigma_sqrt**2   # 元素平方，而不是矩阵平方
        X = rng.multivariate_normal(np.zeros(self.d), Σ, size=n)
        W = X[:, 0]
        ε = rng.normal(scale=self.σ_eps, size=n)
        y = self.β * W + ε
        return X, y

# -- Example 2: Cosine --------------------------------------------------------

@dataclass
class Example2Trig(DGP):
    amp: float = 5.0
    σ_eps: float = 1.0
    d: int = 10

    def generate(self, n: int, rho: float, seed: int | None = None):
        rng = default_rng(seed)
        Σ = make_block_cov(self.d, rho, blocks=[(0, 1)])
        X = rng.multivariate_normal(np.zeros(self.d), Σ, size=n)
        W, Z1 = X[:, 0], X[:, 1]
        ε = rng.normal(scale=self.σ_eps, size=n)
        y = self.amp * np.cos(W) + self.amp * np.cos(Z1) + ε
        return X, y

# -- Example 3: Interaction ---------------------------------------------------

@dataclass
class Example3Interaction(DGP):
    d: int = 10  # W + Z1..Z4
    # σ_eps: float = .7

    def generate(self, n: int, rho: float, seed: int | None = None):
        var_ = (26 + 27 * rho ** 2) / 16
        self.σ_eps = np.sqrt(var_ / 9)

        rng = default_rng(seed)
        # Correlate (W,Z1) and (Z3,Z4)
        Σ = make_block_cov(self.d, rho, blocks=[(0, 1), (3, 4)])
        X = rng.multivariate_normal(np.zeros(self.d), Σ, size=n)
        W, Z1, Z2, Z3, Z4 = [X[:, i] for i in range(5)]
        ε = rng.normal(scale=1.0, size=n)
        y = 1.5 * W * Z1 * (Z2 > 0) + Z3 * Z4 * (Z2 < 0) + ε
        return X, y

# -- Example 4: Low‑density / heteroskedastic --------------------------------

@dataclass
class Example4LowDensity(DGP):
    σ_eps: float = 1.0
    σ_δ: float = 1.0
    d: int = 10  # W + Z1..Z9

    def generate(self, n: int, rho: float, seed: int | None = None):
        rng = default_rng(seed)
        W = rng.normal(size=n)
        δ = rng.normal(scale=self.σ_δ, size=n)
        Z1 = 3 * W ** 2 + δ  # heavy dependence + low‑density horns
        Z_rest = rng.normal(size=(n, self.d - 2))  # independent
        X = np.column_stack([W, Z1, Z_rest])
        y = 5 * W + rng.normal(scale=self.σ_eps, size=n)
        return X, y
    


    


@dataclass
class Exp1:
    d: int = 50   # Total number of features
    rho: float = 0.6  # Correlation coefficient between features

    def generate(self, n: int, rho: float, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data according to the specified model."""
        rng = default_rng(seed)

        # 第一个块 10 维，后续按每 10 维一块继续分块
        Sigma = generate_cov_matrix_blocked(self.d, rho, split_at=10, tail_block_size=10)
        X = rng.multivariate_normal(np.zeros(self.d), Sigma, size=n)

        # 取前 5 个变量
        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        ε = rng.normal(scale=1.0, size=n)

        y = ( np.arctan(X0 + X1) * (X2 > 0) ) + ( np.sin((X3 * X4)) * (X2 < 0) ) + ε

        
        #y = ( np.cos( (X0 + X1) ) * (X2 > 0) ) + ( np.sin( (X3 * X4) ) * (X2 < 0) ) + ε
        

        return X, y
    

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from numpy.random import default_rng



@dataclass
class Exp2:
    d: int = 50          # Total number of features #######
    rho: float = 0.6     # 默认rho（仅作占位/默认值）

    def generate(
        self,
        n: int,
        rho1: Optional[float] = None,          # 第一个高斯分量的相关系数
        rho2: Optional[float] = None,          # 第二个高斯分量的相关系数
        seed: Optional[int] = None,
        mix_weight: float = 0.5,               # 混合权重：P(来自分量1)=mix_weight
        split_at: int = 10,
        tail_block_size: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样流程：
          1) 构造两套协方差矩阵 Sigma1(rho1), Sigma2(rho2)，两者结构相同但 rho 不同；
          2) 以概率 mix_weight 选择分量1，否则分量2；
          3) 按选择的分量从 N(0, Sigma_k) 采样 X；
          4) y = atan(X0+X1)*(X2>0) + sin(X3*X4)*(X2<0) + eps.
        """
        rng = default_rng(seed)

        # 默认：若未指定，rho1 用 self.rho，rho2 给个不同的值（请按需要填写）
        if rho1 is None:
            rho1 = self.rho
        if rho2 is None:
            rho2 = self.rho / 2.0  # 只是默认；建议显式传入你想要的第二个 rho

        # 检查混合权重
        if not (0.0 < mix_weight < 1.0):
            raise ValueError("mix_weight 必须在 (0,1) 内。")

        # 1) 构造两套协方差（均值都为 0）
        Sigma1 = generate_cov_matrix_blocked(self.d, rho1, split_at=split_at, tail_block_size=tail_block_size)
        Sigma2 = generate_cov_matrix_blocked(self.d, rho2, split_at=split_at, tail_block_size=tail_block_size)

        # 2) 采样分量指派
        comp = rng.binomial(n=1, p=mix_weight, size=n)  # 1 表示来自分量1，0 表示分量2

        # 3) 分别采样
        X = np.empty((n, self.d), dtype=float)
        idx1 = np.where(comp == 1)[0]
        idx2 = np.where(comp == 0)[0]
        if idx1.size > 0:
            X[idx1] = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma1, size=idx1.size)
        if idx2.size > 0:
            X[idx2] = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma2, size=idx2.size)

        # 4) 生成 y（保持你的原表达式）
        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        eps = rng.normal(scale=1.0, size=n)
        y = (np.arctan(X0 + X1) * (X2 > 0)) + (np.sin(X3 * X4) * (X2 < 0)) + eps

        return X, y
    




@dataclass
class Correcrion_Mixture:
    d: int = 10          # Total number of features
    rho: float = 0.6     # 默认rho（仅作占位/默认值）

    def generate(
        self,
        n: int,
        rho1: Optional[float] = None,          # 第一个高斯分量的相关系数
        rho2: Optional[float] = None,          # 第二个高斯分量的相关系数
        seed: Optional[int] = None,
        mix_weight: float = 0.5,               # 混合权重：P(来自分量1)=mix_weight
        split_at: int = 2,
        tail_block_size: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样流程：
          1) 构造两套协方差矩阵 Sigma1(rho1), Sigma2(rho2)，两者结构相同但 rho 不同；
          2) 以概率 mix_weight 选择分量1，否则分量2；
          3) 按选择的分量从 N(0, Sigma_k) 采样 X；
          4) y = atan(X0+X1)*(X2>0) + sin(X3*X4)*(X2<0) + eps.
        """
        rng = default_rng(seed)

        # 默认：若未指定，rho1 用 self.rho，rho2 给个不同的值（请按需要填写）
        if rho1 is None:
            rho1 = self.rho
        if rho2 is None:
            rho2 = self.rho / 2.0  # 只是默认；建议显式传入你想要的第二个 rho

        # 检查混合权重
        if not (0.0 < mix_weight < 1.0):
            raise ValueError("mix_weight 必须在 (0,1) 内。")

        # 1) 构造两套协方差（均值都为 0）
        Sigma1 = generate_cov_matrix_blocked(self.d, rho1, split_at=split_at, tail_block_size=tail_block_size)
        Sigma2 = generate_cov_matrix_blocked(self.d, rho2, split_at=split_at, tail_block_size=tail_block_size)

        # 2) 采样分量指派
        comp = rng.binomial(n=1, p=mix_weight, size=n)  # 1 表示来自分量1，0 表示分量2

        # 3) 分别采样
        X = np.empty((n, self.d), dtype=float)
        idx1 = np.where(comp == 1)[0]
        idx2 = np.where(comp == 0)[0]
        if idx1.size > 0:
            X[idx1] = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma1, size=idx1.size)
        if idx2.size > 0:
            X[idx2] = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma2, size=idx2.size)

        # 4) 生成 y（保持你的原表达式）
        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        eps = rng.normal(scale=1.0, size=n)
        y = 5 * X0 + eps

        return X, y
    

@dataclass
class Correction_Gaussian:
    d: int = 10   # Total number of features
    rho: float = 0.6  # Correlation coefficient between features

    def generate(self, n: int, rho: float, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data according to the specified model."""
        rng = default_rng(seed)

        # 第一个块 10 维，后续按每 10 维一块继续分块
        Sigma = generate_cov_matrix_blocked(self.d, rho, split_at=2, tail_block_size=10)
        X = rng.multivariate_normal(np.zeros(self.d), Sigma, size=n)

        # 取前 5 个变量
        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        ε = rng.normal(scale=1.0, size=n)

        y = 5 * X0 + ε


        

        return X, y





@dataclass
class Cls:
    d: int = 50           # 特征维数
    rho: float = 0.6      # 相关系数
    # 下两项不再使用，仅保留以兼容你原来的接口
    threshold: float = 0.0
    noise_sigma: float = 1.0

    def generate(self, n: int, rho: float, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data with binary labels y ∈ {0,1}: y = 1{ arctan(X0+X1) > 0 }."""
        rng = default_rng(seed)

        # 1) 生成 X（与你原实现一致）
        Sigma = generate_cov_matrix_blocked(self.d, rho, split_at=10, tail_block_size=10)
        X = rng.multivariate_normal(np.zeros(self.d), Sigma, size=n)

        # 2) 二值化标签：arctan(X0 + X1) > 0  <=>  X0 + X1 > 0
        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        y = (np.arctan(X0 + X1) > 0).astype(int)


        return X, y
    
from scipy.linalg import sqrtm

@dataclass
class Exp_structure:
    d: int = 10   # Total number of features
    rho: float = 0.6  # Correlation coefficient between features

    def generate(self, n: int, rho: float, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data according to the specified model."""
        rng = default_rng(seed)

        # === 关键修改：调用 block 结构协方差矩阵 ===
        Sigma = generate_cov_matrix_blocked_exp_structure(self.d, rho, block_size=5)

        Sigma_sqrt = sqrtm(Sigma).real

        Sigma_sqrt_sq = Sigma_sqrt**2   # 元素平方，而不是矩阵平方

        # 从多元正态分布采样
        X = rng.multivariate_normal(np.zeros(self.d), Sigma, size=n)

        # 提取前 5 个特征
        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        ε = rng.normal(scale=1.0, size=n)

        y = ( np.arctan(X0) ) + ( np.sin((X2)) ) + ε

        return X, y, Sigma_sqrt_sq



@dataclass
class Example_figure:
    sigma: float = 1.0   # 方差
    d: int = 2           # 只需要二维：X0, X1

    def generate(self, n: int, rho: float, seed: int | None = None):
        rng = default_rng(seed)

        # 构造协方差矩阵
        Sigma = np.array([
            [self.sigma**2, rho * self.sigma**2],
            [rho * self.sigma**2, self.sigma**2]
        ])

        # 生成二维高斯样本
        X = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma, size=n)

        # X0, X1
        X0 = X[:, 0]
        X1 = X[:, 1]

        # 定义 Y
        Y = X0 + X1

        return X, Y, Sigma





        


# Registry --------------------------------------------------------------------
DGP_REGISTRY: Dict[str, DGP] = {
    "1": Example1Linear(),
    "2": Example2Trig(),
    "3": Example3Interaction(),
    "4": Example4LowDensity(),
    "6": Complex_example_2(),
}


import sys
sys.path.append(r'I:\Code for Papers\2025_papers\Flow-Disentangled Feature Importance&ICLR\Flow-Disentangled-Feature-Importance')


from Inference.utils import *



def generate_cov_matrix(d: int, rho: float) -> np.ndarray:

    cov_matrix = np.fromiter((rho**abs(i-j) for i in range(d) for j in range(d)), dtype=float).reshape(d, d)
    np.fill_diagonal(cov_matrix, 1.0)  

    return cov_matrix




def generate_cov_matrix_diag(d: int, rho: float, split_at: int) -> np.ndarray:

    cov_matrix_full = np.fromiter((rho**abs(i-j) for i in range(d) for j in range(d)), dtype=float).reshape(d, d)
    np.fill_diagonal(cov_matrix_full, 1.0)

    cov_matrix_1 = cov_matrix_full[:split_at, :split_at]
    cov_matrix_2 = cov_matrix_full[split_at:, split_at:]

    block_matrix = np.zeros((d, d))
    block_matrix[:split_at, :split_at] = cov_matrix_1
    block_matrix[split_at:, split_at:] = cov_matrix_2

    return block_matrix

def generate_cov_matrix_diag_sec_all_rho(d: int, rho: float, split_at: int) -> np.ndarray:

    cov_matrix_1 = np.fromiter(
        (rho**abs(i - j) for i in range(split_at) for j in range(split_at)),
        dtype=float
    ).reshape(split_at, split_at)
    np.fill_diagonal(cov_matrix_1, 1.0)

    d2 = d - split_at
    cov_matrix_2 = np.full((d2, d2), rho)
    np.fill_diagonal(cov_matrix_2, 1.0)


    block_matrix = np.zeros((d, d))
    block_matrix[:split_at, :split_at] = cov_matrix_1
    block_matrix[split_at:, split_at:] = cov_matrix_2

    return block_matrix

def generate_cov_matrix_diag_all_rho(d: int, rho: float, split_at: int) -> np.ndarray:

    cov_matrix_1 = np.full((split_at, split_at), rho)
    np.fill_diagonal(cov_matrix_1, 1.0)


    d2 = d - split_at
    cov_matrix_2 = np.full((d2, d2), rho)
    np.fill_diagonal(cov_matrix_2, 1.0)


    block_matrix = np.zeros((d, d))
    block_matrix[:split_at, :split_at] = cov_matrix_1
    block_matrix[split_at:, split_at:] = cov_matrix_2

    return block_matrix

import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass
from typing import Tuple

def generate_cov_matrix_blocked(d: int, rho: float, split_at: int, tail_block_size: int = 10) -> np.ndarray:

    if split_at <= 0 or split_at > d:
        raise ValueError("split_at must be in (0, d].")
    if not ( -1.0/(d-1) < rho < 1.0 ):

        pass


    tail_dim = d - split_at
    blocks = [split_at]
    if tail_dim > 0:
        k, rem = divmod(tail_dim, tail_block_size)
        blocks.extend([tail_block_size] * k)
        if rem > 0:
            blocks.append(rem)


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

    if d <= 0:
        raise ValueError("d must be positive.")

    if not (-1.0/4 < rho < 1.0):
        raise ValueError("rho must satisfy -1/4 < rho < 1 to ensure PD for 5x5 compound-symmetry blocks.")

    Sigma = np.zeros((d, d), dtype=float)


    if d >= block_size:

        b0 = slice(0, block_size)

        idxA = np.array([0, 1])
        idxB = np.array([2, 3, 4])


        Sigma[np.ix_(idxA, idxA)] = rho

        Sigma[np.ix_(idxB, idxB)] = rho

        Sigma[np.ix_(idxA, idxB)] = 0.0
        Sigma[np.ix_(idxB, idxA)] = 0.0

        Sigma[idxA, idxA] = 1.0
        Sigma[idxB, idxB] = 1.0

        start = block_size
    else:

        B = np.full((d, d), rho, dtype=float)
        np.fill_diagonal(B, 1.0)
        Sigma[:d, :d] = B
        return Sigma


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
    d: int  

    @abc.abstractmethod
    def generate(self, n: int, rho: float, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        ...

# -- Example 1: Linear --------------------------------------------------------
from scipy.linalg import sqrtm
@dataclass
class Example1Linear(DGP):
    β: float = 5.0
    σ_eps: float = 1.0
    d: int = 10  

    def generate(self, n: int, rho: float, seed: int | None = None):
        rng = default_rng(seed)
        Σ = make_block_cov(self.d, rho, blocks=[(0, 1)])

        Sigma = Σ
        Sigma_sqrt = sqrtm(Sigma).real
        Sigma_sqrt_sq = Sigma_sqrt**2   
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
    d: int = 10  


    def generate(self, n: int, rho: float, seed: int | None = None):
        var_ = (26 + 27 * rho ** 2) / 16
        self.σ_eps = np.sqrt(var_ / 9)

        rng = default_rng(seed)

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
    d: int = 10  

    def generate(self, n: int, rho: float, seed: int | None = None):
        rng = default_rng(seed)
        W = rng.normal(size=n)
        δ = rng.normal(scale=self.σ_δ, size=n)
        Z1 = 3 * W ** 2 + δ  
        Z_rest = rng.normal(size=(n, self.d - 2))  
        X = np.column_stack([W, Z1, Z_rest])
        y = 5 * W + rng.normal(scale=self.σ_eps, size=n)
        return X, y
    


    


@dataclass
class Exp1:
    d: int = 50   
    rho: float = 0.6  

    def generate(self, n: int, rho: float, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = default_rng(seed)

        Sigma = generate_cov_matrix_blocked(self.d, rho, split_at=10, tail_block_size=10)
        X = rng.multivariate_normal(np.zeros(self.d), Sigma, size=n)

        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        ε = rng.normal(scale=1.0, size=n)

        y = ( np.arctan(X0 + X1) * (X2 > 0) ) + ( np.sin((X3 * X4)) * (X2 < 0) ) + ε

        
        return X, y
    

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from numpy.random import default_rng



@dataclass
class Exp2:
    d: int = 50          
    rho: float = 0.6     

    def generate(
        self,
        n: int,
        rho1: Optional[float] = None,        
        rho2: Optional[float] = None,         
        seed: Optional[int] = None,
        mix_weight: float = 0.5,               
        split_at: int = 10,
        tail_block_size: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:

        rng = default_rng(seed)

        if rho1 is None:
            rho1 = self.rho
        if rho2 is None:
            rho2 = self.rho / 2.0  

        if not (0.0 < mix_weight < 1.0):
            raise ValueError("mix_weight has to be within (0,1)")


        Sigma1 = generate_cov_matrix_blocked(self.d, rho1, split_at=split_at, tail_block_size=tail_block_size)
        Sigma2 = generate_cov_matrix_blocked(self.d, rho2, split_at=split_at, tail_block_size=tail_block_size)


        comp = rng.binomial(n=1, p=mix_weight, size=n)  


        X = np.empty((n, self.d), dtype=float)
        idx1 = np.where(comp == 1)[0]
        idx2 = np.where(comp == 0)[0]
        if idx1.size > 0:
            X[idx1] = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma1, size=idx1.size)
        if idx2.size > 0:
            X[idx2] = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma2, size=idx2.size)

        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        eps = rng.normal(scale=1.0, size=n)
        y = (np.arctan(X0 + X1) * (X2 > 0)) + (np.sin(X3 * X4) * (X2 < 0)) + eps

        return X, y
    




@dataclass
class Correcrion_Mixture:
    d: int = 10         
    rho: float = 0.6     

    def generate(
        self,
        n: int,
        rho1: Optional[float] = None,         
        rho2: Optional[float] = None,         
        seed: Optional[int] = None,
        mix_weight: float = 0.5,               
        split_at: int = 2,
        tail_block_size: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:

        rng = default_rng(seed)


        if rho1 is None:
            rho1 = self.rho
        if rho2 is None:
            rho2 = self.rho / 2.0  

      
        if not (0.0 < mix_weight < 1.0):
            raise ValueError("mix_weight has to be within (0,1).")

       
        Sigma1 = generate_cov_matrix_blocked(self.d, rho1, split_at=split_at, tail_block_size=tail_block_size)
        Sigma2 = generate_cov_matrix_blocked(self.d, rho2, split_at=split_at, tail_block_size=tail_block_size)

       
        comp = rng.binomial(n=1, p=mix_weight, size=n)  

      
        X = np.empty((n, self.d), dtype=float)
        idx1 = np.where(comp == 1)[0]
        idx2 = np.where(comp == 0)[0]
        if idx1.size > 0:
            X[idx1] = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma1, size=idx1.size)
        if idx2.size > 0:
            X[idx2] = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma2, size=idx2.size)

     
        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        eps = rng.normal(scale=1.0, size=n)
        y = 5 * X0 + eps

        return X, y
    

@dataclass
class Correction_Gaussian:
    d: int = 10   
    rho: float = 0.6  

    def generate(self, n: int, rho: float, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = default_rng(seed)

 
        Sigma = generate_cov_matrix_blocked(self.d, rho, split_at=2, tail_block_size=10)
        X = rng.multivariate_normal(np.zeros(self.d), Sigma, size=n)

   
        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        ε = rng.normal(scale=1.0, size=n)

        y = 5 * X0 + ε



        return X, y






    
from scipy.linalg import sqrtm

@dataclass
class Exp_structure:
    d: int = 10   
    rho: float = 0.6  

    def generate(self, n: int, rho: float, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = default_rng(seed)

      
        Sigma = generate_cov_matrix_blocked_exp_structure(self.d, rho, block_size=5)

        Sigma_sqrt = sqrtm(Sigma).real

        Sigma_sqrt_sq = Sigma_sqrt**2  

      
        X = rng.multivariate_normal(np.zeros(self.d), Sigma, size=n)

       
        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        ε = rng.normal(scale=1.0, size=n)

        y = ( np.arctan(X0) ) + ( np.sin((X2)) ) + ε

        return X, y, Sigma_sqrt_sq



@dataclass
class Example_figure:
    sigma: float = 1.0   
    d: int = 2           

    def generate(self, n: int, rho: float, seed: int | None = None):
        rng = default_rng(seed)


        Sigma = np.array([
            [self.sigma**2, rho * self.sigma**2],
            [rho * self.sigma**2, self.sigma**2]
        ])


        X = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma, size=n)


        X0 = X[:, 0]
        X1 = X[:, 1]
        Y = X0 + X1

        return X, Y, Sigma



def gaussian_to_t(Z: np.ndarray, df: float, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = default_rng()
    n, d = Z.shape
    W = rng.chisquare(df, size=n)      
    scale = np.sqrt(df / W)[:, None]    

    X = Z * scale
    return X


def generate_cov_matrix_blocked_rho_ij(
    d: int,
    rho: float,
    split_at: int,
    tail_block_size: int = 10,
    return_blocks: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[int]]:

    if split_at <= 0 or split_at > d:
        raise ValueError("split_at must be in (0, d].")

    if not (-1.0 < rho < 1.0):
        print("[Warning] ")

    tail_dim = d - split_at
    blocks: List[int] = [split_at]
    if tail_dim > 0:
        k, rem = divmod(tail_dim, tail_block_size)
        blocks.extend([tail_block_size] * k)
        if rem > 0:
            blocks.append(rem)

    Sigma = np.zeros((d, d), dtype=float)
    start = 0
    for bsz in blocks:
        idx = np.arange(bsz)
        B = rho ** np.abs( (idx[:, None] - idx[None, :]) / 5 ) 
        Sigma[start:start + bsz, start:start + bsz] = B
        start += bsz

    if return_blocks:
        return Sigma, blocks
    else:
        return Sigma


@dataclass
class Exp_SIM:
    d: int = 50       

    def generate(
        self,
        n: int,
        seed: int | None = None,
        rho1: float | None = None,
        rho2: float | None = None,
        mix_weight: float = 0.2,
        df: float = 5.0,
        split_at: int = 10,
        tail_block_size: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:

        rng = default_rng(seed)


        if rho1 is None:
            rho1 = self.rho      
        if rho2 is None:
            rho2 = 0.2            

        if not (0.0 < mix_weight < 1.0):
            raise ValueError("mix_weight has to be within (0,1).")
        Sigma1 = generate_cov_matrix_blocked(
            d=self.d,
            rho=rho1,
            split_at=split_at,
            tail_block_size=tail_block_size,
        )
        Sigma2 = generate_cov_matrix_blocked(
            d=self.d,
            rho=rho2,
            split_at=split_at,
            tail_block_size=tail_block_size,
        )
        comp = rng.binomial(n=1, p=mix_weight, size=n)
        Z = np.empty((n, self.d), dtype=float)
        idx1 = np.where(comp == 1)[0]
        idx2 = np.where(comp == 0)[0]

        if idx1.size > 0:
            Z[idx1] = rng.multivariate_normal(
                mean=np.zeros(self.d),
                cov=Sigma1,
                size=idx1.size,
            )
        if idx2.size > 0:
            Z[idx2] = rng.multivariate_normal(
                mean=np.zeros(self.d),
                cov=Sigma2,
                size=idx2.size,
            )

        X = gaussian_to_t(Z, df=df, rng=rng)

        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        eps = rng.normal(scale=1.0, size=n)

        y = (np.arctan(X0 + X1) * (X2 > 0)) + (np.sin(X3 * X4) * (X2 < 0)) + eps

        return X, y





        




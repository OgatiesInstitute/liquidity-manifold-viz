import torch
import torch.nn as nn
import numpy as np
from scipy.stats import wasserstein_distance

# --- §3.1 Liquidity Score Network ---
class LiquidityScoreNet(nn.Module):
    # Approximates nabla_M log p_t(L) -- eq. (2.2)
    # Input: order book tensor L in R^{P x Q x 2}, time t
    def __init__(self, p=100, q=50, h=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, h, 3, padding=1), nn.SiLU(),  # 3 = 2 sides + time embed
            nn.Conv2d(h, h, 3, padding=1), nn.SiLU(),
            nn.Conv2d(h, 2, 3, padding=1),              # score per bid/ask
        )
    def forward(self, L, t):
        # L: (B, 2, P, Q)  t: (B,)
        t_map = t[:,None,None,None].expand(-1,1,L.shape[2],L.shape[3])
        return self.enc(torch.cat([L, t_map], dim=1))   # nabla_M log p_t

def curvature_proxy(score, L, dx=1.0):
    # Hessian trace as curvature indicator -- eq. (1.4)
    grad_s = torch.autograd.functional.jacobian(
        lambda x: score(x, torch.zeros(x.shape[0])), L)
    return grad_s.diagonal(dim1=-2, dim2=-1).abs().mean() # kappa(x)

# --- §3.2 Reverse SDE Sampler ---
@torch.no_grad()
def reconstruct_liquidity(score_model, shape, T=1.0, n_steps=300):
    # Reverse SDE -- eq. (2.2): dL_bar = [b - Sigma * score] dt + sigma dW_bar
    dt = T / n_steps
    L  = torch.randn(*shape)           # L_T ~ N(0, I): dissolved book

    for i in range(n_steps, 0, -1):
        t     = torch.full((shape[0],), i * dt)
        score = score_model(L, t)      # nabla_M log p_t -- geometric contraction
        drift = -L - score             # b = -L (OU), score correction
        noise = torch.randn_like(L) * dt**0.5
        L     = L - drift * dt + noise # backward Euler
    return L.clamp(min=0)              # liquidity >= 0

# --- §3.3 Optimal Transport ---
def w2_liquidity(book_t, book_t1):
    # W2(p_t, p_{t-1}) -- eq. (2.4): transport cost between consecutive books
    # books: (P, Q, 2) numpy arrays
    bid_t  = book_t[:,:,0].flatten();   bid_t1 = book_t1[:,:,0].flatten()
    ask_t  = book_t[:,:,1].flatten();   ask_t1 = book_t1[:,:,1].flatten()
    w_bid  = wasserstein_distance(bid_t,  bid_t1)
    w_ask  = wasserstein_distance(ask_t,  ask_t1)
    return w_bid + w_ask               # total structural change

def fault_line_score(books, window=20):
    # Rolling W2: spike = fault line forming -- eq. (2.3)
    scores = []
    for i in range(window, len(books)):
        w2 = w2_liquidity(books[i], books[i-1])
        scores.append(w2)
    return scores                      # large values → curvature singularity

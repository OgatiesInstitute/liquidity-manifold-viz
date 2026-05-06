# backend/main.py
import asyncio
import websockets
import json
import torch
import numpy as np

# models.py から定義を読み込みます
from models import LiquidityScoreNet, reconstruct_liquidity, w2_liquidity

async def pipeline(websocket, score_model, simulator):
    book_history = []
    
    while True:
        # 1. Simulate one step (eq. 2.1)
        # ※OrderBookSimulatorは別途定義されている前提です
        raw_book = simulator.step(mu=0.0, sigma=0.12, lambda_jump=0.03)
        book_history.append(raw_book.numpy())

        # 2. Reconstruct via reverse SDE (eq. 2.2)
        noisy = raw_book.unsqueeze(0).permute(0,3,1,2).float()
        recon = reconstruct_liquidity(score_model, noisy.shape, n_steps=100)
        recon_np = recon.squeeze(0).permute(1,2,0).numpy()

        # 3. Fault line detection via W2 (eq. 2.3, 2.4)
        fault_score = 0.0
        fault_level = 0.5
        if len(book_history) > 1:
            w2 = w2_liquidity(book_history[-1], book_history[-2])
            fault_score = float(np.tanh(w2 * 5))     # normalise to [0,1]
            fault_level = float(np.argmax(
                recon_np[:,:,0].mean(axis=1)) / recon_np.shape[0])

        # 4. Stream to three.js (§3.4)
        await websocket.send(json.dumps({
            'bids':        recon_np[:,:,0].tolist(),
            'asks':        recon_np[:,:,1].tolist(),
            'fault_score': fault_score,
            'fault_level': fault_level,
        }))
        await asyncio.sleep(0.016)     # 60 fps

async def main():
    # 外部シミュレーターのインポート（環境に合わせて調整）
    try:
        from build_vol3_final import OrderBookSimulator
    except ImportError:
        # シミュレーターがない場合のプレースホルダ
        class OrderBookSimulator:
            def step(self, **kwargs): return torch.randn(100, 50, 2)
    
    sim   = OrderBookSimulator()
    model = LiquidityScoreNet()
    
    print("WebSocket server started on ws://localhost:8765")
    async with websockets.serve(
        lambda ws: pipeline(ws, model, sim), 'localhost', 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

# prefix_radix — parallel prefix over A5, radix 5 + 1 instruction per layer, on scratch workers

Layout: n inputs, n identity anchors, n workers (worker i at 2n+i). `g` (alphabet 60, an A5 index)
and `ins` (0/1 = instruction, 2 = none) are identity at every non-worker position and are never
changed there, so a worker head whose offset overshoots its own block lands on an anchor and reads
the identity for free — validity never has to be read by the MLP. Heads only exist if some worker
needs them (offset ≤ n-1), so every worker query has a unique match.
Layer 1: 8 heads copy tok from inputs i, i-1, …, i-7 (1 bit each); the MLP reads v1 = min(8, i+1)
(alphabet 9, set from position alone) and folds the valid ones into g (a 2^8 lookup), block = 8.
Layer L ≥ 2 with block b: 4 heads copy g from workers i - m·b (6 bits each), 1 head copies ins from
worker i - 5b (2 bits); the MLP reads own g (5.9 bits) and folds own ∘ p1 ∘ … ∘ p4 ∘ instruction,
31.9 ≤ 32 mlp_bits, block = 5b + 1. Readout: image of cup 0 under g at each worker.

Blocks 8, 41, 206, 1031, …: b_L = (33·5^(L-1) - 1)/4, so layers(n) = 1 for n ≤ 8, else
1 + ⌈log₅((4n+1)/33)⌉ (computed by the integer loop in `blocks`). Versus the doc's 1 for n ≤ 11 /
2 + ⌈log₄(n/55)⌉: the 11-symbol first layer is unattainable with H = 8 (8 symbols), but the
trailing-instruction slot gives 4 layers at n = 1024 where the doc has 5; equal at 16, 64, 256.

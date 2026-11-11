# radix_scan

Every token is an element of the affine monoid {x -> ax+b} over Z_8: the start value is the constant map
(0, x0), a step (op, b) is (1, b), (1, -b) or (b, 0). Element k lives at pos 2k; the prefix product at
element k is the constant map whose b is intermediate k, so the readout is `b` at even positions.
Layer 1 (radix 4): head 0 copies the own op token (pos-1); partner t = 1..3 needs two heads, its op at
pos-2t-1 and its operand at pos-2t (7 heads, MLP reads 24 bits). The MLP composes the 4 elements.
Layer j >= 2 (radix 5): position k holds the product of its block of B = 4*5^(j-2) elements; 4 heads copy
(a, b) from positions k - tB, t = 1..4 (unique match on pos), the MLP composes the 5 blocks in order
(own 6 bits + 4 x 6 head bits = 30 <= mlp_bits). Partners below position 0 are never gated: they are
folded nearest-first and by then the accumulator already contains the start constant, so a = 0 and any
garbage (sym = -1) is annihilated. No position field is read by any MLP; all copies are unique matches.
layers(n) = 1 + ceil(log_5((n+1)/4)) for n >= 4, else 1 (block sizes 4, 20, 100, 500, 2500).
Semantics v2 change: the documented r = heads+1 = 9 needs 9 x 6 = 54 MLP bits; the 32-bit table cap
makes radix 5 the most an MLP can compose (Aff(Z_8) has 64 elements), and the old 2-way-tie first layer
(radix 8) is gone. The doc's ceil(log_9 n) + 1 is not attainable under v2; this construction still
matches it at n = 64, 256, 1024 (3, 4, 5) because the start value is folded for free.

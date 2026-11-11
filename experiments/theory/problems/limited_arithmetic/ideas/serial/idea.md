# serial

Tokens: start value at pos 0, then (op, operand) at pos (2k-1, 2k). Field `x` at pos 0 is the start.
Every layer, at every position: head 0 copies `x` from pos-2, head 1 copies the token at pos-1 (unique
matches on pos); the MLP writes x <- step(op, own operand, x_prev) mod 8, leaving x unchanged where the
copied token is not an operator (pos 0 keeps the start value). By induction x at pos 2k is correct after
layer k, so layers(n) = n; readout reads `x` at the even positions. The MLP reads 10.5 bits, no position.

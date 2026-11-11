# serial — one instruction per layer

Field `cur` at position i is the ball position after instruction i. Layer L (1..n) has one head
copying `cur` from the previous position (offset -1: a unique match in a softmax layer, the free
conv shift in a linear one), and the MLP sets `cur = PERMS[tok][prev]` at position L-1 (prev = 0 in
layer 1, which has no head; every other layer updates a position that has a predecessor). Readout reads
`cur` at positions 0..n-1. layers(n) = n. One pad scratch position is appended only so the
alphabet-5 field fits the 4·n_pos² cap at n = 1.

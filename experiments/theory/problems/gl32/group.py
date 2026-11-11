import numpy as np

from lib.side_tasks.gl32 import _shift

GEN = [tuple(_shift(v) for v in range(8)),
       tuple(v ^ 0b010 if v & 1 else v for v in range(8))]  # instruction 0 = shift, 1 = couple
elems = [tuple(range(8))]
idx = {elems[0]: 0}
for a in elems:
    for g in GEN:
        c = tuple(g[a[x]] for x in range(8))
        if c not in idx:
            idx[c] = len(elems)
            elems.append(c)
MUL = np.array([[idx[tuple(a[b[x]] for x in range(8))] for b in elems] for a in elems])  # MUL[a, b] = a after b
INS = np.array([idx[g] for g in GEN])
APPLY1 = np.array([e[1] for e in elems])
STEP = np.array(GEN)

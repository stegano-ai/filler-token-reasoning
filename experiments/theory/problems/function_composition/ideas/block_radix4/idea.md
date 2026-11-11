# block_radix4 — block-aligned radix-4 scan, delta-rule friendly (qwen)

Same algebra as `hs_radix4` (element = a function on D = 4 points = 8 bits, identity at index 255,
own + 3 partners = 32 MLP bits, x0 carried as element 0 so no final layer), but the scan is
**block-aligned** so a Gated DeltaNet layer can run it: blocks at level L have size B = 2*4^(L-1)
and a position holds the product from its block start up to itself.

The key trick for linear layers: key only on the residue `m_L = (elem // B) % 4`, 4 distinct key
tuples, far under lin_keys = 128. A query with residue q reads the *most recent* position of
residue q - m; residues are constant inside a block and the four residues of a level-(L+1) block
are distinct, so that position is exactly the end of block c - m, and a negative q - m matches
nothing — which is precisely the "partner outside my superblock" case that must act as identity.

| layer | kind | what |
|---|---|---|
| 0 | linear | **gather**: 3 conv heads (pos -1..-3) read this table's other slots; 4 heads keyed on `(par = elem%2 flipped, slot = v)` read the previous table — the delta rule's most recent opposite-parity slot-v position is table k-1. MLP writes g = f_k o f_{k-1} at odd elements, f_k at even ones (block-aligned, B = 2). 18 bits. On a full/window layer the same 4 heads become pos offsets -4..-7. |
| L | linear | **level L**: 3 heads `{m_L: Q(m_L, -m)}`, MLP `acc = own o p1 o p2 o p3`. |
| L | full / window | **level L**: 3 heads `{k_L: Q(k_L,-m), m_L: Q(m_L,-m), e_L: 1}` — `k_L` picks the block, `m_L` keeps it inside the same level-(L+1) block, `e_L` its end. Costs 2 extra fields for that level. |
| — | window, too far | **no-op** when `3 * B * D >= window`. |

Under qwen (linear, linear, linear, full) every layer is usable: levels on linear layers use the
residue head, the level that lands on layer 3 (and 7) uses the three-condition head. Fields at
n = 1024: tok, pos, r, g, slot, par, parq + m1..m5 + k3, e3 = 14 <= F = 16. Because all levels
land on non-linear layers under bf16, the field count only fits there up to n = 64 (the unit-test
range); bf16 is served by `hs_radix4`.

Closed form: `1 + ceil(log4((n+1)/2))` on qwen — 6 layers at n = 1024, model depth 40.

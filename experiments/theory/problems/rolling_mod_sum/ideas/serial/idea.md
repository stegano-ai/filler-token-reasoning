# serial

Layer L (L = 1..n-1): one head copies `s` from position pos-1 (unique match, read as round(frac) of
the 1-bit field: at n_pos = 2 the softmax leak from self is 1/257, which rounds up to 2^-8 and
makes sym() report a tie; round() absorbs it); the MLP at
every position i >= L sets s(i) = s(i-1) + tok(i) mod 2. After layer L position i holds the
parity of tok[i-L..i], so position i is final after i layers and the last position after n-1.
The embedding seeds s = tok, so n = 1 needs no layers. Readout: reasoning = s at every input
position, output = s at the last one (reasoning_in_output = false, as in latent_sft base.yaml).
No scratch positions, one head per layer, no multi-key averaging anywhere.

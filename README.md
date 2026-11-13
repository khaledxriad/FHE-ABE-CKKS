# FHE-ABE: Policy-Bound CKKS Key Domains for Secure IoT Analytics

Reference prototype and benchmark data for FHE-ABE, a construction that binds a CKKS homomorphic-encryption key set to a ciphertext-policy attribute-based access policy, for IoT analytics.

A CKKS key set is derived from a 32-byte epoch seed. The seed is wrapped in a ciphertext-policy ABE envelope (Waters-style LSSS, BN254), and a semi-trusted gateway performs the pairing work so that the user needs one group exponentiation. Revocation is done by epoch change.

Author: Khaled Riad (King Faisal University, Saudi Arabia).

## Contents

| File | Purpose |
|------|---------|
| `bn254.py` | BN254 optimal-ate pairing, with a self-test |
| `abe.py` | CP-ABE key encapsulation, LSSS conversion, blinded transform key |
| `ckks.py` | RNS-CKKS with hybrid key switching and Galois rotations, seed-derived keys |
| `scheme.py` | Glue: provision, rewrap, gateway transform, user finish, epoch change |
| `bench.py` | Experiments E1 to E5; writes `results/*.csv` |
| `plots.py` | Charts in `figs/` (c1, c4, c5, c6, c8, c9), made from `results/` |
| `results/` | Raw measurements |
| `figs/` | Charts generated from the measurements |

## Usage

```bash
pip install -r requirements.txt
python3 bn254.py            # pairing self-test (bilinearity, group order)
python3 bench.py results    # experiments E1-E5, about 10-20 minutes
python3 plots.py            # regenerates the charts in figs/
```

## Notes

- Research prototype: not constant-time, not hardened, no side-channel protection. Do not use it to protect real data.
- All absolute times are those of unoptimised pure Python (single thread). Use them for ratios and scaling, not as performance claims.
- Inputs are synthetic and seeded.

## License

MIT. See `LICENSE`.

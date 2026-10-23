# Policy-Bound Homomorphic Key Domains (FHE-ABE)

Reference prototype and benchmark data for the paper
**"Policy-Bound Homomorphic Key Domains: Integrating CKKS Encryption with Attribute-Based Access Control for IoT Analytics"**
by Khaled Riad (khaled.riad@science.zu.edu.eg and kriad@kfu.edu.sa).

A CKKS key set is derived from a 32-byte epoch seed. The seed is wrapped in a ciphertext-policy ABE envelope
(Waters-style LSSS, BN254), and a semi-trusted gateway performs the pairing work so that the user needs one
group exponentiation. Revocation is done by epoch change.

## Contents

| File | Purpose |
|------|---------|
| `bn254.py` | BN254 optimal-ate pairing, with a self-test |
| `abe.py` | CP-ABE key encapsulation, LSSS conversion, blinded transform key |
| `ckks.py` | RNS-CKKS with hybrid key switching and Galois rotations, seed-derived keys |
| `scheme.py` | Glue: provision, rewrap, gateway transform, user finish, epoch change |
| `bench.py` | Experiments E1 to E5; writes `results/*.csv` |
| `plots.py` | Charts `figs/c1` to `figs/c10` from `results/` |
| `results/` | Raw measurements used in the paper |
| `figs/` | Charts used in the paper |

## Usage

```bash
pip install -r requirements.txt
python3 bn254.py            # pairing self-test (bilinearity, group order)
python3 bench.py results    # experiments E1-E5, about 10-20 minutes
python3 plots.py            # regenerates figs/c1..c10
```

## Notes

- Research prototype: not constant-time, not hardened, no side-channel protection. Do not use it to protect real data.
- All absolute times are those of unoptimised pure Python (single thread). Use them for ratios and scaling, not as performance claims.
- Inputs are synthetic and seeded.

## Citation

If you use this code, please cite the paper (details to be added after publication).

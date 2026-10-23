"""Waters-style LSSS CP-ABE key encapsulation on BN254 (asymmetric adaptation),
with Green-Hohenberger-Waters style outsourced (transform) decryption."""
import random
import bn254 as B

R = B.R

# ------------------------------------------------------------------ policies
# policy: ("attr", name) | ("and", p1, p2, ...) | ("or", p1, p2, ...)
def A(name): return ("attr", name)
def AND(*ps): return ("and",) + tuple(ps)
def OR(*ps): return ("or",) + tuple(ps)

def _binarize(p):
    if p[0] == "attr": return p
    kids = [_binarize(c) for c in p[1:]]
    node = kids[-1]
    for c in reversed(kids[:-1]):
        node = (p[0], c, node)
    return node

def policy_to_lsss(policy):
    """Lewko-Waters conversion for AND/OR trees. Returns (rows, rho, ncols)."""
    policy = _binarize(policy)
    rows, rho = [], []
    counter = [1]
    def rec(node, vec):
        if node[0] == "attr":
            rows.append(vec[:]); rho.append(node[1]); return
        if node[0] == "or":
            rec(node[1], vec[:]); rec(node[2], vec[:])
        else:
            c = counter[0]
            v = vec + [0] * (c - len(vec))
            counter[0] += 1
            rec(node[1], v + [1])
            rec(node[2], [0] * c + [R - 1])
    rec(policy, [1])
    n = counter[0]
    rows = [r + [0] * (n - len(r)) for r in rows]
    return rows, rho, n

def _solve(rows, target_dim):
    """find w with sum_i w_i rows[i] = (1,0,...,0) mod R, or None."""
    m = len(rows); n = target_dim
    # solve M^T w = e1: n equations, m unknowns
    mat = [[rows[i][j] % R for i in range(m)] + [1 if j == 0 else 0] for j in range(n)]
    piv = []; r = 0
    for c in range(m):
        pr = next((i for i in range(r, n) if mat[i][c]), None)
        if pr is None: continue
        mat[r], mat[pr] = mat[pr], mat[r]
        inv = pow(mat[r][c], -1, R)
        mat[r] = [(v * inv) % R for v in mat[r]]
        for i in range(n):
            if i != r and mat[i][c]:
                f = mat[i][c]
                mat[i] = [(a - f * b) % R for a, b in zip(mat[i], mat[r])]
        piv.append(c); r += 1
        if r == n: break
    for i in range(r, n):
        if mat[i][m]: return None
    w = [0] * m
    for i, c in enumerate(piv): w[c] = mat[i][m]
    return w

def satisfies(policy, attrs):
    rows, rho, n = policy_to_lsss(policy)
    idx = [i for i, a in enumerate(rho) if a in attrs]
    if not idx: return False
    return _solve([rows[i] for i in idx], n) is not None

# ------------------------------------------------------------------ scheme
class PK: pass

def setup(universe):
    alpha, a = B.rand_zr(), B.rand_zr()
    pk = PK()
    pk.universe = list(universe)
    pk.g1_a = B.g1_mul(B.G1, a)
    pk.egg_alpha = B.gt_pow(B.pairing(B.G2, B.G1), alpha)
    eta = {x: B.rand_zr() for x in universe}
    pk.h1 = {x: B.g1_mul(B.G1, eta[x]) for x in universe}
    pk.h2 = {x: B.g2_mul(B.G2, eta[x]) for x in universe}
    msk = (alpha, a)
    pk.g2_a = B.g2_mul(B.G2, a)
    return pk, msk

def keygen(pk, msk, attrs, blind=True):
    """Return (sk, z). With blind=True the key is a transformation key and z the retrieval key."""
    alpha, a = msk
    t = B.rand_zr()
    K = B.g2_add(B.g2_mul(B.G2, alpha), B.g2_mul(B.G2, (a * t) % R))
    L = B.g2_mul(B.G2, t)
    Kx = {x: B.g2_mul(pk.h2[x], t) for x in attrs}
    z = 1
    if blind:
        z = B.rand_zr(); zi = pow(z, -1, R)
        K = B.g2_mul(K, zi); L = B.g2_mul(L, zi)
        Kx = {x: B.g2_mul(v, zi) for x, v in Kx.items()}
    return {"K": K, "L": L, "Kx": Kx, "attrs": set(attrs)}, z

def encaps(pk, policy):
    """Return (ciphertext, session_key bytes)."""
    rows, rho, n = policy_to_lsss(policy)
    s = B.rand_zr()
    v = [s] + [B.rand_zr() for _ in range(n - 1)]
    Cp = B.g1_mul(B.G1, s)
    Ci, Di = [], []
    for i, row in enumerate(rows):
        lam = sum(a * b for a, b in zip(row, v)) % R
        ri = B.rand_zr()
        ci = B.g1_add(B.g1_mul(pk.g1_a, lam), B.g1_neg(B.g1_mul(pk.h1[rho[i]], ri)))
        Ci.append(ci); Di.append(B.g1_mul(B.G1, ri))
    key = B.kdf_gt(B.gt_pow(pk.egg_alpha, s))
    return {"policy": policy, "rows": rows, "rho": rho, "n": n, "Cp": Cp, "Ci": Ci, "Di": Di}, key

def transform(ct, sk):
    """Executed by the (untrusted) gateway: returns e(g,g)^{alpha s / z} (as GT element)."""
    idx = [i for i, a in enumerate(ct["rho"]) if a in sk["attrs"]]
    w = _solve([ct["rows"][i] for i in idx], ct["n"])
    if w is None: return None
    pairs = [(ct["Cp"], sk["K"])]
    # prod_i (e(C_i,L) e(D_i,K_rho))^{w_i}  -> move exponents into G1 points; invert via negation
    for wi, i in zip(w, idx):
        if wi == 0: continue
        pairs.append((B.g1_neg(B.g1_mul(ct["Ci"][i], wi)), sk["L"]))
        pairs.append((B.g1_neg(B.g1_mul(ct["Di"][i], wi)), sk["Kx"][ct["rho"][i]]))
    return B.multi_pairing(pairs)

def finish(gt_val, z):
    return B.kdf_gt(B.gt_pow(gt_val, z))

def decaps(ct, sk, z=1):
    g = transform(ct, sk)
    return None if g is None else finish(g, z)

if __name__ == "__main__":
    import time
    uni = ["doctor", "cardio", "nurse", "icu", "analyst", "kfu", "audit", "er"]
    pk, msk = setup(uni)
    pol = AND(OR(A("doctor"), A("nurse")), A("icu"), OR(A("cardio"), A("kfu")))
    ct, key = encaps(pk, pol)
    sk, z = keygen(pk, msk, {"nurse", "icu", "kfu"})
    t = time.time(); k2 = decaps(ct, sk, z); print("decaps ok", k2 == key, time.time() - t)
    sk2, z2 = keygen(pk, msk, {"doctor", "cardio"})
    print("unsat ->", decaps(ct, sk2, z2))
    # collusion: pool two keys' components must not decrypt (structure check only)
    print("satisfies", satisfies(pol, {"nurse", "icu", "kfu"}), satisfies(pol, {"doctor", "cardio"}))

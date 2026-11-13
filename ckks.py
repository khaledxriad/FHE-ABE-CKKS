"""Compact RNS-CKKS implementation (NumPy only) used by the FHE-ABE prototype.
Research prototype: SHAKE-256 based sampling, 31-bit RNS limbs, hybrid key switching
with one special prime.  NOT hardened for production use."""
import hashlib
import os
import numpy as np

# ---------------------------------------------------------------- number theory
def _is_prime(n):
    if n < 2: return False
    for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % sp == 0: return n == sp
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1: break
        else:
            return False
    return True

def gen_primes(count, start, N, exclude=()):
    """primes q = 1 mod 2N, searching downward from `start`."""
    out = []; m = 2 * N
    q = (start // m) * m + 1
    while len(out) < count:
        if q < start and q not in exclude and _is_prime(q): out.append(q)
        q -= m
    return out

def _prim_root_2n(q, N):
    m = 2 * N
    g = 2
    while True:
        psi = pow(g, (q - 1) // m, q)
        if pow(psi, N, q) == q - 1: return psi
        g += 1

def _bitrev(i, bits):
    return int(format(i, "0%db" % bits)[::-1], 2)

# ---------------------------------------------------------------- randomness
class XOF:
    """Deterministic byte stream from a seed (SHAKE-256), for seed-derived keys."""
    def __init__(self, seed): self.seed = seed; self.ctr = 0
    def bytes(self, n):
        self.ctr += 1
        return hashlib.shake_256(self.seed + self.ctr.to_bytes(8, "big")).digest(n)
    def uniform(self, q, N):  # one row of uniform residues
        b = np.frombuffer(self.bytes(8 * N), dtype=np.uint64)
        return (b % np.uint64(q)).astype(np.int64)
    def ternary(self, N):
        b = np.frombuffer(self.bytes(N), dtype=np.uint8)
        return (b % 3).astype(np.int64) - 1
    def cbd(self, N, eta=21):
        raw = np.frombuffer(self.bytes(6 * N), dtype=np.uint8).reshape(N, 6)
        bits = np.unpackbits(raw, axis=1)
        return bits[:, :eta].sum(1).astype(np.int64) - bits[:, eta:2 * eta].sum(1).astype(np.int64)

# ---------------------------------------------------------------- context
class Ctx:
    def __init__(self, N=4096, depth=2):
        self.N = N; self.logN = N.bit_length() - 1
        self.depth = depth
        base = gen_primes(2, 2 ** 31, N)
        resc = gen_primes(depth, 2 ** 30 + 2 ** 20, N, exclude=base) if depth else []
        # choose rescale primes just under 2^30 for scale ~ 2^30
        resc = gen_primes(depth, 2 ** 30, N, exclude=base) if depth else []
        special = gen_primes(1, 2 ** 31 - 2 ** 22, N, exclude=base + resc)
        self.chain = base + resc               # K primes
        self.K = len(self.chain)
        self.allq = self.chain + special      # index K is the special prime
        self.p = special[0]
        self.q = np.array(self.allq, dtype=np.int64)
        self.delta = float(2 ** 30)
        self._tables()
        self.slots = N // 2
        # canonical-embedding index map
        self.rot5 = np.array([pow(5, j, 2 * N) for j in range(self.slots)], dtype=np.int64)

    def _tables(self):
        N, lg = self.N, self.logN
        T = len(self.allq)
        self.psi_rev = np.zeros((T, N), dtype=np.int64)
        self.psi_inv_rev = np.zeros((T, N), dtype=np.int64)
        self.ninv = np.zeros(T, dtype=np.int64)
        br = [_bitrev(i, lg) for i in range(N)]
        for t, q in enumerate(self.allq):
            psi = _prim_root_2n(q, N); psi_inv = pow(psi, -1, q)
            pw = [1] * N; pwi = [1] * N
            for i in range(1, N):
                pw[i] = pw[i - 1] * psi % q; pwi[i] = pwi[i - 1] * psi_inv % q
            self.psi_rev[t] = np.array([pw[br[i]] for i in range(N)], dtype=np.int64)
            self.psi_inv_rev[t] = np.array([pwi[br[i]] for i in range(N)], dtype=np.int64)
            self.ninv[t] = pow(N, -1, q)

    # ---- NTT over a selection of prime indices (rows)
    def ntt(self, a, idx):
        q = self.q[idx][:, None]; W = self.psi_rev[idx]
        a = a.copy(); k = a.shape[0]; N = self.N
        m, t = 1, N
        while m < N:
            t //= 2
            v = a.reshape(k, m, 2, t)
            w = W[:, m:2 * m][:, :, None]
            U = v[:, :, 0, :]; V = (v[:, :, 1, :] * w) % q[:, :, None]
            lo = (U + V) % q[:, :, None]; hi = (U - V) % q[:, :, None]
            v[:, :, 0, :] = lo; v[:, :, 1, :] = hi
            m *= 2
        return a

    def intt(self, a, idx):
        q = self.q[idx][:, None]; W = self.psi_inv_rev[idx]
        a = a.copy(); k = a.shape[0]; N = self.N
        t, m = 1, N
        while m > 1:
            h = m // 2
            v = a.reshape(k, h, 2, t)
            w = W[:, h:2 * h][:, :, None]
            U = v[:, :, 0, :].copy(); V = v[:, :, 1, :].copy()
            v[:, :, 0, :] = (U + V) % q[:, :, None]
            v[:, :, 1, :] = (((U - V) % q[:, :, None]) * w) % q[:, :, None]
            t *= 2; m = h
        return (a * self.ninv[idx][:, None]) % q

    # ---- helpers
    def from_int_coeffs(self, c, idx):
        """small signed integer coefficient vector -> NTT form on rows idx."""
        rows = np.stack([np.mod(c, self.allq[i]) for i in idx]).astype(np.int64)
        return self.ntt(rows, idx)

    # ---- encode / decode (canonical embedding)
    def encode(self, z, scale, idx):
        N = self.N
        z = np.asarray(z, dtype=np.complex128)
        zz = np.zeros(self.slots, dtype=np.complex128); zz[:len(z)] = z
        A = np.zeros(2 * N, dtype=np.complex128)
        A[self.rot5] = zz
        m = (2.0 / N) * np.real(np.fft.fft(A)[:N])
        coeffs = np.rint(m * scale).astype(np.int64)
        return self.from_int_coeffs(coeffs, idx)

    def decode_coeffs(self, bigc, scale):
        N = self.N
        m = np.array([float(v) for v in bigc]) / scale
        B = np.fft.ifft(np.concatenate([m, np.zeros(N)])) * (2 * N)
        return B[self.rot5]

# ---------------------------------------------------------------- objects
class Ciphertext:
    def __init__(self, comps, scale):
        self.c = comps          # list of arrays (k, N), NTT form
        self.scale = scale
    @property
    def k(self): return self.c[0].shape[0]
    def nbytes(self):          # serialised size with 32-bit words
        return sum(x.size for x in self.c) * 4

class KeySet:
    pass

def keygen(ctx, seed, rot_steps=None, relin=True):
    """Deterministic key generation from a 32-byte seed."""
    N, K = ctx.N, ctx.K
    xs = XOF(b"sk" + seed); xa = XOF(b"a" + seed); xe = XOF(b"e" + seed)
    ks = KeySet(); ks.seed = seed
    allidx = list(range(K + 1)); chain = list(range(K))
    s = xs.ternary(N)
    ks.s_ntt = ctx.from_int_coeffs(s, allidx)
    ks.s_coef = s
    # public key (top level)
    a = np.stack([xa.uniform(ctx.allq[i], N) for i in chain])    # already "NTT form" (uniform)
    e = ctx.from_int_coeffs(xe.cbd(N), chain)
    q = ctx.q[chain][:, None]
    ks.pk = ((-(a * ks.s_ntt[chain]) % q + e) % q, a)
    def make_ksk(s_from_ntt):
        ksk = []
        for i in range(K):
            a_i = np.stack([xa.uniform(ctx.allq[j], N) for j in allidx])
            e_i = ctx.from_int_coeffs(xe.cbd(N), allidx)
            qa = ctx.q[allidx][:, None]
            b_i = (-(a_i * ks.s_ntt) % qa + e_i) % qa
            term = (ctx.p % ctx.allq[i]) * s_from_ntt[i] % ctx.allq[i]
            b_i[i] = (b_i[i] + term) % ctx.allq[i]
            ksk.append((b_i, a_i))
        return ksk
    if relin:
        s2 = (ks.s_ntt * ks.s_ntt) % ctx.q[:, None]
        ks.rlk = make_ksk(s2)
    ks.glk = {}
    for r in (rot_steps or []):
        g = pow(5, r, 2 * N)
        s_rot = apply_galois_coef(ks.s_coef, g, N)
        ks.glk[r] = make_ksk(ctx.from_int_coeffs(s_rot, allidx))
    return ks

def apply_galois_coef(a, g, N):
    out = np.zeros_like(a)
    idx = (np.arange(N) * g) % (2 * N)
    pos = idx < N
    out[idx[pos]] = a[pos]
    out[idx[~pos] - N] = -a[~pos]
    return out

def public_view(ks):
    """Everything the cloud may hold: pk + evaluation keys (no secret)."""
    v = KeySet(); v.pk = ks.pk; v.rlk = getattr(ks, "rlk", None); v.glk = ks.glk
    return v

# ---------------------------------------------------------------- operations
def encrypt(ctx, pk, z, scale=None):
    scale = scale or ctx.delta
    K = ctx.K; chain = list(range(K)); N = ctx.N
    x = XOF(os.urandom(32))
    v = ctx.from_int_coeffs(x.ternary(N), chain)
    e0 = ctx.from_int_coeffs(x.cbd(N), chain); e1 = ctx.from_int_coeffs(x.cbd(N), chain)
    m = ctx.encode(z, scale, chain)
    q = ctx.q[chain][:, None]
    c0 = ((v * pk[0]) % q + e0 + m) % q
    c1 = ((v * pk[1]) % q + e1) % q
    return Ciphertext([c0, c1], scale)

def _crt_centered(ctx, rows, idx):
    Q = 1
    for i in idx: Q *= ctx.allq[i]
    acc = np.zeros(ctx.N, dtype=object)
    for r, i in zip(rows, idx):
        qi = ctx.allq[i]; Mi = Q // qi; inv = pow(Mi % qi, -1, qi)
        acc = acc + (np.array(r.tolist(), dtype=object) * inv % qi) * Mi
    acc %= Q
    half = Q // 2
    return np.array([int(v) - Q if v > half else int(v) for v in acc], dtype=object)

def decrypt(ctx, sk_ntt, ct):
    idx = list(range(ct.k)); q = ctx.q[idx][:, None]
    acc = ct.c[0].copy(); sp = sk_ntt[idx]
    pw = sp
    for j in range(1, len(ct.c)):
        acc = (acc + ct.c[j] * pw) % q
        pw = (pw * sp) % q
    coef = ctx.intt(acc, idx)
    big = _crt_centered(ctx, coef, idx)
    return ctx.decode_coeffs(big, ct.scale)

def _trunc(ct, k):
    return Ciphertext([c[:k] for c in ct.c], ct.scale)

def mod_switch_to(ct, k):
    return _trunc(ct, k) if ct.k > k else ct

def add(ctx, a, b):
    k = min(a.k, b.k); a = _trunc(a, k); b = _trunc(b, k)
    q = ctx.q[:k][:, None]
    n = max(len(a.c), len(b.c))
    comps = []
    for i in range(n):
        if i < len(a.c) and i < len(b.c): comps.append((a.c[i] + b.c[i]) % q)
        else: comps.append((a.c[i] if i < len(a.c) else b.c[i]).copy())
    return Ciphertext(comps, a.scale)

def sub(ctx, a, b):
    nb = Ciphertext([(-c) % ctx.q[:b.k][:, None] for c in b.c], b.scale)
    return add(ctx, a, nb)

def add_plain(ctx, a, z):
    m = ctx.encode(z, a.scale, list(range(a.k)))
    q = ctx.q[:a.k][:, None]
    return Ciphertext([(a.c[0] + m) % q] + [c.copy() for c in a.c[1:]], a.scale)

def mul_plain(ctx, a, z, pscale=None):
    pscale = pscale or ctx.delta
    m = ctx.encode(z, pscale, list(range(a.k)))
    q = ctx.q[:a.k][:, None]
    return Ciphertext([(c * m) % q for c in a.c], a.scale * pscale)

def rescale(ctx, a):
    k = a.k
    idx_last = [k - 1]; ql = ctx.allq[k - 1]
    comps = []
    for c in a.c:
        last = ctx.intt(c[k - 1:k], idx_last)[0]
        last = np.where(last > ql // 2, last - ql, last)
        rows = np.stack([np.mod(last, ctx.allq[j]) for j in range(k - 1)])
        rows = ctx.ntt(rows, list(range(k - 1)))
        q = ctx.q[:k - 1][:, None]
        inv = np.array([pow(ql, -1, ctx.allq[j]) for j in range(k - 1)], dtype=np.int64)[:, None]
        comps.append(((c[:k - 1] - rows) % q * inv) % q)
    return Ciphertext(comps, a.scale / ql)

def _keyswitch(ctx, c, ksk):
    """c: (k,N) NTT form on chain rows 0..k-1 -> pair (ks0, ks1) on the same rows."""
    k = c.shape[0]; chain = list(range(k)); ext = chain + [ctx.K]
    coef = ctx.intt(c, chain)
    qext = ctx.q[ext][:, None]
    acc0 = np.zeros((k + 1, ctx.N), dtype=np.int64); acc1 = acc0.copy()
    for i in range(k):
        qi = ctx.allq[i]
        d = np.where(coef[i] > qi // 2, coef[i] - qi, coef[i])
        rows = np.stack([np.mod(d, ctx.allq[j]) for j in ext])
        rows = ctx.ntt(rows, ext)
        b_i, a_i = ksk[i]
        acc0 = (acc0 + (rows * b_i[ext]) % qext) % qext
        acc1 = (acc1 + (rows * a_i[ext]) % qext) % qext
    out = []
    for acc in (acc0, acc1):
        sp = ctx.intt(acc[k:k + 1], [ctx.K])[0]
        sp = np.where(sp > ctx.p // 2, sp - ctx.p, sp)
        rows = np.stack([np.mod(sp, ctx.allq[j]) for j in chain])
        rows = ctx.ntt(rows, chain)
        q = ctx.q[:k][:, None]
        pinv = np.array([pow(ctx.p, -1, ctx.allq[j]) for j in chain], dtype=np.int64)[:, None]
        out.append(((acc[:k] - rows) % q * pinv) % q)
    return out

def mul(ctx, keys, a, b, do_rescale=True):
    k = min(a.k, b.k); a = _trunc(a, k); b = _trunc(b, k)
    q = ctx.q[:k][:, None]
    d0 = (a.c[0] * b.c[0]) % q
    d1 = (a.c[0] * b.c[1] + a.c[1] * b.c[0]) % q
    d2 = (a.c[1] * b.c[1]) % q
    r0, r1 = _keyswitch(ctx, d2, keys.rlk)
    ct = Ciphertext([(d0 + r0) % q, (d1 + r1) % q], a.scale * b.scale)
    return rescale(ctx, ct) if do_rescale else ct

def rotate(ctx, keys, a, r):
    k = a.k; N = ctx.N; chain = list(range(k))
    g = pow(5, r, 2 * N)
    c0 = ctx.ntt(np.stack([apply_galois_coef(row, g, N) for row in ctx.intt(a.c[0], chain)]), chain)
    c1 = ctx.ntt(np.stack([apply_galois_coef(row, g, N) for row in ctx.intt(a.c[1], chain)]), chain)
    r0, r1 = _keyswitch(ctx, c1, keys.glk[r])
    q = ctx.q[:k][:, None]
    return Ciphertext([(c0 + r0) % q, r1], a.scale)

def sum_slots(ctx, keys, a, n):
    """sum of the first n slots (n power of two) replicated into every slot (needs power-of-two rotation keys)."""
    step = 1
    while step < n:
        a = add(ctx, a, rotate(ctx, keys, a, step))
        step *= 2
    return a

def pow2_steps(n):
    s, out = 1, []
    while s < n: out.append(s); s *= 2
    return out

if __name__ == "__main__":
    import time
    ctx = Ctx(4096, 2)
    seed = os.urandom(32)
    t = time.time(); ks = keygen(ctx, seed, rot_steps=pow2_steps(64)); print("keygen", time.time() - t)
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 64); y = rng.uniform(-1, 1, 64)
    cx = encrypt(ctx, ks.pk, x); cy = encrypt(ctx, ks.pk, y)
    print("dec err", np.max(np.abs(decrypt(ctx, ks.s_ntt, cx)[:64].real - x)))
    print("add err", np.max(np.abs(decrypt(ctx, ks.s_ntt, add(ctx, cx, cy))[:64].real - (x + y))))
    t = time.time(); cm = mul(ctx, ks, cx, cy); print("mul", time.time() - t)
    print("mul err", np.max(np.abs(decrypt(ctx, ks.s_ntt, cm)[:64].real - x * y)), cm.k)
    cs = sum_slots(ctx, ks, cx, 64)
    print("sum err", abs(decrypt(ctx, ks.s_ntt, cs)[0].real - x.sum()))
    cp = rescale(ctx, mul_plain(ctx, cx, y))
    print("mulplain err", np.max(np.abs(decrypt(ctx, ks.s_ntt, cp)[:64].real - x * y)))

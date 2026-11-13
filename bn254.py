"""Minimal pure-Python BN254 (alt_bn128) optimal-ate pairing.
Reference implementation written for the FHE-ABE prototype (no external libs).
Correctness is checked by bilinearity / non-degeneracy tests in selftest()."""
import hashlib
import secrets

P = 21888242871839275222246405745257275088696311157297823662689037894645226208583
R = 21888242871839275222246405745257275088548364400416034343698204186575808495617
ATE = 29793968203157093288
FINAL_EXP = (P ** 12 - 1) // R

# ---------- Fp2 = Fp[i]/(i^2+1), elements are tuples (a, b) ----------
def f2add(x, y): return ((x[0] + y[0]) % P, (x[1] + y[1]) % P)
def f2sub(x, y): return ((x[0] - y[0]) % P, (x[1] - y[1]) % P)
def f2neg(x): return ((-x[0]) % P, (-x[1]) % P)
def f2mul(x, y):
    return ((x[0] * y[0] - x[1] * y[1]) % P, (x[0] * y[1] + x[1] * y[0]) % P)
def f2inv(x):
    n = pow((x[0] * x[0] + x[1] * x[1]) % P, -1, P)
    return ((x[0] * n) % P, (-x[1] * n) % P)
def f2sc(x, k): return ((x[0] * k) % P, (x[1] * k) % P)

# curve y^2 = x^3 + 3 over Fp (G1), y^2 = x^3 + 3/(9+i) over Fp2 (G2)
B1 = 3
B2 = f2mul((3, 0), f2inv((9, 1)))
G1 = (1, 2)
G2 = (
    (10857046999023057135944570762232829481370756359578518086990519993285655852781,
     11559732032986387107991004021392285783925812861821192530917403151452391805634),
    (8495653923123431417604973247489272438418190587263600148770280649306958101930,
     4082367875863433681332203403145435568316851327593401208105741076214120093531),
)

# ---------- generic affine group law, parametrised by field ops ----------
class _Ops:
    def __init__(self, add, sub, mul, inv, sc, zero, one):
        self.add, self.sub, self.mul, self.inv, self.sc, self.zero, self.one = add, sub, mul, inv, sc, zero, one

OPS1 = _Ops(lambda a, b: (a + b) % P, lambda a, b: (a - b) % P, lambda a, b: (a * b) % P,
            lambda a: pow(a, -1, P), lambda a, k: (a * k) % P, 0, 1)
OPS2 = _Ops(f2add, f2sub, f2mul, f2inv, f2sc, (0, 0), (1, 0))

def pt_add(Pt, Q, o):
    if Pt is None: return Q
    if Q is None: return Pt
    x1, y1 = Pt; x2, y2 = Q
    if x1 == x2:
        if y1 != y2 or y1 == o.zero: return None
        m = o.mul(o.sc(o.mul(x1, x1), 3), o.inv(o.sc(y1, 2)))
    else:
        m = o.mul(o.sub(y2, y1), o.inv(o.sub(x2, x1)))
    x3 = o.sub(o.sub(o.mul(m, m), x1), x2)
    y3 = o.sub(o.mul(m, o.sub(x1, x3)), y1)
    return (x3, y3)

def pt_neg(Pt, o):
    return None if Pt is None else (Pt[0], o.sub(o.zero, Pt[1]))

def pt_mul(Pt, k, o):
    k %= R
    res = None; add = Pt
    while k:
        if k & 1: res = pt_add(res, add, o)
        add = pt_add(add, add, o)
        k >>= 1
    return res

def g1_mul(Pt, k): return pt_mul(Pt, k, OPS1)
def g1_add(a, b): return pt_add(a, b, OPS1)
def g1_neg(a): return pt_neg(a, OPS1)
def g2_mul(Pt, k): return pt_mul(Pt, k, OPS2)
def g2_add(a, b): return pt_add(a, b, OPS2)
def g2_neg(a): return pt_neg(a, OPS2)

# ---------- Fp12 = Fp[w]/(w^12 - 18 w^6 + 82), lists of 12 ints ----------
ONE12 = [1] + [0] * 11

def f12mul(a, b):
    res = [0] * 23
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    res[i + j] += ai * bj
    for i in range(22, 11, -1):
        c = res[i]
        if c:
            res[i - 6] += 18 * c
            res[i - 12] -= 82 * c
    return [x % P for x in res[:12]]

def f12add(a, b): return [(x + y) % P for x, y in zip(a, b)]
def f12sub(a, b): return [(x - y) % P for x, y in zip(a, b)]
def f12sc(a, k): return [(x * k) % P for x in a]

def _deg(p):
    d = len(p) - 1
    while d >= 0 and p[d] == 0: d -= 1
    return d

def _prdiv(a, b):
    a = a[:]; db = _deg(b); out = [0] * (len(a))
    inv_lead = pow(b[db], -1, P)
    for i in range(_deg(a) - db, -1, -1):
        c = (a[i + db] * inv_lead) % P
        out[i] = c
        for j in range(db + 1):
            a[i + j] = (a[i + j] - c * b[j]) % P
    return out

def f12inv(x):
    lm, hm = ONE12 + [0], [0] * 13
    low, high = list(x) + [0], [82, 0, 0, 0, 0, 0, P - 18, 0, 0, 0, 0, 0, 1]
    while _deg(low) > 0:
        q = _prdiv(high, low) + [0] * 13
        nm, new = hm[:], high[:]
        for i in range(13):
            if lm[i] == 0 and low[i] == 0: continue
            for j in range(13 - i):
                nm[i + j] = (nm[i + j] - lm[i] * q[j]) % P
                new[i + j] = (new[i + j] - low[i] * q[j]) % P
        lm, low, hm, high = nm, new, lm, low
    c = pow(low[0], -1, P)
    return [(v * c) % P for v in lm[:12]]

def f12pow(a, e):
    res = ONE12[:]; base = a[:]
    while e:
        if e & 1: res = f12mul(res, base)
        base = f12mul(base, base)
        e >>= 1
    return res

def f12_from_int(v): return [v % P] + [0] * 11

def twist(Q):
    (x0, x1), (y0, y1) = Q
    nx = [0] * 12; ny = [0] * 12
    nx[2] = (x0 - 9 * x1) % P; nx[8] = x1 % P
    ny[3] = (y0 - 9 * y1) % P; ny[9] = y1 % P
    return (nx, ny)

def cast_g1(Pt): return (f12_from_int(Pt[0]), f12_from_int(Pt[1]))

def _line(x1, y1, x2, y2, xt, yt, m):
    return f12sub(f12mul(m, f12sub(xt, x1)), f12sub(yt, y1))

def miller(Q2, P1):
    """Miller loop value (before final exponentiation) for Q2 in G2, P1 in G1."""
    if Q2 is None or P1 is None: return ONE12[:]
    Q = twist(Q2); Pc = cast_g1(P1)
    xp, yp = Pc
    f = ONE12[:]
    Rx, Ry = Q
    def dbl(Rx, Ry):
        m = f12mul(f12sc(f12mul(Rx, Rx), 3), f12inv(f12sc(Ry, 2)))
        ln = f12sub(f12mul(m, f12sub(xp, Rx)), f12sub(yp, Ry))
        nx = f12sub(f12sub(f12mul(m, m), Rx), Rx)
        ny = f12sub(f12mul(m, f12sub(Rx, nx)), Ry)
        return nx, ny, ln
    def add(Rx, Ry, Sx, Sy):
        if Rx == Sx:
            if Ry == Sy: return dbl(Rx, Ry)
            return None, None, f12sub(xp, Rx)
        m = f12mul(f12sub(Sy, Ry), f12inv(f12sub(Sx, Rx)))
        ln = f12sub(f12mul(m, f12sub(xp, Rx)), f12sub(yp, Ry))
        nx = f12sub(f12sub(f12mul(m, m), Rx), Sx)
        ny = f12sub(f12mul(m, f12sub(Rx, nx)), Ry)
        return nx, ny, ln
    for i in range(63, -1, -1):
        Rx, Ry, ln = dbl(Rx, Ry)
        f = f12mul(f12mul(f, f), ln)
        if ATE & (1 << i):
            Rx, Ry, ln = add(Rx, Ry, Q[0], Q[1])
            f = f12mul(f, ln)
    # Frobenius corrections
    Q1x = f12pow(Q[0], P); Q1y = f12pow(Q[1], P)
    nQ2x = f12pow(Q1x, P); nQ2y = f12sub([0] * 12, f12pow(Q1y, P))
    Rx, Ry, ln = add(Rx, Ry, Q1x, Q1y)
    f = f12mul(f, ln)
    Rx, Ry, ln = add(Rx, Ry, nQ2x, nQ2y)
    f = f12mul(f, ln)
    return f

def final_exp(f): return f12pow(f, FINAL_EXP)

def pairing(Q2, P1): return final_exp(miller(Q2, P1))

def multi_pairing(pairs):
    """prod e(P_i, Q_i) with a single final exponentiation; pairs = [(P1, Q2), ...]."""
    f = ONE12[:]
    for P1, Q2 in pairs:
        f = f12mul(f, miller(Q2, P1))
    return final_exp(f)

def gt_mul(a, b): return f12mul(a, b)
def gt_pow(a, e): return f12pow(a, e % R)
def gt_inv(a): return f12inv(a)
def rand_zr(): return secrets.randbelow(R - 1) + 1

def gt_bytes(a): return b"".join(int(v).to_bytes(32, "big") for v in a)
def kdf_gt(a, n=32): return hashlib.shake_256(gt_bytes(a)).digest(n)

def on_curve_g1(Pt): return (Pt[1] * Pt[1] - Pt[0] ** 3 - B1) % P == 0
def on_curve_g2(Pt):
    x, y = Pt
    return f2sub(f2mul(y, y), f2add(f2mul(x, f2mul(x, x)), B2)) == (0, 0)

def selftest():
    assert on_curve_g1(G1) and on_curve_g2(G2)
    assert g1_mul(G1, R) is None and g2_mul(G2, R) is None, "bad group order"
    e = pairing(G2, G1)
    assert e != ONE12, "degenerate"
    assert f12pow(e, R) == ONE12, "wrong order"
    a, b = 1234567, 7654321
    lhs = pairing(g2_mul(G2, b), g1_mul(G1, a))
    rhs = gt_pow(e, a * b)
    assert lhs == rhs, "not bilinear"
    assert multi_pairing([(g1_mul(G1, a), G2), (G1, g2_mul(G2, b))]) == gt_pow(e, a + b)
    return True

if __name__ == "__main__":
    import time
    t = time.time(); print(selftest(), time.time() - t)
    t = time.time(); pairing(G2, G1); print("pairing", time.time() - t)

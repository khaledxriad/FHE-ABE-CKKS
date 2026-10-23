import csv, json, os, sys, time, hashlib, statistics as st
import numpy as np
import bn254 as B, ckks, abe, scheme
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
os.makedirs(OUT, exist_ok=True)
def now(): return time.perf_counter()
def timeit(f, n):
    xs = []
    for _ in range(n):
        t = now(); r = f(); xs.append((now() - t) * 1000.0)
    return st.mean(xs), (st.stdev(xs) if n > 1 else 0.0), r
def wcsv(name, rows, header):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(r)
def log(*a): print(*a, flush=True)

# ---------------------------------------------------------------- E1 primitives
log("E1 primitives")
P1 = B.g1_mul(B.G1, B.rand_zr()); Q2 = B.g2_mul(B.G2, B.rand_zr())
e = B.pairing(Q2, P1); k = B.rand_zr()
prim = {}
prim["pairing"] = timeit(lambda: B.pairing(Q2, P1), 10)[:2]
prim["miller_loop"] = timeit(lambda: B.miller(Q2, P1), 10)[:2]
prim["final_exp"] = timeit(lambda: B.final_exp(B.miller(Q2, P1)), 5)[:2]
prim["g1_mul"] = timeit(lambda: B.g1_mul(P1, k), 50)[:2]
prim["g2_mul"] = timeit(lambda: B.g2_mul(Q2, k), 30)[:2]
prim["gt_exp"] = timeit(lambda: B.gt_pow(e, k), 20)[:2]
prim["gt_mul"] = timeit(lambda: B.gt_mul(e, e), 200)[:2]
def h2g1():
    x = int.from_bytes(os.urandom(32), "big") % B.P
    while True:
        y2 = (x ** 3 + 3) % B.P
        y = pow(y2, (B.P + 1) // 4, B.P)
        if y * y % B.P == y2: return (x, y)
        x += 1
prim["hash_to_g1"] = timeit(h2g1, 50)[:2]
# symmetric / ECC baselines
key = os.urandom(32); aes = AESGCM(key); data = np.random.default_rng(0).uniform(-1, 1, 4096).tobytes()
nonce = os.urandom(12)
prim["aesgcm_enc_32KB"] = timeit(lambda: aes.encrypt(nonce, data, None), 200)[:2]
ctx_aes = aes.encrypt(nonce, data, None)
prim["aesgcm_dec_32KB"] = timeit(lambda: aes.decrypt(nonce, ctx_aes, None), 200)[:2]
recv = X25519PrivateKey.generate().public_key()
def ecies_wrap():
    eph = X25519PrivateKey.generate()
    sh = eph.exchange(recv)
    kk = HKDF(hashes.SHA256(), 32, None, b"wrap").derive(sh)
    return AESGCM(kk).encrypt(b"\0" * 12, os.urandom(32), None)
prim["ecies_wrap_32B"] = timeit(ecies_wrap, 200)[:2]
wcsv("e1_primitives.csv", [(k_, v[0], v[1]) for k_, v in prim.items()], ["op", "mean_ms", "std_ms"])
log({k_: round(v[0], 3) for k_, v in prim.items()})

# ---------------------------------------------------------------- E2 CKKS micro-benchmarks
log("E2 CKKS")
rows = []
cfgs = [(8192, 1), (8192, 2), (8192, 3), (16384, 2)]
for N, D in cfgs:
    ctx = ckks.Ctx(N, D)
    logqp = sum(int(q).bit_length() for q in ctx.allq)
    seed = os.urandom(32)
    t_kg = timeit(lambda: ckks.keygen(ctx, seed, rot_steps=[]), 2)[0]
    ks = ckks.keygen(ctx, seed, rot_steps=[1])
    t_gk = timeit(lambda: ckks.keygen(ctx, seed, rot_steps=[1]), 2)[0] - t_kg
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, ctx.slots); y = rng.uniform(-1, 1, ctx.slots)
    t_enc, s_enc, cx = timeit(lambda: ckks.encrypt(ctx, ks.pk, x), 5)
    cy = ckks.encrypt(ctx, ks.pk, y)
    t_dec = timeit(lambda: ckks.decrypt(ctx, ks.s_ntt, cx), 5)[0]
    t_add = timeit(lambda: ckks.add(ctx, cx, cy), 20)[0]
    t_mul, _, cm = timeit(lambda: ckks.mul(ctx, ks, cx, cy), 3)
    t_rot = timeit(lambda: ckks.rotate(ctx, ks, cx, 1), 3)[0]
    err_f = float(np.max(np.abs(ckks.decrypt(ctx, ks.s_ntt, cx).real - x)))
    err_m = float(np.max(np.abs(ckks.decrypt(ctx, ks.s_ntt, cm).real - x * y)))
    rows.append((N, D, logqp, ctx.slots, round(t_kg, 1), round(t_gk, 1), round(t_enc, 2), round(t_dec, 2),
                 round(t_add, 3), round(t_mul, 2), round(t_rot, 2), cx.nbytes(), cm.nbytes(), err_f, err_m))
    log(rows[-1])
wcsv("e2_ckks.csv", rows, ["N", "depth", "logQP", "slots", "keygen_ms", "galois_key_ms", "enc_ms", "dec_ms", "add_ms",
                           "mul_relin_rescale_ms", "rotate_ms", "ct_fresh_bytes", "ct_mul_bytes", "err_fresh", "err_mul"])

# ---------------------------------------------------------------- E3 ABE scaling
log("E3 ABE scaling")
uni = ["a%d" % i for i in range(32)]
pk, msk = abe.setup(uni)
rows = []
for kk in [2, 4, 6, 8, 10, 12, 16, 20]:
    attrs = uni[:kk]
    pol = abe.AND(*[abe.A(a) for a in attrs])
    t_kg, _, (sk, z) = timeit(lambda: abe.keygen(pk, msk, set(attrs)), 3)
    t_enc, s_enc, (ct, key) = timeit(lambda: abe.encaps(pk, pol), 5)
    t_tr, s_tr, g = timeit(lambda: abe.transform(ct, sk), 3)
    t_fin, s_fin, key2 = timeit(lambda: abe.finish(g, z), 10)
    assert key2 == key
    ct_bytes = 64 * (1 + 2 * kk) + 48
    sk_bytes = 128 * (2 + kk)
    rows.append((kk, round(t_kg, 1), round(t_enc, 1), round(s_enc, 1), round(t_tr, 1), round(s_tr, 1), round(t_fin, 2), ct_bytes, sk_bytes))
    log(rows[-1])
wcsv("e3_abe.csv", rows, ["k", "keygen_ms", "encaps_ms", "encaps_std", "transform_ms", "transform_std", "finish_ms", "ct_bytes", "sk_bytes"])
json.dump({k_: v[0] for k_, v in prim.items()}, open(os.path.join(OUT, "prim.json"), "w"))

# ---------------------------------------------------------------- E4 end-to-end
log("E4 end-to-end")
N, D = 8192, 2
ctx = ckks.Ctx(N, D)
steps = ckks.pow2_steps(ctx.slots)
kk = 6
attrs = uni[:kk]
pol = abe.AND(abe.OR(abe.A(attrs[0]), abe.A(attrs[1])), *[abe.A(a) for a in attrs[2:]])  # mixed AND/OR policy
t_prov, _, (pub, env, seed) = timeit(lambda: scheme.provision(ctx, pk, pol, steps), 1)
log("provision ms", t_prov)
sk, z = abe.keygen(pk, msk, set(attrs[1:]))           # satisfies via second branch of the OR
t_gw, _, g = timeit(lambda: scheme.gateway_transform(env, sk), 3)
t_fin, _, seed2 = timeit(lambda: scheme.user_finish(env, g, z), 5)
assert seed2 == seed
t_sk, _, s_ntt = timeit(lambda: scheme.derive_sk(ctx, seed2), 5)
sk_bad, zb = abe.keygen(pk, msk, set(attrs[2:]))
assert scheme.open_envelope_local(env, sk_bad, zb) is None
log("unsatisfying key rejected")
rows_e4 = []
rng = np.random.default_rng(7)
slots = ctx.slots
for nct in [1, 2, 4, 8]:
    M = nct * slots
    X = rng.uniform(-1, 1, (nct, slots)); Y = rng.uniform(-1, 1, (nct, slots))
    t0 = now(); cX = [ckks.encrypt(ctx, pub.pk, X[i]) for i in range(nct)]; t_dev = (now() - t0) * 1000
    cY = [ckks.encrypt(ctx, pub.pk, Y[i]) for i in range(nct)]
    up_bytes = sum(c.nbytes() for c in cX)
    def total(cts):
        a = cts[0]
        for c in cts[1:]: a = ckks.add(ctx, a, c)
        return a
    # W1 mean
    t0 = now()
    s = ckks.sum_slots(ctx, pub, total(cX), slots)
    r1 = ckks.rescale(ctx, ckks.mul_plain(ctx, s, np.full(slots, 1.0 / M)))
    t_w1 = (now() - t0) * 1000
    t0 = now(); d1 = ckks.decrypt(ctx, s_ntt, r1); t_d1 = (now() - t0) * 1000
    e1 = abs(d1[0].real - X.mean())
    # W2 variance
    t0 = now()
    sq = [ckks.mul(ctx, pub, c, c) for c in cX]
    ssq = ckks.sum_slots(ctx, pub, total(sq), slots)
    mean_sq = ckks.rescale(ctx, ckks.mul_plain(ctx, ssq, np.full(slots, 1.0 / M)))
    ssum = ckks.sum_slots(ctx, pub, total(cX), slots)
    sm2 = ckks.mul(ctx, pub, ssum, ssum)
    mean2 = ckks.rescale(ctx, ckks.mul_plain(ctx, sm2, np.full(slots, 1.0 / (M * M))))
    var = ckks.sub(ctx, mean_sq, mean2)
    t_w2 = (now() - t0) * 1000
    e2 = abs(ckks.decrypt(ctx, s_ntt, var)[0].real - X.var())
    # W3 linear score over 64 features per record (plain weights)
    w = rng.uniform(-1, 1, 64); wt = np.tile(w, slots // 64)
    t0 = now()
    outs = []
    for c in cX:
        p = ckks.rescale(ctx, ckks.mul_plain(ctx, c, wt))
        outs.append(ckks.sum_slots(ctx, pub, p, 64))
    t_w3 = (now() - t0) * 1000
    dd = ckks.decrypt(ctx, s_ntt, outs[0]).real[::64][: slots // 64]
    ref = (X[0].reshape(-1, 64) * w).sum(1)
    e3 = float(np.max(np.abs(dd - ref)))
    # W4 encrypted dot product
    t0 = now()
    prods = [ckks.mul(ctx, pub, cX[i], cY[i]) for i in range(nct)]
    dot = ckks.sum_slots(ctx, pub, total(prods), slots)
    t_w4 = (now() - t0) * 1000
    e4 = abs(ckks.decrypt(ctx, s_ntt, dot)[0].real - (X * Y).sum())
    rows_e4.append((nct, M, round(t_dev, 1), up_bytes, round(t_w1, 1), round(t_w2, 1), round(t_w3, 1), round(t_w4, 1),
                    round(t_d1, 2), r1.nbytes(), e1, e2, e3, e4 / M))
    log(rows_e4[-1])
wcsv("e4_end2end.csv", rows_e4, ["n_ct", "readings", "device_enc_ms", "upload_bytes", "mean_ms", "variance_ms", "linscore_ms",
                                 "encdot_ms", "user_decrypt_ms", "result_ct_bytes", "err_mean", "err_var", "err_linscore",
                                 "err_dot_per_reading"])
wcsv("e4_user.csv", [(t_gw, t_fin, t_sk, t_prov)], ["gateway_transform_ms", "user_finish_ms", "derive_sk_ms", "provision_ms"])

# ---------------------------------------------------------------- E5 key distribution / revocation
log("E5 distribution")
t_ec = prim["ecies_wrap_32B"][0]
t_encaps6 = [r for r in rows if False]
abe6 = [r for r in csv.reader(open(os.path.join(OUT, "e3_abe.csv")))]
enc6 = float([r for r in abe6[1:] if r[0] == "6"][0][2])
t_keygen_domain = t_prov
rows5 = []
for U in [10, 100, 1000, 10000]:
    rows5.append((U, round(U * t_ec, 2), U * 92, round(enc6, 1), 64 * 13 + 48))
wcsv("e5_distribution.csv", rows5, ["users", "pukw_wrap_ms", "pukw_bytes", "abe_wrap_ms", "abe_bytes"])
log(rows5)
t_rewrap = timeit(lambda: scheme.rewrap(pk, pol, seed), 3)[0]
rows6 = []
for M_ in [10, 100, 1000, 10000]:
    rows6.append((M_, round(t_rewrap, 1), round(t_prov, 1), round(M_ * enc6, 1)))
wcsv("e5_revocation.csv", rows6, ["stored_items", "ours_rewrap_ms", "ours_new_epoch_full_ms", "reencrypt_all_ms"])
log(rows6)
log("DONE")

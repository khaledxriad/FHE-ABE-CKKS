"""FHE-ABE glue: ABE-wrapped FHE seed, policy-bound domains, outsourced unwrap."""
import hashlib
import os
import ckks
import abe

def derive_sk(ctx, seed):
    xs = ckks.XOF(b"sk" + seed)
    return ctx.from_int_coeffs(xs.ternary(ctx.N), list(range(ctx.K + 1)))

def _tag(seed): return hashlib.shake_256(b"tag" + seed).digest(16)

def provision(ctx, abe_pk, policy, rot_steps):
    """Owner / domain authority: fresh epoch seed -> FHE key material + ABE envelope."""
    seed = os.urandom(32)
    ks = ckks.keygen(ctx, seed, rot_steps=rot_steps)
    ct, key = abe.encaps(abe_pk, policy)
    wrap = bytes(a ^ b for a, b in zip(seed, key))
    env = {"abe": ct, "wrap": wrap, "tag": _tag(seed)}
    return ckks.public_view(ks), env, seed

def rewrap(abe_pk, policy, seed):
    """Epoch re-wrap of an existing seed under a (possibly new) policy (no FHE keygen)."""
    ct, key = abe.encaps(abe_pk, policy)
    return {"abe": ct, "wrap": bytes(a ^ b for a, b in zip(seed, key)), "tag": _tag(seed)}

def gateway_transform(env, tk):
    return abe.transform(env["abe"], tk)

def user_finish(env, gt_val, z):
    key = abe.finish(gt_val, z)
    seed = bytes(a ^ b for a, b in zip(env["wrap"], key))
    return seed if _tag(seed) == env["tag"] else None

def open_envelope_local(env, tk, z):
    g = abe.transform(env["abe"], tk)
    return None if g is None else user_finish(env, g, z)

def envelope_bytes(env, k):
    return 64 * (1 + 2 * k) + len(env["wrap"]) + len(env["tag"])

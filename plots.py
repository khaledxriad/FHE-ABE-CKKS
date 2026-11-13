import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"; F = "figs"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1, "lines.markersize": 3.2, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
     "sky": "#56B4E9", "pink": "#CC79A7", "grey": "#555555"}

def rd(name):
    with open(os.path.join(R, name)) as f:
        r = list(csv.DictReader(f))
    return r
prim = json.load(open(os.path.join(R, "prim.json")))
abe = rd("e3_abe.csv"); ck = rd("e2_ckks.csv"); e4 = rd("e4_end2end.csv")
rev = rd("e5_revocation.csv"); usr = rd("e4_user.csv")[0]

ks = np.array([int(r["k"]) for r in abe])
tr = np.array([float(r["transform_ms"]) for r in abe]); fin = np.array([float(r["finish_ms"]) for r in abe])
enc = np.array([float(r["encaps_ms"]) for r in abe]); kg = np.array([float(r["keygen_ms"]) for r in abe])
pair, gtexp, gtmul = prim["pairing"], prim["gt_exp"], prim["gt_mul"]
g1, h2 = prim["g1_mul"], prim["hash_to_g1"]
naive_dec = (2 * ks + 1) * pair + ks * gtexp + 2 * ks * gtmul

def save(fig, name):
    fig.tight_layout(pad=0.4); fig.savefig(os.path.join(F, name), bbox_inches="tight"); plt.close(fig)

# C1 ABE decryption cost on the user side
fig, ax = plt.subplots(figsize=(3.3, 2.1))
ax.plot(ks, naive_dec, "s--", color=C["grey"], label="Per-leaf pairing decryption [BSW07, Waters11] (model)")
ax.plot(ks, tr + fin, "o-", color=C["blue"], label="Ours, local (batched pairing, measured)")
ax.plot(ks, fin, "^-", color=C["verm"], label="Ours, outsourced: user part only (measured)")
ax.set_yscale("log"); ax.set_ylim(5, 2e5); ax.set_xlabel("Number of attributes in the policy (k)"); ax.set_ylabel("Time (ms, log scale)")
ax.legend(frameon=False, loc="upper left"); save(fig, "c1_abe_dec.pdf")

# C4 revocation / rotation
fig, ax = plt.subplots(figsize=(3.3, 2.0))
M = np.array([int(r["stored_items"]) for r in rev])
ax.loglog(M, [float(r["reencrypt_all_ms"]) for r in rev], "s--", color=C["grey"], label="Re-encrypt all stored ABE items (k = 6)")
ax.loglog(M, [float(r["ours_new_epoch_full_ms"]) for r in rev], "o-", color=C["blue"], label="Ours: new epoch, full key refresh")
ax.loglog(M, [float(r["ours_rewrap_ms"]) for r in rev], "^-", color=C["verm"], label="Ours: re-wrap existing seed only")
ax.set_xlabel("Items already stored in the cloud"); ax.set_ylabel("Revocation cost (ms)")
ax.legend(frameon=False, loc="upper left"); save(fig, "c4_revocation.pdf")

# C5 cloud evaluation latency
fig, ax = plt.subplots(figsize=(3.3, 2.1))
rdg = np.array([int(r["readings"]) for r in e4])
for key, lab, col, mk in [("mean_ms", "Mean", C["blue"], "o"), ("variance_ms", "Variance", C["verm"], "s"),
                          ("linscore_ms", "Linear score (64 features)", C["green"], "^"), ("encdot_ms", "Encrypted dot product", C["orange"], "d")]:
    ax.plot(rdg, np.array([float(r[key]) for r in e4]) / 1000, marker=mk, color=col, label=lab)
ax.set_xscale("log", base=2); ax.set_xlabel("Readings in the batch"); ax.set_ylabel("Cloud evaluation time (s)")
ax.legend(frameon=False); save(fig, "c5_cloud_latency.pdf")

# C6 end-to-end breakdown for the four workloads at 32,768 readings
last = e4[-1]
dev = float(last["device_enc_ms"]) / 1000
cloud = [float(last[k]) / 1000 for k in ("mean_ms", "variance_ms", "linscore_ms", "encdot_ms")]
gw = float(usr["gateway_transform_ms"]) / 1000; fin_u = (float(usr["user_finish_ms"]) + float(usr["derive_sk_ms"]) + float(last["user_decrypt_ms"])) / 1000
fig, ax = plt.subplots(figsize=(3.3, 2.1))
names = ["Mean", "Variance", "Linear\nscore", "Encrypted\ndot"]
x = np.arange(4)
ax.bar(x, [dev] * 4, color=C["sky"], label="Device encryption", width=0.55)
ax.bar(x, cloud, bottom=[dev] * 4, color=C["blue"], label="Cloud evaluation", width=0.55)
ax.bar(x, [gw] * 4, bottom=[dev + c for c in cloud], color=C["orange"], label="Gateway ABE transform", width=0.55)
ax.bar(x, [fin_u] * 4, bottom=[dev + c + gw for c in cloud], color=C["verm"], label="User unwrap + decrypt", width=0.55)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Time (s), 32,768 readings")
ax.set_ylim(0, 4.6); ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=5.5); save(fig, "c6_breakdown.pdf")

# C8 precision
fig, ax = plt.subplots(figsize=(3.3, 2.0))
e2 = ck[1]
labs = ["Fresh\nencryption", "Product\n(rescaled)", "Mean", "Variance", "Linear\nscore", "Enc. dot\n(per reading)"]
vals = [float(e2["err_fresh"]), float(e2["err_mul"]), max(float(r["err_mean"]) for r in e4),
        max(float(r["err_var"]) for r in e4), max(float(r["err_linscore"]) for r in e4),
        max(float(r["err_dot_per_reading"]) for r in e4)]
ax.bar(range(6), vals, color=[C["sky"], C["sky"], C["blue"], C["blue"], C["green"], C["orange"]], width=0.6)
ax.set_yscale("log"); ax.set_xticks(range(6)); ax.set_xticklabels(labs, fontsize=5.5)
ax.set_ylabel("Max absolute error (log scale)"); save(fig, "c8_precision.pdf")

# C9 CKKS parameter sets
fig, ax = plt.subplots(figsize=(3.3, 2.1))
labels = ["N=8192\nL=1", "N=8192\nL=2", "N=8192\nL=3", "N=16384\nL=2"]
x = np.arange(4); w = 0.2
for i, (key, lab, col) in enumerate([("enc_ms", "Encrypt", C["sky"]), ("dec_ms", "Decrypt", C["green"]),
                                     ("mul_relin_rescale_ms", "Multiply+relin+rescale", C["blue"]), ("rotate_ms", "Rotate", C["verm"])]):
    ax.bar(x + (i - 1.5) * w, [float(r[key]) for r in ck], w, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Time (ms)"); ax.legend(frameon=False, ncol=2)
save(fig, "c9_ckks.pdf")

print("figures:", sorted(os.listdir(F)))

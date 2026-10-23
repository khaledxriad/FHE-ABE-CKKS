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
dist = rd("e5_distribution.csv"); rev = rd("e5_revocation.csv"); usr = rd("e4_user.csv")[0]

ks = np.array([int(r["k"]) for r in abe])
tr = np.array([float(r["transform_ms"]) for r in abe]); fin = np.array([float(r["finish_ms"]) for r in abe])
enc = np.array([float(r["encaps_ms"]) for r in abe]); kg = np.array([float(r["keygen_ms"]) for r in abe])
pair, gtexp, gtmul = prim["pairing"], prim["gt_exp"], prim["gt_mul"]
g1, h2 = prim["g1_mul"], prim["hash_to_g1"]
naive_dec = (2 * ks + 1) * pair + ks * gtexp + 2 * ks * gtmul
bsw_enc = (2 * ks + 1) * g1 + ks * h2 + gtexp
model = {"ks": ks, "naive_dec": naive_dec, "bsw_enc": bsw_enc}
np.save(os.path.join(R, "model.npy"), model, allow_pickle=True)

def save(fig, name):
    fig.tight_layout(pad=0.4); fig.savefig(os.path.join(F, name), bbox_inches="tight"); plt.close(fig)

# C1 ABE decryption cost on the user side
fig, ax = plt.subplots(figsize=(3.3, 2.1))
ax.plot(ks, naive_dec, "s--", color=C["grey"], label="Per-leaf pairing decryption [BSW07, Waters11] (model)")
ax.plot(ks, tr + fin, "o-", color=C["blue"], label="Ours, local (batched pairing, measured)")
ax.plot(ks, fin, "^-", color=C["verm"], label="Ours, outsourced: user part only (measured)")
ax.set_yscale("log"); ax.set_ylim(5, 2e5); ax.set_xlabel("Number of attributes in the policy (k)"); ax.set_ylabel("Time (ms, log scale)")
ax.legend(frameon=False, loc="upper left"); save(fig, "c1_abe_dec.pdf")

# C2 ABE encryption cost
fig, ax = plt.subplots(figsize=(3.3, 2.0))
ax.plot(ks, enc, "o-", color=C["blue"], label="Ours / Waters-style, 3 G1 exps per leaf (measured)")
ax.plot(ks, bsw_enc, "s--", color=C["grey"], label="BSW07-style, 2 G1 exps + hash per leaf (model)")
ax.set_xlabel("Number of attributes in the policy (k)"); ax.set_ylabel("Time (ms)")
ax.legend(frameon=False, loc="upper left"); save(fig, "c2_abe_enc.pdf")

# C3 key distribution
fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.0))
U = np.array([int(r["users"]) for r in dist])
axs[0].loglog(U, [float(r["pukw_wrap_ms"]) for r in dist], "s--", color=C["grey"], label="Per-user key wrapping (X25519+AES-GCM)")
axs[0].loglog(U, [float(r["abe_wrap_ms"]) for r in dist], "o-", color=C["blue"], label="FHE-ABE envelope (k = 6)")
axs[0].set_xlabel("Authorised users"); axs[0].set_ylabel("Wrapping time (ms)"); axs[0].legend(frameon=False)
axs[1].loglog(U, [float(r["pukw_bytes"]) for r in dist], "s--", color=C["grey"])
axs[1].loglog(U, [float(r["abe_bytes"]) for r in dist], "o-", color=C["blue"])
axs[1].set_xlabel("Authorised users"); axs[1].set_ylabel("Envelope size (bytes)")
save(fig, "c3_distribution.pdf")

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

# C7 user bandwidth
fig, ax = plt.subplots(figsize=(3.3, 2.0))
Ms = np.logspace(3, 7, 30)
res_bytes = float(e4[0]["result_ct_bytes"])
ax.loglog(Ms, Ms * 8 + 28, "-", color=C["grey"], label="Download raw AES-GCM readings (8 B each)")
ax.loglog(Ms, Ms * (float(e4[0]["upload_bytes"]) / 4096), "--", color=C["orange"], label="Download all CKKS ciphertexts (64 B each)")
ax.loglog(Ms, np.full_like(Ms, res_bytes), "-", color=C["blue"], label="FHE-ABE: one result ciphertext")
ax.axvline(res_bytes / 8, color=C["verm"], lw=0.6, ls=":"); ax.set_ylim(1e3, 2e9); ax.text(res_bytes / 8 * 1.15, 2e3, "break-even: 24,576 readings", fontsize=6, color=C["verm"])
ax.set_xlabel("Readings covered by the query"); ax.set_ylabel("Bytes received by the user")
ax.legend(frameon=False, loc="upper left"); save(fig, "c7_bandwidth.pdf")

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

# C10 feature matrix (qualitative; a mark means the property is stated in the cited abstract / paper summary)
feat = ["Attribute-based\npolicy", "Computation on\nciphertexts", "Outsourced\ndecryption", "Multi-authority /\ndecentralised",
        "Lattice-based\nconstruction", "IoT-oriented", "Revocation /\nkey update"]
schemes = ["Riad (2025)", "Wu et al. (2025)", "Guo et al. (2024)", "Fugkeaw et al. (2024)", "Kukut & Sogukpinar (2024)",
           "Emura et al. (2024)", "Zhao et al. (2025)", "Lu et al. (2025)", "Stracovsky et al. (2026)", "FHE-ABE (ours)"]
M_ = np.array([
    [1, 0, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1]])
fig, ax = plt.subplots(figsize=(5.2, 2.7))
ax.imshow(M_, cmap=matplotlib.colors.ListedColormap(["#F2F2F2", "#0072B2"]), aspect="auto")
ax.set_xticks(range(len(feat))); ax.set_xticklabels(feat, fontsize=5.5)
ax.set_yticks(range(len(schemes))); ax.set_yticklabels(schemes, fontsize=6)
ax.xaxis.tick_top()
for i in range(M_.shape[0]):
    for j in range(M_.shape[1]):
        ax.text(j, i, "yes" if M_[i, j] else "-", ha="center", va="center", fontsize=5.5, color="white" if M_[i, j] else "#777777")
ax.set_xticks(np.arange(-.5, M_.shape[1], 1), minor=True); ax.set_yticks(np.arange(-.5, M_.shape[0], 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.0); ax.tick_params(which="both", length=0)
for sp in ax.spines.values(): sp.set_visible(False)
save(fig, "c10_features.pdf")
print("figures:", sorted(os.listdir(F)))

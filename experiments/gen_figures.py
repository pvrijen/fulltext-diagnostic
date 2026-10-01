#!/usr/bin/env python3
"""Regenerate all Python figures with unified colorblind-safe palette.

Palette (Tableau 10 based, colorblind safe):
  C (baseline):     #4E79A7  steel blue
  P@0% (damaged):   #E15759  red
  P@100% (FT eval): #F28E2B  orange
  C_PFD (solution): #59A14F  green
  TE (negative):    #F28E2B  orange (colorblind-safe)
  IE (positive):    #59A14F  green
  Neutral grey:     #BAB0AC
  Positive class:   #E15759  red
  Negative class:   #D3D3D3  light grey
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.metrics import silhouette_score
from umap import UMAP

# === Unified palette ===
COL_C     = '#4E79A7'  # steel blue
COL_P0    = '#E15759'  # red
COL_P100  = '#F28E2B'  # orange
COL_PFD   = '#59A14F'  # green
COL_TE    = '#F28E2B'  # orange (colorblind-safe, with hatching)
COL_IE    = '#59A14F'  # green (positive)
COL_GREY  = '#BAB0AC'  # neutral
COL_POS   = '#E15759'  # positive class dots
COL_NEG   = '#D3D3D3'  # negative class dots

OUT = 'papers/paper_b_biocuration_guide/figures'

# ============================================================
# Figure 2: Waterfall decomposition
# ============================================================
print("Generating waterfall...", flush=True)

fig, ax = plt.subplots(figsize=(4.5, 3.2))

# Data
labels = ['$C$\n(abstract)', 'Training\neffect', 'Inference\neffect', '$P$@100%\n(full text)']
values = [0.697, -0.051, 0.074, 0.720]
bottoms = [0, 0.697 - 0.051, 0.697 - 0.051, 0]  # waterfall bottoms
# C bar: full from 0; TE: hangs from C; IE: hangs from P@0%; P@100%: full from 0
bottoms = [0, 0.646, 0.646, 0]
heights = [0.697, -0.051, 0.074, 0.720]
# Actually let's do proper waterfall
bar_bottoms = [0, 0.697, 0.646, 0]
bar_heights = [0.697, -0.051, 0.074, 0.720]

colors = [COL_GREY, COL_TE, COL_IE, COL_GREY]
hatches = ['', '///', '', '']  # diagonal hatching on TE bar

bars = ax.bar(range(4), bar_heights, bottom=bar_bottoms, color=colors,
              edgecolor='#333333', linewidth=0.6, width=0.6, zorder=3)
# Apply hatching to TE bar (index 1)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)

# Bootstrap 95% CI whiskers on TE and IE bars
# TE: CI [-0.090, -0.016] → bar tip at 0.646, whiskers at 0.697+(-0.090)=0.607 and 0.697+(-0.016)=0.681
# IE: CI [+0.041, +0.111] → bar tip at 0.720, whiskers at 0.646+0.041=0.687 and 0.646+0.111=0.757
te_tip = 0.697 + (-0.051)  # 0.646
ie_tip = 0.646 + 0.074     # 0.720
te_ci_lo, te_ci_hi = 0.697 + (-0.090), 0.697 + (-0.016)  # 0.607, 0.681
ie_ci_lo, ie_ci_hi = 0.646 + 0.041, 0.646 + 0.111        # 0.687, 0.757

ax.errorbar(1, te_tip, yerr=[[te_tip - te_ci_lo], [te_ci_hi - te_tip]],
            fmt='none', ecolor='#333333', capsize=3, capthick=1.0, elinewidth=1.0, zorder=5)
ax.errorbar(2, ie_tip, yerr=[[ie_tip - ie_ci_lo], [ie_ci_hi - ie_tip]],
            fmt='none', ecolor='#333333', capsize=3, capthick=1.0, elinewidth=1.0, zorder=5)

# Connector lines
ax.plot([0.3, 1.3], [0.697, 0.697], '--', color='#666666', linewidth=0.8, zorder=2)
ax.plot([1.3, 2.3], [0.646, 0.646], '--', color='#666666', linewidth=0.8, zorder=2)

# Value labels
ax.text(0, 0.697 + 0.004, '0.697', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.text(1, 0.697 - 0.026, '$-$0.051', ha='center', va='center', fontsize=8,
        fontweight='bold', color='white')
ax.text(2, 0.646 + 0.037, '+0.074', ha='center', va='center', fontsize=8,
        fontweight='bold', color='white')
ax.text(3, 0.720 + 0.004, '0.720', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Total annotation
ax.annotate('', xy=(3, 0.738), xytext=(0, 0.738),
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2))
ax.text(1.5, 0.742, 'Total +0.023, n.s.', ha='center', fontsize=8, color='#555555')

ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('PR-AUC', fontsize=9)
ax.set_ylim(0.58, 0.77)
ax.tick_params(labelsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig(f'{OUT}/decomposition_waterfall.pdf', dpi=300, bbox_inches='tight')
print("  Saved decomposition_waterfall.pdf")

# ============================================================
# Figure 3: UMAP 2x2 with contours + silhouette bar
# ============================================================
print("Generating UMAP 2x2...", flush=True)

cache = np.load(f'{OUT}/umap_cache.npz')
feats_c, feats_p0, feats_p100, feats_pfd = cache['feats_c'], cache['feats_p0'], cache['feats_p100'], cache['feats_pfd']
y_te = cache['y_te']
pos, neg = np.where(y_te == 1)[0], np.where(y_te == 0)[0]

combined = np.vstack([feats_c, feats_p0, feats_p100, feats_pfd])
reducer = UMAP(n_neighbors=15, min_dist=0.3, random_state=42, n_components=2)
emb = reducer.fit_transform(combined)
n = len(y_te)
emb_c, emb_p0, emb_p100, emb_pfd = emb[:n], emb[n:2*n], emb[2*n:3*n], emb[3*n:]

# Silhouette on original 768-dim (cosine)
sil_c = silhouette_score(feats_c, y_te, metric='cosine')
sil_p0 = silhouette_score(feats_p0, y_te, metric='cosine')
sil_p100 = silhouette_score(feats_p100, y_te, metric='cosine')
sil_pfd = silhouette_score(feats_pfd, y_te, metric='cosine')

xlim = [emb[:, 0].min() - 1.5, emb[:, 0].max() + 1.5]
ylim = [emb[:, 1].min() - 1.5, emb[:, 1].max() + 1.5]

def add_contours(ax, e_pos, e_neg, col_pos=COL_POS, col_neg='#999999'):
    if len(e_pos) > 3:
        try:
            kde = gaussian_kde(e_pos.T, bw_method=0.5)
            xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                                  np.linspace(ylim[0], ylim[1], 100))
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=[zz.max() * 0.15], colors=[col_pos],
                      linewidths=1.2, linestyles='--', alpha=0.7)
        except: pass
    if len(e_neg) > 10:
        try:
            kde = gaussian_kde(e_neg.T, bw_method=0.4)
            xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                                  np.linspace(ylim[0], ylim[1], 100))
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=[zz.max() * 0.15], colors=[col_neg],
                      linewidths=1.0, linestyles='-', alpha=0.5)
        except: pass

fig = plt.figure(figsize=(8.5, 5))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.45], hspace=0.38, wspace=0.3,
                      left=0.06, right=0.95, top=0.88, bottom=0.10)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a, sharey=ax_a)
ax_c = fig.add_subplot(gs[1, 0], sharex=ax_a, sharey=ax_a)
ax_d = fig.add_subplot(gs[1, 1], sharex=ax_a, sharey=ax_a)
ax_bar = fig.add_subplot(gs[:, 2])

# Panel border colors match condition
panel_colors = [COL_C, COL_P0, COL_P100, COL_PFD]
umap_panels = [
    (ax_a, emb_c,    '(a) $C$ — abstract baseline', COL_C),
    (ax_b, emb_p0,   '(b) $P$@0% — FT-trained, abs. input', COL_P0),
    (ax_c, emb_p100, '(c) $P$@100% — FT-trained, FT input', COL_P100),
    (ax_d, emb_pfd,  '(d) $C_{\\mathrm{PFD}}$ — PFD-nudged', COL_PFD),
]

for ax, e, title, col in umap_panels:
    ax.scatter(e[neg, 0], e[neg, 1], c=COL_NEG, s=4, alpha=0.35, zorder=1)
    ax.scatter(e[pos, 0], e[pos, 1], c=COL_POS, s=30, zorder=3,
              edgecolors='black', linewidths=0.4)
    add_contours(ax, e[pos], e[neg])
    ax.set_title(title, fontsize=6.5, pad=3, color=col, fontweight='bold')
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.tick_params(labelsize=5)
    # Color the spines to match condition
    for spine in ax.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(1.5)

ax_a.set_ylabel('UMAP 2', fontsize=7)
ax_c.set_ylabel('UMAP 2', fontsize=7)
ax_c.set_xlabel('UMAP 1', fontsize=7)
ax_d.set_xlabel('UMAP 1', fontsize=7)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_POS,
           markeredgecolor='black', markersize=5, label='Positive'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_NEG,
           markersize=4, label='Negative'),
    Line2D([0], [0], color=COL_POS, linestyle='--', lw=1, label='Pos. envelope'),
    Line2D([0], [0], color='#999999', linestyle='-', lw=0.8, label='Neg. envelope'),
]
ax_a.legend(handles=legend_elements, fontsize=4.5, loc='upper right',
            handletextpad=0.3, borderpad=0.3)

# === Silhouette bar chart with matching colors ===
bar_labels = ['(a)\n$C$', '(b)\n$P$@0%', '(c)\n$P$@100%', '(d)\n$C_{\\mathrm{PFD}}$']
sils = [sil_c, sil_p0, sil_p100, sil_pfd]
bar_colors = [COL_C, COL_P0, COL_P100, COL_PFD]

bars = ax_bar.barh(range(4), sils, color=bar_colors, edgecolor='#333333',
                   linewidth=0.5, height=0.6)
ax_bar.set_yticks(range(4))
ax_bar.set_yticklabels(bar_labels, fontsize=7)
ax_bar.set_xlabel('Silhouette score\n(pos. vs neg.)', fontsize=7)
ax_bar.set_xlim(0, 1.0)
ax_bar.tick_params(labelsize=6)
ax_bar.invert_yaxis()
ax_bar.set_title('Class separation', fontsize=7, fontweight='bold')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

for i, (v, bar) in enumerate(zip(sils, bars)):
    ax_bar.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=7, fontweight='bold',
                color=bar_colors[i])

# Top annotation with matching colors
fig.text(0.38, 0.93,
         'a → b: Training effect (TE = $-$0.051, large shift)   |   b ≈ c: IE through classifier head, not CLS',
         ha='center', fontsize=6.5, color='#555555', fontweight='bold')

plt.savefig(f'{OUT}/umap_4panel.pdf', dpi=300, bbox_inches='tight')
print(f"  Saved umap_4panel.pdf  Sil: C={sil_c:.3f} P0={sil_p0:.3f} P100={sil_p100:.3f} PFD={sil_pfd:.3f}")

print("\nDone! Unified palette applied to all Python figures.")
print(f"  Palette: C={COL_C}, P@0%={COL_P0}, P@100%={COL_P100}, PFD={COL_PFD}")

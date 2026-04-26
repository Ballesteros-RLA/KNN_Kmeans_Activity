import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
people   = ['A', 'B', 'C', 'D', 'E', 'F']
ages     = [18, 23, 24, 41, 43, 38]
salaries = [50, 55, 70, 60, 70, 40]
labels   = ['N', 'N', 'N', 'Y', 'Y', 'Y']   # insurance

new_point = np.array([35, 100])   # Person X

# ═══════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════╗
# ║            K-NEAREST NEIGHBOURS (KNN)           ║
# ╚══════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════

print("=" * 55)
print("  K-NEAREST NEIGHBOURS (KNN) – Insurance Prediction")
print("=" * 55)
print(f"\nNew Point X: Age={new_point[0]}, Salary={new_point[1]}")
print("\n{:<8} {:<6} {:<8} {:<12} {:<20}".format(
    "Person", "Age", "Salary", "Insurance", "Euclidean Distance"))
print("-" * 55)

distances = []
for i, p in enumerate(people):
    d = np.sqrt((new_point[0] - ages[i])**2 + (new_point[1] - salaries[i])**2)
    distances.append(d)
    print(f"{p:<8} {ages[i]:<6} {salaries[i]:<8} {labels[i]:<12} {d:.2f}")

# K=3 nearest neighbours
k = 3
sorted_idx  = np.argsort(distances)
knn_indices = sorted_idx[:k]
knn_votes   = [labels[i] for i in knn_indices]
knn_result  = max(set(knn_votes), key=knn_votes.count)

print(f"\nK = {k} nearest neighbours:")
for idx in knn_indices:
    print(f"  Person {people[idx]}  →  distance={distances[idx]:.2f}, label={labels[idx]}")
print(f"\n✔  Prediction: Person X WILL {'GET' if knn_result=='Y' else 'NOT GET'} insurance  ({knn_result})")

# ═══════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════╗
# ║               K-MEANS CLUSTERING                ║
# ╚══════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════

all_ages     = ages + [new_point[0]]
all_salaries = salaries + [new_point[1]]
all_people   = people + ['X']
data = np.column_stack([all_ages, all_salaries]).astype(float)

# Initial centroids as per document
centroids = np.array([[18, 50], [41, 60], [35, 100]], dtype=float)
centroid_names = ['C1 (A)', 'C2 (D)', 'C3 (X)']

print("\n" + "=" * 55)
print("  K-MEANS CLUSTERING")
print("=" * 55)

def assign_clusters(data, centroids):
    dist_matrix = np.array([
        np.sqrt(((data - c) ** 2).sum(axis=1)) for c in centroids
    ]).T
    return np.argmin(dist_matrix, axis=1), dist_matrix

def update_centroids(data, clusters, k):
    return np.array([data[clusters == i].mean(axis=0) for i in range(k)])

cluster_history = []
centroid_history = [centroids.copy()]
max_iter = 10

for it in range(1, max_iter + 1):
    clusters, dist_matrix = assign_clusters(data, centroids)
    cluster_history.append(clusters.copy())

    print(f"\n--- Iteration {it} ---")
    print(f"Centroids: C1={tuple(np.round(centroids[0],2))}, "
          f"C2={tuple(np.round(centroids[1],2))}, "
          f"C3={tuple(np.round(centroids[2],2))}")
    print("{:<8} {:<6} {:<8} {:>12} {:>12} {:>12} {:>8}".format(
        "Person","Age","Salary","Dist→C1","Dist→C2","Dist→C3","Cluster"))
    print("-" * 68)
    for i, p in enumerate(all_people):
        print("{:<8} {:<6} {:<8} {:>12.2f} {:>12.2f} {:>12.2f} {:>8}".format(
            p, int(data[i,0]), int(data[i,1]),
            dist_matrix[i,0], dist_matrix[i,1], dist_matrix[i,2],
            clusters[i]+1))

    new_centroids = update_centroids(data, clusters, k=3)
    centroid_history.append(new_centroids.copy())

    if np.allclose(new_centroids, centroids):
        print(f"\n✔  Converged after {it} iteration(s)!")
        centroids = new_centroids
        break
    centroids = new_centroids

final_clusters = clusters
print("\nFinal Clusters:")
for c in range(3):
    members = [all_people[i] for i in range(len(all_people)) if final_clusters[i] == c]
    print(f"  Cluster {c+1}: {members}  →  Centroid {np.round(centroids[c],2)}")

# ═══════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════╗
# ║                   PLOTS                         ║
# ╚══════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d1117')
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {'N': '#ff4d6d', 'Y': '#4cc9f0'}
CLUSTER_COLORS = ['#f72585', '#4cc9f0', '#7bed9f']
DARK = '#0d1117'
GRID = '#1f2937'
TEXT = '#e2e8f0'

def base_style(ax, title):
    ax.set_facecolor('#111827')
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Age', fontsize=9)
    ax.set_ylabel('Salary (K)', fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

# ── Plot 1: KNN Raw Data ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
base_style(ax1, "① KNN – Dataset")
for i, p in enumerate(people):
    col = COLORS[labels[i]]
    ax1.scatter(ages[i], salaries[i], c=col, s=130, zorder=5,
                edgecolors='white', linewidths=0.8)
    ax1.annotate(p, (ages[i], salaries[i]),
                 textcoords="offset points", xytext=(6, 5),
                 fontsize=9, color=TEXT, fontweight='bold')
ax1.scatter(*new_point, c='#ffd60a', s=200, marker='*', zorder=6,
            edgecolors='white', linewidths=0.8)
ax1.annotate('X (new)', new_point, textcoords="offset points",
             xytext=(6, 5), fontsize=9, color='#ffd60a', fontweight='bold')
legend_handles = [
    mpatches.Patch(color=COLORS['N'], label='No Insurance'),
    mpatches.Patch(color=COLORS['Y'], label='Has Insurance'),
    plt.scatter([], [], c='#ffd60a', marker='*', s=120, label='New Point X'),
]
ax1.legend(handles=legend_handles, loc='lower right',
           facecolor='#1f2937', edgecolor=GRID,
           labelcolor=TEXT, fontsize=8)

# ── Plot 2: KNN Distances ─────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
base_style(ax2, "② KNN – Euclidean Distances")
for i, p in enumerate(people):
    col = COLORS[labels[i]]
    ax2.scatter(ages[i], salaries[i], c=col, s=130, zorder=5,
                edgecolors='white', linewidths=0.8)
    ax2.annotate(p, (ages[i], salaries[i]),
                 textcoords="offset points", xytext=(5, 4),
                 fontsize=8, color=TEXT, fontweight='bold')
    ls = '--' if i in knn_indices else ':'
    lw = 1.8  if i in knn_indices else 0.8
    alpha = 1.0 if i in knn_indices else 0.35
    ax2.plot([new_point[0], ages[i]], [new_point[1], salaries[i]],
             color='#a8a8b3', linewidth=lw, linestyle=ls, alpha=alpha, zorder=3)
    mid = ((new_point[0]+ages[i])/2, (new_point[1]+salaries[i])/2)
    ax2.annotate(f"{distances[i]:.2f}",
                 mid, fontsize=7.5,
                 color='#ffd60a' if i in knn_indices else '#a8a8b3',
                 ha='center')

ax2.scatter(*new_point, c='#ffd60a', s=200, marker='*', zorder=6,
            edgecolors='white', linewidths=0.8)
ax2.annotate('X', new_point, textcoords="offset points",
             xytext=(6, 5), fontsize=9, color='#ffd60a', fontweight='bold')

# ── Plot 3: KNN Decision ──────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
base_style(ax3, f"③ KNN – K={k} Result")
# background shading
from matplotlib.colors import LinearSegmentedColormap
xx, yy = np.meshgrid(np.linspace(10,55,200), np.linspace(30,110,200))
Z = np.zeros(xx.shape)
for xi in range(xx.shape[0]):
    for xj in range(xx.shape[1]):
        pt = np.array([xx[xi,xj], yy[xi,xj]])
        d  = [np.sqrt(((pt - np.array([ages[i], salaries[i]]))**2).sum())
              for i in range(len(people))]
        nn = np.argsort(d)[:k]
        vs = [labels[n] for n in nn]
        Z[xi,xj] = 1 if max(set(vs), key=vs.count) == 'Y' else 0

ax3.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
             colors=['#ff4d6d22', '#4cc9f022'], zorder=1)
ax3.contour(xx, yy, Z, levels=[0.5], colors=['#ffffff44'], linewidths=1, zorder=2)
for i, p in enumerate(people):
    col = COLORS[labels[i]]
    ax3.scatter(ages[i], salaries[i], c=col, s=130, zorder=5,
                edgecolors='white', linewidths=0.8)
    ax3.annotate(p, (ages[i], salaries[i]),
                 textcoords="offset points", xytext=(5, 4),
                 fontsize=9, color=TEXT, fontweight='bold')
ax3.scatter(*new_point, c='#ffd60a', s=240, marker='*', zorder=7,
            edgecolors='white', linewidths=0.8)
ax3.annotate(f"X → {knn_result}", new_point,
             textcoords="offset points", xytext=(6, 5),
             fontsize=9, color='#ffd60a', fontweight='bold')

# ── Plot 4: K-Means Iter 1 ───────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
base_style(ax4, "④ K-Means – Iteration 1")
c0 = cluster_history[0]
init_c = centroid_history[0]
for i, p in enumerate(all_people):
    col = CLUSTER_COLORS[c0[i]]
    ax4.scatter(data[i,0], data[i,1], c=col, s=130, zorder=5,
                edgecolors='white', linewidths=0.8)
    ax4.annotate(p, (data[i,0], data[i,1]),
                 textcoords="offset points", xytext=(5, 4),
                 fontsize=9, color=TEXT, fontweight='bold')
for ci, (cx, cy) in enumerate(init_c):
    ax4.scatter(cx, cy, marker='X', s=220, c=CLUSTER_COLORS[ci],
                edgecolors='white', linewidths=1.2, zorder=7)
    ax4.annotate(f'C{ci+1}', (cx, cy),
                 textcoords="offset points", xytext=(-18, 6),
                 fontsize=9, color=CLUSTER_COLORS[ci], fontweight='bold')

# ── Plot 5: K-Means Final ─────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
base_style(ax5, "⑤ K-Means – Final Clusters (Converged)")
for i, p in enumerate(all_people):
    col = CLUSTER_COLORS[final_clusters[i]]
    ax5.scatter(data[i,0], data[i,1], c=col, s=130, zorder=5,
                edgecolors='white', linewidths=0.8)
    ax5.annotate(p, (data[i,0], data[i,1]),
                 textcoords="offset points", xytext=(5, 4),
                 fontsize=9, color=TEXT, fontweight='bold')
for ci, c in enumerate(centroids):
    ax5.scatter(c[0], c[1], marker='X', s=220, c=CLUSTER_COLORS[ci],
                edgecolors='white', linewidths=1.2, zorder=7)
    ax5.annotate(f'C{ci+1}\n({c[0]:.1f},{c[1]:.1f})', (c[0], c[1]),
                 textcoords="offset points", xytext=(-14, 8),
                 fontsize=7.5, color=CLUSTER_COLORS[ci], fontweight='bold')

# ── Plot 6: Bar – Distances per person ───────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#111827')
ax6.tick_params(colors=TEXT, labelsize=9)
ax6.xaxis.label.set_color(TEXT)
ax6.yaxis.label.set_color(TEXT)
ax6.title.set_color(TEXT)
ax6.set_title("⑥ KNN – Distance to Point X", fontsize=11,
              fontweight='bold', pad=10)
ax6.grid(True, color=GRID, linewidth=0.5, alpha=0.7, axis='y')
for spine in ax6.spines.values():
    spine.set_edgecolor(GRID)

bar_cols = [COLORS[l] for l in labels]
bars = ax6.bar(people, distances, color=bar_cols, width=0.55,
               edgecolor='white', linewidth=0.6, zorder=3)
# mark K=3 nearest
knn_threshold = sorted(distances)[k-1]
for idx, bar in enumerate(bars):
    h = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2, h + 0.8,
             f"{distances[idx]:.2f}", ha='center', va='bottom',
             fontsize=8, color=TEXT)
    if idx in knn_indices:
        ax6.text(bar.get_x() + bar.get_width()/2, h/2,
                 '★', ha='center', va='center',
                 fontsize=14, color='#ffd60a')
ax6.axhline(knn_threshold + 0.01, color='#ffd60a',
            linestyle='--', linewidth=1.2, alpha=0.7,
            label=f'K={k} cutoff')
ax6.set_xlabel('Person', fontsize=9)
ax6.set_ylabel('Euclidean Distance', fontsize=9)
ax6.legend(facecolor='#1f2937', edgecolor=GRID,
           labelcolor=TEXT, fontsize=8)

# ── Super title ───────────────────────────────────────
fig.suptitle("KNN Classification  &  K-Means Clustering  —  Own Dataset",
             fontsize=15, fontweight='bold', color=TEXT, y=0.98)

plt.savefig(r'C:\Users\r3nz3\OneDrive\Desktop\knn_kmeans_analysis.png',
            dpi=150, bbox_inches='tight', facecolor=DARK)
print("\nPlot saved → knn_kmeans_analysis.png")
plt.show()
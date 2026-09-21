import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import gaussian_kde
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 1. Import QuClu
# ----------------------------------------------------------------------
try:
    from QuClu import myfunctions
    QUCLU_AVAILABLE = True
    print("Using QuClu.myfunctions.alg_VS (quantile-based clustering).")
except ImportError:
    QUCLU_AVAILABLE = False
    from sklearn.cluster import KMeans
    print("QuClu not installed. Falling back to KMeans.")
    print("To install: pip install QuClu")

# --------------------------
# 2. Load data
# --------------------------
df = pd.read_csv('eye_data/all_fixations_with_features.csv')
print(f"Loaded {len(df)} fixations")

# Rename columns to match paper
df.rename(columns={
    'duration': 'fixation_duration',
    'Dist_to_closest_obstacles_deg': 'Dist_to_closest_obstacles',
    'dist_to_spaceship_deg': 'distance_to_spaceship'
}, inplace=True)

# Add a binary label for distant fixations (only for PCA visualisation)
df['distant_fixation'] = (df['distance_to_spaceship'] > 5).astype(int)

# Extract participant ID
df['ID'] = df['source_file'].str.extract(r'experiment_([^_]+)_', expand=False)
if df['ID'].isna().all():
    df['ID'] = df['source_file'].apply(lambda x: x.split('_')[1] if len(x.split('_'))>1 else 'unknown')

# Map noise levels
noise_map = {'weak': 1, 'medium': 2, 'strong': 3, 'very_strong': 4}
df['input_noise'] = df['input_noise'].astype(str).map(noise_map).fillna(0).astype(int)

if 'level' not in df.columns:
    df['level'] = 1

# --------------------------
# 3. Prepare features
# --------------------------
features = ["distance_to_spaceship", "fixation_duration", "Dist_to_closest_obstacles"]
df_clean = df.dropna(subset=features).copy()
print(f"Rows before cleaning: {len(df)}")
print(f"Rows after cleaning: {len(df_clean)}")

# Keep X as a DataFrame (QuClu expects a DataFrame)
X = df_clean[features]

# Standardise for silhouette score (we need the array)
X_array = X.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array)

# PCA for visualisation (optional)
pca = PCA(n_components=3)
pca_components = pca.fit_transform(X_scaled)
print("Explained variance ratios:", pca.explained_variance_ratio_)

# --------------------------
# 4. Quantile-based clustering (QuClu)
# --------------------------
k_range = range(2, 11)
silhouette_scores = []
labels_dict = {}

if QUCLU_AVAILABLE:
    print("Running quantile-based clustering (QuClu.alg_VS) ...")
    for k in k_range:
        cluster = myfunctions.alg_VS(X, k=k, B=50)   # X is a DataFrame
        labels = cluster['cl']                       # labels are 0-indexed
        labels_dict[k] = labels
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil)
        print(f"K={k}: silhouette={sil:.4f}")
else:
    print("Falling back to KMeans (on standardised features).")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=36, n_init=10)
        labels = km.fit_predict(X_scaled)
        labels_dict[k] = labels
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil)
        print(f"K={k}: silhouette={sil:.4f}")

# Plot silhouette scores
plt.figure(figsize=(8,5))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True, alpha=0.3)
plt.savefig('silhouette_clustering.png', dpi=150)
plt.show()

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k} (silhouette = {max(silhouette_scores):.4f})")

# Assign cluster labels (use the same name as in the paper: N_qlcu)
df_clean['N_qlcu'] = labels_dict[optimal_k]

# --------------------------
# 5. Descriptive statistics and plots (as in paper)
# --------------------------
# Cluster sizes
print("Cluster sizes:\n", df_clean['N_qlcu'].value_counts())

# Descriptive stats per cluster (mean, mode, 95% CI)
variables = ["distance_to_spaceship", "fixation_duration", "Dist_to_closest_obstacles"]
clusters = sorted(df_clean["N_qlcu"].unique())

summary_list = []
for cluster in clusters:
    cluster_data = df_clean[df_clean["N_qlcu"] == cluster]
    for var in variables:
        values = cluster_data[var].dropna()
        if len(values) < 2:
            continue
        mean = values.mean()
        sd = values.std()
        n = len(values)
        sem = sd / np.sqrt(n)
        ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)
        kde = stats.gaussian_kde(values)
        x_vals = np.linspace(values.min(), values.max(), 1000)
        y_vals = kde(x_vals)
        mode_val = x_vals[np.argmax(y_vals)]
        summary_list.append({
            "cluster": cluster,
            "variable": var,
            "mean": mean,
            "mode": mode_val,
            "sd": sd,
            "ci_low": ci_low,
            "ci_high": ci_high
        })
summary_df = pd.DataFrame(summary_list)
pd.set_option("display.precision", 3)
print(summary_df)

# KDE distributions per cluster (Figure 3 in paper)
n_clusters = len(clusters)
fig, axes = plt.subplots(nrows=n_clusters, ncols=3, figsize=(15, 4*n_clusters), sharex='col', sharey='col')
if n_clusters == 1:
    axes = axes.reshape(1, -1)
for row_idx, cluster in enumerate(clusters):
    cluster_data = df_clean[df_clean["N_qlcu"] == cluster]
    for col_idx, var in enumerate(variables):
        ax = axes[row_idx, col_idx]
        values = cluster_data[var].dropna()
        if values.nunique() > 1:
            sns.kdeplot(x=values, ax=ax, fill=True)
            kde = gaussian_kde(values)
            x_vals = np.linspace(values.min(), values.max(), 500)
            y_vals = kde(x_vals)
            mode_val = x_vals[np.argmax(y_vals)]
            ax.vlines(x=mode_val, ymin=0, ymax=np.max(y_vals), color='red', linestyle='-', linewidth=1)
        if row_idx == 0:
            ax.set_title(var.replace('_', ' ').title(), fontsize=14)
        if col_idx == 0:
            ax.set_ylabel(f'Type {cluster}', fontsize=14, fontweight='bold')
        else:
            ax.set_ylabel('')
plt.tight_layout()
plt.savefig('cluster_distributions.png', dpi=150)
plt.show()

# Trial-level fixation counts per cluster
trial_info = df_clean[['ID', 'trial', 'level', 'input_noise']].drop_duplicates()
fixation_counts = df_clean.groupby(['ID', 'trial', 'N_qlcu']).size().reset_index(name='n_fixations')
fixation_counts_pivot = fixation_counts.pivot(index=['ID','trial'], columns='N_qlcu', values='n_fixations').fillna(0)
fixation_counts_pivot.columns = [f'n_fixations_cluster_{int(col)}' for col in fixation_counts_pivot.columns]
fixation_counts_pivot = fixation_counts_pivot.reset_index()

merged = trial_info.merge(fixation_counts_pivot, on=['ID','trial'])
for col in merged.columns:
    if 'n_fixations_cluster_' in col:
        merged[col] = merged[col].astype(int)

# KDE of fixation counts per cluster (mode estimation)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,5), sharex=True)
for idx, cluster in enumerate([0,1]):
    col = f"n_fixations_cluster_{cluster}"
    values = merged[col].dropna()
    if values.nunique() > 1:
        sns.kdeplot(x=values, ax=axes[idx], fill=True)
        kde = gaussian_kde(values)
        x_vals = np.linspace(values.min(), values.max(), 500)
        y_vals = kde(x_vals)
        mode_val = x_vals[np.argmax(y_vals)]
        print(f"Mode cluster {cluster}: {mode_val}")
        ymax = np.max(y_vals)
        axes[idx].vlines(mode_val, ymin=0, ymax=ymax, color='red', linestyle='-', linewidth=1)
    else:
        axes[idx].text(0.5, 0.5, "No variance", transform=axes[idx].transAxes,
                       ha='center', va='center', fontsize=8, color='gray')
    axes[idx].set_ylabel(f"Cluster {cluster}")
    axes[idx].set_yticks([])
axes[1].set_xlabel("Number of Fixations")
axes[0].set_title("Distribution of Number of Fixations per Cluster")
plt.tight_layout()
plt.savefig('fixation_counts_kde.png', dpi=150)
plt.show()

# --------------------------
# Combined plot: 3 features + N fixations (as in paper's Figure 3)
# --------------------------
variables = ["distance_to_spaceship", "fixation_duration", "Dist_to_closest_obstacles"]
fixation_columns = {0: "n_fixations_cluster_0", 1: "n_fixations_cluster_1"}
column_titles = ["Distance Spaceship", "Fixation Duration", "Distance Closest Obstacle", "N Fixations"]
clusters = sorted(df_clean["N_qlcu"].unique())

# Create 2 rows, 4 columns
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 8), sharex='col', sharey='col')

# Order: [3, 0, 1, 2] – puts N Fixations first, then the three features
plot_order = [3, 0, 1, 2]

for row_idx, cluster in enumerate(clusters):
    data_row = df_clean[df_clean.N_qlcu == cluster]

    for plot_pos, var_idx in enumerate(plot_order):
        ax = axes[row_idx, plot_pos]

        if var_idx < 3:
            var = variables[var_idx]
            values = data_row[var].dropna()
        else:
            # Fixations column
            var = fixation_columns[cluster]
            values = merged[var].dropna()

        if values.nunique() > 1:
            sns.kdeplot(x=values, ax=ax, fill=True)
            kde = gaussian_kde(values)
            x_vals = np.linspace(values.min(), values.max(), 500)
            y_vals = kde(x_vals)
            mode_val = x_vals[np.argmax(y_vals)]
            ax.vlines(x=mode_val, ymin=0, ymax=np.max(y_vals), color='red', linestyle='-', linewidth=1)

        # titles only on top row
        if row_idx == 0:
            ax.set_title(column_titles[var_idx], fontsize=14)

        # row label on the leftmost column
        if plot_pos == 0:
            ax.set_ylabel(f"Type {cluster}", fontsize=14, fontweight='bold')
        else:
            ax.set_ylabel('')

plt.tight_layout()
plt.savefig('figure3_combined.png', dpi=150)
plt.show()
# --------------------------
# 6. Linear Mixed Models (exactly as paper)
# --------------------------
def run_lmm(data, formula, dep_var, transform_func=None):
    if transform_func is not None:
        data[dep_var + '_t'] = transform_func(data[dep_var])
        dep = dep_var + '_t'
    else:
        dep = dep_var
    model = smf.mixedlm(formula.format(dep=dep), data, groups=data["ID"])
    model_fit = model.fit(reml=False)
    print(model_fit.summary())
    return model_fit

# Number of fixations per cluster (log transform)
merged['n_fixations_cluster_0_t'] = np.log(merged.n_fixations_cluster_0 + 1)
merged['n_fixations_cluster_1_t'] = np.log(merged.n_fixations_cluster_1 + 1)
print("\n--- N_fixations cluster 0 ---")
run_lmm(merged, "{dep} ~ input_noise", "n_fixations_cluster_0_t")
print("\n--- N_fixations cluster 1 ---")
run_lmm(merged, "{dep} ~ input_noise", "n_fixations_cluster_1_t")

# Fixation-level metrics per cluster
type_0 = df_clean[df_clean.N_qlcu == 0].copy()
type_1 = df_clean[df_clean.N_qlcu == 1].copy()

print("\n--- Distance to spaceship (type 0) ---")
type_0['dist_ship_t'] = np.sqrt(type_0.distance_to_spaceship)   # λ=0.539
run_lmm(type_0, "{dep} ~ input_noise", "dist_ship_t")

print("\n--- Distance to spaceship (type 1) ---")
type_1['dist_ship_t'] = 1 / type_1.distance_to_spaceship        # λ=-0.909
run_lmm(type_1, "{dep} ~ input_noise", "dist_ship_t")

print("\n--- Fixation duration (type 0) ---")
type_0['dur_t'] = np.log(type_0.fixation_duration)              # λ≈0
run_lmm(type_0, "{dep} ~ input_noise", "dur_t")

print("\n--- Fixation duration (type 1) ---")
type_1['dur_t'] = np.log(type_1.fixation_duration)
run_lmm(type_1, "{dep} ~ input_noise", "dur_t")

print("\n--- Distance to closest obstacle (type 0) ---")
type_0['obs_t'] = np.log(type_0.Dist_to_closest_obstacles)      # λ≈0.16
run_lmm(type_0, "{dep} ~ input_noise", "obs_t")

print("\n--- Distance to closest obstacle (type 1) ---")
type_1['obs_t'] = np.log(type_1.Dist_to_closest_obstacles)      # λ≈0.25
run_lmm(type_1, "{dep} ~ input_noise", "obs_t")

print("Analysis complete.")
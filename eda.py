import os
import pandas as pd
import numpy as np

# 1. Setup paths
data_dir = "C:/Users/leona/PythonScripts/MasterThesis/Tierdaten/"
target_id = 1
other_ids = [3, 4, 8, 10]
all_ids = [target_id] + other_ids


# Helper to load data safely
def load_animal_data(animal_id):
    x_path = os.path.join(data_dir, f"X_anim{animal_id}.csv")
    y_path = os.path.join(data_dir, f"Y_anim{animal_id}.csv")

    # Adjust header=None if your CSVs do not have column names
    X = pd.read_csv(x_path, header=None)
    Y = pd.read_csv(y_path, header=None)

    # Force single column Y to be a Series named 'target'
    Y.columns = ['target']
    return X, Y


# Load all datasets
data = {}
for aid in all_ids:
    try:
        X, Y = load_animal_data(aid)
        data[aid] = (X, Y)
        print(f"Loaded Animal {aid}: X shape {X.shape}, Y shape {Y.shape}")
    except FileNotFoundError as e:
        print(f"Error loading Animal {aid}: {e}")

# 2. Compare Target (Y) Distributions
print("\n" + "=" * 50)
print(" 1. TARGET (Y) DISTRIBUTION COMPARISON")
print("=" * 50)

y_stats = []
for aid in all_ids:
    if aid in data:
        y = data[aid][1]['target']
        y_stats.append({
            'Animal': f"anim{aid}",
            'Count': len(y),
            'Mean': y.mean(),
            'Std': y.std(),
            'Min': y.min(),
            'Max': y.max(),
            'Skew': y.skew()
        })

df_y_stats = pd.DataFrame(y_stats)
print(df_y_stats.to_string(index=False))

# 3. Compare Feature (X) Distributions (Covariate Shift)
print("\n" + "=" * 50)
print(" 2. COVARIATE SHIFT: FEATURE MEANS COMPARISON")
print("=" * 50)

# Calculate mean of each feature for Anim 1 vs average of others
X_1, _ = data[target_id]
X_others = pd.concat([data[aid][0] for aid in other_ids], axis=0)

feature_comparison = pd.DataFrame({
    'Feature_Idx': X_1.columns,
    'Anim1_Mean': X_1.mean(),
    'Anim1_Std': X_1.std(),
    'Others_Mean': X_others.mean(),
    'Others_Std': X_others.std()
})

# Calculate a standardized difference (Cohen's d-like metric) to see which features shifted most
pooled_std = np.sqrt((feature_comparison['Anim1_Std'] ** 2 + feature_comparison['Others_Std'] ** 2) / 2)
feature_comparison['Std_Difference'] = (feature_comparison['Anim1_Mean'] - feature_comparison['Others_Mean']) / (
            pooled_std + 1e-8)

# Sort by absolute difference to show the most shifted features at the top
feature_comparison['Abs_Std_Diff'] = feature_comparison['Std_Difference'].abs()
feature_comparison = feature_comparison.sort_values(by='Abs_Std_Diff', ascending=False).drop(columns=['Abs_Std_Diff'])

print("Features sorted by how much Anim 1 shifts from the others (Standardized Difference):")
print(feature_comparison.to_string(index=False))

# 4. Compare Feature-Target Relationships (Concept Shift)
print("\n" + "=" * 50)
print(" 3. CONCEPT SHIFT: FEATURE-TARGET CORRELATION")
print("=" * 50)

corrs = {}
for aid in all_ids:
    X, Y = data[aid]
    # Compute Pearson correlation of each feature with the target
    corrs[f"anim{aid}_corr"] = X.corrwith(Y['target'])

df_corrs = pd.DataFrame(corrs)
# Calculate correlation difference between Anim 1 and the average correlation of others
df_corrs['Others_Avg_Corr'] = df_corrs[[f"anim{aid}_corr" for aid in other_ids]].mean(axis=1)
df_corrs['Corr_Delta'] = df_corrs[f"anim{target_id}_corr"] - df_corrs['Others_Avg_Corr']
df_corrs['Abs_Corr_Delta'] = df_corrs['Corr_Delta'].abs()

df_corrs = df_corrs.sort_values(by='Abs_Corr_Delta', ascending=False).drop(columns=['Abs_Corr_Delta'])
print("Features sorted by how differently they correlate with the target in Anim 1 vs Others:")
print(df_corrs[[f"anim{target_id}_corr", "Others_Avg_Corr", "Corr_Delta"]])
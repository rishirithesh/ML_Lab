import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

print("=" * 80)
print("EXPERIMENT 7: DIMENSIONALITY REDUCTION & MODEL EVALUATION (WITH AND WITHOUT PCA)")
print("=" * 80)

# 1. Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target") # 0 = malignant, 1 = benign

print(f"Dataset Name       : Wisconsin Diagnostic Breast Cancer (WDBC)")
print(f"Samples (N)        : {X.shape[0]}")
print(f"Features (p)       : {X.shape[1]}")
print(f"Target distribution: Malignant(0)={sum(y==0)}, Benign(1)={sum(y==1)}")
print(f"Missing Values     : {X.isnull().sum().sum()}")

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training Samples   : {X_train.shape[0]}")
print(f"Testing Samples    : {X_test.shape[0]}")

# 2. PCA Analysis
scaler_eda = StandardScaler()
X_train_scaled = scaler_eda.fit_transform(X_train)
X_test_scaled = scaler_eda.transform(X_test)

pca_full = PCA()
pca_full.fit(X_train_scaled)

var_exp = pca_full.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)
n_components_95 = np.argmax(cum_var_exp >= 0.95) + 1
var_95_pct = cum_var_exp[n_components_95 - 1] * 100

print(f"Components for 95% variance: {n_components_95}")
print(f"Actual variance captured   : {var_95_pct:.2f}%")

# ==========================================
# 3. Generate EDA & Dimensionality Figures
# ==========================================

# Figure 1: Class Distribution
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
counts = y.value_counts().sort_index()
bars = ax.bar(['Malignant (0)', 'Benign (1)'], [counts[0], counts[1]], color=['#e74c3c', '#2ecc71'], width=0.5, edgecolor='black', linewidth=1)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5, f'{height} ({height/len(y)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Sample Count')
ax.set_title('Wisconsin Breast Cancer Class Distribution', pad=12, fontweight='bold')
ax.set_ylim(0, 420)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1_class_distribution.png'))
fig.savefig(os.path.join(BASE_DIR, 'fig_class_distribution.png'))
plt.close(fig)

# Figure 2: Scree Plot (Individual Explained Variance)
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
ax.plot(range(1, len(var_exp) + 1), var_exp * 100, marker='o', color='#2980b9', linewidth=2, markersize=6, label='Individual Variance (%)')
ax.axvline(x=n_components_95, color='#e74c3c', linestyle='--', label=f'Chosen n_components={n_components_95} (95% Target)')
ax.set_xlabel('Principal Component Index')
ax.set_ylabel('Explained Variance Ratio (%)')
ax.set_title('PCA Scree Plot (Eigenvalue / Variance Spectrum)', pad=12, fontweight='bold')
ax.set_xticks(range(1, len(var_exp) + 1, 2))
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_scree_plot.png'))
fig.savefig(os.path.join(BASE_DIR, 'fig_scree_plot.png'))
plt.close(fig)

# Figure 3: Cumulative Variance Explained Plot
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
ax.plot(range(1, len(cum_var_exp) + 1), cum_var_exp * 100, marker='s', color='#8e44ad', linewidth=2, markersize=5, label='Cumulative Variance (%)')
ax.axhline(y=95, color='#e74c3c', linestyle=':', label='95% Threshold Line')
ax.axvline(x=n_components_95, color='#27ae60', linestyle='--', label=f'Optimal Cutoff = {n_components_95} PCs ({var_95_pct:.2f}%)')
ax.scatter([n_components_95], [var_95_pct], color='#e74c3c', s=100, zorder=5)
ax.text(n_components_95 + 0.8, var_95_pct - 4, f'10 PCs: {var_95_pct:.2f}%', fontweight='bold', color='#2c3e50')
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance (%)')
ax.set_title('Cumulative Explained Variance vs. Number of Components', pad=12, fontweight='bold')
ax.set_xticks(range(1, len(cum_var_exp) + 1, 2))
ax.set_ylim(40, 102)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_cumulative_variance.png'))
fig.savefig(os.path.join(BASE_DIR, 'fig_cumulative_variance.png'))
plt.close(fig)

# Figure 4: 2D PCA Projection Scatter Plot
pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
scatter = ax.scatter(
    X_train_pca_2d[:, 0], X_train_pca_2d[:, 1],
    c=y_train, cmap='coolwarm', alpha=0.85, edgecolors='k', linewidth=0.5, s=45
)
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}% Variance)')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}% Variance)')
ax.set_title('First Two Principal Components Projection (Train Set)', pad=12, fontweight='bold')
cbar = plt.colorbar(scatter, ticks=[0, 1])
cbar.set_ticklabels(['Malignant (0)', 'Benign (1)'])
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig4_pca_2d_projection.png'))
fig.savefig(os.path.join(BASE_DIR, 'fig_pca_2d_projection.png'))
plt.close(fig)

# Figure 5: Feature Correlation Heatmap (Selected Morphological Descriptors)
fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
top_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'
]
corr = X[top_features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
ax.set_title('Correlation Heatmap of Mean Cytological Features', pad=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig5_correlation_heatmap.png'))
fig.savefig(os.path.join(BASE_DIR, 'fig_correlation_heatmap.png'))
plt.close(fig)

# ==========================================
# 4. Model Definitions & Parameter Grids
# ==========================================
base_stack_estimators = [
    ('svm', SVC(probability=True, random_state=42)),
    ('nb', GaussianNB()),
    ('dt', DecisionTreeClassifier(random_state=42))
]

models_dict = {
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "param_grid": {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto", 0.01, 0.1],
            "model__kernel": ["rbf", "linear"]
        }
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "param_grid": {
            "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "param_grid": {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"]
        }
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__criterion": ["gini", "entropy"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "param_grid": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5]
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "param_grid": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 1.0]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric="logloss", random_state=42),
        "param_grid": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5]
        }
    },
    "Stacking": {
        "model": StackingClassifier(
            estimators=base_stack_estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5
        ),
        "param_grid": {
            "model__final_estimator__C": [0.1, 1.0, 10.0]
        }
    }
}

# ==========================================
# 5. Training & Evaluation Engine
# ==========================================
results_no_pca = {}
results_pca = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def execute_pipeline(name, cfg, use_pca=False):
    start_time = time.time()
    steps = [('scaler', StandardScaler())]
    if use_pca:
        steps.append(('pca', PCA(n_components=n_components_95, random_state=42)))
    steps.append(('model', cfg['model']))
    pipe = Pipeline(steps)
    
    grid = GridSearchCV(pipe, cfg['param_grid'], cv=cv, scoring='accuracy', n_jobs=1)
    grid.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    best_pipe = grid.best_estimator_
    best_params = grid.best_params_
    
    # 5-fold CV on best pipeline
    cv_scores = cross_validate(
        best_pipe, X_train, y_train, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=False
    )
    
    # Test evaluation
    inf_start = time.time()
    y_pred = best_pipe.predict(X_test)
    inf_time = (time.time() - inf_start) / len(X_test) * 1000 # ms per sample
    
    if hasattr(best_pipe, "predict_proba"):
        y_prob = best_pipe.predict_proba(X_test)[:, 1]
    elif hasattr(best_pipe, "decision_function"):
        y_prob = best_pipe.decision_function(X_test)
    else:
        y_prob = y_pred.astype(float)
        
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'best_estimator': best_pipe,
        'best_params': best_params,
        'cv_acc_folds': cv_scores['test_accuracy'].tolist(),
        'cv_acc_mean': float(np.mean(cv_scores['test_accuracy'])),
        'cv_acc_std': float(np.std(cv_scores['test_accuracy'])),
        'cv_acc_var': float(np.var(cv_scores['test_accuracy'])),
        'cv_f1_mean': float(np.mean(cv_scores['test_f1'])),
        'cv_auc_mean': float(np.mean(cv_scores['test_roc_auc'])),
        'test_acc': float(test_acc),
        'test_prec': float(test_prec),
        'test_rec': float(test_rec),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc),
        'fit_time': float(fit_time),
        'inf_time_ms': float(inf_time),
        'cm': cm.tolist(),
        'y_prob': y_prob.tolist(),
        'y_pred': y_pred.tolist()
    }

print("\nRunning Models WITHOUT PCA...")
for name, cfg in models_dict.items():
    print(f"  Training {name} (No-PCA)...")
    results_no_pca[name] = execute_pipeline(name, cfg, use_pca=False)

print("\nRunning Models WITH PCA...")
for name, cfg in models_dict.items():
    print(f"  Training {name} (With-PCA)...")
    results_pca[name] = execute_pipeline(name, cfg, use_pca=True)

# Build Comprehensive Summary DataFrames
summary_rows = []
for name in models_dict.keys():
    r_no = results_no_pca[name]
    r_pca = results_pca[name]
    acc_diff = r_pca['cv_acc_mean'] - r_no['cv_acc_mean']
    var_diff = r_pca['cv_acc_var'] - r_no['cv_acc_var']
    summary_rows.append({
        'Model': name,
        'No_PCA_CV_Acc': r_no['cv_acc_mean'],
        'With_PCA_CV_Acc': r_pca['cv_acc_mean'],
        'CV_Acc_Improvement': acc_diff,
        'No_PCA_Fold1': r_no['cv_acc_folds'][0],
        'No_PCA_Fold2': r_no['cv_acc_folds'][1],
        'No_PCA_Fold3': r_no['cv_acc_folds'][2],
        'No_PCA_Fold4': r_no['cv_acc_folds'][3],
        'No_PCA_Fold5': r_no['cv_acc_folds'][4],
        'With_PCA_Fold1': r_pca['cv_acc_folds'][0],
        'With_PCA_Fold2': r_pca['cv_acc_folds'][1],
        'With_PCA_Fold3': r_pca['cv_acc_folds'][2],
        'With_PCA_Fold4': r_pca['cv_acc_folds'][3],
        'With_PCA_Fold5': r_pca['cv_acc_folds'][4],
        'No_PCA_Var': r_no['cv_acc_var'],
        'With_PCA_Var': r_pca['cv_acc_var'],
        'No_PCA_Test_Acc': r_no['test_acc'],
        'With_PCA_Test_Acc': r_pca['test_acc'],
        'No_PCA_F1': r_no['test_f1'],
        'With_PCA_F1': r_pca['test_f1'],
        'No_PCA_AUC': r_no['test_auc'],
        'With_PCA_AUC': r_pca['test_auc'],
        'No_PCA_Time': r_no['fit_time'],
        'With_PCA_Time': r_pca['fit_time'],
        'Time_Speedup_Pct': ((r_no['fit_time'] - r_pca['fit_time']) / r_no['fit_time']) * 100
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(BASE_DIR, 'PCA_Model_Comparison.csv'), index=False)

# Format serializable results
def make_serializable(res_dict):
    clean_dict = {}
    for k, v in res_dict.items():
        clean_v = {}
        for prop, val in v.items():
            if prop == 'best_estimator':
                clean_v[prop] = str(val)
            else:
                clean_v[prop] = val
        clean_dict[k] = clean_v
    return clean_dict

exp_results = {
    'dataset': {
        'name': 'Wisconsin Diagnostic Breast Cancer (WDBC)',
        'source': 'UCI Machine Learning Repository / Scikit-Learn',
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'classes': ['Malignant (0)', 'Benign (1)'],
        'distribution': {'Malignant': int(counts[0]), 'Benign': int(counts[1])},
        'missing_values': 0,
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0])
    },
    'pca_summary': {
        'original_features': 30,
        'chosen_components': int(n_components_95),
        'variance_target': 95.0,
        'explained_variance_pct': float(var_95_pct),
        'justification': 'Preserves >95% total variance while eliminating 66.7% dimensionality, removing severe multicollinearity among perimeter, radius, and area features.'
    },
    'no_pca_results': make_serializable(results_no_pca),
    'with_pca_results': make_serializable(results_pca),
    'summary_table': summary_rows
}

with open(os.path.join(BASE_DIR, 'results_summary.json'), 'w') as f:
    json.dump(exp_results, f, indent=2)

print("\n" + "=" * 80)
print("FINAL CROSS-VALIDATION SUMMARY (NO-PCA vs WITH-PCA)")
print("=" * 80)
print(df_summary[['Model', 'No_PCA_CV_Acc', 'With_PCA_CV_Acc', 'CV_Acc_Improvement', 'No_PCA_Test_Acc', 'With_PCA_Test_Acc', 'No_PCA_F1', 'With_PCA_F1']].round(4).to_string(index=False))

# ==========================================
# 6. Performance Visualizations
# ==========================================

# Figure 6: Bar Chart Comparison (CV Accuracy with and without PCA)
fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
x = np.arange(len(df_summary))
width = 0.35
bars1 = ax.bar(x - width/2, df_summary['No_PCA_CV_Acc'], width, label='Without PCA (30 Features)', color='#3498db', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, df_summary['With_PCA_CV_Acc'], width, label='With PCA (10 Components)', color='#e67e22', edgecolor='black', linewidth=0.8)

ax.set_ylabel('Mean 5-Fold Stratified Accuracy')
ax.set_title('Cross-Validation Accuracy Benchmark: No-PCA vs. With-PCA', pad=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_summary['Model'], rotation=30, ha='right')
ax.set_ylim(0.90, 1.00)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig6_model_comparison_bar.png'))
fig.savefig(os.path.join(BASE_DIR, 'fig_model_comparison_bar.png'))
plt.close(fig)

# Figure 7: ROC Curves (All Models No-PCA vs With-PCA)
fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), dpi=300)
axes = axes.flatten()

for idx, name in enumerate(models_dict.keys()):
    ax = axes[idx]
    r_no = results_no_pca[name]
    r_pca = results_pca[name]
    
    fpr_no, tpr_no, _ = roc_curve(y_test, r_no['y_prob'])
    fpr_pca, tpr_pca, _ = roc_curve(y_test, r_pca['y_prob'])
    
    ax.plot(fpr_no, tpr_no, label=f"No-PCA (AUC={r_no['test_auc']:.3f})", color='#2980b9', linewidth=1.8)
    ax.plot(fpr_pca, tpr_pca, label=f"With-PCA (AUC={r_pca['test_auc']:.3f})", color='#e74c3c', linestyle='--', linewidth=1.8)
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.5)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Test Set ROC Curves: No-PCA vs. With-PCA Across All 10 Classifiers', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig7_roc_curves_all.png'))
fig.savefig(os.path.join(BASE_DIR, 'fig_roc_curves_all.png'))
plt.close(fig)

# Figure 8: Precision-Recall Curves (All Models)
fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), dpi=300)
axes = axes.flatten()

for idx, name in enumerate(models_dict.keys()):
    ax = axes[idx]
    r_no = results_no_pca[name]
    r_pca = results_pca[name]
    
    p_no, rec_no, _ = precision_recall_curve(y_test, r_no['y_prob'])
    p_pca, rec_pca, _ = precision_recall_curve(y_test, r_pca['y_prob'])
    ap_no = average_precision_score(y_test, r_no['y_prob'])
    ap_pca = average_precision_score(y_test, r_pca['y_prob'])
    
    ax.plot(rec_no, p_no, label=f"No-PCA (AP={ap_no:.3f})", color='#27ae60', linewidth=1.8)
    ax.plot(rec_pca, p_pca, label=f"With-PCA (AP={ap_pca:.3f})", color='#8e44ad', linestyle='--', linewidth=1.8)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Recall', fontsize=9)
    ax.set_ylabel('Precision', fontsize=9)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Test Set Precision-Recall Curves: No-PCA vs. With-PCA', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig8_pr_curves_all.png'))
fig.savefig(os.path.join(BASE_DIR, 'fig_pr_curves_all.png'))
plt.close(fig)

# Figure 9: Confusion Matrices for Key Models (SVM, Logistic Regression, Stacking, XGBoost)
selected_cms = ["SVM", "Logistic Regression", "Stacking", "XGBoost"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=300)

for col, name in enumerate(selected_cms):
    # No PCA
    ax_no = axes[0, col]
    cm_no = np.array(results_no_pca[name]['cm'])
    sns.heatmap(cm_no, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_no, annot_kws={'size': 13, 'fontweight': 'bold'})
    ax_no.set_title(f"{name} (No-PCA)\nAcc: {results_no_pca[name]['test_acc']:.3f}", fontweight='bold')
    ax_no.set_xticklabels(['Malignant', 'Benign'])
    ax_no.set_yticklabels(['Malignant', 'Benign'])
    if col == 0:
        ax_no.set_ylabel('True Label')
        
    # With PCA
    ax_pca = axes[1, col]
    cm_pca = np.array(results_pca[name]['cm'])
    sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax_pca, annot_kws={'size': 13, 'fontweight': 'bold'})
    ax_pca.set_title(f"{name} (With-PCA)\nAcc: {results_pca[name]['test_acc']:.3f}", fontweight='bold')
    ax_pca.set_xticklabels(['Malignant', 'Benign'])
    ax_pca.set_yticklabels(['Malignant', 'Benign'])
    ax_pca.set_xlabel('Predicted Label')
    if col == 0:
        ax_pca.set_ylabel('True Label')

plt.suptitle('Confusion Matrix Comparison on Test Set (No-PCA vs. With-PCA)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig9_confusion_matrices.png'))
fig.savefig(os.path.join(BASE_DIR, 'fig_confusion_matrices.png'))
plt.close(fig)

# Figure 10: Fold Variance & Stability Boxplot
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
var_data = []
for name in models_dict.keys():
    for score in results_no_pca[name]['cv_acc_folds']:
        var_data.append({'Model': name, 'Setting': 'No-PCA (30 Dim)', 'Accuracy': score})
    for score in results_pca[name]['cv_acc_folds']:
        var_data.append({'Model': name, 'Setting': 'With-PCA (10 Dim)', 'Accuracy': score})
df_var = pd.DataFrame(var_data)

sns.boxplot(data=df_var, x='Model', y='Accuracy', hue='Setting', palette={'No-PCA (30 Dim)': '#5dade2', 'With-PCA (10 Dim)': '#f5b041'}, ax=ax)
ax.set_title('Cross-Validation Score Distribution Across 5 Folds (Model Stability)', pad=12, fontweight='bold')
ax.set_ylabel('Fold Accuracy')
plt.xticks(rotation=30, ha='right')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig10_fold_variance_boxplots.png'))
fig.savefig(os.path.join(BASE_DIR, 'fig_fold_variance_boxplots.png'))
plt.close(fig)

# Figure 11: Computational Fit Time Benchmark
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
time_df = df_summary.sort_values('No_PCA_Time', ascending=True)
y_pos = np.arange(len(time_df))
h = 0.35
ax.barh(y_pos - h/2, time_df['No_PCA_Time'], height=h, label='No-PCA Fit Time (s)', color='#34495e', edgecolor='k')
ax.barh(y_pos + h/2, time_df['With_PCA_Time'], height=h, label='With-PCA Fit Time (s)', color='#1abc9c', edgecolor='k')
ax.set_yticks(y_pos)
ax.set_yticklabels(time_df['Model'])
ax.set_xlabel('Total Grid Search & Fitting Time (seconds)')
ax.set_title('Computational Efficiency Comparison: Full vs. Reduced Feature Space', pad=12, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig11_time_computational_cost.png'))
fig.savefig(os.path.join(BASE_DIR, 'fig_time_computational_cost.png'))
plt.close(fig)

print("\nAll Experiment 7 results, figures, and summaries generated successfully!")

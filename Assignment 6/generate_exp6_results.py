"""
Experiment 6 / 7: Bagging, Boosting, and Stacked Ensemble Models
Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
SSN College of Engineering - Machine Learning Algorithms Laboratory
Student: Rishi Rithesh (3122247001049)
Faculty: Dr. Poreddy Ajay Kumar Reddy
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    RandomForestClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report
)
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Set aesthetic styling for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
CSV_PATH = os.path.join(SCRIPT_DIR, 'breast_cancer_wisconsin_diagnostic.csv')
RESULTS_JSON_PATH = os.path.join(SCRIPT_DIR, 'results_summary.json')

def load_and_prepare_data():
    print(">>> 1. Loading Wisconsin Diagnostic Breast Cancer Dataset...", flush=True)
    raw_data = load_breast_cancer(as_frame=True)
    df = raw_data.frame.copy()
    
    # Standardize column names with underscore
    df.columns = [c.replace(' ', '_') for c in df.columns]
    
    # In sklearn target: 0 = Malignant, 1 = Benign.
    # In clinical classification, Malignant is positive (1) and Benign is negative (0).
    diagnosis_col = np.where(raw_data.target == 0, 'M', 'B')
    df['diagnosis'] = diagnosis_col
    df['target_malignant'] = np.where(raw_data.target == 0, 1, 0)
    
    # Add an ID column for realistic dataset format
    np.random.seed(42)
    sample_ids = 842300 + np.arange(len(df))
    df.insert(0, 'id', sample_ids)
    
    # Reorder columns: id, diagnosis, target_malignant, then the 30 features
    feature_cols = [c for c in df.columns if c not in ['id', 'diagnosis', 'target_malignant', 'target']]
    clean_df = df[['id', 'diagnosis', 'target_malignant'] + feature_cols]
    
    # Save CSV
    clean_df.to_csv(CSV_PATH, index=False)
    print(f"Dataset saved to {CSV_PATH} | Shape: {clean_df.shape}", flush=True)
    print(f"Class counts:\n{clean_df['diagnosis'].value_counts()}", flush=True)
    
    X = clean_df[feature_cols].values
    y = clean_df['target_malignant'].values # 1 = Malignant, 0 = Benign
    
    return clean_df, feature_cols, X, y

def generate_eda_plots(df, feature_cols):
    print(">>> 2. Generating Comprehensive Exploratory Data Analysis (EDA) Plots...", flush=True)
    
    # 2.1 Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = df['diagnosis'].value_counts()
    labels = ['Benign (B)', 'Malignant (M)']
    values = [counts['B'], counts['M']]
    colors = ['#2b5c8f', '#d9534f']
    
    sns.barplot(x=labels, y=values, palette=colors, ax=axes[0], edgecolor='black', linewidth=1.2)
    for i, v in enumerate(values):
        axes[0].text(i, v + 8, f"{v} ({v/len(df)*100:.1f}%)", ha='center', fontweight='bold', fontsize=11)
    axes[0].set_title("Class Frequency Count", fontweight='bold')
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_ylim(0, max(values) + 45)
    
    wedges, texts, autotexts = axes[1].pie(
        values, labels=labels, autopct='%1.2f%%', colors=colors,
        startangle=140, explode=(0.04, 0.04),
        wedgeprops=dict(width=0.6, edgecolor='black', linewidth=1.2)
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
    axes[1].set_title("Class Ratio (Donut Chart)", fontweight='bold')
    
    plt.suptitle("Wisconsin Breast Cancer Dataset - Target Class Distribution", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eda_class_distribution.png'), bbox_inches='tight')
    plt.close()

    # 2.2 Feature Distributions (Key Mean Features)
    key_features = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_concavity', 'mean_concave_points']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, feat in enumerate(key_features):
        sns.kdeplot(data=df, x=feat, hue='diagnosis', palette={'M': '#d9534f', 'B': '#2b5c8f'},
                    fill=True, common_norm=False, alpha=0.4, linewidth=1.8, ax=axes[i])
        axes[i].set_title(f"Distribution of {feat.replace('_', ' ').title()}", fontweight='bold', fontsize=11)
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Density")
    
    plt.suptitle("Feature Density Distributions Grouped by Diagnosis (M vs B)", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eda_feature_distributions.png'), bbox_inches='tight')
    plt.close()

    # 2.3 Correlation Matrix (Top Correlated Mean Features)
    mean_cols = [c for c in feature_cols if 'mean' in c]
    corr_mean = df[mean_cols].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_mean, dtype=bool))
    sns.heatmap(corr_mean, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.7, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix of Mean Nuclear Morphology Features", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eda_correlation_matrix.png'), bbox_inches='tight')
    plt.close()

    # 2.4 Feature Outlier Boxplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, feat in enumerate(key_features):
        sns.boxplot(x='diagnosis', y=feat, data=df, palette={'M': '#d9534f', 'B': '#2b5c8f'},
                    ax=axes[i], boxprops=dict(alpha=0.8), showmeans=True,
                    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"7"})
        axes[i].set_title(f"{feat.replace('_', ' ').title()} vs Diagnosis", fontweight='bold', fontsize=11)
        axes[i].set_xlabel("Diagnosis")
        axes[i].set_ylabel(feat)
    
    plt.suptitle("Boxplots Showing Distinct Feature Separability & Outliers", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eda_boxplots_outliers.png'), bbox_inches='tight')
    plt.close()

    # 2.5 2D PCA Dimensionality Reduction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['Diagnosis'] = df['diagnosis']
    
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x='Principal Component 1', y='Principal Component 2', hue='Diagnosis',
        data=pca_df, palette={'M': '#d9534f', 'B': '#2b5c8f'},
        alpha=0.85, s=60, edgecolor='black', linewidth=0.6
    )
    exp_var = pca.explained_variance_ratio_ * 100
    plt.xlabel(f"Principal Component 1 ({exp_var[0]:.1f}% Variance Explained)", fontweight='bold')
    plt.ylabel(f"Principal Component 2 ({exp_var[1]:.1f}% Variance Explained)", fontweight='bold')
    plt.title(f"2D PCA Projection of 30 Features (Total {exp_var.sum():.1f}% Variance)", fontsize=13, fontweight='bold')
    plt.legend(title='Diagnosis', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eda_pca_projection.png'), bbox_inches='tight')
    plt.close()

    # 2.6 Key Scatter Relationships
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(
        x='mean_radius', y='mean_concave_points', hue='diagnosis',
        data=df, palette={'M': '#d9534f', 'B': '#2b5c8f'},
        alpha=0.85, s=55, edgecolor='black', linewidth=0.5, ax=axes[0]
    )
    axes[0].set_title("Mean Radius vs Mean Concave Points", fontweight='bold')
    axes[0].set_xlabel("Mean Radius")
    axes[0].set_ylabel("Mean Concave Points")

    sns.scatterplot(
        x='worst_perimeter', y='worst_area', hue='diagnosis',
        data=df, palette={'M': '#d9534f', 'B': '#2b5c8f'},
        alpha=0.85, s=55, edgecolor='black', linewidth=0.5, ax=axes[1]
    )
    axes[1].set_title("Worst Perimeter vs Worst Area", fontweight='bold')
    axes[1].set_xlabel("Worst Perimeter")
    axes[1].set_ylabel("Worst Area")

    plt.suptitle("Bivariate Morphological Relationships Across Tumor Classes", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eda_scatter_relationships.png'), bbox_inches='tight')
    plt.close()

def evaluate_model_on_test(model, X_train, X_test, y_train, y_test, model_name):
    # Measure Training Time
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = (time.perf_counter() - t0) * 1000 # in ms
    
    # Measure Inference Time
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_time = (time.perf_counter() - t1) * 1000 # in ms
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_score = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-df_score))
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'name': model_name,
        'model': model,
        'accuracy': acc,
        'accuracy_pct': acc * 100,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'train_time_ms': train_time,
        'inf_time_ms': inf_time,
        'confusion_matrix': cm.tolist(),
        'y_pred': y_pred.tolist(),
        'y_prob': y_prob.tolist()
    }

def run_bagging_hyperparameter_study(X_train, y_train, cv):
    print(">>> 3. Running Bagging Hyperparameter Grid Study...", flush=True)
    
    n_estimators_list = [10, 25, 50, 100, 200]
    max_samples_list = [0.5, 0.7, 0.8, 1.0]
    
    results = []
    
    for n_est in n_estimators_list:
        for max_s in max_samples_list:
            base_tree = DecisionTreeClassifier(random_state=42)
            bag = BaggingClassifier(
                estimator=base_tree,
                n_estimators=n_est,
                max_samples=max_s,
                max_features=1.0,
                random_state=42,
                n_jobs=1
            )
            cv_res = cross_validate(bag, X_train, y_train, cv=cv, scoring=['accuracy', 'f1', 'precision', 'recall'])
            acc_mean = np.mean(cv_res['test_accuracy']) * 100
            acc_std = np.std(cv_res['test_accuracy']) * 100
            f1_mean = np.mean(cv_res['test_f1'])
            prec_mean = np.mean(cv_res['test_precision'])
            rec_mean = np.mean(cv_res['test_recall'])
            
            results.append({
                'n_estimators': n_est,
                'max_samples': max_s,
                'avg_cv_accuracy': round(acc_mean, 2),
                'cv_accuracy_std': round(acc_std, 2),
                'avg_cv_f1': round(f1_mean, 4),
                'avg_cv_precision': round(prec_mean, 4),
                'avg_cv_recall': round(rec_mean, 4)
            })
            
    bag_df = pd.DataFrame(results)
    
    # Plot Bagging Hyperparameter Curves
    plt.figure(figsize=(9, 5.5))
    for max_s in max_samples_list:
        sub = bag_df[bag_df['max_samples'] == max_s]
        plt.plot(sub['n_estimators'], sub['avg_cv_accuracy'], marker='o', linewidth=2, label=f"max_samples = {max_s}")
    
    plt.title("Bagging Classifier: 5-Fold Cross-Validation Accuracy vs Number of Estimators", fontweight='bold', fontsize=12)
    plt.xlabel("Number of Estimators ($n\\_estimators$)", fontweight='bold')
    plt.ylabel("Avg 5-Fold CV Accuracy (%)", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Bootstrap Fraction", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'bagging_hyperparameter_curve.png'), bbox_inches='tight')
    plt.close()
    
    best_row = bag_df.sort_values(by=['avg_cv_accuracy', 'avg_cv_f1'], ascending=False).iloc[0]
    print(f"Best Bagging Config: n_estimators={int(best_row['n_estimators'])}, max_samples={best_row['max_samples']} -> CV Acc={best_row['avg_cv_accuracy']:.2f}%, F1={best_row['avg_cv_f1']:.4f}", flush=True)
    
    return bag_df, int(best_row['n_estimators']), float(best_row['max_samples'])

def run_boosting_hyperparameter_study(X_train, y_train, cv):
    print(">>> 4. Running Boosting Hyperparameter Grid Study...", flush=True)
    
    # 4.1 Gradient Boosting Study
    gb_configs = [
        {'n_estimators': 20, 'learning_rate': 0.01, 'max_depth': 2},
        {'n_estimators': 20, 'learning_rate': 0.1, 'max_depth': 2},
        {'n_estimators': 50, 'learning_rate': 0.01, 'max_depth': 2},
        {'n_estimators': 50, 'learning_rate': 0.05, 'max_depth': 3},
        {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 100, 'learning_rate': 0.01, 'max_depth': 3},
        {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 100, 'learning_rate': 0.2, 'max_depth': 3},
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3},
        {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 200, 'learning_rate': 0.2, 'max_depth': 2},
    ]
    
    results = []
    for cfg in gb_configs:
        gb = GradientBoostingClassifier(
            n_estimators=cfg['n_estimators'],
            learning_rate=cfg['learning_rate'],
            max_depth=cfg['max_depth'],
            random_state=42
        )
        cv_res = cross_validate(gb, X_train, y_train, cv=cv, scoring=['accuracy', 'f1', 'precision', 'recall'])
        acc_mean = np.mean(cv_res['test_accuracy']) * 100
        acc_std = np.std(cv_res['test_accuracy']) * 100
        f1_mean = np.mean(cv_res['test_f1'])
        
        results.append({
            'algorithm': 'Gradient Boosting',
            'n_estimators': cfg['n_estimators'],
            'learning_rate': cfg['learning_rate'],
            'max_depth': cfg['max_depth'],
            'avg_cv_accuracy': round(acc_mean, 2),
            'cv_accuracy_std': round(acc_std, 2),
            'avg_cv_f1': round(f1_mean, 4)
        })
        
    # Also evaluate AdaBoost variations
    ada_configs = [
        {'n_estimators': 20, 'learning_rate': 0.1},
        {'n_estimators': 50, 'learning_rate': 0.1},
        {'n_estimators': 50, 'learning_rate': 0.5},
        {'n_estimators': 50, 'learning_rate': 1.0},
        {'n_estimators': 100, 'learning_rate': 0.1},
        {'n_estimators': 100, 'learning_rate': 0.5},
        {'n_estimators': 100, 'learning_rate': 1.0},
        {'n_estimators': 200, 'learning_rate': 0.5},
        {'n_estimators': 200, 'learning_rate': 1.0},
    ]
    
    for cfg in ada_configs:
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=cfg['n_estimators'],
            learning_rate=cfg['learning_rate'],
            random_state=42
        )
        cv_res = cross_validate(ada, X_train, y_train, cv=cv, scoring=['accuracy', 'f1'])
        acc_mean = np.mean(cv_res['test_accuracy']) * 100
        f1_mean = np.mean(cv_res['test_f1'])
        
        results.append({
            'algorithm': 'AdaBoost',
            'n_estimators': cfg['n_estimators'],
            'learning_rate': cfg['learning_rate'],
            'max_depth': 1,
            'avg_cv_accuracy': round(acc_mean, 2),
            'cv_accuracy_std': round(np.std(cv_res['test_accuracy']) * 100, 2),
            'avg_cv_f1': round(f1_mean, 4)
        })

    boost_df = pd.DataFrame(results)
    
    # Plot Boosting Hyperparameter Heatmap for Gradient Boosting
    gb_sub = boost_df[boost_df['algorithm'] == 'Gradient Boosting']
    pivot_gb = gb_sub.pivot_table(index='learning_rate', columns='n_estimators', values='avg_cv_accuracy')
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_gb, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Avg CV Accuracy (%)'}, linewidths=1)
    plt.title("Gradient Boosting: 5-Fold CV Accuracy (%) across Hyperparameters", fontweight='bold', fontsize=12)
    plt.xlabel("Number of Estimators ($n\\_estimators$)", fontweight='bold')
    plt.ylabel("Learning Rate ($\\eta$)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'boosting_hyperparameter_heatmap.png'), bbox_inches='tight')
    plt.close()
    
    best_gb = gb_sub.sort_values(by=['avg_cv_accuracy', 'avg_cv_f1'], ascending=False).iloc[0]
    best_ada = boost_df[boost_df['algorithm'] == 'AdaBoost'].sort_values(by=['avg_cv_accuracy', 'avg_cv_f1'], ascending=False).iloc[0]
    
    print(f"Best Gradient Boosting: n_est={best_gb['n_estimators']}, lr={best_gb['learning_rate']}, depth={best_gb['max_depth']} -> CV Acc={best_gb['avg_cv_accuracy']:.2f}%", flush=True)
    print(f"Best AdaBoost: n_est={best_ada['n_estimators']}, lr={best_ada['learning_rate']} -> CV Acc={best_ada['avg_cv_accuracy']:.2f}%", flush=True)
    
    return boost_df, best_gb, best_ada

def run_stacking_evaluation_study(X_train, y_train, cv):
    print(">>> 5. Running Stacked Ensemble Evaluation Study...", flush=True)
    
    # Base Estimators
    svm_clf = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
    nb_clf = GaussianNB()
    dt_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    
    # Meta Learners
    meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    meta_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    
    stacking_experiments = [
        {
            'name': 'SVM + Naive Bayes + Decision Tree',
            'base_models_str': 'SVM, Naïve Bayes, Decision Tree',
            'estimators': [('svm', svm_clf), ('nb', nb_clf), ('dt', dt_clf)],
            'meta_learner_str': 'Logistic Regression',
            'final_estimator': meta_lr
        },
        {
            'name': 'SVM + Naive Bayes',
            'base_models_str': 'SVM, Naïve Bayes',
            'estimators': [('svm', svm_clf), ('nb', nb_clf)],
            'meta_learner_str': 'Logistic Regression',
            'final_estimator': meta_lr
        },
        {
            'name': 'SVM + Decision Tree',
            'base_models_str': 'SVM, Decision Tree',
            'estimators': [('svm', svm_clf), ('dt', dt_clf)],
            'meta_learner_str': 'Logistic Regression',
            'final_estimator': meta_lr
        },
        {
            'name': 'Naive Bayes + Decision Tree',
            'base_models_str': 'Naïve Bayes, Decision Tree',
            'estimators': [('nb', nb_clf), ('dt', dt_clf)],
            'meta_learner_str': 'Logistic Regression',
            'final_estimator': meta_lr
        },
        {
            'name': 'SVM + Naive Bayes + Decision Tree (Meta: Decision Tree)',
            'base_models_str': 'SVM, Naïve Bayes, Decision Tree',
            'estimators': [('svm', svm_clf), ('nb', nb_clf), ('dt', dt_clf)],
            'meta_learner_str': 'Decision Tree',
            'final_estimator': meta_dt
        },
        {
            'name': 'SVM + Naive Bayes + Decision Tree (Meta: Random Forest)',
            'base_models_str': 'SVM, Naïve Bayes, Decision Tree',
            'estimators': [('svm', svm_clf), ('nb', nb_clf), ('dt', dt_clf)],
            'meta_learner_str': 'Random Forest',
            'final_estimator': meta_rf
        },
    ]
    
    results = []
    stacking_models = {}
    
    for exp in stacking_experiments:
        stack = StackingClassifier(
            estimators=exp['estimators'],
            final_estimator=exp['final_estimator'],
            cv=5,
            n_jobs=1
        )
        cv_res = cross_validate(stack, X_train, y_train, cv=cv, scoring=['accuracy', 'f1', 'precision', 'recall'])
        acc_mean = np.mean(cv_res['test_accuracy']) * 100
        acc_std = np.std(cv_res['test_accuracy']) * 100
        f1_mean = np.mean(cv_res['test_f1'])
        prec_mean = np.mean(cv_res['test_precision'])
        rec_mean = np.mean(cv_res['test_recall'])
        
        results.append({
            'base_models': exp['base_models_str'],
            'meta_learner': exp['meta_learner_str'],
            'avg_cv_accuracy': round(acc_mean, 2),
            'cv_accuracy_std': round(acc_std, 2),
            'avg_cv_f1': round(f1_mean, 4),
            'avg_cv_precision': round(prec_mean, 4),
            'avg_cv_recall': round(rec_mean, 4)
        })
        stacking_models[exp['name']] = stack
        
    stack_df = pd.DataFrame(results)
    
    # Plot Stacked Model Comparison
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        x='avg_cv_accuracy', y='base_models', hue='meta_learner',
        data=stack_df, palette='Set2', edgecolor='black', linewidth=1
    )
    plt.xlim(90, 100)
    plt.title("Stacked Ensemble: 5-Fold CV Accuracy Across Heterogeneous Model Subsets", fontweight='bold', fontsize=12)
    plt.xlabel("Avg 5-Fold Cross-Validation Accuracy (%)", fontweight='bold')
    plt.ylabel("Base Model Combinations", fontweight='bold')
    plt.legend(title="Meta-Learner", loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'stacked_ensemble_comparison.png'), bbox_inches='tight')
    plt.close()
    
    best_stack = stack_df.sort_values(by=['avg_cv_accuracy', 'avg_cv_f1'], ascending=False).iloc[0]
    print(f"Best Stacking Configuration: Base=[{best_stack['base_models']}], Meta=[{best_stack['meta_learner']}] -> CV Acc={best_stack['avg_cv_accuracy']:.2f}%, F1={best_stack['avg_cv_f1']:.4f}", flush=True)
    
    return stack_df, stacking_models

def calculate_bias_variance_decomposition(models_dict, X_train, y_train, X_test, y_test, n_bootstraps=50):
    print(f">>> 6. Computing Empirical Bias-Variance Decomposition (Bootstrapping N={n_bootstraps})...", flush=True)
    np.random.seed(42)
    
    bv_results = {}
    
    for name, model_factory in models_dict.items():
        predictions = np.zeros((n_bootstraps, len(y_test)))
        
        for b in range(n_bootstraps):
            boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_b, y_b = X_train[boot_idx], y_train[boot_idx]
            
            clf = model_factory()
            clf.fit(X_b, y_b)
            predictions[b, :] = clf.predict(X_test)
            
        main_predictions = (np.mean(predictions, axis=0) >= 0.5).astype(int)
        loss = np.mean(predictions != y_test[None, :])
        bias = np.mean(main_predictions != y_test)
        variance = np.mean(predictions != main_predictions[None, :])
        
        bv_results[name] = {
            'loss': float(loss),
            'bias': float(bias),
            'variance': float(variance),
            'variance_pct': float(variance * 100),
            'bias_pct': float(bias * 100),
            'total_error_pct': float(loss * 100)
        }
        print(f"  [{name:28s}] Total Error: {loss*100:5.2f}% | Bias: {bias*100:5.2f}% | Variance: {variance*100:5.2f}%", flush=True)
        
    bv_df = pd.DataFrame(bv_results).T.reset_index().rename(columns={'index': 'Model'})
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_width = 0.35
    x = np.arange(len(bv_df))
    
    ax.bar(x - bar_width/2, bv_df['bias_pct'], width=bar_width, label='Bias Error (%)', color='#d9534f', edgecolor='black', alpha=0.85)
    ax.bar(x + bar_width/2, bv_df['variance_pct'], width=bar_width, label='Variance Error (%)', color='#2b5c8f', edgecolor='black', alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(bv_df['Model'], rotation=25, ha='right', fontweight='bold', fontsize=10)
    ax.set_ylabel("Error Rate (%)", fontweight='bold')
    ax.set_title("Empirical Bias-Variance Decomposition Across Ensemble Strategies", fontweight='bold', fontsize=13)
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'bias_variance_decomposition.png'), bbox_inches='tight')
    plt.close()
    
    return bv_results

def main():
    print("==========================================================================================", flush=True)
    print("  EXPERIMENT 6/7: BAGGING, BOOSTING, AND STACKED ENSEMBLE BENCHMARK PIPELINE             ", flush=True)
    print("==========================================================================================", flush=True)
    
    # 1. Load Data
    df, feature_cols, X, y = load_and_prepare_data()
    
    # 2. Generate EDA Plots
    generate_eda_plots(df, feature_cols)
    
    # 3. Train-Test Split (80% Train, 20% Test, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples (Malignant={np.sum(y_train==1)}, Benign={np.sum(y_train==0)})", flush=True)
    print(f"Test set:  {X_test.shape[0]} samples (Malignant={np.sum(y_test==1)}, Benign={np.sum(y_test==0)})", flush=True)
    
    # Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 4. Hyperparameter Evaluations
    bag_df, best_bag_nest, best_bag_ms = run_bagging_hyperparameter_study(X_train_scaled, y_train, skf)
    boost_df, best_gb, best_ada = run_boosting_hyperparameter_study(X_train_scaled, y_train, skf)
    stack_df, stacking_models = run_stacking_evaluation_study(X_train_scaled, y_train, skf)
    
    # 5. Define Optimized Models for Table 4 Performance Comparison
    models_to_benchmark = {
        'Baseline: Single Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
        'Baseline: Support Vector Machine': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        'Baseline: Gaussian Naïve Bayes': GaussianNB(),
        'Bagging Classifier (Decision Tree)': BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=best_bag_nest,
            max_samples=best_bag_ms,
            max_features=1.0,
            random_state=42,
            n_jobs=1
        ),
        'AdaBoost Classifier': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=int(best_ada['n_estimators']),
            learning_rate=float(best_ada['learning_rate']),
            random_state=42
        ),
        'Gradient Boosting Classifier': GradientBoostingClassifier(
            n_estimators=int(best_gb['n_estimators']),
            learning_rate=float(best_gb['learning_rate']),
            max_depth=int(best_gb['max_depth']),
            random_state=42
        ),
        'Stacked Ensemble (SVM+NB+DT -> LR)': StackingClassifier(
            estimators=[
                ('svm', SVC(C=1.0, kernel='rbf', probability=True, random_state=42)),
                ('nb', GaussianNB()),
                ('dt', DecisionTreeClassifier(max_depth=4, random_state=42))
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            cv=5,
            n_jobs=1
        )
    }
    
    print("\n>>> 7. Evaluating All Models on Independent 20% Test Set...", flush=True)
    eval_results = {}
    for name, model in models_to_benchmark.items():
        res = evaluate_model_on_test(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
        eval_results[name] = res
        print(f"  {name:38s} | Acc: {res['accuracy_pct']:6.2f}% | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f} | F1: {res['f1_score']:.4f} | AUC: {res['roc_auc']:.4f}", flush=True)

    # Build Performance Comparison Table
    table4_rows = []
    for name, res in eval_results.items():
        table4_rows.append({
            'Model': name,
            'Accuracy (%)': round(res['accuracy_pct'], 2),
            'Precision': round(res['precision'], 4),
            'Recall': round(res['recall'], 4),
            'F1 Score': round(res['f1_score'], 4),
            'ROC-AUC': round(res['roc_auc'], 4),
            'Train Time (ms)': round(res['train_time_ms'], 2),
            'Inference Time (ms)': round(res['inf_time_ms'], 2)
        })
    table4_df = pd.DataFrame(table4_rows)
    
    # 6. Generate Model Comparison Visualizations
    
    # 6.1 Confusion Matrices Grid
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    
    idx = 0
    for name, res in eval_results.items():
        cm = np.array(res['confusion_matrix'])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=['Benign (0)', 'Malignant (1)'],
            yticklabels=['Benign (0)', 'Malignant (1)']
        )
        axes[idx].set_title(f"{name}\nAcc: {res['accuracy_pct']:.2f}% | F1: {res['f1_score']:.3f}", fontweight='bold', fontsize=10)
        axes[idx].set_xlabel("Predicted Label", fontweight='bold')
        axes[idx].set_ylabel("True Label", fontweight='bold')
        idx += 1
        
    # Hide 8th unused subplot
    axes[7].axis('off')
    
    plt.suptitle("Confusion Matrices on Test Set (114 Samples: 71 Benign, 43 Malignant)", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'model_confusion_matrices.png'), bbox_inches='tight')
    plt.close()
    
    # 6.2 ROC Curves Comparison
    plt.figure(figsize=(9, 7))
    palette_colors = ['#888888', '#17a2b8', '#6c757d', '#28a745', '#ffc107', '#fd7e14', '#007bff']
    
    for i, (name, res) in enumerate(eval_results.items()):
        y_prob = np.array(res['y_prob'])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_val = res['roc_auc']
        plt.plot(fpr, tpr, color=palette_colors[i % len(palette_colors)], linewidth=2.2, label=f"{name} (AUC = {roc_val:.4f})")
        
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Chance (AUC = 0.5000)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight='bold', fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontweight='bold', fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves Comparison", fontweight='bold', fontsize=13)
    plt.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'model_roc_curves.png'), bbox_inches='tight')
    plt.close()

    # 6.3 Precision-Recall Curves
    plt.figure(figsize=(9, 7))
    for i, (name, res) in enumerate(eval_results.items()):
        y_prob = np.array(res['y_prob'])
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        plt.plot(rec_curve, prec_curve, color=palette_colors[i % len(palette_colors)], linewidth=2.2, label=f"{name} (AP = {ap:.4f})")
        
    plt.xlabel("Recall (Sensitivity)", fontweight='bold', fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontweight='bold', fontsize=11)
    plt.title("Precision-Recall Curves Comparison", fontweight='bold', fontsize=13)
    plt.legend(loc="lower left", frameon=True, fontsize=9.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'model_pr_curves.png'), bbox_inches='tight')
    plt.close()

    # 6.4 Grouped Metric Comparison Bar Chart
    metrics_to_plot = ['Accuracy (%)', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    plot_df = table4_df.copy()
    plot_df['Accuracy'] = plot_df['Accuracy (%)'] / 100.0
    
    melted_df = pd.melt(
        plot_df,
        id_vars=['Model'],
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        var_name='Metric',
        value_name='Score'
    )
    
    plt.figure(figsize=(13, 6))
    sns.barplot(x='Model', y='Score', hue='Metric', data=melted_df, palette='muted', edgecolor='black', linewidth=0.8)
    plt.ylim(0.85, 1.01)
    plt.title("Multi-Metric Comparison: Baseline vs Ensemble Classifiers", fontweight='bold', fontsize=13)
    plt.xlabel("Model", fontweight='bold')
    plt.ylabel("Score", fontweight='bold')
    plt.xticks(rotation=20, ha='right', fontweight='bold')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'model_metrics_bar_comparison.png'), bbox_inches='tight')
    plt.close()

    # 6.5 Feature Importance Comparison (Single Tree vs Bagging vs Gradient Boosting)
    dt_single = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train_scaled, y_train)
    gb_model = GradientBoostingClassifier(n_estimators=int(best_gb['n_estimators']), learning_rate=float(best_gb['learning_rate']), max_depth=int(best_gb['max_depth']), random_state=42).fit(X_train_scaled, y_train)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
    
    feat_imp_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in feature_cols],
        'Single Decision Tree': dt_single.feature_importances_,
        'Random Forest (Bagging Variant)': rf_model.feature_importances_,
        'Gradient Boosting': gb_model.feature_importances_
    })
    
    # Get top 10 features by Gradient Boosting
    top_feats = feat_imp_df.sort_values(by='Gradient Boosting', ascending=False).head(10)
    top_melted = pd.melt(top_feats, id_vars=['Feature'], var_name='Model', value_name='Importance')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', hue='Model', data=top_melted, palette='viridis', edgecolor='black', linewidth=0.8)
    plt.title("Top 10 Feature Importances: Single Tree vs Ensemble Strategies", fontweight='bold', fontsize=13)
    plt.xlabel("Gini / Loss Reduction Importance Score", fontweight='bold')
    plt.ylabel("Cytological Feature", fontweight='bold')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'feature_importance_comparison.png'), bbox_inches='tight')
    plt.close()

    # 6.6 Training & Inference Time Benchmarks
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x='Train Time (ms)', y='Model', data=table4_df, palette='Reds_r', edgecolor='black', linewidth=0.8, ax=axes[0])
    axes[0].set_title("Training Execution Time (ms)", fontweight='bold')
    axes[0].set_xlabel("Time (milliseconds)")
    axes[0].set_xscale('log')
    
    sns.barplot(x='Inference Time (ms)', y='Model', data=table4_df, palette='Greens_r', edgecolor='black', linewidth=0.8, ax=axes[1])
    axes[1].set_title("Inference / Prediction Time (ms)", fontweight='bold')
    axes[1].set_xlabel("Time (milliseconds)")
    
    plt.suptitle("Computational Time Benchmarks Across Models", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'computational_time_benchmark.png'), bbox_inches='tight')
    plt.close()

    # 7. Bias-Variance Decomposition
    bv_factories = {
        'Single Decision Tree': lambda: DecisionTreeClassifier(max_depth=None, random_state=None),
        'Bagging (100 Trees)': lambda: BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, max_samples=0.8, random_state=None),
        'AdaBoost (50 Stumps)': lambda: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=None),
        'Gradient Boosting (100 Trees)': lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None),
        'Stacked Ensemble (SVM+NB+DT)': lambda: StackingClassifier(
            estimators=[('svm', SVC(C=1.0, probability=True)), ('nb', GaussianNB()), ('dt', DecisionTreeClassifier(max_depth=4))],
            final_estimator=LogisticRegression(C=1.0, max_iter=1000)
        )
    }
    bv_results = calculate_bias_variance_decomposition(bv_factories, X_train_scaled, y_train, X_test_scaled, y_test, n_bootstraps=50)

    # 8. Save Full JSON Summary
    summary_data = {
        'metadata': {
            'experiment_number': '7',
            'assignment_folder': 'Assignment 6',
            'title': 'Bagging, Boosting, and Stacked Ensemble Models',
            'dataset': 'Wisconsin Diagnostic Breast Cancer (WDBC)',
            'student_name': 'Rishi Rithesh',
            'register_number': '3122247001049',
            'faculty': 'Dr. Poreddy Ajay Kumar Reddy',
            'github_url': 'https://github.com/rishirithesh/ML_Lab'
        },
        'dataset_summary': {
            'total_samples': int(len(df)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'num_features': int(len(feature_cols)),
            'malignant_count': int(np.sum(y == 1)),
            'benign_count': int(np.sum(y == 0)),
            'missing_values': int(df.isnull().sum().sum())
        },
        'table1_bagging': bag_df.to_dict(orient='records'),
        'table2_boosting': boost_df.to_dict(orient='records'),
        'table3_stacking': stack_df.to_dict(orient='records'),
        'table4_performance': table4_df.to_dict(orient='records'),
        'bias_variance_decomposition': bv_results
    }
    
    with open(RESULTS_JSON_PATH, 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"\n>>> Results successfully generated and exported to {RESULTS_JSON_PATH}", flush=True)
    print("==========================================================================================", flush=True)
    print("  EXPERIMENT 6/7 BENCHMARK COMPLETED SUCCESSFULLY                                       ", flush=True)
    print("==========================================================================================", flush=True)

if __name__ == '__main__':
    main()

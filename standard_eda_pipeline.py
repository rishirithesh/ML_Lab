"""
Standard Exploratory Data Analysis (EDA) Pipeline
=================================================
A universal, reusable Python EDA pipeline designed to run before any Machine Learning task
on any tabular dataset (pandas DataFrame or CSV file).

Key Features:
-------------
1. Dataset Overview & Data Quality Audit (shapes, dtypes, memory, duplicates)
2. Missing Value Analysis & Visualizations
3. Numerical & Categorical Descriptive Statistics (with Skewness & Outliers)
4. Target Variable Analysis (Auto-detects Regression vs Classification)
5. Feature Distributions & Outlier Detection
6. Pairwise Correlation & Multicollinearity Analysis
7. Bivariate Feature-vs-Target Analysis
8. Automated Plot & Summary Exporting

Author: Rishi Rithesh
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Visual style setup
plt.style.use('default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300


class StandardEDAPipeline:
    """
    Automated EDA pipeline class for pre-ML dataset inspection.
    """
    def __init__(self, data_input, target_col=None, output_dir="eda_output"):
        """
        Initialize the EDA Pipeline.

        Parameters:
        -----------
        data_input : str or pd.DataFrame
            Path to CSV file or existing pandas DataFrame.
        target_col : str, optional
            Name of the target variable column (if performing supervised learning).
        output_dir : str
            Directory path to save generated plots and summary reports.
        """
        if isinstance(data_input, str):
            if not os.path.exists(data_input):
                raise FileNotFoundError(f"File not found at path: {data_input}")
            self.df = pd.read_csv(data_input)
            self.dataset_name = os.path.splitext(os.path.basename(data_input))[0]
        elif isinstance(data_input, pd.DataFrame):
            self.df = data_input.copy()
            self.dataset_name = "dataset"
        else:
            raise ValueError("data_input must be a file path (str) or pd.DataFrame")

        self.target_col = target_col
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Classify column types
        self.num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if self.target_col and self.target_col in self.df.columns:
            # Determine problem type
            if self.df[self.target_col].nunique() <= 10 or self.df[self.target_col].dtype in ['object', 'category', 'bool']:
                self.problem_type = 'classification'
            else:
                self.problem_type = 'regression'
        else:
            self.problem_type = 'unsupervised'

    def run_full_pipeline(self):
        """Executes all EDA steps sequentially and exports visual artifacts."""
        print(f"\n========================================================")
        print(f"       RUNNING STANDARD EDA PIPELINE FOR: {self.dataset_name}")
        print(f"========================================================\n")

        summary = {}
        summary['overview'] = self.analyze_overview()
        summary['missing'] = self.analyze_missing_values()
        summary['num_stats'] = self.analyze_numerical_stats()
        summary['cat_stats'] = self.analyze_categorical_stats()
        
        if self.target_col:
            summary['target'] = self.analyze_target()
            
        summary['correlation'] = self.analyze_correlations()
        self.plot_feature_distributions()
        
        if self.target_col:
            self.plot_bivariate_relationships()

        print(f"\n[INFO] EDA Pipeline Complete. All visual plots saved to: {os.path.abspath(self.output_dir)}")
        return summary

    def analyze_overview(self):
        """step 1: Dataset Overview & Health Audit"""
        print("--- Step 1: Dataset Overview ---")
        n_rows, n_cols = self.df.shape
        n_duplicates = self.df.duplicated().sum()
        mem_usage_mb = self.df.memory_usage(deep=True).sum() / (1024 ** 2)

        print(f"Rows: {n_rows} | Columns: {n_cols}")
        print(f"Duplicate Rows: {n_duplicates} ({n_duplicates/n_rows*100:.2f}%)")
        print(f"Memory Usage: {mem_usage_mb:.2f} MB")
        print(f"Numerical Features ({len(self.num_cols)}): {self.num_cols}")
        print(f"Categorical Features ({len(self.cat_cols)}): {self.cat_cols}")
        print(f"Supervised Target: '{self.target_col}' (Problem Type: {self.problem_type.upper()})\n")

        overview_dict = {
            'shape': (n_rows, n_cols),
            'duplicates': n_duplicates,
            'memory_mb': round(mem_usage_mb, 2),
            'num_cols': self.num_cols,
            'cat_cols': self.cat_cols,
            'problem_type': self.problem_type
        }
        return overview_dict

    def analyze_missing_values(self):
        """Step 2: Missing Value Analysis & Bar Chart"""
        print("--- Step 2: Missing Value Audit ---")
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing_count,
            'Missing_Percentage': missing_pct
        }).sort_values(by='Missing_Count', ascending=False)
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0]

        if missing_df.empty:
            print("No missing values found in the dataset.\n")
        else:
            print(f"Found {len(missing_df)} columns with missing values:")
            print(missing_df.to_string())
            print()

            # Plot missing value chart
            fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.3)))
            bars = ax.barh(missing_df.index, missing_df['Missing_Count'], color='#e74c3c', edgecolor='black')
            ax.set_xlabel("Number of Missing Values", fontweight='bold')
            ax.set_title("Missing Values Count per Feature", fontweight='bold', fontsize=12)
            
            for bar in bars:
                w = bar.get_width()
                pct = (w / len(self.df)) * 100
                ax.text(w + (len(self.df)*0.01), bar.get_y() + bar.get_height()/2, f"{int(w)} ({pct:.1f}%)", va='center', fontsize=8)
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "eda_missing_values.png"), bbox_inches='tight')
            plt.close()

        return missing_df

    def analyze_numerical_stats(self):
        """Step 3: Numerical Feature Summary & Skewness"""
        print("--- Step 3: Numerical Feature Statistics ---")
        if not self.num_cols:
            print("No numerical columns present.\n")
            return None

        desc = self.df[self.num_cols].describe().T
        desc['skewness'] = self.df[self.num_cols].skew()
        
        # Calculate IQR outlier counts
        outlier_counts = []
        for col in self.num_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((self.df[col] < (q1 - 1.5 * iqr)) | (self.df[col] > (q3 + 1.5 * iqr))).sum()
            outlier_counts.append(outliers)
            
        desc['IQR_Outliers'] = outlier_counts
        print(desc[['count', 'mean', 'std', 'min', '50%', 'max', 'skewness', 'IQR_Outliers']].round(2).to_string())
        print()
        return desc

    def analyze_categorical_stats(self):
        """Step 4: Categorical Feature Summary"""
        print("--- Step 4: Categorical Feature Statistics ---")
        if not self.cat_cols:
            print("No categorical columns present.\n")
            return None

        cat_summary = []
        for col in self.cat_cols:
            top_val = self.df[col].mode()[0] if not self.df[col].mode().empty else np.nan
            top_freq = self.df[col].value_counts().max() if not self.df[col].dropna().empty else 0
            cat_summary.append({
                'Feature': col,
                'Unique_Values': self.df[col].nunique(),
                'Top_Category': top_val,
                'Top_Frequency': top_freq,
                'Top_Percentage': round((top_freq / len(self.df)) * 100, 2)
            })
            
        cat_df = pd.DataFrame(cat_summary)
        print(cat_df.to_string(index=False))
        print()
        return cat_df

    def analyze_target(self):
        """Step 5: Target Variable Visual Inspection"""
        print(f"--- Step 5: Target Variable Analysis ({self.target_col}) ---")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        if self.problem_type == 'regression':
            ax.hist(self.df[self.target_col].dropna(), bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
            ax.set_title(f"Target Distribution: {self.target_col} (Regression)", fontweight='bold', fontsize=12)
            ax.set_xlabel(self.target_col, fontweight='bold')
            ax.set_ylabel("Frequency", fontweight='bold')
        else:
            val_counts = self.df[self.target_col].value_counts()
            val_counts.plot(kind='bar', ax=ax, color='#3498db', edgecolor='black')
            ax.set_title(f"Target Class Balance: {self.target_col} (Classification)", fontweight='bold', fontsize=12)
            ax.set_xlabel("Classes", fontweight='bold')
            ax.set_ylabel("Count", fontweight='bold')
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "eda_target_distribution.png"), bbox_inches='tight')
        plt.close()
        print(f"Target plot saved: eda_target_distribution.png\n")

    def analyze_correlations(self):
        """Step 6: Correlation Matrix Heatmap & Multicollinearity"""
        print("--- Step 6: Correlation & Multicollinearity Analysis ---")
        if len(self.num_cols) < 2:
            print("Fewer than 2 numerical features; skipping correlation heatmap.\n")
            return None

        corr_matrix = self.df[self.num_cols].corr()

        # Plot Heatmap
        fig, ax = plt.subplots(figsize=(min(12, max(6, len(self.num_cols)*0.8)), min(10, max(5, len(self.num_cols)*0.7))))
        cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)

        ticks = np.arange(0, len(corr_matrix.columns), 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr_matrix.columns, fontsize=8)

        # Annotate values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black' if abs(val) < 0.6 else 'white', fontsize=7)

        plt.title("Correlation Matrix Heatmap", fontweight='bold', fontsize=12, pad=30)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "eda_correlation_heatmap.png"), bbox_inches='tight')
        plt.close()

        # Identify high correlation pairs (|r| > 0.8)
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                r = corr_matrix.iloc[i, j]
                if abs(r) >= 0.8:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], round(r, 4)))

        if high_corr:
            print("High Multicollinearity Warning (|r| >= 0.8):")
            for feat1, feat2, r_val in high_corr:
                print(f"  - {feat1} <--> {feat2}: r = {r_val}")
        else:
            print("No severe multicollinearity (|r| >= 0.8) detected among numeric features.")
        print()

        return corr_matrix

    def plot_feature_distributions(self):
        """Step 7: Univariate Feature Distribution Plots"""
        print("--- Step 7: Plotting Feature Distributions ---")
        
        # Plot Numerical Features Histograms
        if self.num_cols:
            n = len(self.num_cols)
            cols_per_row = 3
            rows = int(np.ceil(n / cols_per_row))
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 3.5 * rows))
            axes = np.array(axes).flatten()

            for i, col in enumerate(self.num_cols):
                axes[i].hist(self.df[col].dropna(), bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
                axes[i].set_title(col, fontweight='bold', fontsize=10)
                
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle("Numerical Feature Distributions", fontweight='bold', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "eda_numerical_distributions.png"), bbox_inches='tight')
            plt.close()

        # Plot Categorical Top Frequencies
        if self.cat_cols:
            n_cat = min(9, len(self.cat_cols))
            cols_per_row = 3
            rows = int(np.ceil(n_cat / cols_per_row))
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))
            axes = np.array(axes).flatten()

            for i, col in enumerate(self.cat_cols[:n_cat]):
                top_v = self.df[col].value_counts().head(7)
                top_v.plot(kind='barh', ax=axes[i], color='#f39c12', edgecolor='black')
                axes[i].set_title(f"{col} (Top Categories)", fontweight='bold', fontsize=10)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle("Categorical Feature Distributions", fontweight='bold', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "eda_categorical_distributions.png"), bbox_inches='tight')
            plt.close()

        print("Distribution plots saved successfully.\n")

    def plot_bivariate_relationships(self):
        """Step 8: Feature vs Target Relationships"""
        print("--- Step 8: Plotting Feature vs Target Bivariate Relationships ---")
        if not self.target_col or self.target_col not in self.df.columns:
            return

        num_features = [c for c in self.num_cols if c != self.target_col]
        if not num_features:
            return

        # Scatter plots for regression or Boxplots for classification
        top_num = num_features[:6]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        for i, col in enumerate(top_num):
            valid = self.df.dropna(subset=[col, self.target_col])
            if self.problem_type == 'regression':
                axes[i].scatter(valid[col], valid[self.target_col], alpha=0.5, color='#34495e', s=20)
                axes[i].set_xlabel(col, fontweight='bold')
                axes[i].set_ylabel(self.target_col, fontweight='bold')
            else:
                valid.boxplot(column=col, by=self.target_col, ax=axes[i])
                axes[i].set_xlabel(self.target_col, fontweight='bold')
                axes[i].set_ylabel(col, fontweight='bold')

            axes[i].set_title(f"{col} vs {self.target_col}", fontweight='bold', fontsize=10)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Bivariate Relationships vs Target ({self.target_col})", fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "eda_bivariate_target_plots.png"), bbox_inches='tight')
        plt.close()
        print("Bivariate target plots saved.\n")


# Execution example for Assignment 3 dataset
if __name__ == "__main__":
    sample_dataset = os.path.join(os.path.dirname(__file__), "loan_amount_prediction_uncleaned_dataset.csv")
    
    if os.path.exists(sample_dataset):
        print("Running EDA Pipeline on Assignment 3 Dataset...")
        eda = StandardEDAPipeline(data_input=sample_dataset, target_col="loan_amount", output_dir="eda_reports")
        eda.run_full_pipeline()
    else:
        print("EDA Pipeline script ready. Import and instantiate StandardEDAPipeline(df, target_col='y') for any dataset.")

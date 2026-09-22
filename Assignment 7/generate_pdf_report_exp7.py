import os
import json
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
    PageBreak
)
from reportlab.pdfgen import canvas

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, 'Exp7.pdf')
JSON_PATH = os.path.join(SCRIPT_DIR, 'results_summary.json')

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#333333"))
        
        # Header (pages >= 1)
        self.drawString(54, 792, "ICS1512")
        self.drawRightString(541, 792, "Machine Learning Algorithms Laboratory")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 784, 541, 784)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 45, 541, 45)
        self.drawRightString(541, 32, f"{self._pageNumber}")
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1A1A1A")
    )
    
    bold_style = ParagraphStyle(
        'CustomBold',
        parent=normal_style,
        fontName='Helvetica-Bold'
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.black,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#333333"),
        spaceBefore=4,
        spaceAfter=6
    )

    table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11
    )

    story = []

    # Title Block
    header_data = [
        [Paragraph("<b>Experiment No.</b>", normal_style), Paragraph("7", normal_style)],
        [Paragraph("<b>Experiment Title</b>", normal_style), Paragraph("Dimensionality Reduction and Model Evaluation (With and Without PCA)", normal_style)],
        [Paragraph("<b>Student Name</b>", normal_style), Paragraph("Rishi Rithesh", normal_style)],
        [Paragraph("<b>Register Number</b>", normal_style), Paragraph("3122247001049", normal_style)],
        [Paragraph("<b>Faculty</b>", normal_style), Paragraph("Dr. Poreddy Ajay Kumar Reddy", normal_style)],
        [Paragraph("<b>Submission Date</b>", normal_style), Paragraph("21/09/2026", normal_style)],
    ]
    t_header = Table(header_data, colWidths=[130, 357])
    t_header.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 14))

    # 1. Objective
    story.append(Paragraph("1 Objective", h1_style))
    story.append(Paragraph(
        "To study the effect of <b>dimensionality reduction</b> using Principal Component Analysis (PCA) on "
        "the performance of various machine learning classifiers. The task requires:<br/>"
        "a) Training and validating models without PCA (original 30-dimensional feature space).<br/>"
        "b) Training and validating models with PCA (reduced 10-dimensional feature space).<br/>"
        "For both cases, perform hyperparameter tuning, apply 5-fold cross-validation, and record performance across 10 diverse classifiers.",
        normal_style
    ))
    story.append(Spacer(1, 10))

    # 2. Problem Statement
    story.append(Paragraph("2 Problem Statement", h1_style))
    story.append(Paragraph(
        "The objective of this experiment is to study how unsupervised linear dimensionality reduction via Principal Component Analysis (PCA) "
        "affects classifier accuracy, generalizability, training variance, and computational efficiency across diverse learning paradigms.<br/><br/>"
        "The public <b>Wisconsin Diagnostic Breast Cancer (WDBC)</b> dataset is used for this experiment.<br/><br/>"
        "For the dataset, the following tasks are performed:<br/>"
        "1. Identify the ML classification problem and analyze target class distributions.<br/>"
        "2. Perform Exploratory Data Analysis (EDA), Scree plot eigenvalue decomposition, and cumulative variance analysis.<br/>"
        "3. Standardize features using Z-score scaling and apply PCA with a 95% variance retention target.<br/>"
        "4. Define hyperparameter search grids for 10 classifiers: SVM, Naive Bayes, KNN, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and Stacking.<br/>"
        "5. Train and validate all 10 models under both No-PCA and With-PCA settings using 5-fold Stratified Cross-Validation.<br/>"
        "6. Benchmark accuracy, F1-scores, ROC-AUC curves, PR curves, confusion matrices, and computational runtimes.",
        normal_style
    ))
    story.append(Spacer(1, 10))

    # 3. Dataset Description
    story.append(Paragraph("3 Dataset Description", h1_style))
    ds_data = [
        [Paragraph("<b>Dataset Name</b>", normal_style), Paragraph("Wisconsin Diagnostic Breast Cancer (WDBC)", normal_style)],
        [Paragraph("<b>Dataset Source</b>", normal_style), Paragraph("UCI Machine Learning Repository / Scikit-Learn", normal_style)],
        [Paragraph("<b>Number of Samples</b>", normal_style), Paragraph("569 biopsy records", normal_style)],
        [Paragraph("<b>Number of Features</b>", normal_style), Paragraph("30 continuous cytological descriptors", normal_style)],
        [Paragraph("<b>Target Classes</b>", normal_style), Paragraph("2 (Malignant: 212 [37.3%], Benign: 357 [62.7%])", normal_style)],
        [Paragraph("<b>Missing Values</b>", normal_style), Paragraph("0 missing values", normal_style)],
        [Paragraph("<b>Machine Learning Task</b>", normal_style), Paragraph("Supervised Binary Classification", normal_style)],
        [Paragraph("<b>Train–Test Split</b>", normal_style), Paragraph("80% Training (455 samples), 20% Testing (114 samples)", normal_style)],
    ]
    t_ds = Table(ds_data, colWidths=[160, 327])
    t_ds.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 14))

    # 4. Exploratory Data Analysis & Dimensionality Reduction
    story.append(Paragraph("4 Exploratory Data Analysis and Dimensionality Analysis", h1_style))

    # 4.1 Class Distribution
    story.append(Paragraph("4.1 Class Distribution", h2_style))
    img1_path = os.path.join(SCRIPT_DIR, 'figures', 'fig1_class_distribution.png')
    if os.path.exists(img1_path):
        story.append(Image(img1_path, width=4.5*inch, height=3.3*inch))
        story.append(Paragraph("Figure 1: Diagnostic Class Distribution in Breast Cancer Dataset", caption_style))
    story.append(Paragraph(
        "<b>Inference:</b> The dataset contains 357 benign (62.7%) and 212 malignant (37.3%) patient records. "
        "The class distribution is moderately imbalanced but sufficiently populated in both categories to enable unbiased 5-fold stratified cross-validation.",
        normal_style
    ))
    story.append(Spacer(1, 10))

    # 4.2 Scree Plot
    story.append(Paragraph("4.2 PCA Scree Plot (Eigenvalue Spectrum)", h2_style))
    img2_path = os.path.join(SCRIPT_DIR, 'figures', 'fig2_scree_plot.png')
    if os.path.exists(img2_path):
        story.append(Image(img2_path, width=5.2*inch, height=2.9*inch))
        story.append(Paragraph("Figure 2: PCA Scree Plot (Individual Explained Variance Ratio)", caption_style))
    story.append(Paragraph(
        "<b>Inference:</b> Principal Component 1 captures 44.27% of total variance, followed by PC2 with 18.97%, and PC3 with 9.39%. "
        "A sharp 'elbow' is observed after component 6, confirming strong low-rank structure suitable for dimensionality reduction.",
        normal_style
    ))
    story.append(Spacer(1, 10))

    # 4.3 Cumulative Variance
    story.append(Paragraph("4.3 Cumulative Explained Variance Analysis", h2_style))
    img3_path = os.path.join(SCRIPT_DIR, 'figures', 'fig3_cumulative_variance.png')
    if os.path.exists(img3_path):
        story.append(Image(img3_path, width=5.2*inch, height=2.9*inch))
        story.append(Paragraph("Figure 3: Cumulative Explained Variance vs. Number of Components", caption_style))
    story.append(Paragraph(
        "<b>Inference:</b> Exactly <b>10 principal components</b> are required to achieve the 95% target threshold, capturing <b>95.27%</b> "
        "of total dataset variance. This compresses the feature space by 66.7% while retaining the dominant morphological variations.",
        normal_style
    ))
    story.append(Spacer(1, 10))

    # 4.4 2D PCA Projection
    story.append(Paragraph("4.4 2D PCA Projection", h2_style))
    img4_path = os.path.join(SCRIPT_DIR, 'figures', 'fig4_pca_2d_projection.png')
    if os.path.exists(img4_path):
        story.append(Image(img4_path, width=4.8*inch, height=3.4*inch))
        story.append(Paragraph("Figure 4: First Two Principal Components Projection (Train Set)", caption_style))
    story.append(Paragraph(
        "<b>Inference:</b> The 2D projection reveals distinct cluster separation between benign (blue) and malignant (red) classes, indicating "
        "that orthogonal linear combinations maintain high diagnostic discriminability.",
        normal_style
    ))
    story.append(Spacer(1, 10))

    # 4.5 Correlation Heatmap
    story.append(Paragraph("4.5 Feature Multicollinearity Analysis", h2_style))
    img5_path = os.path.join(SCRIPT_DIR, 'figures', 'fig5_correlation_heatmap.png')
    if os.path.exists(img5_path):
        story.append(Image(img5_path, width=5.0*inch, height=4.1*inch))
        story.append(Paragraph("Figure 5: Correlation Heatmap of Mean Morphological Features", caption_style))
    story.append(Paragraph(
        "<b>Inference:</b> Extreme multicollinearity (r > 0.90) is evident among size metrics (radius, perimeter, area). PCA transforms these "
        "redundant features into perfectly orthogonal, uncorrelated principal components, preventing numerical instability in linear estimators.",
        normal_style
    ))
    story.append(Spacer(1, 14))

    # 5. PCA Summary Table (Table 1)
    story.append(Paragraph("5 PCA Variance Summary (Table 1)", h1_style))
    pca_table_data = [
        [Paragraph("<b>Setting</b>", table_header), Paragraph("<b>Chosen Components / Variance Target</b>", table_header), Paragraph("<b>Explained Variance (%)</b>", table_header), Paragraph("<b>Justification</b>", table_header)],
        [Paragraph("With-PCA", table_text), Paragraph("10 Components / 95% Variance Target", table_text), Paragraph("95.27%", table_text), Paragraph("Preserves >95% total variance while eliminating 66.7% feature dimensionality, removing severe multicollinearity among nuclear size descriptors and accelerating training throughput.", table_text)]
    ]
    t_pca_summary = Table(pca_table_data, colWidths=[80, 130, 90, 187])
    t_pca_summary.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_pca_summary)
    story.append(Spacer(1, 14))

    # 6. Hyperparameter Tuning Tables (Tables 2, 3, 4, etc.)
    story.append(Paragraph("6 Hyperparameter Tuning Results (All 10 Models)", h1_style))
    
    # Table 2: SVM
    story.append(Paragraph("Table 2: SVM — Hyperparameter Tuning Results", h2_style))
    svm_table = [
        [Paragraph("<b>Kernel</b>", table_header), Paragraph("<b>C Values Tried</b>", table_header), Paragraph("<b>Gamma Values Tried</b>", table_header), Paragraph("<b>Performance (No-PCA)</b>", table_header), Paragraph("<b>Performance (With-PCA)</b>", table_header)],
        [Paragraph("linear, rbf", table_text), Paragraph("0.1, 1, 10, 100", table_text), Paragraph("scale, auto, 0.01, 0.1", table_text), Paragraph("0.9758 (linear, C=0.1)", table_text), Paragraph("0.9802 (linear, C=0.1)", table_text)]
    ]
    t_svm = Table(svm_table, colWidths=[75, 95, 105, 106, 106])
    t_svm.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_svm)
    story.append(Spacer(1, 8))

    # Table 3: Naive Bayes
    story.append(Paragraph("Table 3: Na\"ive Bayes — Smoothing Choices", h2_style))
    nb_table = [
        [Paragraph("<b>Smoothing Parameter (var_smoothing)</b>", table_header), Paragraph("<b>Performance (No-PCA)</b>", table_header), Paragraph("<b>Performance (With-PCA)</b>", table_header)],
        [Paragraph("1e-9, 1e-8, 1e-7, 1e-6, 1e-5", table_text), Paragraph("0.9341 (var=1e-9)", table_text), Paragraph("0.9275 (var=1e-9)", table_text)]
    ]
    t_nb = Table(nb_table, colWidths=[200, 143, 144])
    t_nb.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_nb)
    story.append(Spacer(1, 8))

    # Table 4: KNN
    story.append(Paragraph("Table 4: KNN — Hyperparameter Tuning", h2_style))
    knn_table = [
        [Paragraph("<b>k Values</b>", table_header), Paragraph("<b>Weights</b>", table_header), Paragraph("<b>Distance Metrics</b>", table_header), Paragraph("<b>Performance (No-PCA)</b>", table_header), Paragraph("<b>Performance (With-PCA)</b>", table_header)],
        [Paragraph("3, 5, 7, 9, 11", table_text), Paragraph("uniform, distance", table_text), Paragraph("euclidean, manhattan", table_text), Paragraph("0.9714 (k=9, euclidean)", table_text), Paragraph("0.9670 (k=9, euclidean)", table_text)]
    ]
    t_knn = Table(knn_table, colWidths=[75, 95, 105, 106, 106])
    t_knn.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_knn)
    story.append(Spacer(1, 8))

    # Other Models Table
    story.append(Paragraph("Hyperparameter Tuning Summary for Remaining Models (Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Stacking)", h2_style))
    other_models_data = [
        [Paragraph("<b>Model</b>", table_header), Paragraph("<b>Hyperparameter Grid Searched</b>", table_header), Paragraph("<b>Tuned No-PCA Params</b>", table_header), Paragraph("<b>Tuned With-PCA Params</b>", table_header)],
        [Paragraph("Logistic Reg.", table_text), Paragraph("C: [0.01..100], solver: [lbfgs, liblinear]", table_text), Paragraph("C=0.1, solver='lbfgs' (0.9824)", table_text), Paragraph("C=0.1, solver='lbfgs' (0.9824)", table_text)],
        [Paragraph("Decision Tree", table_text), Paragraph("depth: [3..None], criterion: [gini, entropy]", table_text), Paragraph("depth=3, gini (0.9319)", table_text), Paragraph("depth=3, gini (0.9363)", table_text)],
        [Paragraph("Random Forest", table_text), Paragraph("n_est: [50..200], depth: [3..None]", table_text), Paragraph("n_est=50, depth=None (0.9626)", table_text), Paragraph("n_est=50, depth=None (0.9604)", table_text)],
        [Paragraph("AdaBoost", table_text), Paragraph("n_est: [50..200], lr: [0.01..1.0]", table_text), Paragraph("n_est=200, lr=0.1 (0.9802)", table_text), Paragraph("n_est=200, lr=0.1 (0.9758)", table_text)],
        [Paragraph("Gradient Boost", table_text), Paragraph("n_est: [50..200], lr: [0.01..0.2], depth: [3, 5]", table_text), Paragraph("n_est=100, lr=0.1 (0.9736)", table_text), Paragraph("n_est=50, lr=0.1 (0.9560)", table_text)],
        [Paragraph("XGBoost", table_text), Paragraph("n_est: [50..200], lr: [0.01..0.2], depth: [3, 5]", table_text), Paragraph("n_est=50, lr=0.2 (0.9758)", table_text), Paragraph("n_est=50, lr=0.1 (0.9648)", table_text)],
        [Paragraph("Stacking", table_text), Paragraph("Base: SVM, NB, DT; Meta-C: [0.1, 1, 10]", table_text), Paragraph("meta_C=0.1 (0.9692)", table_text), Paragraph("meta_C=0.1 (0.9714)", table_text)],
    ]
    t_other = Table(other_models_data, colWidths=[80, 160, 123, 124])
    t_other.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_other)
    story.append(Spacer(1, 14))

    # 7. Cross-Validation Results (Table 5)
    story.append(Paragraph("7 5-Fold Cross-Validation Results (No-PCA vs. With-PCA) (Table 5)", h1_style))
    cv_table_data = [
        [Paragraph("<b>Model</b>", table_header), Paragraph("<b>Fold 1</b>", table_header), Paragraph("<b>Fold 2</b>", table_header), Paragraph("<b>Fold 3</b>", table_header), Paragraph("<b>Fold 4</b>", table_header), Paragraph("<b>Fold 5</b>", table_header), Paragraph("<b>Avg (No-PCA)</b>", table_header), Paragraph("<b>Avg (With-PCA)</b>", table_header), Paragraph("<b>Diff</b>", table_header)]
    ]
    for row in data['summary_table']:
        m = row['Model']
        f1 = f"{row['No_PCA_Fold1']:.3f}"
        f2 = f"{row['No_PCA_Fold2']:.3f}"
        f3 = f"{row['No_PCA_Fold3']:.3f}"
        f4 = f"{row['No_PCA_Fold4']:.3f}"
        f5 = f"{row['No_PCA_Fold5']:.3f}"
        no_avg = f"{row['No_PCA_CV_Acc']:.4f}"
        pca_avg = f"{row['With_PCA_CV_Acc']:.4f}"
        diff = f"{row['CV_Acc_Improvement']:+.4f}"
        cv_table_data.append([
            Paragraph(m, table_text),
            Paragraph(f1, table_text),
            Paragraph(f2, table_text),
            Paragraph(f3, table_text),
            Paragraph(f4, table_text),
            Paragraph(f5, table_text),
            Paragraph(no_avg, table_text),
            Paragraph(pca_avg, table_text),
            Paragraph(diff, table_text)
        ])
    t_cv = Table(cv_table_data, colWidths=[75, 43, 43, 43, 43, 43, 68, 72, 57])
    t_cv.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_cv)
    story.append(Spacer(1, 14))

    # 8. Performance Visualizations
    story.append(Paragraph("8 Performance Visualizations", h1_style))

    # Bar chart
    img6_path = os.path.join(SCRIPT_DIR, 'figures', 'fig6_model_comparison_bar.png')
    if os.path.exists(img6_path):
        story.append(Image(img6_path, width=5.4*inch, height=2.6*inch))
        story.append(Paragraph("Figure 6: 5-Fold Cross-Validation Accuracy Comparison (No-PCA vs. With-PCA)", caption_style))
    story.append(Spacer(1, 10))

    # ROC Curves
    img7_path = os.path.join(SCRIPT_DIR, 'figures', 'fig7_roc_curves_all.png')
    if os.path.exists(img7_path):
        story.append(Image(img7_path, width=5.5*inch, height=2.5*inch))
        story.append(Paragraph("Figure 7: Receiver Operating Characteristic (ROC) Curves Across All 10 Classifiers", caption_style))
    story.append(Spacer(1, 10))

    # PR Curves
    img8_path = os.path.join(SCRIPT_DIR, 'figures', 'fig8_pr_curves_all.png')
    if os.path.exists(img8_path):
        story.append(Image(img8_path, width=5.5*inch, height=2.5*inch))
        story.append(Paragraph("Figure 8: Precision-Recall (PR) Curves Across All 10 Classifiers", caption_style))
    story.append(Spacer(1, 10))

    # Confusion Matrices
    img9_path = os.path.join(SCRIPT_DIR, 'figures', 'fig9_confusion_matrices.png')
    if os.path.exists(img9_path):
        story.append(Image(img9_path, width=5.5*inch, height=2.8*inch))
        story.append(Paragraph("Figure 9: Test Set Confusion Matrices for SVM, Logistic Regression, Stacking, and XGBoost", caption_style))
    story.append(Spacer(1, 10))

    # Fold Variance Boxplot
    img10_path = os.path.join(SCRIPT_DIR, 'figures', 'fig10_fold_variance_boxplots.png')
    if os.path.exists(img10_path):
        story.append(Image(img10_path, width=5.4*inch, height=2.4*inch))
        story.append(Paragraph("Figure 10: Score Dispersion Across Folds Indicating Variance and Stability", caption_style))
    story.append(Spacer(1, 10))

    # Computational Cost
    img11_path = os.path.join(SCRIPT_DIR, 'figures', 'fig11_time_computational_cost.png')
    if os.path.exists(img11_path):
        story.append(Image(img11_path, width=5.2*inch, height=2.6*inch))
        story.append(Paragraph("Figure 11: Computational Training Time Comparison (Full vs. Reduced Dimensionality)", caption_style))
    story.append(Spacer(1, 14))

    # 9. Observation Questions
    story.append(Paragraph("9 Answers to Observation Questions", h1_style))
    obs_q = [
        "<b>• Which models improved most with PCA? Which did not? Why?</b><br/>"
        "<b>Improved:</b> SVM (CV accuracy increased from 0.9758 to <b>0.9802</b>), Decision Tree (+0.44%), and Stacking (+0.22%). Logistic Regression matched its optimal 0.9824 score.<br/>"
        "<b>Did not improve:</b> Gradient Boosting (-1.76%), XGBoost (-1.10%), and Gaussian Naive Bayes (-0.66%).<br/>"
        "<b>Reason:</b> Linear classifiers (SVM, Logistic Regression) benefit directly from orthogonalization which eliminates collinearity and conditions the optimization surface. Conversely, decision trees split on axis-aligned original features; dense linear combinations dilute the split purity gain.",

        "<b>• Did PCA reduce variance across folds (more stable results)?</b><br/>"
        "<b>Yes.</b> For SVM, fold-to-fold cross-validation variance was reduced by <b>58.6%</b> (from 1.16e-4 to 0.48e-4). For KNN, variance decreased by <b>25.0%</b>. PCA acts as an effective low-pass filter by discarding noisy, high-frequency dimensions.",

        "<b>• For high-dimensional data, was PCA beneficial in reducing overfitting?</b><br/>"
        "<b>Yes.</b> On unconstrained Decision Trees, training on all 30 features caused variance overfitting. Under PCA with 10 components, cross-validation accuracy improved by +0.44% and test F1 remained high (0.9444), confirming that PCA constrains the hypothesis capacity and curtails overfitting.",

        "<b>• How did linear models (Logistic Regression, SVM) behave compared to ensemble models with PCA?</b><br/>"
        "Linear models exhibited superior responsiveness to PCA. Logistic Regression achieved 0.9824 CV accuracy and 0.9970 ROC-AUC. SVM reached 0.9802 CV accuracy. Tree ensembles (Gradient Boosting, XGBoost) experienced minor drops because linear projections mask individual feature thresholds.",

        "<b>• Did stacking show robustness to dimensionality reduction compared to single models?</b><br/>"
        "<b>Yes.</b> The Stacking Classifier achieved the highest test accuracy (0.9825 without PCA, 0.9737 with PCA) and top ROC-AUC (0.9974). Its CV score improved under PCA (0.9692 to <b>0.9714</b>). Combining diverse base estimators (SVM, Naive Bayes, Decision Tree) via Logistic Regression provides strong regularization against dimensionality shifts."
    ]
    for q in obs_q:
        story.append(Paragraph(q, normal_style))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 10))

    # 10. Discussion & Report Checklist
    story.append(Paragraph("10 Discussion and Report Checklist", h1_style))
    story.append(Paragraph(
        "• <b>Dataset Details and Preprocessing:</b> Standardized 569 WDBC biopsy records using StandardScaler with 80/20 stratified split.<br/>"
        "• <b>PCA Design Choice:</b> 95% variance target requiring exactly 10 principal components (95.27% variance explained), supported by Scree and Cumulative variance curves.<br/>"
        "• <b>Hyperparameter Grids:</b> Fully tuned using GridSearchCV across 10 classifiers under No-PCA and With-PCA settings.<br/>"
        "• <b>Completed Fold-Wise Tables:</b> Fold 1 to Fold 5 scores, mean accuracies, and variances reported in Table 5.<br/>"
        "• <b>ROC/PR Curves and Confusion Matrices:</b> Plotted and evaluated for all models.<br/>"
        "• <b>Conclusion on When PCA Helps:</b> PCA is highly effective for multicollinear data, distance/margin-based classifiers (SVM, KNN, Logistic Regression), and compute-constrained pipelines. For tree ensembles requiring axis-aligned feature interpretability, retaining original features is preferable.",
        normal_style
    ))
    story.append(Spacer(1, 14))

    # 11. Experimental Setup
    story.append(Paragraph("11 Experimental Setup", h1_style))
    setup_data = [
        [Paragraph("Python Version", normal_style), Paragraph("3.11.x", normal_style)],
        [Paragraph("Scikit-Learn Version", normal_style), Paragraph("1.6.1", normal_style)],
        [Paragraph("XGBoost Version", normal_style), Paragraph("3.2.0", normal_style)],
        [Paragraph("IDE", normal_style), Paragraph("Google Colab / Local Jupyter Environment", normal_style)],
        [Paragraph("Operating System", normal_style), Paragraph("Windows 11", normal_style)],
        [Paragraph("Hardware", normal_style), Paragraph("x86_64 Multicore Processor", normal_style)],
    ]
    t_setup = Table(setup_data, colWidths=[160, 327])
    t_setup.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_setup)
    story.append(Spacer(1, 14))

    # 12. References
    story.append(Paragraph("12 References", h1_style))
    refs = [
        "[1] I. T. Jolliffe and J. Cadima, 'Principal component analysis: a review and recent developments,' <i>Philosophical Transactions of the Royal Society A</i>, vol. 374, no. 2065, 2016.",
        "[2] K. P. Murphy, <i>Machine Learning: A Probabilistic Perspective</i>, MIT Press, 2012.",
        "[3] F. Pedregosa et al., 'Scikit-learn: Machine learning in Python,' <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825–2830, 2011.",
        "[4] T. Chen and C. Guestrin, 'XGBoost: A scalable tree boosting system,' in <i>Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining</i>, 2016, pp. 785–794.",
        "[5] D. H. Wolpert, 'Stacked generalization,' <i>Neural Networks</i>, vol. 5, no. 2, pp. 241–259, 1992.",
        "[6] UCI Machine Learning Repository, 'Breast Cancer Wisconsin (Diagnostic) Data Set,' 1995."
    ]
    for ref in refs:
        story.append(Paragraph(ref, normal_style))
        story.append(Spacer(1, 3))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {PDF_PATH}")

if __name__ == '__main__':
    build_pdf()

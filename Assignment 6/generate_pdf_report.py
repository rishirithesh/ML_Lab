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
PDF_PATH = os.path.join(SCRIPT_DIR, 'Exp6.pdf')
PDF_COPY_PATH = os.path.join(SCRIPT_DIR, 'Experiment_6_Report.pdf')
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
        [Paragraph("<b>Experiment No.</b>", normal_style), Paragraph("6", normal_style)],
        [Paragraph("<b>Experiment Title</b>", normal_style), Paragraph("Bagging, Boosting, and Stacked Ensemble Models", normal_style)],
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
        "To understand the theoretical foundations and operational mechanics of primary ensemble learning paradigms: "
        "<b>Bagging</b> (Bootstrap Aggregation), <b>Boosting</b> (Adaptive Boosting and Gradient Tree Boosting), and <b>Stacking</b> (Stacked Generalization).<br/>"
        "• Implement a Bagging Classifier with Decision Tree base estimators and analyze the influence of bootstrap parameters.<br/>"
        "• Implement and evaluate sequential Boosting Classifiers (AdaBoost and Gradient Boosting) across tuned learning rates and depths.<br/>"
        "• Construct a heterogeneous Stacked Ensemble Model integrating Support Vector Machines (SVM), Gaussian Naive Bayes, and Decision Trees via Logistic Regression meta-learner.<br/>"
        "• Benchmark ensemble architectures against individual base classifiers using 5-fold Stratified Cross-Validation and bias-variance decomposition.",
        normal_style
    ))
    story.append(Spacer(1, 10))

    # 2. Problem Statement
    story.append(Paragraph("2 Problem Statement", h1_style))
    story.append(Paragraph(
        "To classify breast tumors as either <i>Malignant</i> or <i>Benign</i> based on cytological attributes extracted from Fine Needle Aspirate (FNA) digitized microscopic biopsy images, comparing how Bagging, Boosting, and Stacking improve predictive reliability, generalization capacity, and stability compared to single baseline learners.<br/><br/>"
        "The public <b>Wisconsin Diagnostic Breast Cancer (WDBC)</b> dataset comprising 569 clinical biopsy records characterized by 30 continuous morphometric features is used for this experiment.",
        normal_style
    ))
    story.append(Spacer(1, 10))

    # 3. Dataset Description
    story.append(Paragraph("3 Dataset Description", h1_style))
    ds_data = [
        [Paragraph("<b>Dataset Name</b>", normal_style), Paragraph("Wisconsin Diagnostic Breast Cancer (WDBC)", normal_style)],
        [Paragraph("<b>Dataset Source</b>", normal_style), Paragraph("UCI Machine Learning Repository / Scikit-Learn", normal_style)],
        [Paragraph("<b>Number of Samples</b>", normal_style), Paragraph("569 patient biopsy cases", normal_style)],
        [Paragraph("<b>Number of Features</b>", normal_style), Paragraph("30 continuous cytological descriptors", normal_style)],
        [Paragraph("<b>Target Classes</b>", normal_style), Paragraph("2 (Malignant: 212 [37.3%], Benign: 357 [62.7%])", normal_style)],
        [Paragraph("<b>Missing Values</b>", normal_style), Paragraph("0 missing records", normal_style)],
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

    # 4. Exploratory Data Analysis
    story.append(Paragraph("4 Exploratory Data Analysis", h1_style))
    img1_path = os.path.join(SCRIPT_DIR, 'figures', 'fig1_class_distribution.png')
    if os.path.exists(img1_path):
        story.append(Image(img1_path, width=4.5*inch, height=3.3*inch))
        story.append(Paragraph("Figure 1: Diagnostic Class Distribution in Breast Cancer Dataset", caption_style))
    story.append(Paragraph(
        "<b>Inference:</b> Target distribution comprises 357 benign and 212 malignant records. Stratified sampling preserves this 62.7% to 37.3% ratio across cross-validation folds.",
        normal_style
    ))
    story.append(Spacer(1, 10))

    img2_path = os.path.join(SCRIPT_DIR, 'figures', 'fig2_correlation_matrix.png')
    if os.path.exists(img2_path):
        story.append(Image(img2_path, width=5.0*inch, height=4.1*inch))
        story.append(Paragraph("Figure 2: Correlation Matrix of Nuclear Morphometric Features", caption_style))
    story.append(Paragraph(
        "<b>Inference:</b> Severe multicollinearity exists among nuclear size descriptors (r > 0.90 between radius, perimeter, and area). Tree ensembles robustly navigate collinearity via random feature subspaces.",
        normal_style
    ))
    story.append(Spacer(1, 14))

    # 5. Hyperparameter Tuning Tables
    story.append(Paragraph("5 Hyperparameter Tuning Results", h1_style))
    
    # Bagging Table
    story.append(Paragraph("Table 1: Bagging Classifier Hyperparameter Tuning", h2_style))
    bag_data = [
        [Paragraph("<b>Base Estimator</b>", table_header), Paragraph("<b>Estimators (M)</b>", table_header), Paragraph("<b>Max Samples</b>", table_header), Paragraph("<b>Max Features</b>", table_header), Paragraph("<b>Validation F1</b>", table_header)],
        [Paragraph("Decision Tree (max_depth=None)", table_text), Paragraph("10", table_text), Paragraph("0.8", table_text), Paragraph("0.8", table_text), Paragraph("0.9571", table_text)],
        [Paragraph("Decision Tree (max_depth=None)", table_text), Paragraph("50", table_text), Paragraph("0.8", table_text), Paragraph("0.8", table_text), Paragraph("0.9682", table_text)],
        [Paragraph("Decision Tree (max_depth=None)", table_text), Paragraph("100", table_text), Paragraph("1.0", table_text), Paragraph("1.0", table_text), Paragraph("0.9726", table_text)],
        [Paragraph("<b>Decision Tree (max_depth=None)</b>", table_text), Paragraph("<b>200</b>", table_text), Paragraph("<b>0.8</b>", table_text), Paragraph("<b>1.0</b>", table_text), Paragraph("<b>0.9789</b>", table_text)],
    ]
    t_bag = Table(bag_data, colWidths=[150, 80, 80, 85, 92])
    t_bag.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_bag)
    story.append(Spacer(1, 8))

    # Boosting Table
    story.append(Paragraph("Table 2: Boosting Classifiers Hyperparameter Tuning", h2_style))
    boost_data = [
        [Paragraph("<b>Model</b>", table_header), Paragraph("<b>Estimators (M)</b>", table_header), Paragraph("<b>Learning Rate (eta)</b>", table_header), Paragraph("<b>Max Depth</b>", table_header), Paragraph("<b>Validation F1</b>", table_header)],
        [Paragraph("AdaBoost", table_text), Paragraph("50", table_text), Paragraph("0.1", table_text), Paragraph("1 (Stump)", table_text), Paragraph("0.9655", table_text)],
        [Paragraph("AdaBoost", table_text), Paragraph("100", table_text), Paragraph("0.5", table_text), Paragraph("1 (Stump)", table_text), Paragraph("0.9722", table_text)],
        [Paragraph("<b>AdaBoost</b>", table_text), Paragraph("<b>200</b>", table_text), Paragraph("<b>1.0</b>", table_text), Paragraph("<b>1 (Stump)</b>", table_text), Paragraph("<b>0.9790</b>", table_text)],
        [Paragraph("Gradient Boosting", table_text), Paragraph("50", table_text), Paragraph("0.05", table_text), Paragraph("3", table_text), Paragraph("0.9650", table_text)],
        [Paragraph("Gradient Boosting", table_text), Paragraph("100", table_text), Paragraph("0.1", table_text), Paragraph("3", table_text), Paragraph("0.9724", table_text)],
        [Paragraph("<b>Gradient Boosting</b>", table_text), Paragraph("<b>200</b>", table_text), Paragraph("<b>0.1</b>", table_text), Paragraph("<b>3</b>", table_text), Paragraph("<b>0.9861</b>", table_text)],
    ]
    t_boost = Table(boost_data, colWidths=[110, 90, 105, 90, 92])
    t_boost.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_boost)
    story.append(Spacer(1, 8))

    # Stacking Table
    story.append(Paragraph("Table 3: Stacking Classifier Architecture Configuration", h2_style))
    stack_data = [
        [Paragraph("<b>Layer</b>", table_header), Paragraph("<b>Constituent Estimators</b>", table_header), Paragraph("<b>Tuned Hyperparameters</b>", table_header)],
        [Paragraph("Level-0 Base Learners", table_text), Paragraph("SVM (SVC(probability=True))", table_text), Paragraph("C=1.0, kernel='rbf', gamma='scale'", table_text)],
        [Paragraph("Level-0 Base Learners", table_text), Paragraph("Gaussian Naive Bayes", table_text), Paragraph("var_smoothing=1e-9", table_text)],
        [Paragraph("Level-0 Base Learners", table_text), Paragraph("Decision Tree", table_text), Paragraph("max_depth=5, criterion='entropy'", table_text)],
        [Paragraph("Level-1 Meta-Learner", table_text), Paragraph("Logistic Regression", table_text), Paragraph("C=1.0, penalty='l2', solver='lbfgs'", table_text)],
    ]
    t_stack = Table(stack_data, colWidths=[120, 180, 187])
    t_stack.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_stack)
    story.append(Spacer(1, 14))

    # 6. Model Evaluation Benchmark Table
    story.append(Paragraph("6 Comprehensive Model Benchmark (Table 4)", h1_style))
    bm_data = [
        [Paragraph("<b>Model Architecture</b>", table_header), Paragraph("<b>Accuracy</b>", table_header), Paragraph("<b>Precision</b>", table_header), Paragraph("<b>Recall</b>", table_header), Paragraph("<b>F1-Score</b>", table_header), Paragraph("<b>ROC-AUC</b>", table_header)],
        [Paragraph("Baseline Decision Tree", table_text), Paragraph("0.9386", table_text), Paragraph("0.9452", table_text), Paragraph("0.9583", table_text), Paragraph("0.9517", table_text), Paragraph("0.9317", table_text)],
        [Paragraph("Support Vector Machine (RBF)", table_text), Paragraph("0.9737", table_text), Paragraph("0.9726", table_text), Paragraph("0.9861", table_text), Paragraph("0.9793", table_text), Paragraph("0.9957", table_text)],
        [Paragraph("Gaussian Naive Bayes", table_text), Paragraph("0.9386", table_text), Paragraph("0.9333", table_text), Paragraph("0.9722", table_text), Paragraph("0.9524", table_text), Paragraph("0.9888", table_text)],
        [Paragraph("<b>Bagging Classifier</b>", table_text), Paragraph("<b>0.9649</b>", table_text), Paragraph("<b>0.9722</b>", table_text), Paragraph("<b>0.9722</b>", table_text), Paragraph("<b>0.9722</b>", table_text), Paragraph("<b>0.9934</b>", table_text)],
        [Paragraph("<b>AdaBoost Classifier</b>", table_text), Paragraph("<b>0.9737</b>", table_text), Paragraph("<b>0.9726</b>", table_text), Paragraph("<b>0.9861</b>", table_text), Paragraph("<b>0.9793</b>", table_text), Paragraph("<b>0.9924</b>", table_text)],
        [Paragraph("<b>Gradient Tree Boosting</b>", table_text), Paragraph("<b>0.9649</b>", table_text), Paragraph("<b>0.9595</b>", table_text), Paragraph("<b>0.9861</b>", table_text), Paragraph("<b>0.9726</b>", table_text), Paragraph("<b>0.9947</b>", table_text)],
        [Paragraph("<b>Stacked Ensemble Model</b>", table_text), Paragraph("<b>0.9825</b>", table_text), Paragraph("<b>0.9861</b>", table_text), Paragraph("<b>0.9861</b>", table_text), Paragraph("<b>0.9861</b>", table_text), Paragraph("<b>0.9974</b>", table_text)],
    ]
    t_bm = Table(bm_data, colWidths=[150, 65, 65, 65, 65, 77])
    t_bm.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_bm)
    story.append(Spacer(1, 14))

    # 7. Performance Figures
    story.append(Paragraph("7 Performance Visualizations", h1_style))
    img3_path = os.path.join(SCRIPT_DIR, 'figures', 'fig3_model_roc_curves.png')
    if os.path.exists(img3_path):
        story.append(Image(img3_path, width=5.4*inch, height=2.8*inch))
        story.append(Paragraph("Figure 3: Multi-Model ROC Curve Comparison for Single vs. Ensemble Learners", caption_style))
    story.append(Spacer(1, 10))

    img4_path = os.path.join(SCRIPT_DIR, 'figures', 'fig4_confusion_matrices.png')
    if os.path.exists(img4_path):
        story.append(Image(img4_path, width=5.5*inch, height=2.8*inch))
        story.append(Paragraph("Figure 4: Confusion Matrices for Decision Tree, Bagging, Boosting, and Stacking Classifiers", caption_style))
    story.append(Spacer(1, 14))

    # 8. Discussion
    story.append(Paragraph("8 Discussion", h1_style))
    story.append(Paragraph(
        "• <b>Observation:</b> Ensemble architectures consistently outperform single baseline decision trees. The stacked classifier achieved the highest accuracy (0.9825) and ROC-AUC (0.9974).<br/>"
        "• <b>Strength:</b> Bagging effectively controls estimator variance, while Boosting sequentially minimizes residual bias. Stacking leverages diverse hypotheses from SVM, Naive Bayes, and Decision Trees.<br/>"
        "• <b>Limitation:</b> Boosting is more sensitive to outliers and label noise than Bagging. Stacking requires multi-fold cross-validation generation of out-of-fold meta-features.<br/>"
        "• <b>Improvement:</b> Computational performance can be enhanced with early stopping in gradient boosting and automated threshold tuning for clinical cost-sensitive classification.",
        normal_style
    ))
    story.append(Spacer(1, 14))

    # 9. Experimental Setup
    story.append(Paragraph("9 Experimental Setup", h1_style))
    setup_data = [
        [Paragraph("Python Version", normal_style), Paragraph("3.11.x", normal_style)],
        [Paragraph("Scikit-Learn Version", normal_style), Paragraph("1.6.1", normal_style)],
        [Paragraph("IDE", normal_style), Paragraph("Google Colab / Local Jupyter Environment", normal_style)],
        [Paragraph("Operating System", normal_style), Paragraph("Windows 11", normal_style)],
        [Paragraph("Hardware Platform", normal_style), Paragraph("x86_64 Multicore Processor", normal_style)],
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

    # 10. References
    story.append(Paragraph("10 References", h1_style))
    refs = [
        "[1] L. Breiman, 'Bagging predictors,' <i>Machine Learning</i>, vol. 24, no. 2, pp. 123–140, 1996.",
        "[2] Y. Freund and R. E. Schapire, 'A decision-theoretic generalization of on-line learning and an application to boosting,' <i>Journal of Computer and System Sciences</i>, vol. 55, no. 1, pp. 119–139, 1997.",
        "[3] D. H. Wolpert, 'Stacked generalization,' <i>Neural Networks</i>, vol. 5, no. 2, pp. 241–259, 1992.",
        "[4] F. Pedregosa et al., 'Scikit-learn: Machine learning in Python,' <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825–2830, 2011.",
        "[5] UCI Machine Learning Repository, 'Breast Cancer Wisconsin (Diagnostic) Data Set,' 1995."
    ]
    for ref in refs:
        story.append(Paragraph(ref, normal_style))
        story.append(Spacer(1, 3))

    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Make copy
    import shutil
    shutil.copyfile(PDF_PATH, PDF_COPY_PATH)
    print(f"PDF successfully generated at: {PDF_PATH} and {PDF_COPY_PATH}")

if __name__ == '__main__':
    build_pdf()

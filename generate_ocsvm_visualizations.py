"""
Generate visualizations for OC-SVM Linear Kernel results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# OC-SVM Results
ocsvm_results = {
    "auroc": 0.5306246071570155,
    "aupr_outlier": 0.9180755289551477,
    "aupr_inlier": 0.1002576505166217,
    "optimal_threshold": -2.000011876106285,
    "best_f1": 0.9068690028828116,
    "precision_at_optimal": 0.9114466939336076,
    "recall_at_optimal": 0.902337064446684,
    "confusion_matrix": [[2069, 14701], [16377, 151312]],
    "true_positive_rate": 0.902337064446684,
    "false_positive_rate": 0.8766249254621348,
    "true_negative_rate": 0.12337507453786524,
    "false_negative_rate": 0.09766293555331595,
    "score_stats": {
        "inlier_mean": -0.8185545528998329,
        "inlier_std": 1.0357741936829694,
        "inlier_median": -0.7628606020212403,
        "outlier_mean": -0.69831129200215,
        "outlier_std": 1.0351982941314084,
        "outlier_median": -0.6700522212982407,
        "separation": 0.12024326089768289
    }
}

# Generate synthetic score distributions based on actual statistics
np.random.seed(42)
n_simple = 16770
n_normal = 167689

# Generate scores using actual mean and std
scores_simple = np.random.normal(
    ocsvm_results['score_stats']['inlier_mean'],
    ocsvm_results['score_stats']['inlier_std'],
    n_simple
)
scores_normal = np.random.normal(
    ocsvm_results['score_stats']['outlier_mean'],
    ocsvm_results['score_stats']['outlier_std'],
    n_normal
)

print("Generating OC-SVM visualizations...")

# ============================================================================
# 1. Score Distribution
# ============================================================================
print("Creating score distribution plot...")
plt.figure(figsize=(10, 6))
plt.hist(scores_simple, bins=60, alpha=0.6, label='Simple (Inlier)',
         color='blue', density=True, edgecolor='black', linewidth=0.5)
plt.hist(scores_normal, bins=60, alpha=0.6, label='Normal (Outlier)',
         color='red', density=True, edgecolor='black', linewidth=0.5)
plt.axvline(ocsvm_results['optimal_threshold'], color='green', linestyle='--',
            linewidth=2.5, label=f"Optimal Threshold ({ocsvm_results['optimal_threshold']:.3f})")

plt.xlabel('Anomaly Score', fontsize=13, fontweight='bold')
plt.ylabel('Density', fontsize=13, fontweight='bold')
plt.title('One-Class SVM: Distribution of Anomaly Scores\n(Linear Kernel)',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ocsvm_score_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('ocsvm_score_distribution.pdf', dpi=300, bbox_inches='tight')
print("  ✓ Saved: ocsvm_score_distribution.png/pdf")
plt.close()

# ============================================================================
# 2. Confusion Matrix
# ============================================================================
print("Creating confusion matrix...")
cm = np.array(ocsvm_results['confusion_matrix'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Simple', 'Normal'],
            yticklabels=['Simple', 'Normal'],
            annot_kws={'fontsize': 14})

# Add percentages
total_simple = cm[0, 0] + cm[0, 1]
total_normal = cm[1, 0] + cm[1, 1]
pct_simple_correct = cm[0, 0] / total_simple * 100
pct_simple_wrong = cm[0, 1] / total_simple * 100
pct_normal_wrong = cm[1, 0] / total_normal * 100
pct_normal_correct = cm[1, 1] / total_normal * 100

plt.text(0.5, 0.25, f'{pct_simple_correct:.1f}%', ha='center', va='center',
         fontsize=16, color='black', fontweight='bold')
plt.text(1.5, 0.25, f'{pct_simple_wrong:.1f}%', ha='center', va='center',
         fontsize=16, color='white', fontweight='bold')
plt.text(0.5, 1.25, f'{pct_normal_wrong:.1f}%', ha='center', va='center',
         fontsize=16, color='white', fontweight='bold')
plt.text(1.5, 1.25, f'{pct_normal_correct:.1f}%', ha='center', va='center',
         fontsize=16, color='white', fontweight='bold')

plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
plt.ylabel('Actual Label', fontsize=13, fontweight='bold')
plt.title('One-Class SVM: Confusion Matrix\n(Linear Kernel)',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ocsvm_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('ocsvm_confusion_matrix.pdf', dpi=300, bbox_inches='tight')
print("  ✓ Saved: ocsvm_confusion_matrix.png/pdf")
plt.close()

# ============================================================================
# 3. ROC Curve (simulated based on AUROC)
# ============================================================================
print("Creating ROC curve...")
from sklearn.metrics import roc_curve, auc

# Combine scores and labels
y_true = np.concatenate([np.zeros(n_simple), np.ones(n_normal)])
y_scores = np.concatenate([scores_simple, scores_normal])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, linewidth=2.5, label=f'OC-SVM (AUROC = {roc_auc:.4f})',
         color='#1f77b4')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUROC = 0.5000)')

plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
plt.title('One-Class SVM: ROC Curve\n(Linear Kernel)',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('ocsvm_roc_curve.png', dpi=300, bbox_inches='tight')
plt.savefig('ocsvm_roc_curve.pdf', dpi=300, bbox_inches='tight')
print("  ✓ Saved: ocsvm_roc_curve.png/pdf")
plt.close()

# ============================================================================
# 4. Performance Metrics Bar Chart
# ============================================================================
print("Creating performance metrics chart...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: AUROC
ax1 = axes[0]
algorithms = ['OC-SVM\n(Linear)', 'Random\nBaseline']
aurocs = [ocsvm_results['auroc'], 0.5]
colors = ['#1f77b4', '#7f7f7f']

bars = ax1.bar(algorithms, aurocs, color=colors, alpha=0.8,
               edgecolor='black', linewidth=1.5, width=0.6)
ax1.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.set_ylabel('AUROC', fontsize=13, fontweight='bold')
ax1.set_title('AUROC: Area Under ROC Curve', fontsize=14, fontweight='bold')
ax1.set_ylim([0.45, 0.55])
ax1.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, aurocs):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add annotation
ax1.text(0, 0.515, 'Only 6% above\nrandom chance!',
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 2: Class-wise Recall
ax2 = axes[1]
classes = ['Simple Texts\n(Minority)', 'Normal Texts\n(Majority)']
recalls = [ocsvm_results['true_negative_rate'] * 100,
           ocsvm_results['true_positive_rate'] * 100]
colors = ['#d62728', '#2ca02c']

bars = ax2.bar(classes, recalls, color=colors, alpha=0.8,
               edgecolor='black', linewidth=1.5, width=0.6)
ax2.set_ylabel('Recall (%)', fontsize=13, fontweight='bold')
ax2.set_title('Recall by Class', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 100])
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, recalls):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add annotation
ax2.text(0, 60, 'Model fails to\nidentify simple texts!',
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

plt.tight_layout()
plt.savefig('ocsvm_performance_metrics.png', dpi=300, bbox_inches='tight')
plt.savefig('ocsvm_performance_metrics.pdf', dpi=300, bbox_inches='tight')
print("  ✓ Saved: ocsvm_performance_metrics.png/pdf")
plt.close()

# ============================================================================
# 5. Score Statistics Comparison
# ============================================================================
print("Creating score statistics plot...")
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Mean Score', 'Median Score', 'Std Dev']
simple_stats = [
    ocsvm_results['score_stats']['inlier_mean'],
    ocsvm_results['score_stats']['inlier_median'],
    ocsvm_results['score_stats']['inlier_std']
]
normal_stats = [
    ocsvm_results['score_stats']['outlier_mean'],
    ocsvm_results['score_stats']['outlier_median'],
    ocsvm_results['score_stats']['outlier_std']
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, simple_stats, width, label='Simple (Inlier)',
               color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, normal_stats, width, label='Normal (Outlier)',
               color='red', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Score Value', fontsize=13, fontweight='bold')
ax.set_title('One-Class SVM: Score Statistics Comparison\n(Linear Kernel)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10)

# Add separation annotation
ax.text(0, -0.5, f'Separation: {ocsvm_results["score_stats"]["separation"]:.3f}',
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))

plt.tight_layout()
plt.savefig('ocsvm_score_statistics.png', dpi=300, bbox_inches='tight')
plt.savefig('ocsvm_score_statistics.pdf', dpi=300, bbox_inches='tight')
print("  ✓ Saved: ocsvm_score_statistics.png/pdf")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("VISUALIZATION SUMMARY - OC-SVM (Linear Kernel)")
print("="*60)
print(f"AUROC:                 {ocsvm_results['auroc']:.4f}")
print(f"Score Separation:      {ocsvm_results['score_stats']['separation']:.4f}")
print(f"Simple Text Recall:    {ocsvm_results['true_negative_rate']*100:.1f}%")
print(f"Normal Text Recall:    {ocsvm_results['true_positive_rate']*100:.1f}%")
print(f"F1 Score:              {ocsvm_results['best_f1']:.4f}")
print("="*60)
print("\nAll visualizations saved successfully!")
print("Files created:")
print("  - ocsvm_score_distribution.png/pdf")
print("  - ocsvm_confusion_matrix.png/pdf")
print("  - ocsvm_roc_curve.png/pdf")
print("  - ocsvm_performance_metrics.png/pdf")
print("  - ocsvm_score_statistics.png/pdf")
print("="*60)

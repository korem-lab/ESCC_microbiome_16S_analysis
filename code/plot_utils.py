import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

import seaborn as sns
import numpy as np
import pandas as pd

from skbio.stats.distance import permanova,anosim
from skbio.stats.ordination import pcoa
from skbio import DistanceMatrix

from sklearn import metrics

from matplotlib.patches import Ellipse

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],  
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'font.weight': 'normal',
    'axes.titleweight': 'bold',
})

def sdi(counts):
    from math import log as ln
    def p(n, N):
        if n == 0:
            return 0
        else:
            return (float(n) / N) * ln(float(n) / N)   
    N = sum(counts)
    return -sum(p(n, N) for n in counts if n != 0)

def taxa_boxplot(feature_table,taxa,p,log,bottom_lim,saveas):
    cancer = [int("CESCC" in sample_id) for sample_id in feature_table.index]
    my_pal = {0: "#00AEEF", 1: "#ED1C24"}
    sns.set_theme(style="white", palette=None)
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(2.7,3.6))
    y_data = feature_table[taxa] + 1e-4
    if log: 
        ax.set_ylim(bottom_lim,2)
        ax.set_yscale("log")
    sns.boxplot(x=cancer, y=y_data,hue = cancer, fliersize=2,linewidth=1.4,palette=my_pal,legend=None)
    sns.swarmplot(x=cancer, y=y_data,color="black",s=4)
    ax.set_ylabel("Relative Abundance") 
    ax.set_xlabel("")
    ax.set_title(f"{taxa}", size=12, fontweight='bold')
    ax.tick_params(labelsize=12,bottom=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'ESCC'], size=12)
    
    p_text = f"$\\mathit{{p}} =$ {p:.4f}"
    
    ax.text(
            0.98, 0.98, p_text,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=11
        )
    fig.tight_layout()
    plt.savefig(saveas)
    plt.close()

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def pcoa_plot(distance_matrix_df,title,permutations,weighted,saveas):
    data = np.ascontiguousarray(distance_matrix_df.values)
    dm = DistanceMatrix(data, ids=distance_matrix_df.index)

    metadata = pd.DataFrame({
        'Cancer': [int('CESCC' in name) for name in distance_matrix_df.index]
    }, index=distance_matrix_df.index)

    result = permanova(dm, metadata, column='Cancer', permutations=permutations)
    p_value = result['p-value']

    pcoa_results = pcoa(dm)
    coords = pcoa_results.samples.iloc[:, :2].copy()
    coords.columns = ['PC1', 'PC2']
    coords['Cancer'] = metadata['Cancer']
    variance_explained = pcoa_results.proportion_explained
    my_pal = {0: "#00AEEF", 1: "#ED1C24"}
    fig, ax = plt.subplots(figsize=(4 ,4))

    for group in [0, 1]:
        pts = coords[coords["Cancer"] == group][["PC1", "PC2"]].values
        ellipse = plot_point_cov(pts, nstd=2, ax=ax)
        rgba = (0, 174/255, 239/255, 0.1) if group == 0 else (190/255, 30/255, 45/255, 0.1)
        ellipse.set_facecolor(rgba)
        ellipse.set_edgecolor('none')
    
    sns.scatterplot(x="PC1", y="PC2", hue="Cancer", data=coords,
                    palette=my_pal, s=25, edgecolor='white', linewidth=0.4, ax=ax)

    if weighted:
        ax.set_ylim(-0.45, 0.3)
        ax.set_xlim(-0.5, 0.55)
    else: 
        ax.set_xlim(-0.3, 0.4)
        ax.set_ylim(-0.3, 0.2)
    
    ax.set_xlabel(f"PC1 ({variance_explained[0]*100:.2f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({variance_explained[1]*100:.2f}%)", fontsize=12)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.legend([], [], frameon=False)

    if p_value < 0.0001:
        p_text = "PERMANOVA $\\mathit{p} < 0.0001$"
    else:
        p_text = f"PERMANOVA $\\mathit{{p}} = $ {p_value:.4f}"

    ax.text(
        0.98, 0.98,
        p_text,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=11
    )
    fig.tight_layout()
    plt.savefig(saveas)
    plt.close()

def plot_roc_curves(predictions,labels,colors,title,legendsize,saveas): 
    assert len(predictions) == len(labels) == len(colors), "Mismatched input lengths"
    sns.set_theme(style="white", palette=None)
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for model, label, color in zip(predictions, labels, colors):
        fpr, tpr, _ = metrics.roc_curve(model['y_test'], model['y_pred'])
        auc = metrics.roc_auc_score(model['y_test'], model['y_pred'])
        ax.plot(fpr, tpr, label=f"{label} (auROC={auc:.2f})", color=color,lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey',linewidth=1.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title}',size=12, fontweight='bold')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.tick_params(labelsize=12)
    font_prop = FontProperties(family='DejaVu Sans', size=legendsize, weight='normal')
    leg = ax.legend(
        fontsize=7,              
        prop=font_prop,
        loc='lower right',
        handlelength=1.5,
        labelspacing=0.5,
        borderaxespad=0.5,
        framealpha=1
    )
    leg.get_frame().set_edgecolor('none') 
    fig.tight_layout()
    plt.savefig(saveas)
    plt.close()

def plot_pr_curves(predictions, labels, colors, title, legendsize,saveas,class_balance):
    assert len(predictions) == len(labels) == len(colors), "Mismatched input lengths"
    assert class_balance in ['none', 'all', 'one'], "class_balance must be 'none', 'all', or 'one'"
    sns.set_theme(style="white", palette=None)
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    
    for model, label, color in zip(predictions, labels, colors): 
        precision, recall, _ = metrics.precision_recall_curve(model['y_test'], model['y_pred'])
        ap = metrics.average_precision_score(model['y_test'], model['y_pred'])
        if class_balance == 'all':
            pos_rate = sum(model['y_test']) / len(model['y_test'])
            ax.axhline(y=pos_rate, linestyle='--', color=color, linewidth=1.5)
            ax.plot(recall, precision, label=f"{label} (auPR={ap:.2f}, class balance={pos_rate:.2f})", color=color,lw=2)
        else: 
            ax.plot(recall, precision, label=f"{label} (auPR={ap:.2f})", color=color,lw=2)

    if class_balance == 'one':
        pos_rate = sum(predictions[0]['y_test']) / len(predictions[0]['y_test'])
        ax.axhline(y=pos_rate, linestyle='--', color='grey', linewidth=1.5,
                   label=f"Class balance (ratio={pos_rate:.2f})")
        
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{title}', size=12, fontweight='bold')
    ax.set_xlim(-0.01, 1.04)
    ax.set_ylim(-0.01, 1.04)
    ax.tick_params(labelsize=12)
    font_prop = FontProperties(family='DejaVu Sans', size=legendsize, weight='normal')
    leg = ax.legend(
        fontsize=7,              
        prop=font_prop,
        loc='lower right',
        handlelength=1.5,
        labelspacing=0.5,
        borderaxespad=0.5,
        framealpha=1
    )
    leg.get_frame().set_edgecolor('none') 
    fig.tight_layout()
    plt.savefig(saveas)
    plt.close()
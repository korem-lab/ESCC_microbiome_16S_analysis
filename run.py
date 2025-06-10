import pandas as pd
import numpy as np
import os 
import sys
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import fdrcorrection
from sklearn import metrics
import shap
import warnings
warnings.filterwarnings("ignore")

BASE_DIR= os.path.dirname(os.path.realpath(__file__))
CODE_DIR=os.path.join(BASE_DIR,'code')
DATA_DIR=os.path.join(BASE_DIR,'data')
RESULTS_DIR=os.path.join(BASE_DIR,'results')

sys.path.append(CODE_DIR)
import delong
import plot_utils

def load_data(): 
    feature_table = pd.read_csv(os.path.join(DATA_DIR, "feature_table_decontaminated.csv"), index_col=0).T
    rarefied_table = pd.read_csv(os.path.join(DATA_DIR, "feature_table_filtered_rarefied.csv"), index_col=0).T
    taxonomy = pd.read_csv(os.path.join(DATA_DIR, "taxonomy.csv"), index_col=0)
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"), index_col=0)

    feature_ra = feature_table.div(feature_table.sum(axis=1),axis=0).sort_index()
    num_samples = 5
    feature_table_filtered = feature_table.loc[:, (feature_table > 0).sum(axis = 0) >= num_samples]
    feature_ra_filtered = feature_table_filtered.div(feature_table_filtered.sum(axis=1),axis=0)
    feature_ra_filtered.sort_index(inplace=True)

    return {
        "rarefied_table": rarefied_table,
        "taxonomy": taxonomy,
        "metadata": metadata,
        "feature_ra": feature_ra_filtered,
    }

def table_S1(metadata):
    clinical = metadata.copy()
    clinical.replace({"Smoking":{2:1,3:0}},inplace=True)
    clinical = clinical.fillna('NA')
    clinical['Education Level'] = clinical['Education'].replace({
        1: "Primary school", 2: "Primary school", 3: "Primary school", 4: "Primary school", 
        5: "Primary school", 6: "Primary school", 7: "Primary school", 
        8: "Secondary school", 9: "Secondary school", 10: "Secondary school", 
        11: "Secondary school", 12: "Secondary school",13:'Higher Education',0:'None'
    })
    for col in clinical.columns:
        if col != 'Age':
            clinical[col] = clinical[col].astype('category')
    results=[]
    for col in clinical.columns:
        if col=='ESCC':
            continue
        if col == 'Age':
            df = clinical[clinical['Age'] != 'NA'].copy()
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            cancer = df[df['ESCC'] == 1]['Age'].dropna()
            ctrl = df[df['ESCC'] == 0]['Age'].dropna()
            stat, p = mannwhitneyu(cancer, ctrl, alternative='two-sided')
            test_type = 'Mann-Whitney U'
        else: 
            contingency_table = clinical.pivot_table(index=col, columns='ESCC', aggfunc='size', fill_value=0)#.drop(index='NA')
            stat, p, *_ = chi2_contingency(contingency_table)
            test_type = 'Chi-squared'
        results.append({'variable':col,'p_value':p,'test':test_type})
    pval_df = pd.DataFrame(results).sort_values('p_value')
    pval_df.to_csv(os.path.join(RESULTS_DIR,'tables','table_S1_stats.csv'),index=False)

def fig_1A(rarefied_table):
    shannon_index = rarefied_table.apply(plot_utils.sdi, axis=1).to_frame(name="shannon")
    shannon_index["Cancer"] = shannon_index.index.str.contains("CESCC").astype(int)

    x = shannon_index["Cancer"].astype(str)
    y = shannon_index["shannon"]
    stat,p_value = mannwhitneyu(shannon_index.loc[shannon_index["Cancer"] == 1,"shannon"],shannon_index.loc[shannon_index["Cancer"] == 0,"shannon"],alternative='two-sided')

    my_pal = {"0": "#00AEEF", "1": "#ED1C24"}
    sns.set_theme(style="white", palette=None)
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(3, 4.5))
    sns.boxplot(x=x,y=y,fliersize=2,linewidth=1.4, hue=x, palette=my_pal,ax=ax)
    sns.swarmplot(x=x,y=y,color="black",s=4,ax=ax)

    ax.set_ylabel("Shannon index")
    ax.set_ylim(1.75,6)
    ax.set_xlabel("")
    ax.set_xticklabels(['Control', 'ESCC'], size=12)
    ax.tick_params(labelsize=12,bottom=False,left=True)
    x1, x2 = 0, 1
    y_max = y.max() + 0.2 
    line_height = y_max + 0.1 
    ax.plot([x1, x1], [y_max, line_height], color='black', linewidth=1.2)
    ax.plot([x2, x2], [y_max, line_height], color='black', linewidth=1.2)
    ax.plot([x1, x2], [line_height, line_height], color='black', linewidth=1.2)

    p_text = f"$\\mathit{{p}} = $ {p_value:.4f}"
    ax.text((x1 + x2) * 0.5, line_height + 0.05, p_text, ha='center', va='bottom', fontsize=11)
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'figures','fig_1A.png'))
    plt.close()

def fig_1B_S1():
    unweighted_matrix = pd.read_csv(os.path.join(RESULTS_DIR,'diversity','unweighted-unifrac-distance-matrix.tsv'),sep='\t',index_col=0)
    weighted_matrix = pd.read_csv(os.path.join(RESULTS_DIR,'diversity','weighted-unifrac-distance-matrix.tsv'), sep='\t', index_col=0)
    plot_utils.pcoa_plot(unweighted_matrix,"",10000,False,os.path.join(RESULTS_DIR,'figures','fig_1B.png'))
    plot_utils.pcoa_plot(weighted_matrix,"",10000,True,os.path.join(RESULTS_DIR,'figures','fig_S1.png'))

def fig_2A(feature_ra,taxonomy):
    save_path = os.path.join(RESULTS_DIR,'figures','fig_2A')
    os.makedirs(save_path,exist_ok=True)

    corncob_genus_batchadj = pd.read_csv(os.path.join(RESULTS_DIR,'corncob','corncob_results_genus_batchadj.csv'),index_col=0)
    corncob_genus_allcovadj = pd.read_csv(os.path.join(RESULTS_DIR,'corncob','corncob_results_genus_allcovadj.csv'),index_col=0)

    batch_sig = corncob_genus_batchadj[corncob_genus_batchadj['p_fdr'] < 0.1].index 
    allcov_sig = corncob_genus_allcovadj.loc[batch_sig].dropna(axis = 0).drop("p_fdr",axis=1) 
    allcov_sig['p_fdr'] = fdrcorrection(allcov_sig["Pr(>|t|)"],alpha=0.05)[1]
    allcov_batch_adj = allcov_sig.sort_values(by='p_fdr')

    genus_results = corncob_genus_batchadj[["Pr(>|t|)","p_fdr"]].dropna(axis=0).rename(columns={"Pr(>|t|)": 'p','p_fdr':'FDR'})
    genus_results = genus_results.join(corncob_genus_allcovadj['Pr(>|t|)']).rename(columns={'Pr(>|t|)':'adj_p'}).dropna(axis=0)
    genus_results = genus_results.join(allcov_batch_adj['p_fdr']).rename(columns={'p_fdr':'adj_FDR'})
    genus_results.sort_values(by='adj_FDR',inplace=True)
    

    genus_ra = feature_ra.T.join(taxonomy["Genus"],how='inner').set_index("Genus")
    genus_ra = genus_ra.groupby(genus_ra.index).sum().T
    genus_ra = genus_ra.div(genus_ra.sum(axis=1),axis=0)

    genus_carriage = []
    for genus in genus_ra.columns: 
        ctrl_mask = genus_ra.index.str.contains('COOC')
        escc_mask = genus_ra.index.str.contains('CESCC')
        ctrl_carry = sum(genus_ra[genus][ctrl_mask] > 0) / sum(ctrl_mask) * 100
        escc_carry = sum(genus_ra[genus][escc_mask] > 0) / sum(escc_mask) * 100
        genus_carriage.append({"Genus": genus, "control_carriage": ctrl_carry, "escc_carriage": escc_carry})
    genus_carriage = pd.DataFrame(genus_carriage).set_index('Genus')
    genus_results = genus_results.join(genus_carriage)
    genus_results.to_csv(os.path.join(RESULTS_DIR,'tables','table_S2.csv'))

    for result in genus_results[genus_results["adj_FDR"] < 0.05].iterrows(): 
        genus = result[0]
        y_lim = 1e-2 if genus == 'Streptococcus' else 1e-4
        p = result[1]['adj_FDR']
        plot_utils.taxa_boxplot(genus_ra,genus,p,True,y_lim,os.path.join(save_path,f'{genus}.png'))
    
def fig_2B(feature_ra,taxonomy):
    save_path = os.path.join(RESULTS_DIR,'figures','fig_2B')
    os.makedirs(save_path,exist_ok=True)

    corncob_batchadj = pd.read_csv(os.path.join(RESULTS_DIR,'corncob','corncob_results_batchadj.csv'),index_col=0)
    corncob_allcovadj = pd.read_csv(os.path.join(RESULTS_DIR,'corncob','corncob_results_allcovadj.csv'),index_col=0)

    batch_sig = corncob_batchadj[corncob_batchadj['p_fdr'] < 0.1].index 
    allcov_sig = corncob_allcovadj.loc[batch_sig].dropna(axis = 0).drop("p_fdr",axis=1) 
    allcov_sig['p_fdr'] = fdrcorrection(allcov_sig["Pr(>|t|)"],alpha=0.05)[1]
    allcov_batch_adj = allcov_sig.sort_values(by='p_fdr')

    asv_results = corncob_batchadj[["Pr(>|t|)","p_fdr"]].dropna(axis=0).rename(columns={"Pr(>|t|)": 'p','p_fdr':'FDR'})
    asv_results = asv_results.join(corncob_allcovadj['Pr(>|t|)']).rename(columns={'Pr(>|t|)':'adj_p'}).dropna(axis=0)
    asv_results = asv_results.join(allcov_batch_adj['p_fdr']).rename(columns={'p_fdr':'adj_FDR'})
    asv_results.sort_values(by='adj_FDR',inplace=True)
    asv_results = asv_results.join(taxonomy[['Genus','Species']])

    asv_carriage = []
    for asv in feature_ra.columns: 
        ctrl_mask = feature_ra.index.str.contains('COOC')
        escc_mask = feature_ra.index.str.contains('CESCC')
        ctrl_carry = sum(feature_ra[asv][ctrl_mask] > 0) / sum(ctrl_mask) * 100
        escc_carry = sum(feature_ra[asv][escc_mask] > 0) / sum(escc_mask) * 100
        asv_carriage.append({"ASV": asv, "control_carriage": ctrl_carry, "escc_carriage": escc_carry})
    asv_carriage = pd.DataFrame(asv_carriage).set_index('ASV')
    asv_results = asv_results.join(asv_carriage)
    asv_results.to_csv(os.path.join(RESULTS_DIR,'tables','table_S3.csv'))

    for result in asv_results[asv_results['adj_FDR'] < 0.05].iterrows():
        taxon = result[0]
        p = result[1]['adj_FDR']
        plot_utils.taxa_boxplot(feature_ra,taxon,p,True,1e-4,os.path.join(save_path,f'{taxon}.png'))

def fig_2C(feature_ra,taxonomy):
    save_path = os.path.join(RESULTS_DIR,'figures','fig_2C')
    os.makedirs(save_path,exist_ok=True)

    corncob_fuso_batchadj = pd.read_csv(os.path.join(RESULTS_DIR,'corncob','corncob_results_fusobacterium_batchadj.csv'),index_col=0)
    corncob_fuso_allcovadj = pd.read_csv(os.path.join(RESULTS_DIR,'corncob','corncob_results_fusobacterium_allcovadj.csv'),index_col=0)

    batch_sig = corncob_fuso_batchadj[corncob_fuso_batchadj['p_fdr'] < 0.1].index 
    allcov_sig = corncob_fuso_allcovadj.loc[batch_sig].dropna(axis = 0).drop("p_fdr",axis=1) 
    allcov_sig['p_fdr'] = fdrcorrection(allcov_sig["Pr(>|t|)"],alpha=0.05)[1]
    allcov_batch_adj = allcov_sig.sort_values(by='p_fdr')

    fuso_results = corncob_fuso_batchadj[["Pr(>|t|)","p_fdr"]].dropna(axis=0).rename(columns={"Pr(>|t|)": 'p','p_fdr':'FDR'})
    fuso_results = fuso_results.join(corncob_fuso_allcovadj['Pr(>|t|)']).rename(columns={'Pr(>|t|)':'adj_p'}).dropna(axis=0)
    fuso_results = fuso_results.join(allcov_batch_adj['p_fdr']).rename(columns={'p_fdr':'adj_FDR'})
    fuso_results.sort_values(by='adj_FDR',inplace=True)
    fuso_results = fuso_results.join(taxonomy[['Genus','Species']])

    fuso_ASVs = taxonomy[taxonomy["Genus"] == "Fusobacterium"].index.intersection(feature_ra.columns)
    fuso_table = feature_ra[fuso_ASVs]
    fuso_ra = fuso_table.div(fuso_table.sum(axis=1),axis=0)

    fuso_carriage = []
    for asv in fuso_ra.columns:  
        ctrl_mask = fuso_ra.index.str.contains('COOC')
        escc_mask = fuso_ra.index.str.contains('CESCC')
        ctrl_carry = sum(fuso_ra[asv][ctrl_mask] > 0) / sum(ctrl_mask) * 100
        escc_carry = sum(fuso_ra[asv][escc_mask] > 0) / sum(escc_mask) * 100
        fuso_carriage.append({"ASV": asv, "control_carriage": ctrl_carry, "escc_carriage": escc_carry})
    fuso_carriage = pd.DataFrame(fuso_carriage).set_index('ASV')
    fuso_results = fuso_results.join(fuso_carriage)
    fuso_results.to_csv(os.path.join(RESULTS_DIR,'tables','table_S4.csv'))

    for result in fuso_results[fuso_results['adj_FDR'] < 0.05].iterrows():
        taxon = result[0]
        p = result[1]['adj_FDR']
        plot_utils.taxa_boxplot(fuso_ra,taxon,p,True,1e-3,os.path.join(save_path,f'{taxon}.png'))

def fig_3A_S3A(): 
    save_path = os.path.join(RESULTS_DIR,'figures')
    pred_asv = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','southafrica','pred_asv.csv'),index_col=0)
    pred_clinical = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','southafrica','pred_clinical.csv'),index_col=0)
    pred_asv_clinical = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','southafrica','pred_asv_clinical.csv'),index_col=0)
    pred_species = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','southafrica','pred_species.csv'),index_col=0)
    predictions = [pred_asv,pred_species,pred_clinical,pred_asv_clinical]
    labels = ["Microbiome,ASV","Microbiome,Species","Clinical","Microbiome,ASV + Clinical"]
    colors = ['C0','C1','C2','C3']
    plot_utils.plot_roc_curves(predictions,labels,colors,"Evaluation of predictors on \n held-out samples",8.1,os.path.join(RESULTS_DIR,'figures','fig_3A.png'))
    plot_utils.plot_pr_curves(predictions,labels,colors,"Evaluation of predictors on \n held-out samples",8.1,os.path.join(RESULTS_DIR,'figures','fig_S3A.png'),'one')

def fig_3B():
    values_df = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','southafrica','shap_values_asv.csv'))
    data_df = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','southafrica','shap_data_asv.csv'))
    shap_values = shap.Explanation(values=values_df.values,
                                data=data_df.values,
                                feature_names=values_df.columns.tolist())
    sns.set_theme(style="white")
    sns.set_style("ticks")
    ax = shap.plots.beeswarm(shap_values, max_display=10, 
                            order=shap.Explanation.abs.mean(0),
                            group_remaining_features=False, 
                            plot_size=(8,4.5),
                            color_bar_label='Relative feature value',
                            show=False)
    new_feature_names = ["ASV2100:Haemophilus parainfleunzae", "ASV7108:Alloprevotella sp.", "ASV505:Streptococcus salivarius","ASV1707:Streptococcus parasanguinis","ASV1169:Veillonella atypica","ASV10234:Veillonella parvula","ASV2404:Pseudoleptotrichia sp.","ASV5088:Fusobacterium nucleatum","ASV582:Veillonella sp.",'ASV11255:Veillonella sp.']  # Custom names
    ax.set_yticklabels(new_feature_names[::-1], fontsize=12) 
    fig = plt.gcf()
    ax = fig.gca()
    ax.set_xlabel('SHAP value', fontsize=12)
    ax.set_title('Microbiome top 10 predictive features', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'figures','fig_3B.png'))
    plt.close()

def fig_S2A():
    values_df = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','southafrica','shap_values_clinical.csv'))
    data_df = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','southafrica','shap_data_clinical.csv'))
    shap_values = shap.Explanation(values=values_df.values,
                               data=data_df.values,
                               feature_names=values_df.columns.tolist())
    sns.set_theme(style="white")
    sns.set_style("ticks")
    ax = shap.plots.beeswarm(shap_values, max_display=20, 
                            order=shap.Explanation.abs.mean(0),
                            group_remaining_features=False, 
                            plot_size=(8,4.5),
                            color_bar_label='Relative feature value',
                            show=False)
    fig = plt.gcf()
    ax = fig.gca()
    ax.set_xlabel('SHAP value', fontsize=12)
    ax.set_title('Clinical model predictive features', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'figures','fig_S2A.png'))
    plt.close()
    
def fig_S2B():
    values_df = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','southafrica','shap_values_asv_clinical.csv'))
    data_df = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','southafrica','shap_data_asv_clinical.csv'))
    shap_values = shap.Explanation(values=values_df.values,
                                data=data_df.values,
                                feature_names=values_df.columns.tolist())
    sns.set_theme(style="white")
    sns.set_style("ticks")
    ax = shap.plots.beeswarm(shap_values, max_display=10, 
                            order=shap.Explanation.abs.mean(0),
                            group_remaining_features=False, 
                            plot_size=(8,4.5),
                            color_bar_label='Relative feature value',
                            show=False)
    new_feature_names = ["Age","ASV1707:Streptococcus parasanguinis","ASV2100:Haemophilus parainfleunzae","ASV7108:Alloprevotella sp.","ASV8526:Prevotella melaninogenica","ASV5088:Fusobacterium nucleatum","ASV11255:Veillonella sp.","ASV3991:Rothia mucilaginosa","ASV7958:Streptococcus parasanguinis","ASV616:Streptococcus mitis"]
    ax.set_yticklabels(new_feature_names[::-1], fontsize=12) 
    fig = plt.gcf()
    ax = fig.gca()
    ax.set_xlabel('SHAP value', fontsize=12)
    ax.set_title('Combined model predictive features', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'figures','fig_S2B.png'))
    plt.close()

def fig_3C_S3B(): 
    preds = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','external_val','preds_external_iters.csv'),index_col=0)
    mean_fpr = np.linspace(0,1,100)
    mean_recall = np.linspace(0,1,100)
    roc_results = {}
    pr_results = {}
    study_colors = {"Wang et al. 2019":"C0","Zhao et al. 2020":"C1","Chen et al. 2024":"C2"}
    study_order = ["Zhao et al. 2020", "Wang et al. 2019", "Chen et al. 2024"]

    for study_name in study_order:
        study_data = preds[preds['Study']==study_name]
        tprs = []
        aucs = []
        precisions = []
        auprs = []
        for iteration in study_data['iteration'].unique():
            iter_data = study_data[study_data['iteration']==iteration]

            fpr, tpr, _ = metrics.roc_curve(iter_data['y_true'], iter_data['y_pred'])
            auc_score = metrics.roc_auc_score(iter_data['y_true'], iter_data['y_pred'])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc_score)

            precision,recall,_ = metrics.precision_recall_curve(iter_data['y_true'],iter_data['y_pred'])
            interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1])
            aupr = metrics.average_precision_score(iter_data['y_true'], iter_data['y_pred'])
            precisions.append(interp_prec)
            auprs.append(aupr)
        roc_results[study_name] = {'tprs': tprs, 'aucs': aucs}
        pr_results[study_name] = {'precisions': precisions, 'auprs': auprs}
    
    sns.set_theme(style='white',palette=None)
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for study_name in study_order:
        data = roc_results[study_name]
        color = study_colors.get(study_name,'black')
        mean_tpr = np.mean(data['tprs'], axis=0)
        std_tpr = np.std(data['tprs'],axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.1,color=color)
        ax.plot(mean_fpr, mean_tpr,
                label=f'{study_name} (auROC={mean_auc:.2f})',
                lw=2, alpha=0.8,color=color)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Generalization of South Africa microbiome \n predictor on external studies', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.tick_params(labelsize=12)
    plt.plot([0, 1], [0, 1], '--',color='grey')
    font_prop = FontProperties(family='DejaVu Sans', size=8.5, weight='normal')
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
    plt.savefig(os.path.join(RESULTS_DIR,'figures','fig_3C.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for study_name in study_order:
        data = pr_results[study_name]
        study_data = preds[preds['Study'] == study_name]
        pos_frac = study_data['y_true'].mean() 

        color = study_colors.get(study_name, 'black')
        mean_prec = np.mean(data['precisions'], axis=0)
        std_prec = np.std(data['precisions'], axis=0)
        mean_aupr = np.mean(data['auprs'])

        ax.fill_between(mean_recall, np.maximum(mean_prec - std_prec, 0),
                        np.minimum(mean_prec + std_prec, 1), color=color, alpha=0.1)
        ax.axhline(y=pos_frac, linestyle='--', color=color, linewidth=2)
            #    label=f'Class balance (ratio={pos_frac:.2f})')
        ax.plot(mean_recall, mean_prec,
                label=f'{study_name} (auPR={mean_aupr:.2f}, class balance ={pos_frac:.2f})',
                lw=2, alpha=0.8, color=color)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('Recall') 
    ax.set_ylabel('Precision')
    ax.set_title('Generalization of South Africa microbiome \n predictor on external studies', fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=12)
    font_prop = FontProperties(family='DejaVu Sans', size=8.5, weight='normal')
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
    plt.savefig(os.path.join(RESULTS_DIR, 'figures', 'fig_S3B.png'))
    plt.close()

def fig_3D_S3C():
    preds = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','cross_study_val','pred_species_leaveonestudyout.csv'),index_col=0)
    y_vals = preds['y_true'].values
    y_preds = preds['y_pred'].values
    y_studies = preds['heldout_study'].values
    study_colors = {'Chen2024':'C2','SouthAfrica':'C3','Wang2019':'C0','Zhao2020':'C1'}
    study_names = {'Chen2024': 'Chen et al. 2024','SouthAfrica': 'South Africa','Wang2019':'Wang et al. 2019','Zhao2020':'Zhao et al. 2020'}
    
    sns.set_theme(style="white")
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    for study in ['SouthAfrica','Zhao2020','Wang2019','Chen2024']:
        study_mask = (y_studies == study)
        y_true = y_vals[study_mask]
        y_pred = y_preds[study_mask]
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        study_label = study_names[study]
        ax.plot(fpr, tpr, label=f'{study_label} (auROC={roc_auc:.2f})', color=study_colors.get(study, 'black'),lw=2)
    ax.plot([0, 1], [0, 1], '--', color='grey',lw=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Generalization of microbiome predictor on \n held-out study', size=12, fontweight='bold')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.tick_params(labelsize=12)
    font_prop = FontProperties(family='DejaVu Sans', size=8.5, weight='normal')
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
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'figures', 'fig_3D.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for study in np.unique(y_studies):
        study_mask = (y_studies == study)
        y_true = y_vals[study_mask]
        y_pred = y_preds[study_mask]
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
        aupr = metrics.average_precision_score(y_true, y_pred)
        pos_frac = np.mean(y_true)
        color = study_colors.get(study, 'black')
        study_label = study_names[study]
        ax.plot(recall, precision, label=f'{study_label} (auPR={aupr:.2f}, class balance={pos_frac:.2f})', color=color,lw=2)
        ax.axhline(y=pos_frac, linestyle='--', color=color, linewidth=2)
            #    label=f'Class balance (ratio={pos_frac:.2f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Generalization of microbiome predictor on \n held-out study', size=12, fontweight='bold')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.tick_params(labelsize=12)
    font_prop = FontProperties(family='DejaVu Sans', size=8.5, weight='normal')
    leg = ax.legend(
        fontsize=7,              
        prop=font_prop,
        loc='lower left',
        handlelength=1.5,
        labelspacing=0.5,
        borderaxespad=0.5,
        framealpha=1
    )
    leg.get_frame().set_edgecolor('none') 
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'figures', 'fig_S3C.png'))
    plt.close()

def fig_S4AB(): 
    save_path = os.path.join(RESULTS_DIR,'figures')
    pred_zhao = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','zhao2020','pred_species.csv'),index_col=0)
    pred_wang = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','wang2019','pred_species.csv'),index_col=0)
    pred_chen = pd.read_csv(os.path.join(RESULTS_DIR,'predictions','chen2024','pred_species.csv'),index_col=0)
    predictions = [pred_zhao,pred_wang,pred_chen]
    labels = ["Zhao et al. 2020","Wang et al. 2019","Chen et al. 2024"]
    colors = ['C1','C0','C2']
    plot_utils.plot_roc_curves(predictions,labels,colors,"Evaluation of predictors on \n held-out samples",8.5,os.path.join(RESULTS_DIR,'figures','fig_S4A.png'))
    plot_utils.plot_pr_curves(predictions,labels,colors,"Evaluation of predictors on \n held-out samples",8.5,os.path.join(RESULTS_DIR,'figures','fig_S4B.png'),'all')

def main():
    data = load_data()
    table_S1(data['metadata'])
    # diversity
    fig_1A(data['rarefied_table'])
    fig_1B_S1()
    # differential abundance
    fig_2A(data['feature_ra'],data['taxonomy'])
    fig_2B(data['feature_ra'],data['taxonomy'])
    fig_2C(data['feature_ra'],data['taxonomy'])
    # predictions
    fig_3A_S3A()
    fig_3B()
    fig_S2A()
    fig_S2B()
    fig_3C_S3B()
    fig_3D_S3C()
    fig_S4AB()

if __name__ == "__main__":
    main() 
"""

ESCC_microbiome_predictions.py

Author: Shahd ElNaggar

Purpose: use nested-CV to build clinical and microbiome-based logistic models predicting ESCC. Since the data was processed in batches, I will group the samples by batch in order to avoid leakage.

Date: 3.21.25

Last edited: 3.21.25

"""
import pandas as pd
import numpy as np
from PredictionPipelineV3 import *
from sklearn import metrics

def run_clinical_pred(X,md,splits,num_nests,num_folds,num_robust,out_path,iterations = 500,on_cluster=False):
    run_params = {
        'steps': [
            ['Imputer'],
        ],
        'imputer_method':['mean','median'],
    }
    model_params = Defaults.LOGISTIC
    RUN_NAME = 'clinical_logistic'
    pred = Prediction(
        run_name=RUN_NAME,
        X=X,
        md=md,
        model_type=ModelType.LOGISTIC,
        run_hp_space=run_params,
        model_hp_space=model_params
    )
    pred.run(
        outer_folds=num_nests,
        inner_folds=num_folds,
        robust=num_robust,
        iterations=iterations,
        stratified=True,
        on_cluster=on_cluster,
        n_jobs=50,
        cpus=6,
        mem='24G',
        hours=24,
        seed=1,
        out_file=f'{out_path}/{RUN_NAME}.pkl',
        custom_splits = splits, 
        run_nested_evaluation=False, 
        run_shap=False,
        conda_env='pp3'
    )
    pred.run_nested_evaluation(plot=False)
    nested_res, train_test_scores = pred.get_nested_predictions(plot=False)
    nested_res.sort_index(inplace=True)
    return nested_res

def run_microbiome_pred(X,md,splits,num_nests,num_folds,num_robust,out_path,iterations = 500,on_cluster=False):

    run_params = {
        'steps': [
            ['CountFilter','VarianceFilter','Preprocessor','FeatureSelector'], 
        ],    
        'count_filter_threshold': [1,5,10],
        'count_filter_num_samples': [0,0.01,0.05,0.1,0.5],
         
        'preprocessing_method': [
            PreprocessingMethods.CLR,
        ],

        'feature_selection_method':[  
            [FeatureSelectionMethods.UNIVAR] 
        ],
        'univar_score_func': ['f'], 
        'univar_mode': ['k_best'],
        'univar_param': [0.01,0.05,0.1,0.25,0.5,0.75,-1]
    }

    model_params = Defaults.LOGISTIC

    RUN_NAME = 'microbiome_logistic'
    pred = Prediction(
        run_name=RUN_NAME,
        X=X,
        md=md,
        model_type=ModelType.LOGISTIC,
        run_hp_space=run_params,
        model_hp_space=model_params
    )

    pred.run(
        outer_folds=num_nests, 
        inner_folds=num_folds, 
        robust=num_robust, 
        iterations=iterations,
        stratified=True,
        seed=1,
        out_file=f'{out_path}/{RUN_NAME}.pkl',
        run_shap=False, 
        on_cluster=on_cluster,
        n_jobs=50,
        cpus=6,
        mem='24G',
        hours=24,
        seed=1,
        run_nested_evaluation=False, 
        custom_splits = splits,
        conda_env='pp3'
    )
    pred.run_nested_evaluation(plot=False)
    nested_res, train_test_scores = pred.get_nested_predictions(plot=False)
    nested_res.sort_index(inplace=True)
    return nested_res

def run_genus_pred(X,md,splits,num_nests,num_folds,num_robust,out_path,iterations = 500,on_cluster=False):
    run_params = {
        'steps': [
            ['CountFilter','VarianceFilter','Preprocessor','FeatureSelector'], 
        ],    
        'count_filter_threshold': [1,5,10,20],
        'count_filter_num_samples': [0,0.01,0.05,0.1,0.5],
         
        'preprocessing_method': [
            PreprocessingMethods.CLR,
        ],

        'feature_selection_method':[  
            [FeatureSelectionMethods.UNIVAR] 
        ],
        'univar_score_func': ['f'], 
        'univar_mode': ['k_best'],
        'univar_param': [0.01,0.05,0.1,0.25,0.5,0.75,-1]
    }

    model_params = Defaults.LOGISTIC

    RUN_NAME = 'genus_logistic'
    pred = Prediction(
        run_name=RUN_NAME,
        X=X,
        md=md,
        model_type=ModelType.LOGISTIC,
        run_hp_space=run_params,
        model_hp_space=model_params
    )

    pred.run(
        outer_folds=num_nests, 
        inner_folds=num_folds, 
        robust=num_robust, 
        iterations=iterations,
        stratified=True,
        seed=1,
        out_file=f'{out_path}/{RUN_NAME}.pkl',
        run_shap=False, 
        on_cluster=on_cluster,
        n_jobs=50,
        cpus=6,
        mem='24G',
        hours=24,
        seed=1,
        run_nested_evaluation=False, 
        custom_splits = splits,
        conda_env='pp3'
    )
    pred.run_nested_evaluation(plot=False)
    nested_res, train_test_scores = pred.get_nested_predictions(plot=False)
    nested_res.sort_index(inplace=True)
    return nested_res

def create_splits(md,X,group,num_folds,num_robust):
    splits = []
    for nest,batch in enumerate(sorted(md[group].unique())):
        outer_test_samples = list(md[md[group] == batch].index)
        outer_train_samples = list(md[md[group]!= batch].index)
        for robust in range(num_robust):
            for fold,(train,test) in enumerate(get_kfold(X.loc[outer_train_samples],md.loc[outer_train_samples],folds=num_folds,stratified=True,stratify_by='outcome',group_by=group,seed=robust*nest)):
                assert set(outer_test_samples).intersection(set(train))==set()
                assert set(outer_test_samples).intersection(set(test))==set()
                splits.append(
                    {'key':f"{nest}_{robust}_{fold}",'nest':nest,'robust':robust,'fold':fold,'train_samples':train,'test_samples':test,'outer_test_samples':outer_test_samples}
                )
    splits = pd.DataFrame(splits).set_index('key')            
    return splits

def main(out_path = 'predictions/'):

    metadata = pd.read_csv("data/metadata.csv",index_col=0).sort_index()
    md = metadata[['ESCC','Batch']].rename(columns={'ESCC':'outcome'})

    clinical_features = metadata.drop(columns=['ESCC','Batch'])
    category_cols = ['Sex','Residence','Marriage Status','Hot Beverage Consumption','Diabetes','Smoking','Alcohol','HIV','Cooking Location']
    clinical_features[category_cols] = clinical_features[category_cols].apply(lambda x: x.astype('category'))

    microbiome_features = pd.read_csv("data/feature_table_decontaminated.csv",index_col=0).T.reindex(md.index)
    taxonomy = pd.read_csv('data/taxonomy.csv',index_col=0)
    genus_features = microbiome_features.T.join(taxonomy['Genus'],how='inner').set_index('Genus')
    genus_features = genus_features.groupby(genus_features.index).sum().T.reindex(md.index)

    on_cluster = True
    num_folds = 5
    num_robust = 3
    group_by = 'Batch'
    num_nests = len(md[group_by].unique())

    splits = create_splits(md,clinical_features,group_by,num_folds,num_robust)

    clinical_res = run_clinical_pred(clinical_features,md,splits,num_nests,num_folds,num_robust,out_path,1000,on_cluster)
    microbiome_res = run_microbiome_pred(microbiome_features,md,splits,num_nests,num_folds,num_robust,out_path,1000,on_cluster)
    genus_res = run_genus_pred(genus_features,md,splits,num_nests,num_folds,num_robust,out_path,1000,on_cluster)

    clinical_res.to_csv(f'{out_path}/clinical_res.csv',index=True)
    microbiome_res.to_csv(f'{out_path}/microbiome_res.csv',index=True)
    genus_res.to_csv(f'{out_path}/genus_res.csv',index=True)

if __name__=='__main__':
    main()
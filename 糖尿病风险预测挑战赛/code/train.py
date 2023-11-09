# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2023-09-30
# @Description: 模型训练

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import joblib
import argparse
import warnings
warnings.filterwarnings('ignore')


def cross_features(df):
    """特征交叉"""
    df['BMI_mul_HighBP'] = df['BMI'] * df['HighBP']
    df['BMI_mul_HighChol'] = df['BMI'] * df['HighChol']
    df['BMI_mul_Smoker'] = df['BMI'] * df['Smoker']
    df['BMI_mul_Sex'] = df['BMI'] * df['Sex']
    df['BMI_mul_Fruits'] = df['BMI'] * df['Fruits']
    df['BMI_mul_PhysActivity'] = df['BMI'] * df['PhysActivity']
    df['BMI_mul_Veggies'] = df['BMI'] * df['Veggies']
    df['BMI_mul_DiffWalk'] = df['BMI'] * df['DiffWalk']
    df['HighChol_mul_HvyAlcoholConsump'] = df['HighChol'] * df[
        'HvyAlcoholConsump']
    df['PhysHlth_add_MentHlth_mean'] = (df['PhysHlth'] + df['MentHlth']) / 2

    tmp_sum = np.sum(df[['PhysActivity', 'Fruits', 'Veggies']], axis=1)
    tmp_mean = np.mean(df[['PhysActivity', 'Fruits', 'Veggies']], axis=1)
    tmp_std = np.std(df[['PhysActivity', 'Fruits', 'Veggies']], axis=1)
    df['PFV_sum'] = tmp_sum
    df['PFV_mean'] = tmp_mean
    df['PFV_std'] = tmp_std
    df['PFV_cv'] = tmp_std / tmp_mean

    tmp_sum = np.sum(df[['GenHlth', 'MentHlth', 'PhysHlth']], axis=1)
    tmp_mean = np.mean(df[['GenHlth', 'MentHlth', 'PhysHlth']], axis=1)
    tmp_std = np.std(df[['GenHlth', 'MentHlth', 'PhysHlth']], axis=1)
    df['GMP_sum'] = tmp_sum
    df['GMP_mean'] = tmp_mean
    df['GMP_std'] = tmp_std
    df['GMP_cv'] = tmp_std / tmp_mean

    return df


def argg_features(df):
    """特征聚合"""
    stats = ['mean', 'std', 'skew']
    groups = ['Age', 'PhysHlth', 'MentHlth', 'Income']
    agg_cols = ['BMI']
    for cat in groups:
        for num in agg_cols:
            map_df = df.groupby(cat)[num].agg(stats)
            for s in stats:
                df[cat + '_' + num + '_' + s] = df[cat].map(map_df[s])
            df[cat + '_' + num + '_cv'] = df[cat + '_' + num + '_std'] / df[
                cat + '_' + num + '_mean']
            df[cat + '_' + num + '_diff'] = df.groupby(cat)[num].transform(
                'max') - df.groupby(cat)[num].transform('min')

    pair_groups = [['PhysHlth', 'Education'], ['PhysHlth', 'GenHlth'],
                   ['GenHlth', 'Education'], ['Age', 'Income'],
                   ['Age', 'GenHlth'], ['GenHlth', 'Income'],
                   ['Age', 'Education']]
    for cat in pair_groups:
        for num in agg_cols:
            map_df = df.groupby(cat)[num].agg(stats)
            map_df.fillna(0, inplace=True)
            map_df.columns = [cat[0] + '_' + cat[1] + '_' + num + '_' + i for i
                              in map_df.columns]
            map_df = map_df.reset_index()
            df = df.merge(map_df, on=cat, how='left')

            df[cat[0] + '_' + cat[1] + '_' + num + '_cv'] = df[cat[0] + '_' + cat[1] + '_' + num + '_std'] / df[cat[0] + '_' + cat[1] + '_' + num + '_mean']
            df[cat[0] + '_' + cat[1] + '_' + num + '_diff'] = df.groupby(cat)[num].transform('max') - df.groupby(cat)[num].transform('min')

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../xfdata/糖尿病风险预测挑战赛公开数据/', type=str)
    parser.add_argument('--fold', default=5, type=int)
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--depth', default=8, type=int)
    parser.add_argument('--seed', default=2222, type=int)
    args = parser.parse_args()

    print("==========读取原始数据集==========")
    # 读取原始数据集并删除无用的id标识特征
    train_df = pd.read_csv('../xfdata/糖尿病风险预测挑战赛公开数据/train.csv')
    del train_df['id']
    test_df = pd.read_csv('../xfdata/糖尿病风险预测挑战赛公开数据/test.csv')
    del test_df['id']
    # 合并训练集和测试集数据，便于后续构造特征
    df = pd.concat([train_df, test_df])

    print("==========构造特征：特征交叉、特征统计聚合等操作==========")
    # 构造特征：特征交叉
    df = cross_features(df)
    # 构造特征：不同维度特征进行聚合统计计算
    df = argg_features(df)

    print("==========划分训练特征数据和测试特征数据并保存起来==========")
    train_df = df[~df['target'].isnull()]
    test_df = df[df['target'].isnull()]
    x = train_df.drop(columns=['target'])
    y = train_df['target']
    test_x = test_df.drop(columns=['target'])
    # 保存特征数据集
    train_df.to_csv('../user_data/tmp_data/train_data.csv', index=False)
    test_x.to_csv('../user_data/tmp_data/test_data.csv', index=False)

    print("==========定义训练参数和模型超参数==========")
    seed = args.seed
    n_fold = args.fold
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    feats_weight = pd.DataFrame(x.columns.values, columns=['feat_name'])
    feats_weight['importance'] = 0
    # lightgbm模型参数设置
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'learning_rate': 0.03,
        'metric': 'multi_logloss',
        'max_depth': 8,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 4,
        'verbose': -1,
        'seed': seed,
        'class_weight': 'balanced'
    }

    print("=========开始训练模型：五折交叉训练验证==========")
    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        print(f'==========fold {fold}==========')
        train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
        val_x, val_y = x.iloc[val_idx], y.iloc[val_idx]

        # 模型定义、训练
        model = lgb.LGBMClassifier(**lgb_params, n_estimators=30000, n_jobs=-1)
        model.fit(train_x, train_y, eval_set=(val_x, val_y),
                  eval_metric='multi_logloss', verbose=2000,
                  early_stopping_rounds=200)

        # 保存模型文件
        joblib.dump(model, f'../user_data/model_data/model_{fold}.pkl')

        oof[val_idx] = model.predict(val_x, num_iteration=model.best_iteration_)
        feats_weight['importance'] += model.feature_importances_ / n_fold

    print("==========输出模型分类验证报告和模型特征重要性==========")
    print(classification_report(y, oof))
    print(feats_weight.sort_values(by='importance', ascending=False))


if __name__ == '__main__':
    main()


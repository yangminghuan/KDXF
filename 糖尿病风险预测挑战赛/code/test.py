# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2023-09-30
# @Description: 模型预测

import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')


def main():
    print("==========读取预测数据集==========")
    # 读取处理好的预测数据集
    test_x = pd.read_csv('../user_data/tmp_data/test_data.csv')

    print("==========加载模型文件、进行模型预测==========")
    pred_y = pd.DataFrame()
    for i in range(5):
        model = joblib.load(f'../user_data/model_data/model_{i}.pkl')
        pred_y[f'fold_{i}'] = model.predict(test_x, num_iteration=model.best_iteration_)

    print("==========集成模型结果：投票方式==========")
    pred_y = pred_y.mode(axis=1)

    print("==========输出并保存模型预测结果==========")
    submit_df = pd.DataFrame(data={'id': range(300000, 400000)})
    submit_df['target'] = pred_y[0]
    submit_df.to_csv('../prediction_result/result.csv', index=False)


if __name__ == '__main__':
    main()

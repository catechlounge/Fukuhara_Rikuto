import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def preprocess(csv_path, base_columns, visualize: bool):
    covtype_dict = {}

    covtype_data = pd.read_csv(csv_path)
    print(covtype_data)

    for covtype_index in base_columns:
        # すべての値を正の値に変換
        offset = abs(covtype_data[covtype_index].min()) + 1
        adjusted_data = covtype_data[covtype_index] + offset

        # Log1p変換（自然対数の変換, log(1 + x)）
        transformed_data = np.log1p(adjusted_data)
        
        df = pd.DataFrame({"{} original".format(covtype_index): covtype_data[covtype_index], "log1p({})".format(covtype_index): transformed_data})

        original_skew = skew(df["{} original".format(covtype_index)])
        log1p_skew = skew(df["log1p({})".format(covtype_index)])
        print('{} skew:'.format(covtype_index), original_skew)
        print('log1p({}):'.format(covtype_index), log1p_skew)

        # 偏度が小さい方のデータを保持
        if original_skew < log1p_skew:
            covtype_dict[covtype_index] = df["{} original".format(covtype_index)]
        else:
            covtype_dict["log1p_" + covtype_index] = df["log1p({})".format(covtype_index)]

        if visualize:
            df.hist()
            plt.text(0.5, 0.5, 'Original skew: {:.2f}\nLog skew: {:.2f}'.format(original_skew, log1p_skew), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.savefig("./log1p_distribution_before_after_{}.png".format(covtype_index))
            plt.show()

    log_covtype_df = pd.DataFrame.from_dict(covtype_dict)
    mns = MinMaxScaler()
    log_covtype_df_normalize = mns.fit_transform(log_covtype_df)
    log_covtype_df_normalize = pd.DataFrame(log_covtype_df_normalize, columns=log_covtype_df.columns)
    return log_covtype_df_normalize


if __name__ == '__main__':
    csv_path = '../covtype.csv'
    covtype_data = pd.read_csv(csv_path)
    base_columns = covtype_data.columns
    x = preprocess(csv_path=csv_path, base_columns=base_columns, visualize=False)
    print(x)

import csv
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


df = pd.read_csv('../covtype.csv')

# ターゲット列をダミー変数に変換
dummy_target = pd.get_dummies(df["Cover_Type"])
df = pd.concat([df.drop("Cover_Type", axis=1), dummy_target], axis=1)

# ヒートマップを作成
plt.figure(figsize=(80, 80))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.savefig(f'./all心拍指標ヒートマップ.png', dpi=600)
plt.show()

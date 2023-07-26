# pearson,spearman相関係数
## データの加工
JGLUE-STS内のtrain-v1.1.jsonから、pandasを利用して、sentence1,sentence2を取得した。それらをpandas.seriesの形にまとめ、まとめたものをfor文で回した。

## 相関係数の演算
//bertを使用してtensorに直したという説明をする
その後、トークン化した2文を引数にとり、scipyのperasonrでPearson相関係数を計算し、scipyのspearmanrでSpearman相関係数を演算した。また、0-5の範囲内で示すため各結果に5をかけた。
## 出力
それぞれの相関係数の種類と結果を提示されているフォーマットに従い、jsonの形で書き出した。









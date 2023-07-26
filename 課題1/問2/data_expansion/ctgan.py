from sdv.single_table import CTGANSynthesizer
import pandas as pd
from sdv.metadata import SingleTableMetadata

# データ数をふやすためにCTGAN使ってみる。

# データを読みこむ
covtypedata_df = pd.read_csv('../../covtype.csv')
covtypedata_df = covtypedata_df[1:]

# metadataの設定
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=covtypedata_df)

# CTGANの学習
synthesizer = CTGANSynthesizer(
    metadata, # required
    enforce_rounding=False,
    epochs=1000,
    #logの出力Trueにしておいて間違いはない。
    verbose=True,
    cuda = False
)
metadata = synthesizer.get_metadata()
synthesizer.get_parameters()
synthesizer.fit(covtypedata_df)

#GANで生成 DataFrameの形で書き出し
synthetic_data = synthesizer.sample(num_rows=1000)

#csvに保存
synthetic_data.to_csv('ctgan_make_dataset.csv')

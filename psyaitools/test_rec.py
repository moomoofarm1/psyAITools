# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from functional.representation.factDecomMachine import *
np.random.seed(42)

# All personality + preEmo
path1 =  '../../comDat/tabFile.csv'
path2 =  '../../comDat/CLAPerEm.csv'
path3 = '../../comDat/preEmoEmbed.csv'

# TODO: change column names to numbers.
def recommend():
    print("Hello world!")

    spark, logger = getSpark()
    logger.LogManager.getRootLogger().setLevel(logger.Level.WARN)
    
    # start the recommendation
    df1 = getRecDat(path1)
    extTab = [0,2] + [4,5,6,7,8,9]
    df1 = df1.iloc[:, extTab]
    df1.columns = ['procID','pPlace', "extraversion", "agreeableness","consciensiousness",
            "emoStability", "openness",'prePANAS']
    df1 = df1.sample(frac=1)
    melted_df1 = pd.melt(df1, id_vars=['procID'], value_vars=['pPlace','extraversion', 'agreeableness', 'consciensiousness', 'emoStability', 'openness', 'prePANAS'], var_name='item', value_name='rating')
    print(melted_df1.head())

    model = factDecomMachine(melted_df1, numDims=5, backend='spark')
    print("Model trained!")

    df_CLAPer = getRecDat(path2)
    col_CLAPer = ['CLA_Dim' + str(i) for i in range(1, df_CLAPer.shape[1]+1)]
    df_CLAPer.columns = col_CLAPer
    df_preEmoEmb = getRecDat(path3)
    col_preEmoEmb = ['preEmoEmb_Dim' + str(i) for i in range(1, df_preEmoEmb.shape[1]+1)]
    df_preEmoEmb.columns = col_preEmoEmb
    df2 = pd.concat([df1.loc[:,['procID','pPlace']],df_CLAPer, df_preEmoEmb], axis=1)
    df2 = df2.sample(frac=1)
    melted_df2 = pd.melt(df2, id_vars=['procID'], value_vars=['pPlace'] + df_CLAPer.columns.tolist() + df_preEmoEmb.columns.tolist(), 
                         var_name='item', value_name='rating')
    print(melted_df2.head())
    
    #spark_nonEmb = spark.createDataFrame(df1)
    #spark_nonEmb.show()
    #spark_df_emb = spark.createDataFrame(df2)
    #spark_df_emb.show()


    print("Goodbye world!")

    return 0

if __name__ == "__main__":
    recommend()
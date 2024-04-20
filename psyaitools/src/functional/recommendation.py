# -*- coding: utf-8 -*-
import numpy as np

# https://spark.apache.org/docs/3.3.1/api/python/reference/pyspark.ml.html#pipeline-apis
# https://spark.apache.org/docs/3.3.1/api/python/reference/api/pyspark.ml.recommendation.ALSModel.html#pyspark.ml.recommendation.ALSModel
# https://spark.apache.org/docs/3.3.1/api/python/reference/api/pyspark.ml.recommendation.ALS.html#pyspark.ml.recommendation.ALS
# https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
# indirect SVD-based recommendation

def recommendation(df):
    '''
    df : pyspark.sql.dataframe.DataFrame
    '''
    print("Recommendation: You should try the new restaurant in town!")

    return 0
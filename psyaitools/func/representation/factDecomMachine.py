# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer

def _getSpark():
    spark = SparkSession.builder.getOrCreate()
    logger = spark.sparkContext._jvm.org.apache.log4j
    logger.LogManager.getRootLogger().setLevel(logger.Level.WARN)
    return spark, logger

def factDecomMachine(df,colNames=["usrId","item","rating"],
                     numDims = 5, backend='spark'):
    '''
    df : pandas dataframe #pyspark.sql.dataframe.DataFram
    colNames : list
    numDims : int
        number of dimensions for the latent factors, default 5
    backend : str
        'spark' or 'surprise', default 'spark', 'surprise' is not implemented yet.
    '''
    
    if backend == 'spark':
        spark, logger = _getSpark()
        logger.LogManager.getRootLogger().setLevel(logger.Level.WARN)
        spark_df = spark.createDataFrame(df)
        als = ALS(rank=numDims, maxIter=10, regParam=0.01, userCol=colNames[0], 
                  itemCol=colNames[1], ratingCol=colNames[2],
                  coldStartStrategy="drop")
        model = als.fit(spark_df)
        return model
    else:
        spark, logger = _getSpark()
        logger.LogManager.getRootLogger().setLevel(logger.Level.WARN)
        spark_df = spark.createDataFrame(df)
        als = ALS(rank=numDims, maxIter=10, regParam=0.01, userCol=colNames[0], 
                  itemCol=colNames[1], ratingCol=colNames[2],
                  coldStartStrategy="drop")
        model = als.fit(spark_df)
        return model
    

def factDecomPredict(model, df, colNames=["usrId","item","rating"], backend='spark'):
    '''
    model : ALS model
    df : pandas dataframe
    colNames : list
    backend : str
        'spark' or 'surprise', default 'spark', 'surprise' is not implemented yet.
    '''
    if backend == 'spark':
        spark, logger = _getSpark()
        logger.LogManager.getRootLogger().setLevel(logger.Level.WARN)
        spark_df = spark.createDataFrame(df)
        return model.transform(spark_df).toPandas()
    else:
        spark, logger = _getSpark()
        logger.LogManager.getRootLogger().setLevel(logger.Level.WARN)
        spark_df = spark.createDataFrame(df)
        return model.transform(spark_df).toPandas()
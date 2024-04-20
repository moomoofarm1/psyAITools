# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

path =  '../tidyDatMIP_copy.csv'
def getRecDat(path):
    df1 = pd.read_csv(path)
    df2 = df1.loc[:, "procID":"pDate"]
    df2["prePANAS"] = df1.loc[:, "pPrePANAS_01_Interested":"pPrePANAS_20_Afraid"].sum(axis=1)
    df2["postPANAS"] = df1.loc[:, "pPostPANAS_01_Interested":"pPostPANAS_20_Afraid"].sum(axis=1)
    df1.loc[:, ["pTIPI01", "pTIPI04", "pTIPI06", "pTIPI09"]] = 1 + df1.loc[:, ["pTIPI01", "pTIPI04", "pTIPI06", "pTIPI09"]]
    df1.loc[:, ["pTIPI04","pTIPI09"]] = 8 - df1.loc[:, ["pTIPI04","pTIPI09"]]
    df2["extraversion"] = df1.loc[:, ["pTIPI01", "pTIPI06"]].mean(axis=1)
    df2["emoStability"] = df1.loc[:, ["pTIPI04", "pTIPI09"]].mean(axis=1)
    return df2

def main():
    print("Hello world!")

    # Initialize SparkSession
    spark = SparkSession.builder.getOrCreate()
    # Get the logger
    logger = spark.sparkContext._jvm.org.apache.log4j
    # Set default log level to "WARN"
    logger.LogManager.getRootLogger().setLevel(logger.Level.WARN)
    
    # start the recommendation
    df = getRecDat(path)
    spark_df = spark.createDataFrame(df)
    spark_df.show()

    print("Goodbye world!")

    return 0

# if __name__ == "__main__":
#     main()
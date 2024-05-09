from psyaitools.func.representation.factDecomMachine import *
import pandas as pd

from pyspark.sql import SparkSession
#from pyspark.ml.recommendation import ALS



df = pd.read_csv('../../comDat/onlineMIPNoProfID.csv')
df = df.iloc[:1011,:] # PANAS na data
# 56 - 75 prePANAS
# 84 - 103 postPANAS
pos_list_pre = ["pPrePANAS_01_Interested", "pPrePANAS_04_Alert", "pPrePANAS_05_Excited", "pPrePANAS_08_Inspired", "pPrePANAS_09_Strong", "pPrePANAS_12_Determined",
            "pPrePANAS_14_Attentive", "pPrePANAS_17_Enthusiastic", "pPrePANAS_18_Active", "pPrePANAS_19_Proud"]
neg_list_pre = ["pPrePANAS_02_Irritable", "pPrePANAS_03_Distressed", "pPrePANAS_06_Ashamed", "pPrePANAS_07_Upset",
            "pPrePANAS_10_Nervous", "pPrePANAS_11_Guilty", "pPrePANAS_13_Scared","pPrePANAS_15_Hostile", "pPrePANAS_16_Jittery", "pPrePANAS_20_Afraid"]
pos_list_post = ["pPostPANAS_01_Interested", "pPostPANAS_04_Alert", "pPostPANAS_05_Excited", "pPostPANAS_08_Inspired", "pPostPANAS_09_Strong", "pPostPANAS_12_Determined",
            "pPostPANAS_14_Attentive", "pPostPANAS_17_Enthusiastic", "pPostPANAS_18_Active", "pPostPANAS_19_Proud"]
neg_list_post = ["pPostPANAS_02_Irritable", "pPostPANAS_03_Distressed", "pPostPANAS_06_Ashamed", "pPostPANAS_07_Upset",
            "pPostPANAS_10_Nervous", "pPostPANAS_11_Guilty", "pPostPANAS_13_Scared","pPostPANAS_15_Hostile", "pPostPANAS_16_Jittery", "pPostPANAS_20_Afraid"]

df.loc[:,pos_list_pre] = df.loc[:,pos_list_pre] + 1
df.loc[:,neg_list_pre] = df.loc[:,neg_list_pre] + 1
df.loc[:, neg_list_pre] = 6 - df.loc[:, neg_list_pre]
df["pPrePANAS_pos"] = df.loc[:,pos_list_pre].sum(axis=1)
df["pPrePANAS_neg"] = df.loc[:,neg_list_pre].sum(axis=1)
df.loc[:,pos_list_post] = df.loc[:,pos_list_post] + 1
df.loc[:,neg_list_post] = df.loc[:,neg_list_post] + 1
df.loc[:, neg_list_post] = 6 - df.loc[:, neg_list_post]
df["pPostPANAS_pos"] = df.loc[:,pos_list_post].sum(axis=1)
df["pPostPANAS_neg"] = df.loc[:,neg_list_post].sum(axis=1)
# print(df.iloc[:,56:76].columns)
# print(df.iloc[:,84:104].columns)
to_ana = ["pPrePANAS_pos", "pPrePANAS_neg", "pPostPANAS_pos", "pPostPANAS_neg"]
df["pPlace_code"] = df["pPlace"].astype('category').cat.codes
df = df.loc[:,["pPlace_code","pAge"] + to_ana]
df.reset_index(inplace=True)
df_leave = df.iloc[-2:,:]
df_train = df.iloc[:-2,:]
df_melt = df_train.melt(id_vars='index', var_name='item', value_name='rating')
df_melt["item"] = df_melt["item"].astype('category').cat.codes # item has to be numeric.
df_leave_melt = df_leave.melt(id_vars='index', var_name='item', value_name='rating')
df_leave_melt["item"] = df_leave_melt["item"].astype('category').cat.codes 

#print(df_melt.head())

mod1 = factDecomMachine(df_melt,colNames=["index","item","rating"], 
                        numDims = 3,
                        backend='spark')
print(mod1.extractParamMap())

spark = SparkSession.builder.getOrCreate()
spark_df_leave = spark.createDataFrame(df_leave_melt)
aaa = mod1.transform(spark_df_leave).toPandas()
print(aaa.head())

''' Numerical indexer for categorical item names
First, create a mapping from the categorical item names to numerical indices. You can do this using the StringIndexer function in PySpark:

Python

from pyspark.ml.feature import StringIndexer

# Create a StringIndexer
indexer = StringIndexer(inputCol="item", outputCol="itemIndex")

# Fit the indexer to the data
indexer_model = indexer.fit(df_melt)

# Create the indexed data
df_melt = indexer_model.transform(df_melt)
AI-generated code. Review and use carefully. More info on FAQ.
Now, df_melt has an additional column itemIndex that contains a numerical index for each unique item in the item column.

You can do the same for df_leave_melt:

Python

df_leave_melt = indexer_model.transform(df_leave_melt)
AI-generated code. Review and use carefully. More info on FAQ.
Now, you can use itemIndex as the input to your ALS model:

Python

mod1 = factDecomMachine(df_melt,colNames=["index","itemIndex","rating"], 
                        numDims = 3,
                        backend='spark')
AI-generated code. Review and use carefully. More info on FAQ.
And for prediction:

Python

spark_df_leave = spark.createDataFrame(df_leave_melt)
aaa = mod1.transform(spark_df_leave).toPandas()

'''
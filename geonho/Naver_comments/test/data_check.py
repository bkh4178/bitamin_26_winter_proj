#%%
import pandas as pd
import numpy as np

comments = pd.read_csv('/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/comments/comments_2025_adj.csv')
articles = pd.read_csv('/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/article/articles_2025_financial.csv')
#%%
articles.rename(columns={'url':'article_url'}, inplace=True)
df = articles.merge(comments, on='article_url', how='left')
print(df)

#%%
df['is_financial'].sum()
len(df)
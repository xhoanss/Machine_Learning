import pandas as pd

df1 = pd.read_csv("listings.csv")
df1.head()
if __name__ == '__main__':
    print(df1.head()['name'])
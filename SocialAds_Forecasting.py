file = 'https://github.com/theleadio/datascience_demo/blob/master/social-ads-raw.xlsx?raw=true'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc = {'figure.figsize': (5, 5)})

df = pd.read_excel(file)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Initial Exploration

df_long = df.melt(id_vars = ['segment', 'sales', 'size', 'area'], value_vars = ['google', 'facebook', 'instagram'], var_name = 'platform', value_name = 'expenses')

fig, ax = plt.subplots(2, 3, figsize = (21, 10))
cat = ['size', 'area', 'platform']

for i in range(len(cat)):
  sns.scatterplot(data = df_long, x = 'expenses', y = 'sales', hue = cat[i], ax = ax[0,i]);

for i in range(len(cat)):
  sns.boxplot(data = df_long, y = "expenses", x = cat[i], ax = ax[1,i]);
  
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Dummify categorical columns

size_dummies = pd.get_dummies(df['size'])
area_dummies = pd.get_dummies(df['area'])
df = df.drop(['segment', 'size', 'area'], axis = 1).join(area_dummies).join(size_dummies)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Correlation Heatmap

sns.set(rc = {'figure.figsize':(8, 5)})
sns.heatmap(df.corr('pearson'), cmap = 'PuOr', fmt = '.2f', annot = True, vmin = -1, vmax = 1, center = 0);

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Linear Regression Modelling

from sklearn.linear_model import LinearRegression
model = LinearRegression()
y = df['sales']

# --- Consider advertising on a single platform:

# google
X = df[['google']]
model.fit(X, y)
print(f"R^2 = {model.score(X, y):.4f}")
print(f"y = {model.intercept_:.4f} + {model.coef_[0]:.4f} X")

# facebook
X = df[['facebook']]
model.fit(X, y)
print(f"R^2 = {model.score(X, y):.4f}")
print(f"y = {model.intercept_:.4f} + {model.coef_[0]:.4f} X")

# instagram
X = df[['instagram']]
model.fit(X, y)
print(f"R^2 = {model.score(X, y):.4f}")
print(f"y = {model.intercept_:.4f} + {model.coef_[0]:.4f} X")

# --- The overall model
X = df[['google', 'facebook', 'instagram', 'small', 'large', 'rural', 'suburban', 'urban']]
model.fit(X, y)
print(f"R^2 = {model.score(X, y)}")
print(f"Intercept = {model.intercept_}")
print(f"Coefficients = {model.coef_}")

# --- Excluding "small", "rural", and "suburban"
X = df[['google', 'facebook', 'instagram', 'large', 'urban']]
model.fit(X, y)
print(f"R^2 = {model.score(X, y)}")
print(f"Intercept = {model.intercept_}")
print(f"Coefficients = {model.coef_}")

# Example Prediction
y = model.predict([[50, 30, 20, 1, 1]])

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Given ads budget of $100k, how would I spend it on the 3 platforms?
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#

import random as rd

gg = 10
fb = 6
ins = 4

ggnew = rd.randint(0, 20)
fbnew = rd.randint(0, 20-ggnew)
insnew = 20-ggnew-fbnew

i = 0

while i < 500:

  y = model.predict([[gg*5, fb*5, ins*5, 1, 1]])

  # print(i)
  # print(f"The current combination:\ngoogle {gg*5}\nfacebook {fb*5}\ninstagram {ins*5}\n")
  # print(f"Predicted sales = {y[0]}\n")

  ynew = model.predict([[ggnew*5, fbnew*5, insnew*5, 1, 1]])

  # print(f"The new combination:\ngoogle {ggnew*5}\nfacebook {fbnew*5}\ninstagram {insnew*5}\n")
  # print(f"New predicted sales = {ynew[0]}\n\n")

  if (ynew > y and fbnew <= 10 and insnew != 0):
    gg = ggnew
    fb = fbnew
    ins = insnew
  else:
    gg = gg
    fb = fb
    ins = ins
    
  ggnew = rd.randint(0, 20)
  fbnew = rd.randint(0, 20-ggnew)
  insnew = 20-ggnew-fbnew
  
  i = i + 1

print(f"The best combination after {i} iterations:\ngoogle {gg*5}\nfacebook {fb*5}\ninstagram {ins*5}\n")
print(f"Predicted sales = {model.predict([[gg*5, fb*5, ins*5, 1, 1]])[0]}")

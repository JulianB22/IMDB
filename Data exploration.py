#Import libraries
import pandas as pd

import seaborn as sns

import statsmodels.api as sm

#Load datasets
Ratings = pd.read_csv ('title_ratings.csv')
Basics = pd.read_csv ('title_basics_2018.csv')

#1.	How many 2018 films were categorized as a Comedy? 
print(
      str(sum(Basics['genres'].str.count('Comedy')| (Basics.year == '2018')))
      + " comedies in 2018")

#2.	How many 2018 films got a score of 8.0 or higher? 
 
#Join rating information to basic film information in new dataframe
df = pd.merge(Basics, Ratings, how="inner", on='tconst')

#Count 2018 films where average rating is greater or equal to 8.0
print(
      str(sum((df.year == '2018') | (df.averageRating >= 8.0))) 
      + " films in 2018 got a score of 8.0 or higher")

#3.	What was the best film of 2018?

#Return primaryTitle based on the max value of averageRating
print(
      str(df.loc[df['averageRating'] == max(df[(df.year == 2018)].averageRating), 
                 'primaryTitle'].iloc[0]) 
      + " was the best film of 2018")


#4.	Do audiences prefer longer films, or shorter films?
#Define predictor and response variables
y = df['averageRating']
x = df['runtimeMinutes']
#Add constant to predictor variables
x = sm.add_constant(x)

#Fit linear regression model
model = sm.OLS(y, x).fit()

#View model summary
print(model.summary())

#Plot scatter graph with regression line
sns.lmplot(x = "runtimeMinutes", y = "averageRating", data = df, ci=None, 
           scatter_kws={"s": 2}, line_kws={"color": "C3"})

# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')

dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# fitting linear regression to the dataset
lin_reg = lm(formula = Salary ~ .,
             data = dataset)
# fitting polynomial regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)
# Visualizing the Linear Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'red')+
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), color = 'blue')+
  ggtitle('Truth or Bluff(Linear Regression)')+
  xlab('Level')+
  ylab('Salary')

  
# Visualizing the polynomial regression results

library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'red')+
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), color = 'blue')+
  ggtitle('Truth or Bluff(Polynomial Regression)')+
  xlab('Level')+
  ylab('Salary')

# predicting a new result with linear regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# predicting a new result with Polynomial regression
y_pred = predict(poly_reg, data.frame(Level=6.5,
                                      Level2=6.5^2,
                                      Level3=6.5^3,
                                      Level4=6.4^4))

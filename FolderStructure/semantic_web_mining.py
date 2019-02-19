import sys
import numpy as ny
import pandas as ps
from sklearn.linear_model import Ridge

def LinReg(Xtrain, ytrain, Xtest=None, alpha=1):
    model = Ridge(fit_intercept=False, alpha=alpha)
    model.fit(Xtrain, ytrain)
    pred = model.predict(Xtrain)
    if Xtest is None:
        return pred
    else:
        pred_test = model.predict(Xtest)
    return pred, pred_test
    
def rmse(y, yhat):
    err = (y - yhat) *(y - yhat)
    return ny.sqrt(err.mean())   
    

def startProgram(folder):    
    train=folder+'ua.base'
    test=folder+'ua.test'
    item=folder+'u.itemcustominzed'
    user=folder+'u.user'
    datatrain = ps.read_table(train).drop('timestamp', axis=1)
    datatest = ps.read_table(test).drop('timestamp', axis=1)
    	
    			
    ratingtrain = datatrain.pivot(index='userID', columns='movieID').rating
    userIDtrain = ratingtrain.index
    movieIDtrain = ratingtrain.columns
    ratingtest = datatest.pivot(index='userID', columns='movieID').rating
    userIDtest = ratingtest.index
    movieIDtest = ratingtest.columns	
    movies_info = ps.read_table(item, sep='|')
  
    #movies_info.to_csv('E:\Lectures\\movies_info.csv')

    users_info = ps.read_table(user, sep='|').set_index('userID').\
                drop(['occupation','zipcode'], axis=1)
                 
    users_info['age_group'] = ps.cut(users_info.age, [0,18,29,45,100])
    users_info=users_info.drop('age', axis=1)
    users_info.head()    
    
                        
    joinedtrain = datatrain.join(users_info, on='userID').join(movies_info.iloc[:,3:], on='movieID')
    joinedtest = datatest.join(users_info, on='userID').join(movies_info.iloc[:,3:], on='movieID')    
    joinedtrain.sort_values(['userID', 'movieID'], inplace=True)
    joinedtest.sort_values(['userID', 'movieID'], inplace=True)
    
    #joined.to_csv('E:\\Lectures\\joined.csv')
  
    joinedtrain.head()
    joinedtest.head()
    avgtrain = joinedtrain.groupby(['movieID', 'gender', 'age_group']).rating.mean()
    avgtest = joinedtest.groupby(['movieID', 'gender', 'age_group']).rating.mean()
    avgtrain.head()
    avgtest.head()
    
    #avg.to_csv('E:\\Lectures\\avg.csv')
    
    newdatatrain = joinedtrain.join(avgtrain, rsuffix='_avg', on=['movieID', 'gender', 'age_group'])
    newdatatrain=newdatatrain.drop(['gender', 'age_group'], axis=1)
    newdatatrain.set_index(['userID', 'movieID'], inplace=True)
    
    newdatatest = joinedtest.join(avgtest, rsuffix='_avg', on=['movieID', 'gender', 'age_group'])
    newdatatest=newdatatest.drop(['gender', 'age_group'], axis=1)
    newdatatest.set_index(['userID', 'movieID'], inplace=True)
    
    
    #newdata.to_csv('E:\\Lectures\\newdata.csv')
    Y = newdatatrain.rating - newdatatrain.rating_avg
    X = newdatatrain.drop(['rating', 'rating_avg'], axis=1)
    
    
    #Y = newdatatrain.rating - newdatatrain.rating_avg
    Xtest = newdatatest.drop(['rating', 'rating_avg'], axis=1)
    
    #Y.to_csv('E:\\Lectures\\Y.csv')
    #X.to_csv('E:\\Lectures\\X.csv')
    
    where_are_NaNs = ny.isnan(X)
    X[where_are_NaNs]=21546
    
    alpha = 1.0
    err = []
    for user in userIDtrain:
        x, y = X.ix[user], Y.ix[user]
        pred = LinReg(x, y, alpha=alpha)
        #print pred
        err.append(rmse(y, pred))
    print('|***********************************|')
    print( 'RMSE =', sum(err) / len(err))

    
    model = Ridge(fit_intercept=False, alpha=1)
    user = 6
    model.fit(X.ix[user], Y.ix[user])
    yhat = model.predict(Xtest.ix[user])
            
    beta = ps.Series(model.coef_)
    count=0
    maxam=sys.float_info.min
    index='hello'
    beta.index = X.columns
    
    print ('|************Rating Categories************|')
    print (beta)
    allindex=list(beta.index)
    for var in beta:
        if(var > maxam):  
            maxam=var   
            index=allindex[count]             
        count=count+1  
    print ('Highest Category :', index)
    #print  maxam,index     
    
    moviesrec = movies_info['movieID'][(movies_info[index] ==1)]   
    print ('|****5 movies from the highest category********|')
    print (moviesrec.head())
    
def main():
    startProgram('ml-100k/')     
	

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()

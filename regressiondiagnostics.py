from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
import scipy.stats as stats
from numpy.linalg import inv


'''Library that can be used for both simple and multiple linear regression
diagnostics. It currently deals with normality and leverage points.

The "regress" method runs the linear regression with the user specifying
the target variable and the predictors. After running that method, the user
can call the "summary" attribute to see the results.

The "plot_errors" method plots the residuals in the model.

The "is_normal" method runs the Shapiro-Wilk test for normality, plots the
residuals with respect to a normal qq-plot, and also plots a histogram of the
residuals.

The "leverage" method requires the user to run the "hatter" method, and that
will print observations that have leverages more than twice the average and
give those point's leverages.

The "cooks_distance" method calculates all of the Cook's distances. They can
be viewed by accessing the "cooks_list" attribute after running this method.

The "re_regress" method removes the data point with the largest Cook's distance
and then refits the model.

The "structure" method removes each one of the predictors and replots the
residuals.

It it built on top of the statsmodels.api library.

An included example runs the library over the milk data set from the text
Regression Analysis by Example by Chatterjee and Hadi. They host a website
for the textbook here: http://www1.aucegypt.edu/faculty/hadi/RABE5/#Download
and the data can be downloaded (or copy-pasted or whatever)at this address:
http://www1.aucegypt.edu/faculty/hadi/RABE5/Data5/P004.txt

'''

class Diagnostics(object):

    def __init__(self, target, *predictors):
        '''Get the data in and get it into a matrix that we can work with'''

        self.target = np.array(target)
        const = np.ones([len(target), 1])

        predictors_list = []
        for predictor in predictors:
            predictors_list.append(predictor)

        predictors_array = np.array(predictors_list)

        # predictors comes out the wrong shape so I transpose them
        predictors_array = predictors_array.transpose()

        # this OLS does not automatically add a constant
        predictors_array = np.concatenate((const, predictors_array), axis = 1)
        self.predictors_array = predictors_array


    def regress(self): # add transformations (log, interaction)
        '''Runs the linear regression. If you want to see the output you
        run the "summary" method over the model'''

        self.model= OLS(self.target, self.predictors_array)
        self.results = self.model.fit()
        self.summary = self.results.summary()
        self.residuals = self.results.resid
        self.fitted = self.results.fittedvalues
        return self


    def plot_errors(self):
        '''Scatter plot of the fitted values vs. the residuals'''

        plt.scatter(self.fitted, self.residuals)
        plt.show()

    def is_normal(self):
        '''A series of Normal tests including qqplot, histogram
        and Shaprio-Wilks'''

        stats.probplot(self.residuals, dist = 'norm', plot = plt)
        plt.show()

        plt.hist(self.residuals)
        plt.show()

        print "The Shapiro-Wilk p-value is {}\n".format(stats.shapiro(self.residuals)[1])

    def hatter(self):
        '''Get the hat matrix for the model'''

        matrix = self.predictors_array
        interior = matrix.transpose().dot(matrix)
        left_side = matrix.dot(inv(interior))

        self.hat_matrix = left_side.dot(matrix.transpose())
        return self

    def leverage(self):
        '''Identify the leverages for each point and alert me to any that are
        more than 2p/n, where p is the number of predictors and n the number
        of observations.
        '''

        self.leverages = np.diag(self.hat_matrix)
        hat_average = self.hat_matrix.trace() / len(self.fitted)

        high_leverages = []
        for i in range(len(self.leverages)):
            if self.leverages[i] > 2*hat_average:
                high_leverages.append((self.leverages[i], i))

        # Pull names of observations out of the original data
        for leverage, i in high_leverages:
            print "Observation {0} has leverage {1}".format(i, leverage)

    def cooks_distance(self):
        '''Calculates the Cook's distances. To access the Cook's distances
        use the "cooks_list" attribute'''

        cooks_list = []
        for i in range(len(self.leverages)):
            # Cooks distance formula
            distance = 1/self.hat_matrix.trace()  * (self.residuals[i])**2 * \
                        ((self.leverages[i]) / (1 - self.leverages[i]))
            cooks_list.append([distance, i])

        self.cooks_list = cooks_list


    def re_regress(self): # ALLOW THIS TO TAKE OUT MORE THAN ONE
        # Return the percent change in the regression coefficients.
        '''Runs another regression, leaving out the observation with
        the largets Cook's distance
        '''

        self.biggest_cook = max(self.cooks_list)[1]
        new_data= np.delete(self.predictors_array, self.biggest_cook, 0)
        new_target = np.delete(self.target, self.biggest_cook, 0)
        self.new_model = OLS(new_target, new_data)
        self.new_results = self.new_model.fit()
        self.new_summary = self.new_results.summary()

    def structure(self): # Make the chart label which predictor was removed
        '''Reruns the regression by removing one of the predictor columns and
        then plots the residuals versus the target'''

        # The length of the transpose of the predictors array
        # gives the number of predictors in the model
        model_list = []
        for i in range(1, len(self.predictors_array.transpose())):
            temp_target = self.predictors_array[:, i].reshape([len(
                                self.predictors_array), 1])
            temp_model = OLS(temp_target,
                             np.delete(self.predictors_array, i, 1))
            temp_results = temp_model.fit()
            model_list.append(temp_results)
            del temp_target
            del temp_model

        for model in model_list:
            plt.scatter(model.fittedvalues, model.resid)
            plt.show()

# ADD A METHOD TO FIND AUTOCORRELATION DURBIN-WATSON
# ADD A METHOD TO FIND CORRELATION BETWEEN THE PREDICTORS
# ADD A METHOD TO PLOT AUTOCORRLATION OF A GIVEN DEGREE
#-------RUNNING MY SCRIPT OVER MY DATA SET----------------

if __name__ == "__main__":
    moo = pd.read_csv('~/Downloads/milk.txt', delimiter = '\t')
    moo.head()
    moo_preds = [moo.Previous, moo.Fat, moo.Protein, moo.Days, moo.Lactation, moo.I79]
    moo_model = Diagnostics(moo.CurrentMilk, *moo_preds)
    moo_model.regress()
    moo_model.plot_errors()
    moo_model.is_normal()
    moo_model.hatter()
    moo_model.leverage()
    moo_model.cooks_distance()
    moo_model.re_regress()
    print moo_model.summary
    print moo_model.new_summary
    moo_model.structure()

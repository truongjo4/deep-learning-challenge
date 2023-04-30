## Overview of the Analysis

This analysis was an attempt to create a binary classifer that can predict whether applicants will be successful in funded by Alphabet Soup.

A CSV of ~34,000 organisations that received funding from Alphabet Soup was provided. Within this dataset, approximately 53% of companies were considered 'successful' after receiving funding from Alphabet Soup (18k:16k split). An attempt to better classify these groups and lower risk of funding companies that would be unsuccessful was made with a dense neural network.

The dataset provided the following factors:
• EIN and NAME — Identification columns 
• APPLICATION_TYPE - Alphabet soup application type 
• AFFILIATION — Afflliated sector of industry 
• CLASSIFICATION — Government organisation classification 
• USE_CASE — Use case for funding 
• ORGANIZATION - Organisation type 
• STATUS - Active status 
• INCOME_AMT — lncome classification 
• SPECIAL_CONSlDERATlONS - Special considerations for application 
• ASK_AMT — Funding amount requested 

For our neural network, certain categorical variables within [APPLICATION_TYPE, CLASSIFICATION] were binned together given their low value counts. These, as well as other categorical variables (e.g. INCOME_AMT, AFFILIATION, USE_CASE etc.) were then turned into True/False dummies.

Numerical data (i.e., ASK_AMT) were scaled prior to analysis.

The target variable for this dataset was
• IS_SUCCESSFUL — Was the money used effectively (1 = success, 0 = unsuccessful)

The target of our model was a 75% accuracy score.

## Results

The initial model created had a single hidden layer with the following characteristics:
- 22 neurons in the first/second layer
- First/second layer used activation function 'relu'.
- Output neuron used 'sigmoid' as its function.
- Epoch of 100.

Model evaluation outputted the following:
- Loss = 0.55
- Accuracy = 0.73

To increase our model's efficacy, the following was attempted:
- Epoch increase from 100 -> 300
- Added a third layer with activation function 'relu' and 11 neurons.
- Removed all datapoints with a ASK_AMT of > 1,000,000 (culling outliers)

Only the second action provided marginal improvement (accuracy increase from 0.73 -> 0.74). Unfortunately, the target of 75% accuracy was not reached.

## Suggestions

In terms of the neural network:
- Perhaps other columns such as ASK_AMT could have been binned, or INCOME_AMT be re-binned given value count disparities.

Overall, given the relatively even split of the "IS_SUCCESSFUL" target variable, an accuracy score of 0.74 is still not too bad. It is possible that the data provided may not be sufficient for a neural network. Given the nature of the problem and the even splitting, perhaps a decision tree/random forests model may be better suited for this problem due to its openness. The client's need to justify funding of certain companies requires this openness, which is difficult in a neural network with possible many neurons/hidden layers.


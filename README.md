# ML-text-classification
Book genre classification using ML algorithms. 

## Dataset 
The goal here is to classify a book by genre using machine learning algorithms. <br />
The Dataset for the project has been taken from [CMU Book Summary Dataset](http://www.cs.cmu.edu/~dbamman/booksummaries.html)

## Used libraries
- pandas
- numpy
- tqdm
- matplotlib
- sklearn

## Flow
### Data extraction
The purpose is to extract data from file keeping only 4 columns
```
book_id = []
book_name = []
summary = []
genre = []
books = pd.DataFrame({'book_id': book_id, 'book_name': book_name,
                       'genre': genre, 'summary': summary})
```
### Data pre-processing
This stage is preparing *summary* column to be ready for data prediction. The following preprocessing text functions being used:
- to lowercase
- remove everething except letters
- remove whitespaces
- remove stop words
- text stemming (Porter) 
### Feature extraction
Using Chi-square feature selection K features with highest chi-squared statistics will selected
### Training model (using x10 cross validation)
Before train model split data into training and test sets (90%/10%)
- Naive Bayes Classifier (MultinomialNB) <br />
The results of testing hiper-parametr alpha of the MultinomialNB classifier
```
Best alpha:  {'alpha': 0.1}
Best score:  0.7516945493793848
Accurancy 0.7487603305785124
```
- SVM (using the RBF (Gaussian) kernel)
The model has the best evaluation parametrs with SVM hyperparameters: Gamma = 0.1, C = 10
```
# x10 cross validation
[0.73211009 0.74264706 0.71875    0.73713235 0.73161765 0.73161765
 0.73345588 0.72977941 0.73897059 0.75      ]
best model: 9
              precision    recall  f1-score   support

           0       0.72      0.65      0.68        66
           1       0.84      0.90      0.87       228
           2       0.70      0.46      0.55        81
           3       0.71      0.78      0.74       187
           4       0.59      0.60      0.60        43

    accuracy                           0.75       605
   macro avg       0.71      0.68      0.69       605
weighted avg       0.75      0.75      0.75       605

Accurancy 0.7537190082644628
```

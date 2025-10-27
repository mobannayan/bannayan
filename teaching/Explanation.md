Why Exploratory Data Analysis Important?

    Helps to understand the dataset by showing how many features it has, what type of data each feature contains and how the data is distributed. 
    Helps to identify hidden patterns and relationships between different data points which help us in and model building.
    Allows to identify errors or unusual data points (outliers) that could affect our results.
    The insights gained from EDA help us to identify most important features for building models and guide us on how to prepare them for better performance.
    By understanding the data it helps us in choosing best modeling techniques and adjusting them for better results

    1. Univariate Analysis: 
    Summary statistics like mean, median, mode,variance and standard deviation helps in describing the central tendency and spread of the data
    2. Bivariate Analysis

     focuses on identifying relationship between two variables to find connections, correlations and dependencies. It helps to understand how two variables interact with each other. Some key techniques include:
        Scatter plots which visualize the relationship between two continuous variables.
        Correlation coefficient measures how strongly two variables are related which commonly use Pearson's correlation for linear relationships.
        Cross-tabulation or contingency tables shows the frequency distribution of two categorical variables and help to understand their relationship.
        Line graphs are useful for comparing two variables over time in time series data to identify trends or patterns. 
        Covariance measures how two variables change together but it is paired with the correlation coefficient for a clearer and more standardized understanding of the relationship.
    3. Multivariate Analysis

    Multivariate Analysis identify relationships between two or more variables in the dataset and aims to understand how variables interact with one another which is important for statistical modeling techniques. It include techniques like:
        Pair plots which shows the relationships between multiple variables at once and helps in understanding how they interact. 
        Another technique is Principal Component Analysis (PCA) which reduces the complexity of large datasets by simplifying them while keeping the most important information.
        Spatial Analysis is used for geographical data by using maps and spatial plotting to understand the geographical distribution of variables.
        Time Series Analysis is used for datasets that involve time-based data and it involves understanding and modeling patterns and trends over time. Common techniques include line plots, autocorrelation analysis, moving averages and ARIMA models.

        Module 2: Supervised Learning 

Supervised learning algorithms are generally categorized into two main types: 

    Classification - where the goal is to predict discrete labels or categories 
    Regression - where the aim is to predict continuous numerical values.

Classification teaches a machine to sort things into categories. It learns by looking at examples with labels (like emails marked "spam" or "not spam"). After learning, it can decide which category new items belong to, like identifying if a new email is spam or not.
Types of Classification

When we talk about classification in machine learning, we’re talking about the process of sorting data into categories based on specific features or characteristics. There are different types of classification problems depending on how many categories (or classes) we are working with and how they are organized. There are two main classification types in machine learning:
1. Binary Classification

This is the simplest kind of classification. In binary classification, the goal is to sort the data into two distinct categories. Think of it like a simple choice between two options. Imagine a system that sorts emails into either spam or not spam. It works by looking at different features of the email like certain keywords or sender details, and decides whether it’s spam or not. It only chooses between these two options.
2. Multiclass Classification

Here, instead of just two categories, the data needs to be sorted into more than two categories. The model picks the one that best matches the input. Think of an image recognition system that sorts pictures of animals into categories like cat, dog, and bird. 

Basically, machine looks at the features in the image (like shape, color, or texture) and chooses which animal the picture is most likely to be based on the training it received. 
Binary vs Multi class classification -GeeksforgeeksBinary classification vs Multi class classification
3. Multi-Label Classification
In multi-label classification single piece of data can belong to multiple categories at once. Unlike multiclass classification where each data point belongs to only one class, multi-label classification allows datapoints to belong to multiple classes. A movie recommendation system could tag a movie as both action and comedy. The system checks various features (like movie plot, actors, or genre tags) and assigns multiple labels to a single piece of data, rather than just one.

Examples of Machine Learning Classification in Real Life 

Classification algorithms are widely used in many real-world applications across various domains, including:

    Email spam filtering
    Credit risk assessment: Algorithms predict whether a loan applicant is likely to default by analyzing factors such as credit score, income, and loan history. This helps banks make informed lending decisions and minimize financial risk.
    Medical diagnosis : Machine learning models classify whether a patient has a certain condition (e.g., cancer or diabetes) based on medical data such as test results, symptoms, and patient history. This aids doctors in making quicker, more accurate diagnoses, improving patient care.
    Image classification : Applied in fields such as facial recognition, autonomous driving, and medical imaging.
    Sentiment analysis: Determining whether the sentiment of a piece of text is positive, negative, or neutral. Businesses use this to understand customer opinions, helping to improve products and services.
    Fraud detection : Algorithms detect fraudulent activities by analyzing transaction patterns and identifying anomalies crucial in protecting against credit card fraud and other financial crimes.
    Recommendation systems : Used to recommend products or content based on past user behavior, such as suggesting movies on Netflix or products on Amazon. This personalization boosts user satisfaction and sales for businesses.

Classification Algorithms

Now, for implementation of any classification model it is essential to understand Logistic Regression, which is one of the most fundamental and widely used algorithms in machine learning for classification tasks. There are various types of classifiers algorithms. Some of them are : 

Linear Classifiers: Linear classifier models create a linear decision boundary between classes. They are simple and computationally efficient. Some of the linear classification models are as follows: 

    Logistic Regression
    Support Vector Machines having kernel = 'linear'
    Single-layer Perceptron
    Stochastic Gradient Descent (SGD) Classifier

Non-linear Classifiers: Non-linear models create a non-linear decision boundary between classes. They can capture more complex relationships between input features and target variable. Some of the non-linear classification models are as follows:

    K-Nearest Neighbours
    Kernel SVM
    Naive Bayes
    Decision Tree Classification
    Ensemble learning classifiers: 
    Random Forests, 
    AdaBoost, 
    Bagging Classifier, 
    Voting Classifier, 
    Extra Trees Classifier
    Multi-layer Artificial Neural Networks

    Regression in machine learning
    Last Updated : 13 Jan, 2025

    Regression in machine learning refers to a supervised learning technique where the goal is to predict a continuous numerical value based on one or more independent features. It finds relationships between variables so that predictions can be made. we have two types of variables present in regression:
        Dependent Variable (Target): The variable we are trying to predict e.g house price.
        Independent Variables (Features): The input variables that influence the prediction e.g locality, number of rooms.

    Regression analysis problem works with if output variable is a real or continuous value such as “salary” or “weight”. Many different regression models can be used but the simplest model in them is linear regression.
    Types of Regression

    Regression can be classified into different types based on the number of predictor variables and the nature of the relationship between variables:
    1. Simple Linear Regression 

    Linear regression is one of the simplest and most widely used statistical models. This assumes that there is a linear relationship between the independent and dependent variables. This means that the change in the dependent variable is proportional to the change in the independent variables. For example predicting the price of a house based on its size.
    2. Multiple Linear Regression

    Multiple linear regression extends simple linear regression by using multiple independent variables to predict target variable. For example predicting the price of a house based on multiple features such as size, location, number of rooms, etc.
    3. Polynomial Regression 

    Polynomial regression is used to model with non-linear relationships between the dependent variable and the independent variables. It adds polynomial terms to the linear regression model to capture more complex relationships. For example when we want to predict a non-linear trend like population growth over time we use polynomial regression.
    4. Ridge & Lasso Regression 

    Ridge & lasso regression are regularized versions of linear regression that help avoid overfitting by penalizing large coefficients. When there’s a risk of overfitting due to too many features we use these type of regression algorithms.
    5. Support Vector Regression (SVR)

    SVR is a type of regression algorithm that is based on the Support Vector Machine (SVM)algorithm. SVM is a type of algorithm that is used for classification tasks but it can also be used for regression tasks. SVR works by finding a hyperplane that minimizes the sum of the squared residuals between the predicted and actual values.
    6. Decision Tree Regression

    Decision tree Uses a tree-like structure to make decisions where each branch of tree represents a decision and leaves represent outcomes. For example predicting customer behavior based on features like age, income, etc there we use decison tree regression.
    7. Random Forest Regression

    Random Forest is a ensemble method that builds multiple decision trees and each tree is trained on a different subset of the training data. The final prediction is made by averaging the predictions of all of the trees. For example customer churn or sales data using this
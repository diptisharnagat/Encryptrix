*****Encryptrix Data Science Internship - Task Submission*****

__Internship Overview__

This repository contains the projects I completed as part of my Data Science Internship at Encryptrix. The internship involved working on real-world datasets and building machine learning models to solve practical business problems. Out of the five tasks provided, I successfully completed three tasks, which are outlined below.


__Completed Tasks:-__

1. __Titanic Survival Prediction__
- Objective: Predict whether a passenger on the Titanic survived based on passenger data like age, gender, ticket class, etc.

- Dataset: Titanic Dataset (linked in the project brief).

- Implementation:

      1.Import Libraries:

       Key libraries like pandas, numpy, matplotlib, seaborn, sklearn are imported for data handling, visualization, and model building.

      2.Load Dataset: 

       The Titanic dataset is loaded from a CSV file using pandas.

      3.Exploratory Data Analysis (EDA):

      - head(), info(), and describe() functions provide an overview of the data.
  
      - A countplot is used to visualize the distribution of survived vs. non-survived passengers.

      - The total number of survivors is printed.

     4.Handling Missing Values:

       Missing values in the Sex column are filled using the mode (most frequent value).

     5.Feature Engineering:

    - A new feature Title is created by extracting titles from the Name column.

    - Sex_Name combines the Sex and Name columns.
    
    -  Categorical columns like Sex, Name, Embarked, Title, and Sex_Name are label-encoded into numeric values.
  
     6.Split Data:

      The features X and target y (Survived) are defined, excluding irrelevant columns like Cabin, Ticket, Fare, and Age. Data is split into training and testing sets.

     7.Logistic Regression Model:

      A logistic regression model is created, trained on the training data, and used to predict survival on the test set.

     8.Model Evaluation:

     The accuracy of the model is calculated and displayed using accuracy_score.

- Methodology: Implemented a classification model using Logistic Regression to predict survival based on features like Age, Fare, Sex, Pclass, etc.

- Key Techniques: Data preprocessing (handling missing values, encoding categorical features), model training, evaluation using accuracy score.

- Outcome: Achieved an accuracy of over 80% on the test dataset. This task helped me understand basic classification techniques and exploratory data analysis.

- Link to project files:







2. __Movie Rating Prediction__


- Objective: Predict movie ratings based on features like genre, director, and actors.

- Dataset: Movie Data (linked in the project brief).

- Implementation:

   1.Data Preparation:

   The dataset is loaded, and columns like Year, Votes, and Duration are converted to numeric.
Missing values are handled by filling them with median or default values (e.g., 0 for votes, median for Year and Duration, mean for Rating).
Categorical variables like Genre, Director, and actors are one-hot encoded to convert them into numeric features for the model.

   2.Model Training:

   The data is split into training and testing sets (80% train, 20% test).
A Random Forest Regressor is trained on the training data to predict the movie Rating based on the other features.

    3.Model Evaluation:

     Predictions (y_pred) are made on the test set (y_test), and the performance is measured using Mean Squared Error (MSE) and R² score.
In this case, the MSE is 0.621 (lower is better), and the R² score is 0.35 (closer to 1 is better), indicating a modest fit.

  4.Visualization:

     A scatter plot shows the actual vs. predicted movie ratings. A diagonal red line represents perfect predictions, and the blue points indicate how closely the model's predictions match actual ratings.
The scatter plot helps visualize prediction accuracy. If points are close to the diagonal line, the predictions are good.
Finally, a plt.figure() call is made to resize the scatter plot, making it clearer by showing how well the model predicted ratings.

- Methodology: Built a Random Forest Regression model to estimate movie ratings. Applied feature engineering, handled missing data, and used one-hot encoding for categorical variables like Genre and Director.

- Key Techniques: Regression modeling, one-hot encoding, feature selection.

- Outcome: Achieved an R² score of 0.35 and fine-tuned the model for better performance. This task enhanced my understanding of regression techniques and model evaluation.

- Link to project files:

3. __Sales Prediction Using Python__

- Objective: Forecast product sales based on advertising spend across multiple channels like TV, Radio, and Newspaper.

- Dataset: Sales Data (linked in the project brief).

- Implementation:

     1.Data Loading:

      The dataset SalesData.csv is loaded into a DataFrame (df). The columns are TV, Radio, Newspaper (advertising spend), and Sales (the dependent variable, i.e., the target).

     2.Feature Selection:

      The features X include advertising expenditures for TV, Radio, and Newspaper.
      The target variable y is the Sales, which we want to predict.

      3.Data Splitting:

       The data is split into training (80%) and testing (20%) sets using train_test_split, with X_train and y_train used to train the model, and X_test and y_test used for evaluation.

      4.Model Creation and Training:

        A Linear Regression model is instantiated and trained on the training data (X_train, y_train) using the fit() method. This trains the model to understand the relationship between advertising spend and sales.

      5.Prediction:

        The trained model is used to make predictions on the test data (X_test) using the predict() method, storing the predictions in y_pred.

      6.Model Evaluation:

  The Mean Squared Error (MSE) and R² score are calculated to assess the model’s performance:
MSE measures the average squared difference between actual and predicted values (lower MSE means better performance). In this case, it's 2.91, indicating a small error.
R² score shows how well the model explains the variability in the data (closer to 1 is better). An R² score of 0.91 indicates the model explains 91% of the variability in sales based on advertising spend.


     7.Visualization:

       A scatter plot is created to visualize the actual sales (y_test) vs. predicted sales (y_pred).
The red dashed line represents perfect predictions (where actual equals predicted). Points closer to the line indicate better predictions.

- Methodology: Used Linear Regression to predict sales based on advertising budgets. Split data into training and test sets, evaluated the model using Mean Squared Error (MSE) and R² score.

- Key Techniques: Data splitting, linear regression modeling, performance evaluation using MSE and R².

- Outcome: Achieved an MSE of 2.91 and an R² score of 0.91, indicating strong model performance. This task deepened my knowledge of linear regression and real-world sales data analysis.

- Link to project files:

__Instructions for Review:__

Please refer to the source code files for each task (Titanic_Survival_Prediction.ipynb, Movie_Rating_Prediction.ipynb, Sales_Prediction.ipynb), and the video demonstration available on LinkedIn.


__#encryptix, #internship, #webdevelopment__

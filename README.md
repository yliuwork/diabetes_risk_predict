### Diabetes Risk Prediction

    **Shirley Liu**

#### Executive summary
    This project aims to predict the risk of diabetes using survey data from the Behavioral Risk Factor Surveillance System (BRFSS) collected by the CDC in 2015. The dataset, sourced from Kaggle, contains responses from 253,680 individuals and features 21 variables. Through data preprocessing, exploratory data analysis, and machine learning modeling, we develop predictive models to classify individuals into three categories: no diabetes, prediabetes, and diabetes. The results identify key risk factors that are most predictive of diabetes risk, providing actionable insights for healthcare providers and public health officials.

#### Rationale
    Diabetes is one of the most prevalent chronic diseases in the United States, affecting millions and imposing a significant financial burden on the economy. Early diagnosis and risk assessment are crucial for preventing complications and improving patient outcomes. By leveraging survey data, this research aims to provide an efficient method for predicting diabetes risk, thus enabling early interventions and personalized care.

#### Research Question
    Can survey questions from the Behavioral Risk Factor Surveillance System (BRFSS) provide accurate predictions of whether an individual has diabetes or is at high risk of developing diabetes?

#### Data Sources
    The primary data source for this project is the diabetes-related dataset from Kaggle, which includes survey data from the Behavioral Risk Factor Surveillance System (BRFSS) collected by the CDC in 2015. This dataset contains responses from 253,680 individuals and features 21 variables. https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data

#### Methodology
    1. Data Preprocessing: Data cleaning and handling missing values, feature selection, and engineering.
    2. Exploratory Data Analysis (EDA): Statistical analysis and visualization of survey responses, analysis of class distribution and imbalance
       2.1 Analyze the distribution of the data of each feature using countplot
        2.1.1 Health Behaviors:Most individuals in the dataset engage in positive health behaviors, including regular physical activity, fruit and    vegetable consumption, and healthcare check-ups.The majority do not engage in heavy alcohol consumption, and the smoking status is nearly evenly split between smokers and non-smokers.
        2.1.2 Chronic Conditions:Conditions such as high blood pressure and high cholesterol are prevalent in a significant portion of the population.
        2.1.3 Stroke and heart disease/attack are less common but present in a notable minority.
        Socioeconomic Factors:There is a diverse distribution of education and income levels, with significant representation across all levels.
        Both lower and higher income and education brackets are well-represented in the dataset.
        2.1.4 Accessibility:Most individuals have access to healthcare, but a small subset avoids seeing a doctor due to cost concerns.
            Regular cholesterol checks are common among the population.
        2.1.5 Gender Distribution:
            The dataset shows a balanced gender distribution, providing an equitable basis for analysis across genders.

        2.2 Analyze data range of each feature using boxplot
         2.2.1 BMI and Health-Related Features:BMI shows a significant number of outliers, indicating a subset of individuals with considerably higher BMI values.Both mental and physical health variables exhibit a wide range and numerous outliers, suggesting varied health conditions across the population.
         2.2.2 Age and Socioeconomic Factors:The age distribution is fairly symmetric, encompassing a wide range of ages.
    Education and income levels also show wide ranges, highlighting the diversity in socioeconomic status within the population.
         2.2.3Binary Health Indicators:Many health-related features, such as high blood pressure (HighBP), high cholesterol (HighChol), and smoking status, are binary.These indicators clearly show the presence or absence of specific health conditions or behaviors.

        2.3 Analyze the relationship among a few critical features using pair plots and correlation matrix
         2.3.1 Correlation Matrix:Positive correlations between age and chronic conditions such as diabetes, high blood pressure, and high cholesterol.High BMI is correlated with higher likelihoods of diabetes and high blood pressure.
    General health is strongly related to both mental and physical health.
         2.3.2 Pair Plot Analysis:Shows relationships between features like HighBP, HighChol, BMI, Age, and Sex, highlighting clustering of individuals with diabetes in relation to these factors.

        2.4 Distribution of Diabetes:
    Highlights the imbalanced nature of the dataset, with most individuals not having diabetes and smaller portions having pre-diabetes or diabetes.

    3. Modeling:
        Training and evaluating multiple classification models were conducted, including Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines (SVM). The process involved grid search for hyperparameter tuning and cross-validation for performance evaluation. The results of the training were as follows:
        Logistic Regression:
            Successfully trained and evaluated.
            Best parameters: {'C': 1}.
        Decision Trees:
            Successfully trained and evaluated.
            Best parameters: {'max_depth': 10}.
        Random Forests:
            Successfully trained and evaluated.
            Best parameters: {'n_estimators': 300}.
        Support Vector Machines (SVM):
            Training was not completed successfully after running for more than 20 hours using paid Google Colab service.
            Suspected cause: Highly imbalanced training data leading to increased computational complexity and prolonged convergence time.
    4. Evaluation:
        Performance metrics were used to evaluate and compare the models, including accuracy, precision, recall, F1-score, and ROC-AUC. The evaluation of the successfully trained models is summarized in the result section. 


#### Results
    The analysis aimed to determine if survey responses from the BRFSS can accurately predict diabetes risk. Various machine learning models were evaluated to assess their predictive capabilities. The results are summarized below:

    Logistic Regression:
        Best Parameters: {'C': 1}
        Accuracy: 84.57%
        Precision: 79.81%
        Recall: 84.57%
        F1-Score: 80.73%
        ROC AUC Score: 78.28%
        The model performed well for the majority class (No Diabetes) but struggled to accurately predict the minority classes (Pre-Diabetes and Diabetes).
        
    Decision Tree:
        Best Parameters: {'max_depth': 10}
        Accuracy: 84.56%
        Precision: 80.06%
        Recall: 84.56%
        F1-Score: 80.99%
        ROC AUC Score: 76.19%
        This model also showed high performance for the majority class but had poor performance for the minority classes.
    
    Random Forest:
        Best Parameters: {'n_estimators': 300}
        Accuracy: 84.33%
        Precision: 79.69%
        Recall: 84.33%
        F1-Score: 80.79%
        ROC AUC Score: 75.04%
        Despite being computationally intensive, the Random Forest model provided the best overall performance among the evaluated models but still had difficulty with minority class predictions.
        Support Vector Machine (SVM):
    SVM:
        The training and grid search for the SVM model were too computationally intensive to complete within a reasonable time frame, even using paid Google Colab resources.
    
    Key Findings
        Predictive Capabilities: Survey responses from the BRFSS can predict diabetes risk with reasonable accuracy, particularly for the majority class (No Diabetes).
        Key Predictors: BMI, age, physical activity, and general health status were identified as key predictors of diabetes risk.
        Class Imbalance: Three models performed well for the majority class but struggled with accurately predicting the minority classes (Pre-Diabetes and Diabetes). It also impacted the SVM model, leading to prolonged training times and inability to complete the grid search. All these issues highlights the need for handling class imbalance effectively.

#### Next steps
    Model Refinement:Further tuning of hyperparameters and exploration of additional features to improve model accuracy and performance on minority classes.Consideration of advanced techniques to handle class imbalance, such as oversampling, undersampling, and class weighting.
    Data Expansion:Incorporating additional years of BRFSS (Behavioral Risk Factor Surveillance System) data to enhance model robustness and accuracy. More data can help improve the model's ability to generalize and perform better on minority classes.
    Alternative Models:Exploration of other machine learning algorithms such as Gradient Boosting, XGBoost, and ensemble methods that might handle class imbalance better and provide improved performance.

#### Outline of project

- [Link to notebook 1]()
- [Link to notebook 2]()
- [Link to notebook 3]()


##### Contact and Further Information
    yliuwork@gmail.com


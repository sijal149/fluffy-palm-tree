import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import ttest_ind
from sklearn.tree import plot_tree

dataset = pd.read_csv("/Users/sijalrupakheti/Desktop/SleepDataAnalysis/Sleep_health_and_lifestyle_dataset.csv")


dataset['Sleep Disorder'] = dataset['Sleep Disorder'].fillna('No Disorder')
#print(dataset)
missing_columns = dataset.columns[dataset.isnull().any()]
missing = dataset.isnull().any().sum()
# print(missing_columns)
# print(missing)

#Age Distribution-Histogram
plt.figure(figsize=(8,6))
sns.histplot(dataset['Age'], bins=20, kde=True, color='Blue')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.close()


#Occupation vs Sleep- Box Plot
plt.figure(figsize=(14,6))
sns.boxplot(x = 'Occupation', y = 'Sleep Duration', data=dataset)
plt.xlabel('Occupation')
plt.ylabel('Average Sleep Hours')
plt.title('Average sleeping hours by occupation')
plt.close()


#barplots for categorical variables - Bar Plots
plt.figure(figsize=(8, 6))
sns.countplot(x='Sleep Disorder', data=dataset)
plt.title('Count of Sleep Disorders')
plt.xlabel('Sleep Disorder')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.close()

#line plot for identifying trends - Line Plot 
average_sleep_by_age = dataset.groupby('Age')['Sleep Duration'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='Age', y='Sleep Duration', data=average_sleep_by_age, marker='o')
plt.title('Average Hours of Sleep by Age')
plt.xlabel('Age')
plt.ylabel('Average Hours of Sleep')
plt.close()


#Mapping gender and bmi category to numerical scale 
dataset['Gender'] = dataset['Gender'].map({'Male': 1, 'Female': 0})

dataset['BMI Category'] = dataset['BMI Category'].replace('Normal Weight', 'Normal')
bmi_mapping = {
    'Underweight': 0, 
    'Normal': 1, 
    'Overweight': 2, 
    'Obese': 3}
dataset['BMI Category'] = dataset['BMI Category'].map(bmi_mapping)

mapping = {
    'Sleep Apnea': 1,
    'Insomnia': 1,
    'No Disorder': 0
}

dataset['Sleep Disorder'] = dataset['Sleep Disorder'].map(mapping)

#violin-plot : distribution of sleep duration vs sleep disorder
plt.figure(figsize=(10, 6))
sns.violinplot(x='Sleep Disorder', y='Sleep Duration', data=dataset)
plt.title('Distribution of Sleep Duration by Sleep Disorder')
plt.xlabel('Sleep Disorder')
plt.ylabel('Sleep Duration')
plt.close()

#SPLIT BP INTO SYSTOLIC AND DIASTOLIC
split_bp = dataset['Blood Pressure'].str.split('/', expand = True)
dataset['Systolic BP'] = split_bp[0]
dataset['Diastolic BP'] = split_bp[1]
dataset = dataset.drop(columns=['Blood Pressure'])
dataset['Systolic BP'] = pd.to_numeric(dataset['Systolic BP'], errors='coerce')
dataset['Diastolic BP'] = pd.to_numeric(dataset['Diastolic BP'], errors='coerce')
non_numeric_colums = dataset.select_dtypes(include=['object']).columns
# print(non_numeric_colums)
# print(dataset.head())

#correlation_matrix_heatmap
dataset_for_corr = dataset.drop(columns=['Occupation'])
corr_matrix = dataset_for_corr.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.close()

#correlation_matrix_clustermap
sns.clustermap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Clustered Correlation Matrix')
plt.close()

#PA vs SDu scattterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Physical Activity Level', y='Sleep Duration', data=dataset)
plt.title('Physical Activity Level vs. Sleep Duration')
plt.xlabel('Physical Activity Level')
plt.ylabel('Sleep Duration')
plt.show()

#correlation between PA and sleep duration
correlation = dataset['Physical Activity Level'].corr(dataset['Sleep Duration'])
#print(f"Correlation between Physical Activity Level and Sleep Duration: {correlation:.2f}")

#t-test between high and low Physical Activity Levels
median_activity = dataset['Physical Activity Level'].median()
high_activity = dataset[dataset['Physical Activity Level'] > median_activity]['Sleep Duration']
low_activity = dataset[dataset['Physical Activity Level'] <= median_activity]['Sleep Duration']

t_stat, p_value = ttest_ind(high_activity, low_activity)
# print(f"T-test results for Sleep Duration between High and Low Physical Activity Levels:")
# print(f"T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

#Linear Regression Model- prediction of Sleep Duration based on Physical Activity Level
X = dataset[['Physical Activity Level']]  
y = dataset['Sleep Duration']             

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"Coefficients: {model.coef_}")
# print(f"Intercept: {model.intercept_}")

# Scatter plot of actual vs predicted values-linear regression
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.plot(X_test, y_pred, color='black', linewidth=2, label='Regression Line')
plt.xlabel('Physical Activity Level')
plt.ylabel('Sleep Duration')
plt.legend()
plt.title('Linear Regression: Sleep Duration vs Physical Activity Level')
plt.close()


#Exporting processed dataset
#dataset.to_csv('/Users/sijalrupakheti/Desktop/SleepDataAnalysis/Processed_Sleep_health_and_lifestyle_dataset.csv')






import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


# Задача 3

df = pd.read_csv('appl_score_sample.csv', sep=';')
print(df.head())
print(df.shape)

x = df.drop('Target', axis=1)
y = df['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


#Линейная регрессия

model = LinearRegression()
model.fit(x_train, y_train)

prediction_l = model.predict(x_test)
err_l = (mean_squared_error(y_test, prediction_l))**0.5

print(f'Корень из квадратичной ошибки, линейная регрессия: {err_l}')


#Линейная регрессия с полиномиальными признаками

pf = PolynomialFeatures(degree=2)

pf.fit(x_train)

x_train_pf = pf.transform(x_train)
x_test_pf = pf.transform(x_test)

model.fit(x_train_pf, y_train)

prediction_f = model.predict(x_test_pf)

err_f = (mean_squared_error(y_test, prediction_f))**0.5

print(f'Корень из квадратичной ошибки, линейная регрессия c полиномиальными признаками (степень 2): {err_f}')

print('''В задании передан не достаточный для использования
полиномиальных признаков объем данных (даже для полинома 2й степени).
Использование полиномиальных признаков вызывает
переобучение модели, квадратичная ошибка увеличивается.''')


# Задача 5

# 5.1
subset_1 = df.iloc[:20, [0, 4]]

# 5.2
subset_2 = df[(df['STANDING_IN_MONTHS_LAST'] > 50) & (df['SEX'] == 1)]

# 5.3
df['CI'] = df['DCI'] + df['UCI']
df['CI_ln'] = df['CI'].apply(lambda x: np.log(x) if x > 0 else None)

# 5.4
subset_4 = df.groupby('EDUCATION')['CI'].mean().reset_index()
subset_4.rename(columns={'CI': 'MEAN_CI'}, inplace=True)

# 5.5
median_transport = df[df['TRANSPORT_AMOUNT'] >= 0]['TRANSPORT_AMOUNT'].median()
df['TRANSPORT_AMOUNT'] = df['TRANSPORT_AMOUNT'].apply(lambda x: median_transport if x < 0 else x)

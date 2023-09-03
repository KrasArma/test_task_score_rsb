
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sb

warnings.filterwarnings('ignore')


# Задача 3

df = pd.read_csv('appl_score_sample.csv', sep=';')
print(df.head())
print(df.shape)


corr_matrix = df.corr()
corr = df.corr()
sb.heatmap(corr, cmap="Blues", annot=True)
plt.show();

print('Два наиболее зависимых признака - DCI и PROFIT_FAMILY')
print('EDUCATION имеет наибольшую по модулю корреляцию с таргетом')

print(f'Классы сбаллансированы, можно использовать метрику класса accuarcy: {y.value_counts()}')

x = df.drop('Target', axis=1)
y = df['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123)
lr = LogisticRegression()

lr.fit(x_train, y_train)

prediction = lr.predict(x_test)

print(f'LogisticRegression: {accuracy_score(y_test, prediction)}')

print(lr.coef_, lr.intercept_)


for i in range(1, 10, 2):
  
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(x_train, y_train)
  
    prediction_n = clf.predict(x_test)
  
    print(f'KNN (k= {i}): {accuracy_score(y_test, prediction_n)}')

print('''На предоставленных в задании данных линейная модель работает более качественно, чем KNN.
Возжно, проблема в недостаточном объеме обучающей выборки. 
Кроме того, точность предсказания KNN возрастает при уменьшении k.''')


scaler = StandardScaler()
scaler.fit(x_train)

x_train = pd.DataFrame(scaler.transform(x_train), columns=x.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x.columns)

model = SVC(kernel='linear', probability=True)
model.fit(x_train, y_train)

prediction_r = model.predict_proba(x_test)[:,1]

print(f'SVC по ROC AUC: {roc_auc_score(y_test, prediction_r)}')


fpr, tpr, threshold = roc_curve(y_test, prediction_r)
roc_auc = auc(fpr, tpr)


plt.title('Receiver operating characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')

plt.xlim([0,1])
plt.ylim([0,1])

plt.ylabel('True positive rate')
plt.xlabel('False positive rate')

plt.show()


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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xlrd as xlrd
from IPython.display import display

"""**Загрузка данных:**"""

#для загрузки через колаб, выбрать файл "exam"

# from google.colab import files
# uploaded = files.upload()

exam = pd.ExcelFile(open('exam.xlsx','rb'))

#для загрузки через anaconda

exam = pd.ExcelFile('exam.xlsx')

#создадим отдельный dataframe для каждого листа excel

var_1 = exam.parse('Вариант 1')
var_2 = exam.parse('Вариант 2')
var_3 = exam.parse('Вариант 3')
var_4 = exam.parse('Вариант 4')
conse = exam.parse('Соответствие вариантов')[0:40]

display(var_1.head(3))

"""**Получение рубежных баллов для квантильных оценок:**"""

#соберем статистику количества правильных ответов для каждого студента
right_answers = pd.DataFrame()

for x in [var_1,var_2,var_3,var_4]:
    answers = x['Верный ответ'].values
    x_t = x.iloc[:,2:].T
    right_answers = right_answers._append(pd.DataFrame((x_t == answers).sum(axis = 1)))

right_answers = right_answers.reset_index()
right_answers.columns = ['ФИО','правильные ответы']

#составим таблицу рубежных баллов для квантильных оценок
quantiles = [np.quantile(right_answers['правильные ответы'], a, interpolation = 'lower') for a in [8/9,7/9,2/3,1/2,1/3,1/6]]

est = pd.DataFrame({'левая граница':quantiles +
                                    [8,5,3,1,0],
                    'правая граница':[var_1.iloc[:,0].size] +
                                     quantiles +
                                     [8,5,3,1],
                    '10 балльная шкала':range(10,-1,-1),
                    'традиционная шкала':['отлично' for i in range(8,11)] +
                                         ['хорошо' for i in range(6,8)] +
                                         ['удовлетворительно' for i in range(4,6)] +
                                         ['неудовлетворительно' for i in range(0,4)]})

est['интервал ответов'] = est[['левая граница','правая граница']].astype(str).apply(lambda x: ' - '.join(x), axis = 1)

#таблица рубежных баллов и соответствующих оценок
display(est[['интервал ответов','10 балльная шкала','традиционная шкала']])

"""**Присвоение полученных баллов и оценки каждому студенту:**"""

def estimate_1(x):
    place = np.array(est['левая граница'] <= x) * np.array(est['правая граница'] > x)
    return est[place][['10 балльная шкала']].iloc[0,0]

def estimate_2(x):
    place = np.array(est['левая граница'] <= x) * np.array(est['правая граница'] > x)
    return est[place][['традиционная шкала']].iloc[0,0]

right_answers['баллы'] = right_answers.loc[:,'правильные ответы'].apply(estimate_1)
right_answers['оценка'] = right_answers.loc[:,'правильные ответы'].apply(estimate_2)

display(right_answers)

"""**Результат в виде отдельных таблиц для оценок «отлично», «хорошо», «удовлетворительно», «неудовлетворительно»:**"""

#докажем, что студентов с неудовлетворительной оценкой не существует
print('наименьшая оценка на курсе:',estimate_2(right_answers['правильные ответы'].min()))

#упорядочим исходную таблицу по столбцу баллов, а затем по столбцу ФИО
ra_sorted = right_answers.sort_values(by = ['баллы','ФИО'], ascending = [False,True])

#создадим таблицы для соответствующих оценок
great = ra_sorted[ra_sorted['оценка'] == 'отлично'][['ФИО','баллы','оценка']].reset_index(drop = True)
good = ra_sorted[ra_sorted['оценка'] == 'хорошо'][['ФИО','баллы','оценка']].reset_index(drop = True)
not_bad = ra_sorted[ra_sorted['оценка'] == 'удовлетворительно'][['ФИО','баллы','оценка']].reset_index(drop = True)

#таблицы соответствующих оценок

print("\nОТЛИЧНО")
display(great)

print("\nХОРОШО")
display(good)

print("\nУДОВЛЕТВОРИТЕЛЬНО")
display(not_bad)

print("\nотметок НЕУДОВЛЕТВОРИТЕЛЬНО на экзамене не было")

"""**Анализ статистики ответов по различным дисциплинам:**"""

#переназначим номера вопросов в 2,3 и 4 вариантах, ориентируясь на вариант 1
var_2['Номер вопроса'] = conse['Вариант 2'].values
var_3['Номер вопроса'] = conse['Вариант 3'].values
var_4['Номер вопроса'] = conse['Вариант 4'].values

#отсортируем по порядку
var_1_q = var_1.reset_index(drop = True)
var_2_q = var_2.sort_values(by = 'Номер вопроса').reset_index(drop = True)
var_3_q = var_3.sort_values(by = 'Номер вопроса').reset_index(drop = True)
var_4_q = var_4.sort_values(by = 'Номер вопроса').reset_index(drop = True)

#посчитаем количество людей, ответивших верно на каждый из вопросов суммарно по всем вариантам
questions = pd.DataFrame({'номер вопроса':var_1['Номер вопроса'],
                          'правильные ответы':0})

for x in [var_1_q,var_2_q,var_3_q,var_4_q]:
    answers = x['Верный ответ'].values
    x_t = x.iloc[:,2:].T
    questions['правильные ответы'] = questions['правильные ответы'] + (x_t == answers).sum(axis = 0)

#поделим таблицу на дисциплины и упорядочим по выполняемости (сложности вопросов), а затем по номеру вопроса
micro = questions[0:10].sort_values(by = ['правильные ответы','номер вопроса'],
                                    ascending = [False,True]).set_index('номер вопроса')
macro = questions[10:20].sort_values(by = ['правильные ответы','номер вопроса'],
                                     ascending = [False,True]).set_index('номер вопроса')
econo = questions[20:30].sort_values(by = ['правильные ответы','номер вопроса'],
                                     ascending = [False,True]).set_index('номер вопроса')
finan = questions[30:40].sort_values(by = ['правильные ответы','номер вопроса'],
                                     ascending = [False,True]).set_index('номер вопроса')

#посчитаем процент выполняемости конкретного задания среди студентов (процент правильно ответивших)
n = right_answers['ФИО'].size

for x in [micro,macro,econo,finan]:
    x['выполняемость'] = (x.iloc[:,0]/n*100).round(2).astype(str) + "%"

#результаты анализа статистики ответов по различным дисциплинам

print("дисциплина: микроэкономика",
      "\nнаиболее сложный вопрос: #",micro.index[9],
      "\nнаиболее простой вопрос: #",micro.index[0],
      "\nвыполняемость дисциплины в целом: ",(micro.iloc[:,0].sum(axis = 0)*10/n).round(2),"%")

micro = micro.reset_index()
display(micro)

print("дисциплина: макроэкономика",
      "\nнаиболее сложный вопрос: #",macro.index[9],
      "\nнаиболее простой вопрос: #",macro.index[0],
      "\nвыполняемость дисциплины в целом: ",(macro.iloc[:,0].sum(axis = 0)*10/n).round(2),"%")

macro = macro.reset_index()
display(macro)

print("дисциплина: эконометрика",
      "\nнаиболее сложный вопрос: #",econo.index[9],
      "\nнаиболее простой вопрос: #",econo.index[0],
      "\nвыполняемость дисциплины в целом: ",(econo.iloc[:,0].sum(axis = 0)*10/n).round(2),"%")

econo = econo.reset_index()
display(econo)

print("дисциплина: финансы",
      "\nнаиболее сложный вопрос: #",finan.index[9],
      "\nнаиболее простой вопрос: #",finan.index[0],
      "\nвыполняемость дисциплины в целом: ",(finan.iloc[:,0].sum(axis = 0)*10/n).round(2),"%")

finan = finan.reset_index()
display(finan)

"""**Визуализация статистики ответов по различным дисциплинам:**"""

#данные для графика
exam_plot = pd.concat({'микроэкономика':micro['правильные ответы'],
                       'макроэкономика':macro['правильные ответы'],
                       'эконометрика':econo['правильные ответы'],
                       'финансы':finan['правильные ответы']}, axis = 1)/n*100

plt.figure(figsize = (11,7))
ax = sns.boxplot(data = exam_plot);
ax = sns.swarmplot(data = exam_plot, color = "black");
ax.set(xlabel = "дисциплина", ylabel = "процент правильных ответов");

plt.figure(figsize = (11,7))
plt.show()
ax = sns.violinplot(data = exam_plot, inner = None, cut = 0.5);
ax = sns.swarmplot(data = exam_plot, color = "black");
ax.set(xlabel = "дисциплина", ylabel = "процент правильных ответов");
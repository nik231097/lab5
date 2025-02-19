import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy.stats import pearsonr


# Функция для чтения данных из файла CSV с заменой значений в столбце gender
def read_data(file_path):
    data = pd.read_csv(file_path, sep=',')  # Чтение файла с разделителем запятая

    # Замена значений в столбце 'gender': 'male' -> 0, 'female' -> 1
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'Male' else 1)


    y = data['Calories_Burned']
    X = data.drop(columns=['Calories_Burned', 'Workout_Type'])
    return X, y


# Функция для выполнения линейной регрессии
def linear_regression(X, y):
    X_train = sm.add_constant(X)  # Добавляем константу для модели
    model = sm.OLS(y, X_train).fit()
    return model


# Функция для анализа значимости факторов
def select_significant_factors(model, alpha):
    summary = model.summary2().tables[1]
    significant_factors = summary[summary['P>|t|'] < alpha].index.tolist()
    if "const" in significant_factors:
        significant_factors.remove("const")
    return significant_factors


# Функция для оценки корреляции между факторами и откликом
def correlation_analysis(X, y):
    correlations = {}
    for col in X.columns:
        corr, _ = pearsonr(X[col], y)
        correlations[col] = corr
    return correlations


# Функция для вычисления ошибки RMSE
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Проверка мультиколлинеарности между факторами.
def check_multicollinearity(X, threshold=0.8):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [
        (column, row)
        for column in upper.columns
        for row in upper.index
        if upper.loc[row, column] > threshold
    ]
    return high_corr

def predict_new(model, new_data_path):
    """
    Предсказание новых значений отклика на основе введённых факторов.
    """
    try:
        new_data = pd.read_csv(new_data_path, sep=None, engine="python")
        predictions = model.predict(new_data)
        return predictions
    except Exception as e:
        if len(new_data_path) >0:
            print(f"Ошибка при чтении файла с новыми данными: {e}")
        sys.exit(1)


# Основная функция программы
def main(file_path, alpha):
    # Чтение данных
    X, y = read_data(file_path)


    # Оценка параметров ЛМФМ на тренировочной выборке
    model = linear_regression(X, y)
    print(model.summary())

    # Отбор значимых факторов
    significant_factors = select_significant_factors(model, alpha)

    print(f"Значимые факторы при уровне значимости {alpha}: {significant_factors}")

    # Проверка мультиколлинеарности
    if significant_factors:
        high_corr = check_multicollinearity(X[significant_factors])

        if high_corr:
            print("Факторы с высокой корреляцией:")
            for pair in high_corr:
                print(f"{pair[0]} и {pair[1]}")
        else:
            print("Мультиколлинеарности между факторами нет")
    else:
        print("Нет значимых факторов для проверки мультиколлинеарности.")

    # Анализ корреляции
    correlations = correlation_analysis(X, y)
    print(f"Коэффициенты корреляции факторов с откликом: {correlations}")

    # Предсказание на основе модели для тестовой выборки
    X_with_const = sm.add_constant(X)
    y_pred = model.predict(X_with_const)

    # Оценка адекватности модели на тестовой выборке
    r2 = r2_score(y, y_pred)
    rmse = compute_rmse(y, y_pred)
    print(f"Коэффициент детерминации (R^2) на тестовых данных: {r2}")
    print(f"RMSE на тестовых данных: {rmse}")

    # Оценка F-статистики и адекватности модели
    f_stat = model.fvalue
    print(f"F-статистика: {f_stat}")

    f_pval = model.f_pvalue

    if f_pval < alpha:
        print("Модель адекватна")
    else:
        print("Модель неадекватна")

    file_path_new = input("Введите введите название нового файла для предсказания, если не нужно просто нажмите Enter:")  # Путь к новому файлу с данными

    print(predict_new(model, file_path_new))
    return y_pred




# Вызов основной функции программы
if __name__ == '__main__':
    file_path = ('gym_data.csv')# Путь к файлу с данными

    try:
        alpha = float(
            input("Введите уровень значимости (например, 0.05): ")
        )
    except ValueError:
        print("Некорректное значение. По умолчанию установлено значение 0.05.")
        alpha = 0.05
    predictions = main(file_path, alpha)


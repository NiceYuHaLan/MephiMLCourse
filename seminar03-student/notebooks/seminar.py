# ---
# jupyter:
#   audience: student
#   jupytext:
#     cell_metadata_filter: id,tags,start_minute,duration_minutes,end_minute
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,language_info,audience,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     name: python
# ---

# %% [markdown] tags=["block-SETUP"]
# # Семинар 3. Проверяем число, оцениваем вероятность, выбираем действие
#
# Работаем с **Auto MPG**: сначала прогнозируем экономичность автомобиля, затем вероятность
# события «экономичность ниже 20 mpg». Это две задачи на одних измерениях, не две метрики одной задачи.
#
# Распакуйте **весь комплект**, откройте этот notebook в Jupyter. Данные лежат локально:
# `auto_mpg_train.csv`, `auto_mpg_valid.csv`, `auto_mpg_test.csv`. Загрузка из сети на занятии не нужна.

# %% tags=["block-SETUP", "imports", "provided"]
import math
import sys
from pathlib import Path

# Готовый поиск корня комплекта; не упражнение по Python.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(
    (
        p
        for p in [HERE, *HERE.parents]
        if (
            p / "materials/03-probability-and-decisions/code/seminar3_support.py"
        ).is_file()
        or (
            (p / "seminar3_support.py").is_file()
            and (p / "data/auto_mpg_train.csv").is_file()
        )
    ),
    None,
)
if ROOT is None:
    raise FileNotFoundError(
        "Откройте notebook в репозитории или распакуйте весь архив семинара."
    )
CODE_DIR = (
    ROOT / "materials/03-probability-and-decisions/code"
    if (ROOT / "materials/03-probability-and-decisions/code").is_dir()
    else ROOT
)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from collections.abc import Sequence

import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.display import display
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

if get_ipython() is not None:
    get_ipython().run_line_magic("matplotlib", "inline")
from seminar3_support import (
    C_FN,
    C_FP,
    CUTOFF_MPG,
    DATA_DIR,
    FEATURES,
    classification_context,
    comparison_table,
    decision_report,
    final_report,
    load_part,
    make_classifier,
    make_regressor,
    plot_budget,
    plot_costs,
    plot_data,
    threshold_table,
    try_task,
)
from sklearn.model_selection import KFold, cross_val_score
from threadpoolctl import threadpool_limits

checks = {}
frozen_regression = None
chosen_tau = None
theoretical_tau = None
OPEN_TEST = False

# %% [markdown] duration_minutes=10 end_minute=10 start_minute=0 tags=["block-B00", "block-header"]
# ## B00. Данные и две содержательные задачи

# %% [markdown] tags=["block-B00"]
# **Задача.** Для описанного автомобиля, до измерения его экономичности в городском цикле,
# спрогнозировать `mpg` — число миль на галлон. **Больше mpg — экономичнее.**
#
# Одна строка — наблюдение об автомобиле определённой модели и года. В полном источнике 398 строк.
# В нашем эксперименте 238 строк train, 80 validation и 80 test; разбиение дано готовым.
#
# | Поле | Смысл | Роль здесь |
# |---|---|---|
# | `weight` | Масса, фунты | Признак |
# | `horsepower` | Мощность; встречаются пропуски | Признак |
# | `model_year` | Код модельного года, 70–82 | Признак |
# | `mpg` | Экономичность в городском цикле | Ответ регрессии |
# | `name`, `row_id` | Название и идентификатор строки | Не признаки |
#
# Считаем, что три выбранных признака известны в момент прогноза. Другие столбцы оставлены
# для чтения таблицы, но не подаются моделям.

# %% tags=["block-B00", "provided", "data-loading", "visualization"]
train = load_part("train")  # Validation и test здесь не читаем.
display(train[["name", *FEATURES, "mpg"]].head(6))
display(train[FEATURES].isna().sum().rename("Пропуски в train"))
plot_data(train)

# %% [markdown] duration_minutes=5 end_minute=15 start_minute=10 tags=["block-B01", "block-header"]
# ## B01. Что означает одна строка CV-отчёта

# %% [markdown] tags=["block-B01", "theory-recap"]
# ### Вспоминаем лекцию 4
#
# $f$ — уже обученный предсказатель; $y_i$ — наблюдённый ответ; $m$ — размер проверки.
# Эмпирический риск — средняя потеря на конкретной таблице:
#
# $$\widehat R_S(f)=\frac1m\sum_{i=1}^m\ell(y_i,f(x_i)).$$
#
# Сегодня в отчёте выбираем $\ell(y,\widehat y)=|y-\widehat y|$, поэтому считаем **MAE в mpg**.
# Обучение линейной регрессии по квадратам и отчёт по MAE не противоречат друг другу.
#
# Пример **трёх** фолдов только внутри train; A, B, C — непересекающиеся группы строк:
#
# | Запуск | Fit всего pipeline | Проверка |
# |---|---|---|
# | 1 | B + C | A |
# | 2 | A + C | B |
# | 3 | A + B | C |
#
# Каждый запуск начинает с необученного pipeline: медиана пропусков, масштабы, коэффициенты.
# В отчёте — среднее ошибок фолдов. Это не ансамбль трёх моделей.
#
# **Наш протокол:** train — EDA, CV регрессии и обучение классификатора;
# validation — выбор порога классификатора; test — одна заключительная проверка **обоих** зафиксированных решений.
# Поле `test_score` в CV означает проверочный fold, а не наш отдельный файл test.

# %% [markdown] duration_minutes=12 end_minute=27 start_minute=15 tags=["block-B02", "block-header"]
# ## B02. Простые регрессии на одинаковых фолдах

# %% [markdown] tags=["block-B02"]
# Заранее фиксируем четыре решения: медиана train; прямая по массе; линейная модель по трём
# признакам; Ridge по тем же трём признакам с фиксированным `alpha=30`.
# Медианная константа соответствует нашей отчётной MAE. Модели учатся по своим квадратным критериям.
#
# **Самостоятельно, 3–4 строки.** В `mean_cv_mae(model, X, y)` создайте
# `KFold(n_splits=4, shuffle=True, random_state=17)`, вызовите `cross_val_score`
# с `scoring="neg_mean_absolute_error"`, верните положительную среднюю MAE.
#
# `model` уже содержит всю предобработку. Вам не нужны индексы фолдов, клонирование модели или NumPy-матрицы.
# Общий вид вызова: `cross_val_score(model, X, y, cv=..., scoring=..., n_jobs=1)`.
# Минус в имени scorer нужен scikit-learn соглашению «больше score — лучше».


# %% tags=["block-B02", "student-only", "exercise", "definitions"]
def mean_cv_mae(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
    """Средняя положительная MAE в mpg по четырём фолдам внутри train.

    model — необученный estimator с предобработкой; X — таблица (n, p),
    y — ответы mpg длины n в том же порядке. Исходный model не обучается."""
    # Библиотечная CV всего pipeline; вернуть положительную MAE.
    raise NotImplementedError("B02")


# %% tags=["block-B02", "checks"]
def check_B02():
    tr = load_part("train")
    with threadpool_limits(limits=1):
        result = mean_cv_mae(make_regressor("linear"), tr[FEATURES], tr.mpg)
        reference = comparison_table()["linear"].mean()
    assert math.isclose(result, reference, rel_tol=1e-10)
    assert result > 0
    return result


checks["B02"] = try_task(check_B02)
if checks["B02"] is not None:
    print("MAE собственного вызова:", checks["B02"])

# %% tags=["block-B02", "provided", "reference-experiment"]
# Отдельный готовый библиотечный эксперимент, не результат незаполненной функции.
reference_folds = comparison_table(k=4, seed=17)
display(reference_folds.rename_axis("Проверочный fold"))
reference_means = reference_folds.mean().rename("Средняя CV MAE, mpg")
display(reference_means)
frozen_regression = str(reference_means.idxmin())
print(
    "Выбор в этой демонстрации по заранее заданному правилу min CV MAE:",
    frozen_regression,
)

# %% [markdown] duration_minutes=6 end_minute=33 start_minute=27 tags=["block-B04D", "block-header"]
# ## B04D. Сколько фолдов стоит оплачивать: короткая демонстрация

# %% [markdown] tags=["block-B04D", "theory-recap"]
# Одна фиксированная линейная модель; меняем **только протокол её оценки**.
#
# При увеличении K в train каждого fit остаётся больше строк: в идеализированном равномерном
# разбиении доля равна $1-1/K$. Для 3 и 4 это $2/3$ и $3/4$.
# Ниже реальный опыт в референсном окружении.

# %% tags=["block-B04D", "provided", "visualization"]
k_reference = pd.read_csv(DATA_DIR / "cv_k_reference.csv")
display(k_reference[["K", "mae", "seconds", "cumulative_fits"]])
plot_budget(k_reference)

# %% [markdown] duration_minutes=6 end_minute=39 start_minute=33 tags=["block-B06", "block-header"]
# ## B06. Из численного ответа — событие


# %% [markdown] tags=["block-B06", "theory-recap"]
# ### Вспоминаем мостик из лекции 5
#
# Пусть T — экономичность в mpg. Определим новое наблюдаемое событие:
#
# $$Y=\mathbf1[T<20].$$
#
# Теперь хотим $\pi_\theta(x)$ — оценку вероятности $Y=1$, а не среднего значения T.
# Число 20 выбрано заранее **для учебной задачи**.
#
# **Самостоятельно:** преобразуйте список mpg в список меток. Строгое неравенство важно:
# 20 mpg относится к классу 0. Достаточно цикла или спискового выражения.
#


# %% tags=["block-B06", "student-only", "exercise", "definitions"]
def event_labels(values: Sequence[float]) -> list[int]:
    """Значения mpg длины n → список из n меток 0/1; событие строго mpg < 20."""
    # Класс 1, когда значение строго меньше 20 mpg.
    raise NotImplementedError("B06")


# %% tags=["block-B06", "checks"]
def check_B06():
    assert event_labels([19.9, 20, 21]) == [1, 0, 0]
    labels = event_labels(list(load_part("train").mpg))
    return sum(labels) / len(labels)


checks["B06"] = try_task(check_B06)
if checks["B06"] is not None:
    print("Доля класса 1 на train:", checks["B06"])


# %% [markdown] tags=["block-B06"]
# При построении X нельзя оставить ни `mpg`, ни `1/mpg`, ни уже вычисленный класс.
# Мы продолжаем пользоваться только массой, мощностью и модельным годом.
# **20 mpg и τ=0.5 — разные величины в разных единицах.**

# %% [markdown] duration_minutes=8 end_minute=47 start_minute=39 tags=["block-B07", "block-header"]
# ## B07. Вероятности одной обученной LogReg


# %% [markdown] tags=["block-B07", "theory-recap"]
# ### Теория, которую используем сейчас
#
# $x$ — исходные признаки; $z=\varphi(x)$ — признаки после заполнения пропусков и масштабирования.
# $b,w$ — обучаемые коэффициенты. Цепочка из лекции 5:
#
# $$a=b+w^\top z,\qquad \pi_\theta(x)=\sigma(a)=\frac1{1+e^{-a}},
# \qquad \widehat y_\tau=\mathbf1[\pi_\theta(x)\ge\tau].$$
#
# LogReg обучается по log loss с регуляризацией, а не по числу ошибок после порога.
# Используем **готовый** solver и фиксированное C=1. Предобработка — внутри pipeline.
#
# **Самостоятельно:** `fit_probabilities(model, X_train, y_train, X_valid)` должна обучить
# model на train, найти столбец класса 1 через `list(model.classes_).index(1)` и вернуть
# его `predict_proba` на validation. Объяснение `[:, positive]` уже дано: все строки, один столбец.


# %% tags=["block-B07", "student-only", "exercise", "definitions"]
def fit_probabilities(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame
) -> NDArray[np.float64]:
    """Обучить model на train и вернуть P(Y=1) для validation, массив (n_valid,).

    X_train и X_valid содержат одинаковые признаки; y_train — метки 0/1
    длины n_train. Функция изменяет model через fit; validation не участвует в fit."""
    # Обучение только train; вернуть вероятности класса 1 на validation.
    raise NotImplementedError("B07")


# %% tags=["block-B07", "checks"]
def check_B07():
    tr, va = load_part("train"), load_part("valid")
    student_model = make_classifier()
    values = fit_probabilities(
        student_model, tr[FEATURES], (tr.mpg < CUTOFF_MPG).astype(int), va[FEATURES]
    )
    _, reference, _ = classification_context()
    np.testing.assert_allclose(values, reference, atol=1e-8)
    return values


checks["B07"] = try_task(check_B07)
if checks["B07"] is not None:
    print("Первые пять вероятностей:", checks["B07"][:5])

# Независимый готовый референс для следующих блоков. Не подмена student_model.
reference_classifier, pi_valid, y_valid_class = classification_context()
print("Далее фиксированный библиотечный референс; valid-строк:", len(pi_valid))

# %% [markdown] tags=["block-B07"]
# **Проверка понимания:** при изменении τ с 0.5 на 0.2 что нужно переобучить?

# %% [markdown] duration_minutes=12 end_minute=59 start_minute=47 tags=["block-B08", "block-header"]
# ## B08. Ошибки как действия и их цена

# %% [markdown] tags=["block-B08", "theory-recap"]
# Для каждого объекта есть факт y и вероятность $\pi$. Действие:
# $\widehat y=\mathbf1[\pi\ge\tau]$. Ложное предупреждение (FP) стоит 1 условную единицу,
# пропуск события (FN) — 4. Верные решения бесплатны.
#
# $$\text{Cost}=C_{FP}\,FP+C_{FN}\,FN=FP+4FN.$$
#
# | Факт / действие | 0 | 1 |
# |---|---|---|
# | 0 | TN | FP |
# | 1 | FN | TP |
#
# **Сначала вручную:** в таблице ниже при τ=0.5 найдите один FP и один FN.
# Затем замените τ на 0.2, не меняя вероятности.

# %% tags=["block-B08", "provided"]
small_y = [1, 0, 1, 0, 1, 0]
small_pi = [0.85, 0.75, 0.55, 0.35, 0.25, 0.05]
display(pd.DataFrame({"y": small_y, "pi": small_pi}))


# %% tags=["block-B08", "student-only", "exercise", "definitions"]
def decisions(
    probabilities: Sequence[float] | NDArray[np.float64], tau: float
) -> list[int]:
    """Вероятности длины n из [0, 1] и порог tau → n меток; равенство даёт 1."""
    # Вернуть список меток; равенство порогу даёт 1.
    raise NotImplementedError("B08")


# %% tags=["block-B08", "checks"]
def check_B08_decisions():
    assert decisions([0.2, 0.5, 0.8], 0.5) == [0, 1, 1]
    assert decisions([0, 1], 0.2) == [0, 1]
    return True


checks["B08_decisions"] = try_task(check_B08_decisions)


# %% tags=["block-B08", "student-only", "exercise", "definitions"]
def error_cost(
    y_true: Sequence[int], y_pred: Sequence[int], c_fp: float = 1, c_fn: float = 4
) -> float:
    """Суммарная цена ошибок в условных единицах, без деления на число строк.

    y_true и y_pred — согласованные по порядку списки меток 0/1 одинаковой
    длины; c_fp и c_fn — положительные цены FP и FN. Верные действия бесплатны."""
    # Сумма цены ложных предупреждений и пропусков; входы согласованы.
    raise NotImplementedError("B08")


# %% tags=["block-B08", "checks"]
def check_B08_cost():
    assert error_cost([0, 1, 1], [1, 0, 1]) == 5
    assert error_cost([0, 1], [0, 1]) == 0
    assert error_cost([0, 1], [1, 0], c_fp=2, c_fn=7) == 9
    return True


checks["B08_cost"] = try_task(check_B08_cost)
if checks["B08_decisions"] and checks["B08_cost"]:
    for tau in [0.5, 0.2, 0.8]:
        labels = decisions(small_pi, tau)
        print("tau =", tau, "метки:", labels, "цена:", error_cost(small_y, labels))
        display(pd.Series(decision_report(small_y, labels)))


# %% [markdown] tags=["block-B08"]
# - $\mathrm{Precision}=TP/(TP+FP)$ — доля верных среди предупреждений;
# - $\mathrm{Recall}=TP/(TP+FN)$ — доля найденных событий.
#
# При нуле предупреждений precision не определена; код показывает NaN. При отсутствии событий так же
# не определена recall.

# %% [markdown] duration_minutes=10 end_minute=69 start_minute=59 tags=["block-B09", "block-header"]
# ## B09. Главная формула: вывести порог из цены ошибок

# %% [markdown] tags=["block-B09", "theory-recap", "math-core", "exercise"]
# ### Зачем нужна математика
#
# Число 0.5 не учитывает цену ошибки. Хотим выбрать действие с меньшим ожидаемым ущербом.
#
# $x$ фиксирован; $q=P(Y=1\mid x)$ — **истинная** вероятность события.
# $u\in\{0,1\}$ — действие; $\mathcal R(u\mid x)$ — его условный ожидаемый ущерб.
# $C_{FP}>0$, $C_{FN}>0$ — постоянные цены; верные действия бесплатны, ограничения на число
# предупреждений нет. В модели используем лишь оценку $\pi_\theta(x)$ неизвестного q.
#
# На лекции уже было:
#
# $$\mathcal R(1\mid x)=C_{FP}(1-q),\qquad
# \mathcal R(0\mid x)=C_{FN}q.$$
#
# **Самостоятельно:** решите неравенство
# $C_{FP}(1-q)\le C_{FN}q$ относительно q. Что произойдёт с порогом, если пропуск станет дороже?
# Затем перенесите формулу в `bayes_threshold`: одна строка.

# %% tags=["block-B09", "student-only", "exercise", "definitions"]
def bayes_threshold(c_fp: float, c_fn: float) -> float:
    """Положительные цены FP и FN → теоретический порог вероятности из (0, 1)."""
    # Цены положительны; верные действия бесплатны.
    raise NotImplementedError("B09")


# %% tags=["block-B09", "checks"]
def check_B09():
    assert math.isclose(bayes_threshold(1, 4), 0.2)
    assert math.isclose(bayes_threshold(3, 3), 0.5)
    assert math.isclose(bayes_threshold(10, 40), 0.2)
    assert bayes_threshold(1, 9) < bayes_threshold(1, 4)
    return bayes_threshold(C_FP, C_FN)


checks["B09"] = try_task(check_B09)
if checks["B09"] is not None:
    theoretical_tau = checks["B09"]
    print("Теоретический порог:", theoretical_tau)

# %% [markdown] duration_minutes=8 end_minute=77 start_minute=69 tags=["block-B10", "block-header"]
# ## B10. Порог по validation, а не по test


# %% [markdown] tags=["block-B10", "theory-recap"]
# На практике q неизвестно. Другой способ — сравнить действия по сохранённым вероятностям validation.
# Модель уже обучена; кандидаты τ = 0.1, 0.2, …, 0.9 и цена FP+4FN объявлены заранее.
#
# **Самостоятельно:** `select_threshold(rows)` получает обычный список словарей с ключами
# `tau` и `cost`, упорядоченный по τ. Верните τ первой строки с минимальной ценой.
# Достаточно `min(rows, key=...)`; при равенстве остаётся первая строка.
#
# После выбора фиксируем и регрессионную модель, и классификатор, и τ.
# Test до этого не открываем ни для одной из задач: они используют связанные ответы на одних строках.

# %% tags=["block-B10", "provided", "reference-experiment", "visualization"]
_, pi_valid, y_valid_class = classification_context()
validation_table = threshold_table(y_valid_class, pi_valid)
display(validation_table)
plot_costs(validation_table)


# %% tags=["block-B10", "student-only", "exercise", "definitions"]
def select_threshold(rows: list[dict[str, float]]) -> float:
    """Непустой список строк с tau и cost → tau первой строки с минимальной cost.

    Строки упорядочены по tau; при равенстве цен сохраняется первая строка.
    Дополнительные ключи отчёта не используются."""
    # Минимум cost, первая строка при равенстве.
    raise NotImplementedError("B10")


# %% tags=["block-B10", "checks"]
def check_B10():
    assert select_threshold([{"tau": 0.2, "cost": 3}, {"tau": 0.4, "cost": 3}]) == 0.2
    assert select_threshold([{"tau": 0.2, "cost": 5}, {"tau": 0.4, "cost": 3}]) == 0.4
    return float(select_threshold(validation_table.to_dict("records")))


checks["B10"] = try_task(check_B10)
if checks["B10"] is not None:
    chosen_tau = checks["B10"]
    print("Выбрано по validation; модель не переобучаем:", chosen_tau)

# %% [markdown] duration_minutes=5 end_minute=82 start_minute=77 tags=["block-B15", "block-header"]
# ## B15. Финальная проверка


# %% [markdown] tags=["block-B15"]
# **До открытия test** запишите: какая регрессионная модель выбрана по CV, какой классификатор обучен,
# по какому правилу взят τ. После открытия не менять ни представление, ни модель, ни τ.
#
# В итоговой демонстрации используем один фиксированный библиотечный pipeline классификатора.
# Он **не переобучается** после выбора τ.
#
# Когда оба решения зафиксированы, поменяйте `OPEN_TEST` на True.

# %% tags=["block-B15", "student-only", "configuration"]
OPEN_TEST = False  # Только после фиксации обоих решений.

# %% tags=["block-B15", "provided", "final-evaluation"]
fixed_tau = chosen_tau  # Маршрут A: порог зафиксирован по validation в B10.
if OPEN_TEST:
    if frozen_regression is None or fixed_tau is None:
        raise RuntimeError(
            "Сначала завершите выбор регрессии и порога. Test не открыт."
        )
    final = final_report(tau=fixed_tau, regression_kind=frozen_regression)
    print(
        "Регрессия MAE / baseline, mpg:", final["regression_mae"], final["baseline_mae"]
    )
    display(final["policies"])
    print(
        "Одинаковые вероятности при разных tau: log loss =",
        final["log_loss"],
        "; ROC-AUC =",
        final["roc_auc"],
    )
else:
    print("Test не открыт: OPEN_TEST=False")
print(
    "Самостоятельные части:",
    {name: value is not None for name, value in checks.items()},
)

# %% [markdown] tags=["block-B15"]
# **Самопроверка.** Если дать вдвое больше времени на K-fold,
# обучить ту же модель на том же train и затем изменить порог, какие из следующих величин
# обязаны измениться:
# - веса
# - качество оценки по CV
# - вероятности
# - метки
# - цена ошибок?
#
# На выбор — дополнить: почему нельзя подбирать τ по test, даже если все коэффициенты уже фиксированы?

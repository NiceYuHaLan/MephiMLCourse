"""Готовая инфраструктура семинара 3, только маршрут A. Здесь нет учебных реализаций ЛР1.

CV выполняется только библиотечным cross_val_score. В блоке B04D читается
готовая таблица измерений; планировщика бюджетных запусков здесь нет.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA_DIR = (ROOT.parent / 'datasets/auto-mpg'
            if (ROOT.parent / 'datasets/auto-mpg').is_dir() else ROOT / 'data')
FEATURES = ['weight', 'horsepower', 'model_year']
CUTOFF_MPG = 20.0
C_FP, C_FN = 1.0, 4.0
THRESHOLDS = [i / 10 for i in range(1, 10)]


def load_part(part: str = 'train') -> pd.DataFrame:
    """Читает только названную часть; test не открывается побочным вызовом."""
    if part not in {'train', 'valid', 'test'}:
        raise ValueError('part must be train, valid or test')
    path = DATA_DIR / f'auto_mpg_{part}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Распакуйте весь комплект: отсутствует {path.name}')
    return pd.read_csv(path)


def make_regressor(kind: str = 'linear'):
    """Четыре заранее заданных кандидата; никаких сеток lambda и своих solver."""
    if kind == 'baseline':
        return DummyRegressor(strategy='median')
    if kind not in {'weight', 'linear', 'ridge'}:
        raise ValueError('Unknown model kind')
    columns = ['weight'] if kind == 'weight' else FEATURES
    prep = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), columns)
    ], remainder='drop')
    model = Ridge(alpha=30.0) if kind == 'ridge' else LinearRegression()
    return make_pipeline(prep, model)


def make_classifier():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, tol=1e-9))


def cv_errors(kind='linear', k=4, seed=17):
    """Готовый вызов библиотеки для самостоятельных демонстрационных блоков."""
    train = load_part('train')
    with threadpool_limits(limits=1):
        return -cross_val_score(make_regressor(kind), train[FEATURES], train.mpg,
              cv=KFold(n_splits=k, shuffle=True, random_state=seed),
              scoring='neg_mean_absolute_error', n_jobs=1, error_score='raise')


def comparison_table(k=4, seed=17):
    """Библиотечный референс: не результат незавершённого студенческого задания."""
    return pd.DataFrame({name: cv_errors(name,k,seed)
                         for name in ('baseline','weight','linear','ridge')})


@lru_cache(maxsize=1)
def classification_context():
    """Фиксированный референс для независимости блоков. Не студенческое решение."""
    tr, va = load_part('train'), load_part('valid')
    clf = make_classifier().fit(tr[FEATURES], (tr.mpg < CUTOFF_MPG).astype(int))
    positive = list(clf.classes_).index(1)
    p = clf.predict_proba(va[FEATURES])[:, positive]
    return clf, p, (va.mpg < CUTOFF_MPG).astype(int).to_numpy()


def decision_report(y, decisions, c_fp=C_FP, c_fn=C_FN):
    y, decisions = np.asarray(y), np.asarray(decisions)
    if y.ndim != 1 or y.shape != decisions.shape or not len(y):
        raise ValueError('Expected nonempty equal-length vectors')
    if not set(y).issubset({0,1}) or not set(decisions).issubset({0,1}):
        raise ValueError('Labels and decisions must be binary')
    if c_fp <= 0 or c_fn <= 0:
        raise ValueError('Positive costs required')
    tn, fp, fn, tp = map(int, confusion_matrix(y, decisions, labels=[0,1]).ravel())
    return {'TN':tn, 'FP':fp, 'FN':fn, 'TP':tp,
            'accuracy': (tn+tp)/len(y),
            'precision': tp/(tp+fp) if tp+fp else np.nan,
            'recall': tp/(tp+fn) if tp+fn else np.nan,
            'cost': c_fp*fp+c_fn*fn}


def threshold_table(y, p, thresholds=THRESHOLDS, c_fp=C_FP, c_fn=C_FN):
    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('Expected probabilities between 0 and 1')
    return pd.DataFrame([{'tau':float(t), **decision_report(y,p>=t,c_fp,c_fn)}
                         for t in thresholds])


def plot_data(train):
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.scatter(train.weight,train.mpg,s=18,alpha=.5)
    ax.set(xlabel='Масса, lb (фунты)',ylabel='Экономичность, mpg',
           title='Только train: одинаковая масса — разные mpg')
    fig.tight_layout(); plt.show()
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.hist(train.mpg,bins=16)
    ax.axvline(CUTOFF_MPG,linestyle='--',label='Учебная граница события: 20 mpg')
    ax.set(xlabel='Экономичность, mpg',ylabel='Число строк',title='Только train')
    ax.legend();fig.tight_layout();plt.show()


def plot_budget(table):
    for y,label,title in [('seconds','Время полного K-fold запуска, с','Каждая точка — K новых обучений'),
                         ('mae','Средняя MAE по folds, mpg','Больше K ≠ гарантированно меньше ошибка')]:
        fig,ax=plt.subplots(figsize=(7,3.5))
        ax.plot(table.K,table[y],'o-')
        ax.set(xscale='log',xlabel='K, логарифмическая ось',ylabel=label,title=title)
        ax.grid(alpha=.2);fig.tight_layout();plt.show()


def plot_costs(table):
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.plot(table.tau,table.cost,'o-')
    ax.set(xlabel='Порог решения τ',ylabel='FP + 4 FN, условные единицы',
           title='Validation: сравниваем заранее заданные пороги')
    ax.grid(alpha=.2);fig.tight_layout();plt.show()


def final_report(tau=.2, regression_kind='linear'):
    """Вызывается лишь после фиксации процедуры. Classifier НЕ переобучается."""
    if not 0 < tau < 1: raise ValueError('0 < tau < 1')
    train, test = load_part('train'), load_part('test')
    reg=make_regressor(regression_kind).fit(train[FEATURES],train.mpg)
    reg_base=make_regressor('baseline').fit(train[FEATURES],train.mpg)
    clf,_,_=classification_context()
    p=clf.predict_proba(test[FEATURES])[:,list(clf.classes_).index(1)]
    y=(test.mpg<CUTOFF_MPG).astype(int).to_numpy()
    rows=[]
    for name,d in [('Всегда 0',np.zeros(len(y),dtype=int)),
                   ('Всегда 1',np.ones(len(y),dtype=int)),
                   ('LogReg: τ=0.5',p>=.5),('LogReg: фиксированный τ',p>=tau)]:
        rows.append({'policy':name,**decision_report(y,d)})
    train_prior=float((train.mpg<CUTOFF_MPG).mean())
    return {'regression_mae':float(mean_absolute_error(test.mpg,reg.predict(test[FEATURES]))),
            'baseline_mae':float(mean_absolute_error(test.mpg,reg_base.predict(test[FEATURES]))),
            'tau':float(tau),'policies':pd.DataFrame(rows),
            'log_loss':float(log_loss(y,p,labels=[0,1])),
            'prior_log_loss':float(log_loss(y,np.full(len(y),train_prior),labels=[0,1])),
            'roc_auc':float(roc_auc_score(y,p))}


def try_task(function, *args, **kwargs):
    """Ловим только явно незавершённое задание. Неверные ответы не скрываем."""
    try:
        return function(*args,**kwargs)
    except NotImplementedError:
        print('Задание ещё не выполнено. Референс не подставлен.')
        return None

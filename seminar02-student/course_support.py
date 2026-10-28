"""Ready-to-use infrastructure. No student optimizer, Ridge solver or CV loop here."""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

TARGET='resolution_minutes'
NUMERIC=['message_length','queue_at_open','account_age_days','attachments_count',
         'active_agents','is_weekend','initial_payload_bytes']
CATEGORICAL=['channel','topic']
FEATURES=NUMERIC+CATEGORICAL
BLUE='#002395'
REPOSITORY_DATA = Path('materials/02-data/datasets/support-tickets/student')
REPOSITORY_SUPPORT = Path('materials/01-regression/code/course_support.py')


def data_root(root: Path) -> Path:
    root = Path(root)
    for relative in (REPOSITORY_DATA, Path('data')):
        candidate = root / relative
        if (candidate / 'tickets_lab1_dev.csv').is_file():
            return candidate
    raise FileNotFoundError('Не найдены открытые данные ЛР1.')


def locate_root(start: Path | None=None) -> Path:
    start=Path.cwd() if start is None else Path(start)
    for p in [start,*start.parents]:
        repository = (p / REPOSITORY_DATA / 'tickets_lab1_dev.csv').is_file() and (p / REPOSITORY_SUPPORT).is_file()
        archive = (p / 'data/tickets_lab1_dev.csv').is_file() and (p / 'course_support.py').is_file()
        if repository or archive:
            return p
    raise FileNotFoundError('Распакуйте весь комплект: нужны данные ЛР1 и course_support.py.')


def load_tables(root: Path):
    data=data_root(root)
    dev=pd.read_csv(data/'tickets_lab1_dev.csv')
    test=pd.read_csv(data/'tickets_lab1_test.csv')
    split=pd.read_csv(data/'lab1_split.csv')
    assert dev.ticket_id.is_unique and test.ticket_id.is_unique and split.ticket_id.is_unique
    assert set(dev.ticket_id)==set(split.ticket_id)
    assert set(dev.ticket_id).isdisjoint(set(test.ticket_id))
    assert TARGET not in test.columns
    merged=dev.merge(split,on='ticket_id',how='left',validate='one_to_one')
    train=merged.loc[merged['split']=='train'].drop(columns='split').reset_index(drop=True)
    valid=merged.loc[merged['split']=='validation'].drop(columns='split').reset_index(drop=True)
    assert len(train)+len(valid)==len(dev)
    return train,valid,test


def select_features(frame: pd.DataFrame, mode: str='base') -> pd.DataFrame:
    if mode not in {'base','queue'}: raise ValueError('mode: base или queue')
    result=frame[FEATURES].copy()
    if mode=='queue': result['queue_excess_sq']=np.maximum(result['queue_at_open']-18,0)**2
    return result


def make_preprocessor(mode: str='base', scale: bool=True) -> Pipeline:
    numeric=NUMERIC+(['queue_excess_sq'] if mode=='queue' else [])
    num_steps=[('imputer',SimpleImputer(strategy='median'))]
    if scale: num_steps.append(('scaler',StandardScaler()))
    cat=make_pipeline(SimpleImputer(strategy='most_frequent'),
                      OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))
    columns=ColumnTransformer([('numeric',Pipeline(num_steps),numeric),
                               ('category',cat,CATEGORICAL)],remainder='drop',sparse_threshold=0)
    return make_pipeline(FunctionTransformer(select_features,kw_args={'mode':mode},validate=False),columns)


def augment(X):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or not np.isfinite(X).all(): raise ValueError('Нужна конечная числовая матрица (n,d).')
    return np.column_stack([np.ones(len(X)),X])


def check_arrays(Z,y):
    Z=np.asarray(Z,dtype=float); y=np.asarray(y,dtype=float)
    if Z.ndim!=2 or y.ndim!=1 or len(Z)!=len(y) or len(y)==0:
        raise ValueError('Ожидаются Z:(n,d+1), y:(n,), n>0.')
    if not np.isfinite(Z).all() or not np.isfinite(y).all(): raise ValueError('NaN/inf на входе оптимизатора.')
    if not np.allclose(Z[:,0],1): raise ValueError('Нулевой столбец Z должен состоять из единиц.')
    return Z,y


def record(history, theta, loss, updates, seen, **extra):
    if not np.isfinite(loss) or not np.isfinite(theta).all():
        raise FloatingPointError('Параметры/критерий стали нечисловыми: проверьте знак и длину шага.')
    history.append({'updates':int(updates),'seen':int(seen),'loss':float(loss),**extra})


def result_dict(theta,history,updates,seen,**extra):
    return {'theta':np.asarray(theta,dtype=float).copy(), 'history':history,
            'updates':int(updates),'seen':int(seen),**extra}


def run_checks(label: str, test: Callable[[],None]) -> bool:
    """Only an explicit unfinished exercise is skipped. Actual bugs fail loudly."""
    try: test()
    except NotImplementedError as exc:
        print(f'{label}: НЕ ВЫПОЛНЕНО — {exc}'); return False
    print(f'{label}: проверки пройдены (это не автоматическая итоговая оценка).')
    return True


def mse(y,pred):
    y=np.asarray(y,dtype=float); pred=np.asarray(pred,dtype=float)
    if y.ndim!=1 or pred.shape!=y.shape: raise ValueError('MSE ожидает совпадающие формы (n,).')
    return float(np.mean((y-pred)**2))


def metrics(y,pred):
    y=np.asarray(y,dtype=float); pred=np.asarray(pred,dtype=float)
    return {'MSE, мин²':mse(y,pred),'RMSE, мин':float(np.sqrt(mse(y,pred))),
            'MAE, мин':float(np.mean(np.abs(y-pred)))}


class NormalizedRidge(RegressorMixin, BaseEstimator):
    """Library baseline for MSE + l2 * ||w||². Not a custom solver."""
    def __init__(self,l2=0.0): self.l2=l2
    def fit(self,X,y):
        if self.l2<0: raise ValueError('l2 >= 0')
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float)
        self.model_=Ridge(alpha=len(y)*self.l2,solver='svd').fit(X,y)
        self.coef_=self.model_.coef_.copy(); self.intercept_=float(self.model_.intercept_)
        self.n_features_in_=X.shape[1]
        return self
    def predict(self,X):
        check_is_fitted(self,'model_'); return self.model_.predict(X)


class IterativeRegressor(RegressorMixin, BaseEstimator):
    """Given adapter: student solver accepts Z with ones in column 0."""
    def __init__(self,solver,solver_params=None):
        self.solver=solver; self.solver_params=solver_params
    def fit(self,X,y):
        Z,y=check_arrays(augment(X),y)
        self.result_=self.solver(Z,y,**(self.solver_params or {}))
        theta=np.asarray(self.result_['theta'])
        if theta.shape!=(Z.shape[1],) or not np.isfinite(theta).all():
            raise ValueError('solver должен вернуть конечный theta формы (d+1,).')
        self.intercept_=float(theta[0]); self.coef_=theta[1:].copy()
        self.n_features_in_=Z.shape[1]-1
        return self
    def predict(self,X):
        check_is_fitted(self,'coef_'); X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_: raise ValueError('Неверная форма X.')
        return self.intercept_+X@self.coef_


def baseline_pipeline(l2=0.0,mode='base'):
    return make_pipeline(make_preprocessor(mode),NormalizedRidge(l2=l2))


def plot_history(named_results, x='seen', y='loss', title='Ход обучения'):
    fig,ax=plt.subplots(figsize=(8,4.2))
    for i,(name,res) in enumerate(named_results.items()):
        table=pd.DataFrame(res['history'])
        if y not in table: continue
        kwargs={'color':BLUE} if i==0 else {}
        ax.plot(table[x],table[y],label=name,**kwargs)
    ax.set(title=title,xlabel={'seen':'Объектов в обучающих градиентах','updates':'Обновлений'}.get(x,x),
           ylabel={'loss':'J на train, мин²','valid_mse':'MSE на validation, мин²'}.get(y,y))
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); plt.show()


def plot_residuals(frame,y,pred,feature='queue_at_open',title='Остатки: факт − прогноз'):
    fig,ax=plt.subplots(figsize=(8,4.2))
    ax.scatter(frame[feature],np.asarray(y)-np.asarray(pred),s=13,alpha=.35,color=BLUE)
    ax.axhline(0,linestyle='--',linewidth=1)
    ax.set(title=title,xlabel=feature,ylabel='Остаток, мин'); ax.grid(alpha=.15)
    fig.tight_layout(); plt.show()


def plot_target(frame):
    fig,ax=plt.subplots(figsize=(8,3.8))
    ax.hist(frame[TARGET],bins=35,color=BLUE,alpha=.8)
    ax.set(xlabel='Время решения, мин',ylabel='Обращений',title='Учебные обращения: распределение target')
    fig.tight_layout(); plt.show()


def export_predictions(root,test,pred,filename):
    pred=np.asarray(pred,dtype=float)
    if pred.shape!=(len(test),) or not np.isfinite(pred).all(): raise ValueError('Невалидные прогнозы.')
    out=root/'submissions'; out.mkdir(exist_ok=True)
    dest=out/filename
    if dest.parent!=out or dest.suffix!='.csv': raise ValueError('Укажите простое имя .csv')
    pd.DataFrame({'ticket_id':test.ticket_id.to_numpy(),'prediction':pred}).to_csv(dest,index=False)
    print('Сохранено:',dest)
    return dest

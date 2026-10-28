# Источники маршрута A

Этот архив — сокращение ранее подготовленного комплекта семинара 3. Данные, задачи,
порог события, цены ошибок и выбранная траектория сохранены. Новое в этой редакции —
детальные преподавательские временные метки, подсказки и упаковка только одного занятия.
Дополнительная внешняя проверка источников при сокращении архива не проводилась.

## Опора на материалы преподавателя

| Блок | Материал |
|---|---|
| B00–B02 | Лекции 2–3: линейная модель, метрики, предобработка и Ridge; лекция 4: риск, роли данных и K-fold |
| B04D | Лекция 4: K-fold и ограничения оценки; сохранённый ресурсный эксперимент исходного комплекта |
| B06–B07 | Черновик лекции 5: target, why-not-mean, notation, outputs, sigmoid, logloss |
| B08 | Лекция 5: confusion, metrics и цена ошибок |
| B09 | Лекция 5: decision-costs, conditional-risk, bayes-rule, probability-not-oracle |
| B10 и B15 | Лекции 4–5: выбор по validation и независимая финальная проверка |

По приложенным lab-v1…lab-v4 сохранена граница: нет собственных GD, Momentum,
решателя Ridge, ручного управления фолдами или сборки OOF. Сложность кода уменьшена
по сравнению с приложенным «Семинаром 2»; цели и обозначения не заменены новым курсом.

## Открытые референсы, использованные в исходном комплекте

- UCI Auto MPG, DOI 10.24432/C5859H: https://archive.ics.uci.edu/dataset/9/auto%2Bmpg
- CSV-зеркало: https://github.com/mwaskom/seaborn-data/blob/master/mpg.csv
- scikit-learn, Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn, Decision threshold: https://scikit-learn.org/stable/modules/classification_threshold.html
- Jupytext, percent format: https://jupytext.readthedocs.io/en/latest/formats-scripts.html
- ВШЭ, пример библиотечной CV в sem04-features.ipynb:
  https://github.com/esokolov/ml-course-hse/blob/master/2025-fall/seminars/sem04-features.ipynb

Готовый notebook ВШЭ не копируется. Временные бюджеты и реплики преподавателя —
предложенный сценарий, а не данные о фактическом проведении занятия в ВШЭ.
Подробное происхождение данных, лицензия и преобразования — в [datasets/auto-mpg/DATA_CARD.md](datasets/auto-mpg/DATA_CARD.md) в репозитории;
в распакованном ZIP откройте локальный `data/DATA_CARD.md`.


## Интеграция в курс

Импортировано из локального `seminar3_A/`. Данные и сохранённая таблица опыта с K
перенесены без изменения байтов. Вспомогательный код адаптирован к расположению в
репозитории и ZIP; учебные алгоритмы и численные ориентиры сохранены.
Канонический notebook-источник — `teacher/seminars/03-probability-and-decisions/solution.py`.
Студенческие пары производятся общей системой синхронизации. Типы и docstrings
добавлены к семи заготовкам и соответствующим решениям.
Внешние ссылки здесь — происхождение исходного комплекта; при интеграции новой
внешней проверки источников не проводилось.

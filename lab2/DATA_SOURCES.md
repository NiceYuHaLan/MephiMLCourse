# Исходные данные

В комплект включены полные raw CSV, полученные с UCI без преобразования содержимого.
Они доступны отдельно от notebook и читаются обычным pandas.read_csv.

| Файл | Датасет | Прямой URL | SHA-256 |
|---|---|---|---|
| raisin.csv | Raisin, 900 × 7 + Class | https://archive.ics.uci.edu/static/public/850/data.csv | 175ecbc52d351e0e0d9aa00605de09425b3a4826c5db294fabcbc24bcba955ab |
| dry_bean.csv | Dry Bean, 13 611 × 16 + Class | https://archive.ics.uci.edu/static/public/602/data.csv | c91a086bb4357e69409d23287ec22be1be9512145cead1d8ca988c918be38fcb |

Raisin: Çınar, İ., Köklü, M., Taşdemir, Ş. (2020), UCI Machine Learning Repository,
https://doi.org/10.24432/C5660T.

Dry Bean: Köklü, M., Özkan, İ. A. (2020), UCI Machine Learning Repository,
https://doi.org/10.24432/C50S4B.

Страницы источников:
https://archive.ics.uci.edu/dataset/850/raisin
https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

Лицензия обоих датасетов: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/.

Загрузчик удаляет полные дубликаты Dry Bean и делает стратифицированную подвыборку
только в памяти. Исходный CSV сохраняется полностью. В notebook нет синтетического
обучающего датасета; маленькие искусственные массивы служат числовыми проверками функций.

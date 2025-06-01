# Skip-gram with negative sampling

Проект выполнялся в рамках тренировок Яндекса ML2.0 по NLP и был доработан в последствии.

### Описание

В репозитории содержится реализация модели Skip-gram with negative sampling из статьи [Миколова и Суцкевера](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf). Данная модель — это доработанная версия Skip-gram, она позволяет выявлять близкие по контексту слова на основе анализа различных контекстных пар, а также использования отрицательных примеров (negative sampling).

### Структура

 * [Папка `data`](https://github.com/SupNek/Skip-gram_with_neg_sampling/tree/main/data) — содержит данные, на которых обучалась и тестировалась модель. Все данные содержатся в файле `quora.txt`;
 * [Папка `models`](https://github.com/SupNek/Skip-gram_with_neg_sampling/tree/main/models) — содержит сохраненные версии обученных моделей Skip-gram with negative sampling в виде файлов `.pth`
 * [Папка `skip-gram`](https://github.com/SupNek/Skip-gram_with_neg_sampling/tree/main/skip-gram) — содержит все функции необходимые для создания, обучения и созранения модели. Структура:
    * [`data_preprocessing.py`](https://github.com/SupNek/Skip-gram_with_neg_sampling/blob/main/skip-gram/data_preprocessing.py) — отвечает за предобработку данных: составление словаря допустимых слов, выделение контекстных пар, реализации функций для positive sampling и negative sampling;
    * [`data_setup.py`](https://github.com/SupNek/Skip-gram_with_neg_sampling/blob/main/skip-gram/data_setup.py) — отвечает за преобразование предобработанных данных в `torch.utils.data.Dataset` — `Word2VecDataset`, после чего используется `DataLoader` для эффективного генерирования батчей для обучения модели;
    * [`engine.py`](https://github.com/SupNek/Skip-gram_with_neg_sampling/blob/main/skip-gram/engine.py) — содержит функцию `train` отвечающую за обучение модели;
    * [`model_builder.py`](https://github.com/SupNek/Skip-gram_with_neg_sampling/blob/main/skip-gram/model_builder.py) — содержит реализацию модели Skip-gram with negative sampling;
    * [`utils.py`](https://github.com/SupNek/Skip-gram_with_neg_sampling/blob/main/skip-gram/utils.py) — содержит функцию, реализующую сохранение обученной модели;
    * [`main.py`](https://github.com/SupNek/Skip-gram_with_neg_sampling/blob/main/skip-gram/main.py) — основной файл, вызывает все вышеперечисленные, осуществляющий полный пайплайн обучения модели — от предобработки данных, создания датасета и даталоадера до обучения модели и ее созранения.
* [`Ноутбук Skip-gram.ipynb`](https://github.com/SupNek/Skip-gram_with_neg_sampling/blob/main/Skip-gram.ipynb) — содержит весь пайплайн создания и обучения модели с комментариями, также в нем присутствует визуализация тестирования модели;
* [`Ноутбук Skip-gram-going-modular.ipynb`](https://github.com/SupNek/Skip-gram_with_neg_sampling/blob/main/Skip-gram-going-modular.ipynb) — использовался для перехода от jupyter-notebook к модульному оформлению проекта, с его помощью удобно редактировать содержимое всех рабочих скриптов проекта.

### Запуск программы

Запуск осуществляется через запуск файла `./skip-gram/main.py`

``` sh
python ./skip-gram/main.py
```

### Библиотеки

* string, collections.Counter, itertools.chain, numpy — утилиты для подготовки данных
* nltk.tokenize.WordPunctTokenizer — токенизация предложений
* torch: torch.autograd, torch.nn, torch.nn.functional, torch.optim, torch.optim.lr_scheduler — основная библиотека для построения модели, ее обучения и тестирования
* tqdm.auto

### Известные проблемы и необходимые доработки

* На данный момент использование вероятностей для выбора примеров в обучающую выборку (positive sampling) приводит к некорректным результатам. Необходимо исследовать данную проблему, так как использование positive sampling теоретически должно улучшить точность работы модели и ее эффективность;

* Необходимо добавить использование списка стоп-слов (например, с использованием nltk.corpus.stopwords). Это бы уменьшило влияние стоп-слов на составление контекстных пар, подсчет вероятностей при positive sampling и улучшило бы итоговое качество модели;

* Возможно добавление передачи параметров модели при запуске через командную строку.


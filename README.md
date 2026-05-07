# Music Recommendation System on Yambda Dataset

## Описание проекта

Проект посвящён построению и сравнению нескольких подходов к рекомендательным системам на музыкальном датасете Yambda. В рамках работы были реализованы:

* preprocessing и очистка interaction-данных;
* построение sparse user-item matrix;
* визуальный анализ embedding-пространства пользователей;
* baseline collaborative filtering модель на ALS;
* neural recommendation model на DeepFM;
* pairwise ranking модель BPR;
* hyperparameter optimization через Optuna;
* сравнение моделей по ranking-метрикам.

Проект ориентирован на implicit feedback сценарий и демонстрирует полный ML pipeline для задачи recommendation systems.

---

# Используемый стек

## Основные библиотеки

* Python
* PyTorch
* Pandas
* NumPy
* SciPy Sparse
* Scikit-learn
* Optuna
* RecBole
* implicit
* Matplotlib

## Методы и алгоритмы

* Collaborative Filtering
* Matrix Factorization
* ALS (Alternating Least Squares)
* DeepFM
* BPR (Bayesian Personalized Ranking)
* Truncated SVD
* PCA
* t-SNE
* Spectral Embedding

---

# Dataset

Использовался датасет Yambda с музыкальными пользовательскими взаимодействиями.

Типы событий:

* listen
* like
* dislike
* unlike
* undislike

После загрузки данных выполнялась фильтрация:

* удалялись пользователи с количеством событий < 10;
* удалялись треки с количеством уникальных пользователей < 10;
* фильтрация применялась итеративно 2 раза.

Это позволило уменьшить sparsity и повысить стабильность обучения моделей.

---

# Feature Engineering

## Взвешивание implicit feedback

Для различных типов событий использовались разные веса:

| Event     | Score |
| --------- | ----- |
| like      | 5.0   |
| dislike   | 0.001 |
| unlike    | 0.01  |
| undislike | 1.0   |

Для события `listen` вес рассчитывался динамически на основе времени прослушивания:

* чем больше duration, тем выше interaction weight;
* значения clipping ограничивали влияние аномалий.

Таким образом был построен weighted implicit feedback.

---

# Train / Validation / Test Split

Разделение производилось по времени взаимодействий.

Для каждого пользователя:

* первые 80% interactions → train;
* следующие 10% → validation;
* последние 10% → test.

Такой temporal split корректно моделирует production recommendation scenario.

---

# Построение Sparse Matrix

Была реализована функция создания согласованных user/item индексов:

* categorical encoding пользователей;
* categorical encoding треков;
* построение CSR sparse matrix.

Использовались:

* user-item matrix;
* item-user matrix.

Это позволило эффективно обучать ALS и проводить retrieval.

---

# Анализ embedding-пространства

Для исследования структуры пользовательских взаимодействий были применены методы снижения размерности.

## Truncated SVD

* количество компонент: 30;
* анализ cumulative explained variance.

## PCA

Выполнена 3D-визуализация пользователей после SVD.

## t-SNE

Построено нелинейное embedding-пространство пользователей.

## Spectral Embedding

Проведён manifold-анализ пользовательских взаимодействий.

Эти визуализации использовались для исследования кластеризации пользовательских предпочтений.

---

# ALS Model

## Подход

Использована реализация ALS из библиотеки `implicit`.

Модель обучалась на implicit feedback interactions.

## Hyperparameter Optimization

Оптимизация выполнялась через Optuna.

Подбирались параметры:

* factors;
* regularization;
* alpha.

## Лучшие параметры

```python
factors = 173
regularization = 0.06247838089752494
alpha = 1.2696487957456881
```

## Лучший результат

```text
NDCG@20 = 0.0752
```

## Финальные метрики ALS

| Metric       | Score  |
| ------------ | ------ |
| MAP@20       | 0.0212 |
| Precision@20 | 0.0713 |
| NDCG@20      | 0.0706 |

ALS показал лучший результат среди моделей, обучавшихся в режиме full ranking.

---

# DeepFM Model

## Подход

Для neural recommendation использовалась модель DeepFM через библиотеку RecBole.

DeepFM сочетает:

* factorization machines;
* deep neural network;
* feature interaction learning.

## Подготовка данных

Данные были конвертированы в формат RecBole:

* user_id:token
* item_id:token
* label:float
* timestamp:float

## Hyperparameter Optimization

Подбирались:

* embedding_size;
* learning_rate;
* dropout_prob;
* architecture MLP.

## Лучшие параметры

```python
embedding_size = 128
learning_rate = 0.0005
dropout_prob = 0.3
mlp_hidden = [128, 128, 128]
```

## Лучший результат

```text
NDCG@20 = 0.0480
```

## Финальные метрики DeepFM

| Metric       | Score  |
| ------------ | ------ |
| MAP@20       | 0.0145 |
| Precision@20 | 0.0264 |
| NDCG@20      | 0.0332 |

В рамках данного датасета DeepFM уступил ALS, что характерно для sparse implicit feedback сценариев без rich side-features.

---

# BPR Model

## Подход

Также была реализована модель BPR (Bayesian Personalized Ranking).

BPR обучается через pairwise ranking loss:

* positive interaction;
* sampled negative interaction.

## Hyperparameter Optimization

Подбирались:

* embedding_size;
* learning_rate.

## Лучшие параметры

```python
embedding_size = 128
learning_rate = 0.0003834013351473748
```

## Лучший результат

```text
NDCG@20 = 0.7526
```

## Финальные метрики BPR

| Metric       | Score  |
| ------------ | ------ |
| MAP@20       | 0.8079 |
| Precision@20 | 0.8007 |
| NDCG@20      | 0.8758 |

---

# Важное замечание по сравнению моделей

Метрики BPR значительно выше ALS и DeepFM по причине различий в evaluation protocol.

## ALS и DeepFM

Обучались и оценивались в режиме:

```python
mode = 'full'
```

Модель искала релевантный трек среди всего каталога ≈ 148000 треков.

## BPR

Оценивался в режиме:

```python
mode = 'uni100'
```

Модель выбирала 1 релевантный объект среди 100 негативных примеров всего 101 кандидат.

Метрики значительно разнятся и это отражено в solution.ipynb.

---

# Сравнение моделей

## Full Ranking Scenario

| Metric       | ALS    | DeepFM |
| ------------ | ------ | ------ |
| MAP@20       | 0.0212 | 0.0145 |
| Precision@20 | 0.0713 | 0.0264 |
| NDCG@20      | 0.0706 | 0.0332 |

## Вывод

На данном датасете ALS показал себя лучше DeepFM:

* устойчивее к sparsity;
* проще оптимизируется;
* лучше работает на implicit interactions без дополнительных features.

DeepFM требует:

* richer feature space;
* side information;
* большего объёма dense interactions.

---

# Основные результаты проекта

* Реализован end-to-end recommendation pipeline.
* Построены baseline и neural recommendation модели.
* Проведено сравнение collaborative filtering подходов.
* Реализована автоматическая оптимизация гиперпараметров.
* Выполнен анализ embedding-пространства пользователей.
* Проведён корректный temporal split для offline evaluation.
* Исследованы особенности implicit feedback recommendation.

# Breast Cancer Wisconsin (Diagnostic) — классификация B/M

Этот ноутбук решает задачу бинарной классификации: по 30 числовым признакам опухоли предсказать диагноз  
**B (benign, доброкачественная) = 0** и **M (malignant, злокачественная) = 1**.

> В задаче упор делается на **recall для класса 1 (malignant)** — важно *не пропустить* злокачественные случаи.

---

## Датасет

Ноутбук скачивает данные командой `wget` и читает CSV:

- файл: `breast_cancer_wisconsin.csv`
- источник: ссылка зашита в ноутбуке (скачивание происходит автоматически)

---

## Что внутри ноутбука

### 1) Подготовка данных
- Загрузка датасета и базовый просмотр
- `diagnosis` маппится в бинарный таргет: `{'B': 0, 'M': 1}`
- Удаляются столбцы `id`, `diagnosis`, `Unnamed: 32`
- Train/Test split: `test_size=0.50`, `stratify=y`, `random_state=42`
- `reset_index(drop=True)` для единых индексов после split

### 2) EDA и визуализация распределений
- Проверка пропусков (в т.ч. `missingno`)
- Гистограммы распределений признаков с разбиением по классам (`hue=y`)
- Boxplot’ы по группам признаков:
  - `*_mean`
  - `*_se`
  - `*_worst`
- Кластерная тепловая карта корреляций (`sns.clustermap`) на сырых и преобразованных признаках

### 3) Нормализация / преобразование признаков
Используется:
- `PowerTransformer(method="yeo-johnson")` внутри `Pipeline`

⚠️ Важно: `PowerTransformer` **по умолчанию делает стандартизацию** (`standardize=True`), поэтому отдельный `StandardScaler` обычно не нужен.

### 4) Статистический анализ признаков
- Независимый двухвыборочный t-тест (`ttest_ind`) по каждому признаку
- Таблица `diff_df` с колонками:
  - `T`, `P`, `neglogP = -log10(P)`
  - `difference = mean(class=1) - mean(class=0)`
- Сортировка по `neglogP` для поиска наиболее различающихся признаков

### 5) Модели и оценка качества
- Базовый классификатор: `SGDClassifier(loss='log_loss')` (логистическая регрессия через SGD)
- Эксперименты с PCA + CV (ROC-AUC)
- Снижение размерности через **PLSRegression(n_components=2)** + `SGDClassifier`
- Матрица ошибок (Confusion Matrix) и визуализация ошибочных точек на плоскости компонент

---

## Метрики (что получилось в ноутбуке)

В ноутбуке считается набор метрик:
- `balanced_accuracy`, `recall`, `precision`, `f1_score`, `accuracy`, `roc_auc`
- + confusion matrix

Пример результатов из `df_res`:

- **LogReg_base (на X_tr_norm):**
  - recall ≈ **0.953**
  - roc_auc ≈ **0.994**
  - accuracy ≈ **0.965**

- **LogReg_PLS (PLS 2 компоненты + LogReg):**
  - recall ≈ **0.991**
  - roc_auc ≈ **0.999**
  - accuracy ≈ **0.979**

---

## Как запустить

### Вариант A — Google Colab
1. Открой `.ipynb` в Colab
2. Выполни `Runtime → Run all`
3. CSV скачается автоматически командой `wget`

### Вариант B — локально
```bash
pip install -U numpy pandas scipy matplotlib seaborn scikit-learn missingno

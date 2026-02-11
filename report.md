## Содержание

- [Описание шагов](#описание-шагов)
- [Описание и результаты каждого эксперимента](#описание-и-результаты-каждого-эксперимента)
- [Как воспроизвести мою модель?](#как-воспроизвести-мою-модель)
- [Что сработало, а что нет?](#что-сработало-а-что-нет)

---

# Описание шагов

## 1. Как извлекаются pitch/energy/duration

### **Pitch (F0) - Основная частота**
- **Метод**: `torchaudio.functional.detect_pitch_frequency()`
- **Параметры**:
  - `sample_rate`: 22050 Hz
  - `hop_length`: 256 samples
  - `f_min`: 80 Hz
  - `f_max`: 400 Hz


### **Energy - Энергия**
- **Метод**: L2-норма по мел-спектрограмме
- **Формула**: $energy = ||mel||_2$ (сумма по мел-бинам для каждого кадра)

### **Duration - Длительность**
- **Метод**: Montreal Forced Aligner (MFA) textgrid файлы
- **Процесс**:
  1. Читаются textgrid alignments
  2. Конвертация времени в кадры: `frame = time * sample_rate / hop_length`
  3. Длительность = количество кадров для каждой фонемы
  4. Минимум: 1 кадр на фонему

---

## 2. Как нормализуются/обрабатываются эти параметры

### **Pitch нормализация**
- **Метод**: Z-score нормализация -> диапазон `[-3, 3]`
- **Формула**: `normalized_pitch = (pitch_hz - mean) / std`
- **Статистика**:
  - Вычисляется глобально по всему датасету только для вокализованных кадров
  - Сохраняется в `data/pitch_stats.json`
  - Текущие значения: `mean=132.83 Hz`, `std=36.07 Hz`
- **Обработка**:
  - Невокализованные кадры (pitch=0) не нормализуются
  - Клиппинг выбросов: [80, 400] Hz
  - Хранится в нормализованном виде в `data/features/pitch/*.npy`
- **Денормализация при инференсе**

### **Energy нормализация**
- **Метод**: Min-max нормализация -> диапазон `[0, 1]`
- **Формула**: `normalized_energy = (energy - min) / (max - min)`
- **Особенности**:
  - Нормализуется для каждого семпла отдельно (а не глобально)
  - Фиксированные пределы в конфиге: `energy_min=0.0`, `energy_max=1.0`
- **Денормализация при инференсе**

### **Duration обработка**
- **Метод**: log-трансформация
- **Обратное преобразование при инференсе**

---


# Описание и результаты каждого эксперимента

## Эксперимент 1: One-Batch Test

**Цель**: Проверить, что модель способна идеально переобучиться на маленьком датасете (4 сэмпла)

**Конфигурация**: `onebatch_test.yaml`

**Итоговые метрики (эпоха 6)**:
- Val Loss: **0.0265**
- Mel MAE: **0.00539**
- Duration Accuracy: **100%**
- Pitch MAE: **0.071**
- Energy MAE: **0.024**
- Emotion Loss: **0.0** (идеальная классификация)

**Время обучения**: ~1.5 минуты на эпоху (CPU)

**Comet**: https://www.comet.com/ldvrn01/tts-project/89d13f495cb0441d81be8bfa8f0d73be?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step

**Вывод**: Модель успешно переобучается, достигая около нулевого лосса -> архитектура и лосс функции работают корректно.

---

## Эксперимент 2: Baseline Training

**Цель**: Обучить базовую модель FastSpeech2 на всем датасете RUSLAN.

**Конфигурация**: `fastspeech2_train.yaml`, но веса всех фичей =1 и максимальная длительность аудио 36 секунд.


**Лучшие метрики (эпоха 20)**:
- Val Loss: **0.1557**
- Mel MAE: **0.0669**
- Duration Accuracy: **88.3%**
- Pitch MAE: **0.559**
- Energy MAE: **0.139**

**Comet**: https://www.comet.com/ldvrn01/tts-project/mm75in8uiys01z93ql23m14f8tcvlbng?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step

**Вывод**: Обучение прервано на 20 эпохе, так как плохо обучалась, было принято решение поменять веса фичей (lambda_mel, lambda_duration, lambda_pitch, lambda_energy) и брать датасет с максимальной длительностью аудио в 10 секунд, минимальной - 2 секунды.

---

## Эксперимент 3: Best Training


**Цель**: Обучить базовую модель FastSpeech2 на всем датасете RUSLAN с учетом предыдущих ошибок.

**Конфигурация**: `fastspeech2_train.yaml`. В процессе прерывал обучение и перебалансировал веса + менял размер батч, поэтому кривые метрик такие странные.


**Лучшие метрики (эпоха 186)**:
- Val Loss: **0.789**
- Mel MAE: **0.104**
- Duration Accuracy: **97.5%**
- Pitch MAE: **0.425**
- Energy MAE: **0.088**

**Comet**: https://www.comet.com/ldvrn01/tts-project/kmd4l81rsqqtjik7i9h3y0xkta26y0me?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=duration

**pdf**: https://storage.yandexcloud.net/fastspeech2-russian/cometml_report.pdf

**Вывод**: получил неплохую модель на выходе, но был потенциал для дальнейшего улучшения.


---

## Эксперимент 4: Fine-tuning

**Цель**: Дотюнить обученную модель FastSpeech2

**Конфигурация**: `fastspeech2_v1_finetuning.yaml`. Ещееее больше веса мел-лоссу.

**Лучшие метрики (эпоха 251)**:
- Val Loss: **1.060**
- Mel MAE: **0.101**
- Duration Accuracy: **97.8%**
- Pitch MAE: **0.422**
- Energy MAE: **0.088**

**Вывод**: Было прервано на 251 эпохе (деньги на датасфере заканчивались -_- ). Так как менял вес mel-лосса, то итоговый лосс вырос, хотя отдельные составлящие упали, что безусловно хорошо.

---

## Эксперимент 5: Emotion Fine-Tuning

**Цель**: Добавить контроль эмоций к предобученной модели без потери качества

### Фаза № 1 - Заморожены: encoder, decoder, все variance predictors (кроме emotion) (10 эпох)

**Конфигурация**: `fastspeech2_emotion_finetune_phase1.yaml`

**Лучшие метрики**:
- Val Loss: **2.377**
- Mel MAE: **0.132**
- Duration Accuracy: **97.8%**
- Pitch MAE: **0.423**
- Energy MAE: **0.094**


### Фаза № 2 - Полный файнтюнинг

**Конфигурация**: `fastspeech2_emotion_finetune_phase2.yaml`

**Лучшие метрики (прервано на 13 эпохе)**:
- Val Loss: **2.029**
- Mel MAE: **0.102**
- Duration Accuracy: **97.8**
- Pitch MAE: **0.420**
- Energy MAE: **0.094**

**Вывод**: удалось добавить тюнинг эмоций в задаче TTS.

---

# Как воспроизвести мою модель?

```bash
# Подготовка окружения
git clone https://github.com/yourusername/fastspeech2-russian.git
cd fastspeech2-russian
conda create -n fs2 python=3.10
conda activate fs2
pip install -r requirements.txt
```

```bash
# Получение данных через DVC
dvc pull data/features
dvc pull data/phoneme_vocabulary.txt
dvc pull data/pitch_stats.json.dvc
```

(Либо подготовка features с нуля с использованием MFA):

```bash
# 1. MFA alignment
mfa align data/ruslan_mfa_corpus russian_mfa russian_mfa \
    data/ruslan_alignments --clean --num_jobs 8

# 2. Извлечение duration
python scripts/preprocessing/extract_duration_from_mfa.py \
    --alignments_dir data/ruslan_alignments \
    --output_dir data/features/duration

# 3. Извлечение mel/pitch/energy
python scripts/preprocessing/preprocess_ruslan.py \
    --output_dir data/features \
    --num_workers 4

# 4. Генерация text_tokens
python scripts/preprocessing/generate_text_tokens.py
```

```bash
# One-batch test (для проверки)
python train.py -cn onebatch_test

# Обучение базовой модели
python train.py -cn fastspeech2_train \
    trainer.n_epochs=30 \
    writer.run_name="fastspeech2_baseline"
```


```bash
# Fine-tuning с эмоциями
# фаза 1: Только emotion компоненты
python train.py -cn fastspeech2_emotion_finetune_phase1 \
    trainer.resume_from="saved/fastspeech2_train/model_best.pth" \
    writer.run_name="emotion_phase1"

# фаза 2: Полное дообучение
python train.py -cn fastspeech2_emotion_finetune_phase2 \
    trainer.resume_from="saved/fastspeech2_emotion_finetune_phase1/model_best.pth" \
    writer.run_name="emotion_phase2"
```


```bash
# Базовый inference
python inference.py \
    --checkpoint saved/fastspeech2_emotion_finetune_phase2/model_best.pth \
    --text "Привет, как дела?" \
    --output output.npy

# С управлением эмоциями и параметрами
python inference.py \
    --checkpoint saved/fastspeech2_emotion_finetune_phase2/model_best.pth \
    --text "Это потрясающе!" \
    --duration_control 0.9 \
    --pitch_control 1.1 \
    --energy_control 1.2 \
    --emotion_control 1.0 \
    --output output.npy
```

---


# Что сработало, а что нет?

## Что сработало

- Архитектура FastSpeech2 работает корректно, модель обучается ~~и даже что-то предсказывает~~. Доказательством тому one-batch test.
- MFA завелся и отлично показал себя.
- Отдельные фичи в variance adaptor хорошо предсказываются (MAE < 0.5, местами < 0.1).
- Успешно интегрирован вокодер WaveGlow (на мел-спектрограммах RUSLAN хорошо воспроизводит аудио).
- Суммарное обучение на A100 финальной модели уложилось в 30 часов.
- Удалось внедрить DVC для хранения данных и моделей.
- Удалось добавить дополнительный регулятор эмоции в variance adaptor.
- Отлично сработала двухфазное дообучене на контроль эмоций без переобучения всей модели.
- Удалось залогировать все обучение в CometML (метрики, мел-спектрограммы, аудио).


---

## Что не сработало / основные сложности

- CometML перестал работать без впн....
- Пришлось прерывать обучение и менять конфиг дабы сбалансировать веса в финальном лоссе.
- На CPU обучение слишком долгое, пришлось переехать на GPU.
- На выходе финальной модели все еще много шума, разборчивы только 2-3 слова в начале предложения.
- Недостаток GPU ресурсов (после 30ч на датасфере деняк не осталось).
- Не успел протестировать расширенную архитектуру.


В целом эти 2 недели были очень непростыми, но опыт очень крутой!


---

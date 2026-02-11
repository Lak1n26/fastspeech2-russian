# Text-to-Speech (TTS)


<p align="center">
  <a href="#установка">Установка</a> •
  <a href="#быстрый-старт">Быстрый старт</a> •
  <a href="#демо">Демо</a> •
  <a href="#доступные-конфигурации">Доступные конфигурации</a> •
  <a href="#структура-проекта">Структура проекта</a> •
  <a href="#результаты-финальной-модели">Результаты</a> •
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
<a href="https://github.com/Lak1n26/fastspeech2-russian/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
<a href="https://github.com/Lak1n26/fastspeech2-russian/blob/main/CITATION.cff">
   <img src="https://img.shields.io/badge/cite-this%20repo-purple">
</a>
</p>

## О проекте

Реализация с нуля модели FastSpeech2 для синтеза речи на русском языке. Модель обучена на наборе данных RUSLAN. Используется предобученный вокодер - WaveGlow.

> [!IMPORTANT]
> **Эта модель поддерживает только русский язык.**

## Установка

1. Клонирование репозиторий:

```bash
git clone https://github.com/Lak1n26/fastspeech2-russian.git
cd fastspeech2-russian
```

2. Создание виртуальной среды:

```bash
conda create -n fs2 python=3.10
conda activate fs2
```

3. Установка зависимостей:

```bash
pip install -r requirements.txt
```

4. (Опционально) Получить данные через DVC

```bash
# Для inference - только модель
dvc pull saved/fastspeech2_training data/phoneme_vocabulary.txt

# Для обучения - features
dvc pull data/features data/phoneme_vocabulary.txt

# Для полной разработки - всё
dvc pull

# Подробнее см. раздел "Data Version Control (DVC)"
```

5. (опционально) Установка pre-commit hooks:

```bash
pre-commit install
```

6. (опционально) Настройка ключей API для запуска обучения:

```bash
export COMET_API_KEY=YOUR_API_KEY
```

7. (Опционально) Установить Montreal Forced Aligner

> [!IMPORTANT]
> **MFA должен быть установлен через conda, НЕ через pip!**

```bash
# Создать отдельное окружение для MFA
conda create -n mfa python=3.10
conda activate mfa
conda install -c conda-forge montreal-forced-aligner

# Проверить установку
mfa version

# Загрузить модели для русского языка
mfa model download acoustic russian_mfa
mfa model download dictionary russian_mfa
```

## Подготовка данных

```bash
# 1. Подготовить корпус для MFA
python prepare_mfa_corpus.py \
    --metadata data/metadata_RUSLAN_22200.csv \
    --audio_dir data/RUSLAN \
    --output_dir data/ruslan_mfa_corpus

# 2. Запустить MFA alignment
mfa align \
    data/ruslan_mfa_corpus \
    russian_mfa \
    russian_mfa \
    data/ruslan_alignments \
    --clean \
    --num_jobs 8

# 3. Извлечь duration из alignments
python extract_duration_from_mfa.py \
    --alignments_dir data/ruslan_alignments \
    --output_dir data/features/duration \
    --hop_length 256 \
    --sample_rate 22050

# 4. Создать словарь фонем
python create_phoneme_vocabulary.py \
    --phonemes_dir data/features/phonemes \
    --output_path data/phoneme_vocabulary.txt

# 5. Извлечь остальные features
python preprocess_ruslan.py \
    --output_dir data/features \
    --num_workers 4

# 6. Генерация text_tokens из phonemes
python generate_text_tokens.py
```



## Быстрый старт

### One-Batch Test

```bash
python3 train.py --config-name=onebatch_test
```


### Обучение на RUSLAN:

```bash
python3 train.py -cn fastspeech2_train
```


### Обучение с пользовательскими параметрами

```bash
python train.py -cn fastspeech2_train \
    trainer.n_epochs=200 \
    optimizer.lr=1e-3 \
    dataloader.batch_size=32
```


### Инференс

```bash
# простая генерация
python inference.py \
    --checkpoint saved/onebatch_test/model_best.pth \
     --text "Кого интересуют признания литературного неудачника?" \
    --output output.npy
```

```bash
# с управлением характеристиками
python inference.py \
    --checkpoint saved/models/fastspeech2/best_model.pth \
    --text "Привет, как дела?" \
    --output output.npy \
    --duration_control 0.8 \  # 20% быстрее
    --pitch_control 1.2 \     # 20% выше
    --energy_control 1.1      # 10% громче
```


```bash
# batch inference по файлу
python inference.py \
    --checkpoint saved/models/fastspeech2/best_model.pth \
    --text_file inference_texts.txt \
    --output_dir outputs/
```

## Демо

Пример работы с проектом представлен в [demo.ipynb](demo.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1k66tV3N9oRbSrvUoDNajxskdjwaTUW3w?usp=sharing?usp%3Dsharing)



## Доступные конфигурации

### Основные конфигурации

| Конфигурация | Описание |
|--------------|----------|
| `onebatch_test` | One-Batch Test |
| `fastspeech2_train` | Обучение основной модели |
| `fastspeech2_v1_finetuning` | Дообучение основной модели |
| `fastspeech2_emotion_finetune_phase1` | Добавление дополнительного контроля эмоций (бОльшая часть весов заморожена) |
| `fastspeech2_emotion_finetune_phase2` | Добавление дополнительного контроля эмоций (все веса разморожены, низкий LR) |

### Параметры управления

- **duration_control**: скорость речи
  - `< 1.0` - быстрее (0.8 = на 20% быстрее)
  - `= 1.0` - нормально
  - `> 1.0` - медленнее (1.2 = на 20% медленнее)

- **pitch_control**: высота тона
  - `< 1.0` - ниже (мужской голос)
  - `= 1.0` - нормально
  - `> 1.0` - выше (женский голос)

- **energy_control**: громкость
  - `< 1.0` - тише
  - `= 1.0` - нормально
  - `> 1.0` - громче

- **emotion_control**: эмоция
   - `= 0.0` - нейтрально
   - `= 1.0` - нормально



## Структура проекта


```
fastspeech2-russian/
├── fastspeech2/                     # Основной модуль
│   ├── configs/                     # Hydra конфигурации
│   │   ├── dataloader/              # Настройки DataLoader
│   │   │   ├── overfit.yaml
│   │   │   ├── tts_features.yaml
│   │   │   ├── tts_inference.yaml
│   │   │   └── tts_train.yaml
│   │   ├── datasets/                # Настройки датасетов
│   │   │   ├── ruslan_features.yaml
│   │   │   ├── ruslan_features_emotion.yaml
│   │   │   └── ruslan_overfit_split.yaml
│   │   ├── metrics/                 # Настройки метрик
│   │   │   └── metrics.yaml
│   │   ├── model/                   # Настройки моделей
│   │   │   ├── fastspeech2.yaml
│   │   │   └── fastspeech2_large.yaml
│   │   ├── writer/                  # Настройки логгеров
│   │   │   ├── cometml.yaml
│   │   │   └── wandb.yaml
│   │   ├── onebatch_test.yaml       # One-batch overfitting test
│   │   ├── fastspeech2_train.yaml   # Основное обучение
│   │   ├── fastspeech2_v1_finetuning.yaml      # Fine-tuning v1
│   │   ├── fastspeech2_emotion_finetune_phase1.yaml  # Emotion: Phase 1
│   │   └── fastspeech2_emotion_finetune_phase2.yaml  # Emotion: Phase 2
│   ├── datasets/                    # Dataset классы
│   │   ├── base_dataset.py          # Базовый датасет
│   │   ├── collate.py               # Collate функции
│   │   ├── data_utils.py            # Утилиты для данных
│   │   ├── example.py               # Пример датасета
│   │   ├── feature_extraction.py    # Извлечение признаков
│   │   ├── ruslan_dataset.py        # Датасет RUSLAN (raw audio)
│   │   ├── ruslan_feature_dataset.py # Датасет RUSLAN (features)
│   │   └── tts_collate.py           # TTS collate функции
│   ├── logger/                      # Система логирования
│   │   ├── logger.py                # Базовый логгер
│   │   ├── logger_config.json       # Конфигурация логирования
│   │   ├── utils.py                 # Утилиты логирования
│   │   ├── cometml.py               # CometML интеграция
│   │   └── wandb.py                 # Weights & Biases интеграция
│   ├── loss/                        # Loss функции
│   │   └── fastspeech2_loss.py      # FastSpeech2 loss
│   ├── metrics/                     # Метрики
│   │   ├── base_metric.py           # Базовая метрика
│   │   ├── tracker.py               # Трекер метрик
│   │   └── tts_metrics.py           # TTS метрики
│   ├── model/                       # Модель FastSpeech2
│   │   ├── blocks.py                # Multi-Head Attention, FFT Block
│   │   ├── encoder_decoder.py       # Encoder & Decoder
│   │   ├── variance_adaptor.py      # Duration/Pitch/Energy/Emotion predictors
│   │   └── fastspeech2.py           # Полная модель FastSpeech2
│   ├── trainer/                     # Training loop
│   │   ├── base_trainer.py          # Базовый трейнер
│   │   ├── trainer.py               # TTS трейнер
│   │   └── inferencer.py            # Inference класс
│   └── utils/                       # Утилиты
│       ├── audio_utils.py           # Аудио утилиты
│       ├── init_utils.py            # Инициализация
│       ├── io_utils.py              # I/O утилиты
│       ├── train_utils.py           # Утилиты обучения (freezing/unfreezing)
│       └── tts_utils.py             # TTS утилиты
├── data/                            # Данные (→ DVC)
│   ├── RUSLAN.dvc                   # Аудио файлы RUSLAN
│   ├── metadata_RUSLAN_22200.csv.dvc # Метаданные датасета
│   ├── features.dvc                 # Извлеченные признаки (mel/pitch/energy/duration)
│   ├── ruslan_alignments.dvc        # MFA alignments (TextGrid)
│   ├── phoneme_vocabulary.txt.dvc   # Словарь фонем
│   ├── phoneme_to_idx.txt.dvc       # Маппинг фонем в индексы
│   ├── pitch_stats.json.dvc         # Статистика pitch для нормализации
│   ├── dataset_audio_stats.json.dvc # Статистика аудио
│   ├── transcriptions_clean_ruslan.txt.dvc  # Чистые транскрипции
│   ├── emotion_labels_hf_test.json.dvc      # Эмоциональные метки
│   └── .gitignore                   # Игнорируемые файлы данных
├── scripts/                         # Вспомогательные скрипты
│   ├── annotation/                  # Скрипты аннотации
│   │   └── annotate_emotions_hf.py  # Аннотация эмоций через HuggingFace
│   └── preprocessing/               # Скрипты предобработки
│       ├── prepare_mfa_corpus.py    # Подготовка корпуса для MFA
│       ├── extract_duration_from_mfa.py  # Извлечение duration из MFA
│       ├── preprocess_ruslan.py     # Предобработка датасета RUSLAN
│       ├── create_phoneme_vocabulary.py  # Создание словаря фонем
│       ├── generate_text_tokens.py  # Генерация токенов текста
│       └── prepare_transcriptions.py  # Подготовка транскрипций
├── waveglow/                        # WaveGlow вокодер
│   ├── model.py                     # Модель WaveGlow
│   ├── modules.py                   # Модули WaveGlow
│   ├── dataset.py                   # Датасет для WaveGlow
│   └── logging.py                   # Логирование WaveGlow
├── .dvc/                            # DVC конфигурация
│   ├── config                       # Remote storage настройки
│   └── .gitignore                   # DVC служебные файлы
├── train.py                         # Скрипт обучения
├── inference.py                     # Скрипт inference
├── demo.ipynb                       # Демо ноутбук
├── report.md                        # Отчет об экспериментах
├── requirements.txt                 # Python зависимости
├── waveglow_params.json             # Параметры WaveGlow
├── CITATION.cff                     # Информация для цитирования
├── LICENSE                          # MIT License
├── README.md                        # Этот файл
├── .gitignore                       # Git игнорируемые файлы
└── .dvcignore                       # DVC игнорируемые файлы
```


## Результаты финальной модели

| Метрика           | Значение  |
|-------------------|-----------|
| val loss          |    2.029  |
| mel MAE           |    0.102  |
| Duration Accuracy |    97.8%  |
| Pitch MAE         |    0.420  |
| Energy MAE        |    0.094  |

Более подробно результаты экспериментов описаны в [Отчете](report.md)

## Лицензия

MIT License

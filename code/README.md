# Код воспроизведения Exclusive Self Attention

## Состав

| Файл в `fineweb/` | Назначение |
|---|---|
| `model.py` | Transformer с переключением SA/XSA |
| `prepare.py` | FineWeb-100BT, GPT-2 токенизация, разделение документов |
| `verify_data.py` | Проверка SHA256 и отсутствия пересечения документов между split |
| `train.py` | Обучение, одинаковые target tokens, validation/test, checkpoints и resume |
| `suite.py` | Выбор LR и все 12 парных обучений по фиксированному плану |
| `benchmark.py` | Время и память блока; native BF16 и FP32-проекция |
| `diagnostics.py` | Диагностика attention и численной ортогональности |

## Запуск

Нужны Python 3.11+, CUDA GPU с BF16 и зависимости из `requirements.txt`. В выполненном эксперименте использовалась RTX 3090 24 GiB.


```bash
pip install -r requirements.txt
python -m fineweb.prepare
python -m fineweb.verify_data
python -m fineweb.benchmark
python -m fineweb.suite --hours 8
python -m fineweb.diagnostics
```

Самостоятельный запуск одной модели:

```bash
python -m fineweb.train --out results/single_xsa --mode XSA --seed 17 --context 2048 --lr 0.001
```

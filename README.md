# Гиперсети

## Описание

В данном репозитории находится код и результаты экспериментов для курсового проекта по теме "Гиперсети". Репозиторий имеет следующую структуру:

## Структура

Репозиторий имеет следующую файловую структуру:

- experiments - результаты экспериментов для разных архитектур. 

    - experiments/{architecture}/accuracies - значения accuracy для разных конфигураций моделей в течение континуального обучения

    - experiments/{architecture}/plots - столбчатые диаграммы, демонстрирующие работу разных конфигураций моделей в континуальном обучении

- notebooks - ноутбуки для запуска экспериментов и построения графиков

- utils - вспомогательные инструменты для реализации гиперсетей

- src - исходный код

    - src/hnet_lib - код для гиперсети и для гипер-версий используемых слоёв (HyperLinear, HyperConv2d, HyperBatchNorm2d)

    - src/mlp - реализация полносвязной нейросети с поддержкой гиперсетей

    - src/resnet - реализация ResNet-18 с поддержкой гиперсетей

    - src/vit/custom - реализация ViT с поддержкой гиперсетей

    - src/vit/pretrained - модификация предобученной модели ViT с поддержкой гиперсетей для обучения последнего линейного слоя

В корне репозитория также находятся скрипты {model}_pipeline.py, которые позволяют запускать обучение моделей. Узнать список возможных аргументов можно с помощью команды:

```bash
./{model}_pipeline.py
```


#!/bin/bash

# 1. Определяем корень проекта (папка alex)
# Скрипт поймет, где он находится, и поднимется на уровень выше
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ALEX_ROOT="$SCRIPT_DIR/.."
cd "$ALEX_ROOT"

# 2. Настройки путей (теперь точно по результатам find)
SOURCE="./solver/default/solver.cpp"
TARGET="./solver/default/solver.exe"

echo "=== Окружение ==="
echo "Корень проекта: $ALEX_ROOT"
echo "Ищем файл: $SOURCE"

# 3. Проверка наличия файла перед компиляцией
if [ ! -f "$SOURCE" ]; then
    echo "ОШИБКА: Файл $SOURCE не найден!"
    echo "Содержимое папки solver/default/:"
    ls -la ./solver/default/
    exit 1
fi

echo "=== Компиляция ==="

# 4. Флаги компиляции:
# -O3: Максимальная скорость
# -mstackrealign: КРИТИЧНО для Windows (исправляет Segmentation fault)
# -march=native: Использовать инструкции вашего процессора (AVX/AVX2)
# -I .: Искать папку Eigen в текущей директории
g++ -O3 -mstackrealign -march=native -I . "$SOURCE" -o "$TARGET"

if [ $? -eq 0 ]; then
    echo "=== Запуск программы ==="
    echo "---------------------------"
    "$TARGET"
    echo "---------------------------"
else
    echo "--- ОШИБКА КОМПИЛЯЦИИ ---"
    exit 1
fi
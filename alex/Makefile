CXX = g++
CXXFLAGS = -O3 -march=native -I.

ifeq ($(OS),Windows_NT)
    CXXFLAGS += -mstackrealign
    EXE = .exe
else
    UNAME_S := $(shell uname -s)
    EXE =
endif

TARGET = a$(EXE)

SOURCES = main.cpp task1/task1.cpp solver/default/solver.cpp
HEADERS = task1/task1.hpp solver/default/solver.hpp

OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "=== Линковка ==="
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET)
	@echo "=== Сборка завершена ==="

%.o: %.cpp $(HEADERS)
	@echo "Компиляция: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	@echo "=== Запуск ==="
	@./$(TARGET)

run-test: $(TARGET)
	@echo "=== Запуск с тестовыми параметрами ==="
	@./$(TARGET) --Nx 100 --Ny 100 --method ldlt

clean:
	@echo "=== Очистка ==="
	rm -f $(TARGET) $(OBJECTS) solution.txt
	@echo "=== Очистка завершена ==="

rebuild: clean all

help:
	@echo "Доступные цели:"
	@echo "  make all      - Сборка проекта"
	@echo "  make run      - Сборка и запуск"
	@echo "  make run-test - Запуск с тестовыми параметрами (100x100)"
	@echo "  make clean    - Удалить файлы сборки"
	@echo "  make rebuild  - Полная пересборка"
	@echo "  make help     - Показать справку"

.PHONY: all run run-test clean rebuild help

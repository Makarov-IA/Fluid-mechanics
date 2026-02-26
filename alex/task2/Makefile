CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native

ifeq ($(OS),Windows_NT)
    CXXFLAGS += -mstackrealign
    EXE = .exe
else
    EXE =
endif

TARGET = task2$(EXE)
SOURCES = main.cpp solver.cpp
HEADERS = solver.hpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(ARGS)

run-test: $(TARGET)
	./$(TARGET) --Nx 51 --Ny 51 --nu 0.01 --lid 1 --forcing zero --tol 1e-5 --max-iter 12000 --output-dir results_test

test: $(TARGET)
	bash ./scripts/test.sh

plot:
	bash ./scripts/plot.sh --input-dir results --output-dir plots --field all --save --no-show

clean:
	rm -f $(TARGET) $(OBJECTS)

clean-results:
	rm -rf results results_test plots plots_test test_runs

help:
	@echo "make all                - build solver"
	@echo "make run ARGS='...'     - run with custom args"
	@echo "make run-test           - single quick run"
	@echo "make test               - run automated test scenarios"
	@echo "make plot               - build standard plots from results"
	@echo "make clean              - remove build files"
	@echo "make clean-results      - remove generated data/plots"

.PHONY: all run run-test test plot clean clean-results help

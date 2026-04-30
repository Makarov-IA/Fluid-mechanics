CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2
CPPFLAGS += -I. -I../External_libs

BUILD_DIR := build
TARGET := $(BUILD_DIR)/solver_app.exe
SOURCES := main.cpp solver/solver.cpp
HEADERS := solver/solver.hpp solver/solver_utils.hpp utils/config.hpp forcing/omega_forcing.hpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(SOURCES) -o $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(BUILD_DIR)/solver_app

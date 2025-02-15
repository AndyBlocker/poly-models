TARGET = demo

CXX = g++

CXXFLAGS = -std=c++11 -O3 -Iinclude -MMD -MP

SRC_DIRS = src/common src/layers src/models src

SRCS  = $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.cpp))

OBJS = $(SRCS:.cpp=.o)

DEPS = $(SRCS:.cpp=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)
CC= g++
CPPFLAGS= -std=c++11 -fopenmp -lm -Ofast
OBJS= app
LIBS= -L ../bin -lproNet

all: $(OBJS)

$(OBJS):
	$(CC) $(CPPFLAGS) $@.cpp $(LIBS) -o $@
clean:
	rm $(OBJS)

CC = g++
CPPFLAGS = -std=c++11 -fPIC -fopenmp -lm -Ofast
OBJECTS = util.o random.o APP-Modified.o ./model/APP.o
all: $(OBJECTS)
	mkdir -p ../bin
	ar rcs ../bin/libproNet.a $(OBJECTS)

clean:
	rm $(OBJECTS)
	rm ../bin/libproNet.a

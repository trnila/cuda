CUDADIR=/opt/cuda
PROJECT=program

all: $(PROJECT)

%.o: %.cu
	nvcc -c $<

%.o: %.cpp
	g++ -std=c++11 -I$(CUDADIR)/include -c $<

$(PROJECT): main.o check.o rotate.o grayscale.o resize.o border.o complex.o
	g++ $^ -lcudart -lcuda -lopencv_core -lopencv_highgui -L$(CUDADIR)/lib64 -o $@

clean:
	rm -f *.o $(PROJECT)


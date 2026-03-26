NVCC = nvcc
TARGET = bin/project

all:
	$(NVCC) src/main.cu -o $(TARGET)

clean:
	rm -f $(TARGET)

# Nome: Abdelrahman
# Cognome: Abdelrahman
# Matricola: SM3201670

CC = gcc
TARGET = tensorforth
CFLAGS = -std=c11 -O3 -Wall -Wextra -pedantic -fopenmp
LDFLAGS = -fopenmp -lm
SRCS = main.c stack.c checks.c supporto.c operandi.c scrivibin.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm -f $(OBJS) $(TARGET)
	rm -f examples/tensor.bin
	rm -f examples/*-bw.pgm
	rm -f examples/*-blurred.pgm
	rm -f examples/*-edges.pgm

.PHONY: all clean

// Nome: Abdelrahman
// Cognome: Abdelrahman
// Matricola: SM3201670

#ifndef STRUTTURE_H
#define STRUTTURE_H

#include <stddef.h>
#include <stdint.h>

#define MAX_DIM 2
#define INITIAL_STACK_CAPACITY 64

/* Tipo di oggetto gestito dallo stack TensorForth. */
typedef enum {
    OBJ_TENSOR = 1,
    OBJ_STRING = 2
} ObjectType;

/* Rappresentazione in memoria di un tensore 1D o 2D. */
typedef struct Tensor {
    int ndim;                    // 1 oppure 2
    int32_t shape[MAX_DIM];      // shape[0]=len se 1D; shape[0]=righe, shape[1]=colonne se 2D
    size_t size;                 // numero totale di float
    float *data;                 // dati in row-major
    int owns_data;               // 1 se data va liberato con free
    void *mmap_addr;             // indirizzo mmap, se il tensore arriva da file mappato
    size_t mmap_len;             // dimensione mapping
} Tensor;

/* Oggetto generico sullo stack: tensore o filename. */
typedef struct Object {
    ObjectType type;
    int refcount;
    union {
        Tensor tensor;
        char *string;
    } as;
} Object;

/* Stack dinamico di puntatori a Object. */
typedef struct Stack {
    Object **items;
    size_t size;
    size_t capacity;
} Stack;

/* Header richiesto per il formato binario su disco dei tensori. */
typedef struct on_disk_tensor {
    int32_t shape[MAX_DIM];
    int32_t ndim;
    int64_t data_offset;
} on_disk_tensor;

#endif

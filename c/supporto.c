// Nome: Abdelrahman
// Cognome: Abdelrahman
// Matricola: SM3201670

#include "supporto.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/mman.h>
#endif

void die(const char *msg) {
    fprintf(stderr, "Errore: %s\n", msg);
    exit(1);
}

void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) die("malloc fallita");
    return p;
}

void *xcalloc(size_t count, size_t size) {
    void *p = calloc(count, size);
    if (!p) die("calloc fallita");
    return p;
}

void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n);
    if (!p) die("realloc fallita");
    return p;
}

char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = xmalloc(n);
    memcpy(out, s, n);
    return out;
}

size_t tensor_numel(int ndim, const int32_t shape[MAX_DIM]) {
    if (ndim == 1) return (size_t)shape[0];
    if (ndim == 2) return (size_t)shape[0] * (size_t)shape[1];
    return 0;
}

int tensor_same_shape(const Tensor *a, const Tensor *b) {
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

int tensor_is_binary(const Tensor *t) {
    for (size_t i = 0; i < t->size; i++) {
        if (!(t->data[i] == 0.0f || t->data[i] == 1.0f)) return 0;
    }
    return 1;
}

Object *object_new_tensor_from_data(int ndim, const int32_t shape[MAX_DIM], float *data, int owns_data) {
    if (ndim < 1 || ndim > MAX_DIM) die("ndim non valido");
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) die("shape non valida");
    }

    Object *obj = xcalloc(1, sizeof(Object));
    obj->type = OBJ_TENSOR;
    obj->refcount = 1;
    obj->as.tensor.ndim = ndim;
    obj->as.tensor.shape[0] = shape[0];
    obj->as.tensor.shape[1] = (ndim == 2) ? shape[1] : 1;
    obj->as.tensor.size = tensor_numel(ndim, obj->as.tensor.shape);
    obj->as.tensor.data = data;
    obj->as.tensor.owns_data = owns_data;
    obj->as.tensor.mmap_addr = NULL;
    obj->as.tensor.mmap_len = 0;
    return obj;
}

Object *object_new_tensor(int ndim, const int32_t shape[MAX_DIM]) {
    size_t n = tensor_numel(ndim, shape);
    float *data = xcalloc(n, sizeof(float));
    return object_new_tensor_from_data(ndim, shape, data, 1);
}

Object *object_new_string(const char *s) {
    Object *obj = xcalloc(1, sizeof(Object));
    obj->type = OBJ_STRING;
    obj->refcount = 1;
    obj->as.string = xstrdup(s);
    return obj;
}

Object *object_retain(Object *obj) {
    if (obj) obj->refcount++;
    return obj;
}

void object_release(Object *obj) {
    if (!obj) return;
    obj->refcount--;
    if (obj->refcount > 0) return;

    if (obj->type == OBJ_TENSOR) {
        Tensor *t = &obj->as.tensor;
        if (t->mmap_addr && t->mmap_len > 0) {
#ifndef _WIN32
            munmap(t->mmap_addr, t->mmap_len);
#else
            free(t->mmap_addr);
#endif
        } else if (t->owns_data) {
            free(t->data);
        }
    } else if (obj->type == OBJ_STRING) {
        free(obj->as.string);
    }
    free(obj);
}

Object *object_clone_tensor(const Object *obj) {
    if (!obj || obj->type != OBJ_TENSOR) die("clone richiesto su oggetto non tensoriale");
    const Tensor *t = &obj->as.tensor;
    float *data = xmalloc(sizeof(float) * t->size);
    memcpy(data, t->data, sizeof(float) * t->size);
    return object_new_tensor_from_data(t->ndim, t->shape, data, 1);
}

/* Legge un tensore testuale dopo aver già letto il token "[". */
Object *parse_tensor(FILE *fp) {
    size_t cap = 16;
    size_t len = 0;
    float *data = xmalloc(cap * sizeof(float));
    char tok[256];

    while (fscanf(fp, "%255s", tok) == 1) {
        if (strcmp(tok, "]") == 0) {
            if (len == 0) die("tensore vuoto non valido");
            int32_t shape[MAX_DIM] = {(int32_t)len, 1};
            data = xrealloc(data, len * sizeof(float));
            return object_new_tensor_from_data(1, shape, data, 1);
        }

        char *end = NULL;
        errno = 0;
        float v = strtof(tok, &end);
        if (errno || end == tok || *end != '\0') die("numero non valido dentro un tensore");

        if (len == cap) {
            cap *= 2;
            data = xrealloc(data, cap * sizeof(float));
        }
        data[len++] = v;
    }

    free(data);
    die("tensore senza parentesi di chiusura ]");
    return NULL;
}

Object *parse_string_token(const char *token) {
    size_t n = strlen(token);
    if (n < 2 || token[0] != '"' || token[n - 1] != '"') die("filename non valido");

    char *tmp = xmalloc(n - 1);
    memcpy(tmp, token + 1, n - 2);
    tmp[n - 2] = '\0';

    Object *obj = object_new_string(tmp);
    free(tmp);
    return obj;
}

void print_tensor(const Tensor *t) {
    printf("Tensor(shape=[");
    if (t->ndim == 1) {
        printf("%d", t->shape[0]);
    } else {
        printf("%d %d", t->shape[0], t->shape[1]);
    }
    printf("], data=[");
    for (size_t i = 0; i < t->size; i++) {
        if (i) printf(" ");
        printf("%g", t->data[i]);
    }
    printf("])\n");
}

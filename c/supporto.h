// Nome: Abdelrahman
// Cognome: Abdelrahman
// Matricola: SM3201670

#ifndef SUPPORTO_H
#define SUPPORTO_H

#include <stdio.h>
#include "strutture.h"

void die(const char *msg);
void *xmalloc(size_t n);
void *xcalloc(size_t count, size_t size);
void *xrealloc(void *ptr, size_t n);
char *xstrdup(const char *s);

size_t tensor_numel(int ndim, const int32_t shape[MAX_DIM]);
int tensor_same_shape(const Tensor *a, const Tensor *b);
int tensor_is_binary(const Tensor *t);

Object *object_new_tensor(int ndim, const int32_t shape[MAX_DIM]);
Object *object_new_tensor_from_data(int ndim, const int32_t shape[MAX_DIM], float *data, int owns_data);
Object *object_new_string(const char *s);
Object *object_retain(Object *obj);
void object_release(Object *obj);
Object *object_clone_tensor(const Object *obj);

Object *parse_tensor(FILE *fp);
Object *parse_string_token(const char *token);
void print_tensor(const Tensor *t);

#endif

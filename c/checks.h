// Nome: Abdelrahman
// Cognome: Abdelrahman
// Matricola: SM3201670

#ifndef CHECKS_H
#define CHECKS_H

#include "strutture.h"

void require_stack(Stack *s, size_t n);
Tensor *require_tensor(Object *obj);
char *require_string(Object *obj);
void require_same_shape(const Tensor *a, const Tensor *b);
void require_binary_tensor(const Tensor *t);
void require_1d(const Tensor *t);
void require_2d(const Tensor *t);

#endif

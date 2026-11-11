// Nome: Abdelrahman
// Cognome: Abdelrahman
// Matricola: SM3201670

#ifndef STACK_H
#define STACK_H

#include "strutture.h"

void stack_init(Stack *s);
void stack_free(Stack *s);
void stack_push(Stack *s, Object *obj);
Object *stack_pop(Stack *s);
Object *stack_peek(Stack *s, size_t depth);

void op_dup(Stack *s);
void op_swap(Stack *s);
void op_over(Stack *s);
void op_drop(Stack *s);
/* Funzione di debug: stampa tutto lo stack senza modificarlo.
   L'operatore richiesto dal progetto per stampare un tensore è invece p. */
void stack_print(const Stack *s);

#endif

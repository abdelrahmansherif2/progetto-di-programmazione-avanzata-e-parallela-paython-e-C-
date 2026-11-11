// Nome: Abdelrahman
// Cognome: Abdelrahman
// Matricola: SM3201670

#include "stack.h"
#include "supporto.h"

#include <stdio.h>
#include <stdlib.h>

void stack_init(Stack *s) {
    s->capacity = INITIAL_STACK_CAPACITY;
    s->size = 0;
    s->items = xcalloc(s->capacity, sizeof(Object *));
}

void stack_free(Stack *s) {
    for (size_t i = 0; i < s->size; i++) object_release(s->items[i]);
    free(s->items);
    s->items = NULL;
    s->size = 0;
    s->capacity = 0;
}

void stack_push(Stack *s, Object *obj) {
    if (!obj) die("push di oggetto NULL");
    if (s->size == s->capacity) {
        s->capacity *= 2;
        s->items = xrealloc(s->items, s->capacity * sizeof(Object *));
    }
    s->items[s->size++] = obj;
}

Object *stack_pop(Stack *s) {
    if (s->size == 0) die("stack vuoto");
    return s->items[--s->size];
}

/* depth=0 cima, depth=1 elemento sotto la cima. */
Object *stack_peek(Stack *s, size_t depth) {
    if (s->size <= depth) die("stack con operandi insufficienti");
    return s->items[s->size - 1 - depth];
}

void op_dup(Stack *s) {
    Object *a = stack_peek(s, 0);
    stack_push(s, object_retain(a));
}

void op_swap(Stack *s) {
    if (s->size < 2) die("swap richiede due elementi");
    Object *a = s->items[s->size - 1];
    s->items[s->size - 1] = s->items[s->size - 2];
    s->items[s->size - 2] = a;
}

void op_over(Stack *s) {
    Object *b = stack_peek(s, 1);
    stack_push(s, object_retain(b));
}

void op_drop(Stack *s) {
    Object *a = stack_pop(s);
    object_release(a);
}

void stack_print(const Stack *s) {
    printf("Stack size = %zu\n", s->size);
    for (size_t i = s->size; i > 0; i--) {
        Object *obj = s->items[i - 1];
        printf("[%zu] ", i - 1);
        if (obj->type == OBJ_TENSOR) print_tensor(&obj->as.tensor);
        else printf("String(\"%s\")\n", obj->as.string);
    }
}

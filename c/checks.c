// Nome: Abdelrahman
// Cognome: Abdelrahman
// Matricola: SM3201670

#include "checks.h"
#include "supporto.h"

void require_stack(Stack *s, size_t n) {
    if (s->size < n) die("operandi insufficienti sullo stack");
}

Tensor *require_tensor(Object *obj) {
    if (!obj || obj->type != OBJ_TENSOR) die("tipo errato: atteso tensore");
    return &obj->as.tensor;
}

char *require_string(Object *obj) {
    if (!obj || obj->type != OBJ_STRING) die("tipo errato: atteso filename/stringa");
    return obj->as.string;
}

void require_same_shape(const Tensor *a, const Tensor *b) {
    if (!tensor_same_shape(a, b)) die("dimensioni incompatibili tra tensori");
}

void require_binary_tensor(const Tensor *t) {
    if (!tensor_is_binary(t)) die("tensore logico non composto solo da 0.0 e 1.0");
}

void require_1d(const Tensor *t) {
    if (t->ndim != 1) die("atteso tensore 1D");
}

void require_2d(const Tensor *t) {
    if (t->ndim != 2) die("atteso tensore 2D");
}

// Nome: Abdelrahman
// Cognome: Abdelrahman
// Matricola: SM3201670

#include "operandi.h"
#include "checks.h"
#include "scrivibin.h"
#include "stack.h"
#include "supporto.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int is_binary_op_symbol(const char *op) {
    return strcmp(op, "+") == 0 || strcmp(op, "-") == 0 || strcmp(op, "*") == 0 ||
           strcmp(op, "<") == 0 || strcmp(op, ">") == 0 || strcmp(op, "=") == 0 ||
           strcmp(op, "&") == 0 || strcmp(op, "|") == 0 || strcmp(op, "\\|") == 0 || strcmp(op, "m") == 0 || strcmp(op, "M") == 0;
}

/* Operazioni elemento per elemento: + - * < > = & | m M. */
static void op_elementwise(Stack *s, const char *op) {
    require_stack(s, 2);
    Object *oa = stack_pop(s); // a = cima
    Object *ob = stack_pop(s); // b = sotto
    Tensor *a = require_tensor(oa);
    Tensor *b = require_tensor(ob);
    require_same_shape(a, b);

    if ((strcmp(op, "&") == 0 || strcmp(op, "|") == 0 || strcmp(op, "\\|") == 0)) {
        require_binary_tensor(a);
        require_binary_tensor(b);
    }

    float *out = xmalloc(a->size * sizeof(float));
    #pragma omp parallel for
    for (size_t i = 0; i < a->size; i++) {
        float av = a->data[i];
        float bv = b->data[i];
        if (strcmp(op, "+") == 0) out[i] = av + bv;
        else if (strcmp(op, "-") == 0) out[i] = av - bv;
        else if (strcmp(op, "*") == 0) out[i] = av * bv;
        else if (strcmp(op, "<") == 0) out[i] = av < bv ? 1.0f : 0.0f;
        else if (strcmp(op, ">") == 0) out[i] = av > bv ? 1.0f : 0.0f;
        else if (strcmp(op, "=") == 0) out[i] = av == bv ? 1.0f : 0.0f;
        else if (strcmp(op, "&") == 0) out[i] = (av == 1.0f && bv == 1.0f) ? 1.0f : 0.0f;
        else if (strcmp(op, "|") == 0 || strcmp(op, "\\|") == 0) out[i] = (av == 1.0f || bv == 1.0f) ? 1.0f : 0.0f;
        else if (strcmp(op, "m") == 0) out[i] = av < bv ? av : bv;
        else if (strcmp(op, "M") == 0) out[i] = av > bv ? av : bv;
    }

    Object *res = object_new_tensor_from_data(a->ndim, a->shape, out, 1);
    object_release(oa);
    object_release(ob);
    stack_push(s, res);
}

/* ! : negazione logica elemento per elemento. */
static void op_not(Stack *s) {
    require_stack(s, 1);
    Object *oa = stack_pop(s);
    Tensor *a = require_tensor(oa);
    require_binary_tensor(a);

    float *out = xmalloc(a->size * sizeof(float));
    #pragma omp parallel for
    for (size_t i = 0; i < a->size; i++) out[i] = (a->data[i] == 1.0f) ? 0.0f : 1.0f;

    Object *res = object_new_tensor_from_data(a->ndim, a->shape, out, 1);
    object_release(oa);
    stack_push(s, res);
}

/* $ : ( b a m -- m?a:b ). */
static void op_select(Stack *s) {
    require_stack(s, 3);
    Object *om = stack_pop(s);
    Object *oa = stack_pop(s);
    Object *ob = stack_pop(s);
    Tensor *m = require_tensor(om);
    Tensor *a = require_tensor(oa);
    Tensor *b = require_tensor(ob);
    require_same_shape(a, b);
    require_same_shape(a, m);
    require_binary_tensor(m);

    float *out = xmalloc(a->size * sizeof(float));
    #pragma omp parallel for
    for (size_t i = 0; i < a->size; i++) out[i] = (m->data[i] == 1.0f) ? a->data[i] : b->data[i];

    Object *res = object_new_tensor_from_data(a->ndim, a->shape, out, 1);
    object_release(om); object_release(oa); object_release(ob);
    stack_push(s, res);
}

/* @ : (b a -- a@b), prodotto tra matrici. */
static void op_matmul(Stack *s) {
    require_stack(s, 2);
    Object *oa = stack_pop(s); // sinistra
    Object *ob = stack_pop(s); // destra
    Tensor *a = require_tensor(oa);
    Tensor *b = require_tensor(ob);
    require_2d(a);
    require_2d(b);
    if (a->shape[1] != b->shape[0]) die("dimensioni incompatibili per prodotto matriciale");

    int rows = a->shape[0];
    int inner = a->shape[1];
    int cols = b->shape[1];
    int32_t shape[MAX_DIM] = {rows, cols};
    float *out = xcalloc((size_t)rows * cols, sizeof(float));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < inner; k++) {
                sum += a->data[i * inner + k] * b->data[k * cols + j];
            }
            out[i * cols + j] = sum;
        }
    }

    Object *res = object_new_tensor_from_data(2, shape, out, 1);
    object_release(oa); object_release(ob);
    stack_push(s, res);
}

/* . : prodotto interno tra vettori 1D. */
static void op_dot(Stack *s) {
    require_stack(s, 2);
    Object *oa = stack_pop(s);
    Object *ob = stack_pop(s);
    Tensor *a = require_tensor(oa);
    Tensor *b = require_tensor(ob);
    require_1d(a);
    require_1d(b);
    require_same_shape(a, b);

    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < a->size; i++) sum += a->data[i] * b->data[i];

    int32_t shape[MAX_DIM] = {1, 1};
    float *out = xmalloc(sizeof(float));
    out[0] = sum;
    Object *res = object_new_tensor_from_data(1, shape, out, 1);
    object_release(oa); object_release(ob);
    stack_push(s, res);
}

/* c : convoluzione 2D con padding zero e stride 1, output stessa shape di a. */
static void op_conv2d(Stack *s) {
    require_stack(s, 2);
    Object *ok = stack_pop(s);
    Object *oa = stack_pop(s);
    Tensor *k = require_tensor(ok);
    Tensor *a = require_tensor(oa);
    require_2d(a);
    require_2d(k);

    int h = a->shape[0], w = a->shape[1];
    int kh = k->shape[0], kw = k->shape[1];
    int ah = kh / 2, aw = kw / 2;
    float *out = xcalloc(a->size, sizeof(float));

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    int yy = y + ky - ah;
                    int xx = x + kx - aw;
                    if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
                        sum += a->data[yy * w + xx] * k->data[ky * kw + kx];
                    }
                }
            }
            out[y * w + x] = sum;
        }
    }

    Object *res = object_new_tensor_from_data(2, a->shape, out, 1);
    object_release(ok); object_release(oa);
    stack_push(s, res);
}

static void shape_to_dims(const Tensor *shape_t, int *ndim, int32_t shape[MAX_DIM]) {
    require_1d(shape_t);
    if (!(shape_t->size == 1 || shape_t->size == 2)) die("shape deve contenere 1 o 2 dimensioni");
    *ndim = (int)shape_t->size;
    for (int i = 0; i < *ndim; i++) {
        int v = (int)shape_t->data[i];
        if (v <= 0 || fabsf(shape_t->data[i] - (float)v) > 1e-6f) die("dimensione shape non valida");
        shape[i] = v;
    }
    if (*ndim == 1) shape[1] = 1;
}

/* r : reshape, (a s -- a'). */
static void op_reshape(Stack *s) {
    require_stack(s, 2);
    Object *os = stack_pop(s);
    Object *oa = stack_pop(s);
    Tensor *shape_t = require_tensor(os);
    Tensor *a = require_tensor(oa);

    int ndim;
    int32_t shape[MAX_DIM];
    shape_to_dims(shape_t, &ndim, shape);
    if (tensor_numel(ndim, shape) != a->size) die("reshape con numero elementi incompatibile");

    float *out = xmalloc(a->size * sizeof(float));
    memcpy(out, a->data, a->size * sizeof(float));
    Object *res = object_new_tensor_from_data(ndim, shape, out, 1);
    object_release(os); object_release(oa);
    stack_push(s, res);
}

/* _ : ravel, appiattisce in 1D. */
static void op_flatten(Stack *s) {
    require_stack(s, 1);
    Object *oa = stack_pop(s);
    Tensor *a = require_tensor(oa);
    int32_t shape[MAX_DIM] = {(int32_t)a->size, 1};
    float *out = xmalloc(a->size * sizeof(float));
    memcpy(out, a->data, a->size * sizeof(float));
    Object *res = object_new_tensor_from_data(1, shape, out, 1);
    object_release(oa);
    stack_push(s, res);
}

/* # : ritorna shape come tensore 1D. */
static void op_shape(Stack *s) {
    require_stack(s, 1);
    Object *oa = stack_pop(s);
    Tensor *a = require_tensor(oa);
    int32_t shape[MAX_DIM] = {a->ndim, 1};
    float *out = xmalloc((size_t)a->ndim * sizeof(float));
    for (int i = 0; i < a->ndim; i++) out[i] = (float)a->shape[i];
    Object *res = object_new_tensor_from_data(1, shape, out, 1);
    object_release(oa);
    stack_push(s, res);
}

/* ? : crea tensore casuale dalle dimensioni indicate. */
static void op_random(Stack *s) {
    require_stack(s, 1);
    Object *os = stack_pop(s);
    Tensor *shape_t = require_tensor(os);
    int ndim;
    int32_t shape[MAX_DIM];
    shape_to_dims(shape_t, &ndim, shape);

    size_t n = tensor_numel(ndim, shape);
    float *out = xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) out[i] = (float)rand() / (float)RAND_MAX;
    Object *res = object_new_tensor_from_data(ndim, shape, out, 1);
    object_release(os);
    stack_push(s, res);
}

/* R : ReLU. */
static void op_relu(Stack *s) {
    require_stack(s, 1);
    Object *oa = stack_pop(s);
    Tensor *a = require_tensor(oa);
    float *out = xmalloc(a->size * sizeof(float));
    #pragma omp parallel for
    for (size_t i = 0; i < a->size; i++) out[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
    Object *res = object_new_tensor_from_data(a->ndim, a->shape, out, 1);
    object_release(oa);
    stack_push(s, res);
}

/* S : riduzione somma, risultato tensore 1D con 1 elemento. */
static void op_sum(Stack *s) {
    require_stack(s, 1);
    Object *oa = stack_pop(s);
    Tensor *a = require_tensor(oa);
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < a->size; i++) sum += a->data[i];
    int32_t shape[MAX_DIM] = {1, 1};
    float *out = xmalloc(sizeof(float));
    out[0] = sum;
    Object *res = object_new_tensor_from_data(1, shape, out, 1);
    object_release(oa);
    stack_push(s, res);
}

/* f : fill, (s v -- a). */
static void op_fill(Stack *s) {
    require_stack(s, 2);
    Object *ov = stack_pop(s);
    Object *os = stack_pop(s);
    Tensor *v = require_tensor(ov);
    Tensor *shape_t = require_tensor(os);
    int ndim;
    int32_t shape[MAX_DIM];
    shape_to_dims(shape_t, &ndim, shape);

    size_t n = tensor_numel(ndim, shape);
    float *out = xmalloc(n * sizeof(float));
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) out[i] = v->data[i % v->size];
    Object *res = object_new_tensor_from_data(ndim, shape, out, 1);
    object_release(ov); object_release(os);
    stack_push(s, res);
}

static void op_print(Stack *s) {
    require_stack(s, 1);
    Object *oa = stack_pop(s);
    Tensor *a = require_tensor(oa);
    print_tensor(a);
    object_release(oa);
}

static void op_read_pgm(Stack *s) {
    require_stack(s, 1);
    Object *of = stack_pop(s);
    char *path = require_string(of);
    Object *tensor = read_pgm_as_tensor(path);
    object_release(of);
    stack_push(s, tensor);
}

static void op_write_pgm(Stack *s) {
    require_stack(s, 2);
    Object *of = stack_pop(s);
    Object *oa = stack_pop(s);
    char *path = require_string(of);
    Tensor *a = require_tensor(oa);
    write_tensor_as_pgm(a, path);
    object_release(of); object_release(oa);
}

static void op_load_tensor(Stack *s) {
    require_stack(s, 1);
    Object *of = stack_pop(s);
    char *path = require_string(of);
    Object *tensor = mmap_tensor_from_file(path);
    object_release(of);
    stack_push(s, tensor);
}

static void op_store_tensor(Stack *s) {
    require_stack(s, 2);
    Object *of = stack_pop(s);
    Object *oa = stack_pop(s);
    char *path = require_string(of);
    Tensor *a = require_tensor(oa);
    write_tensor_to_file(a, path);
    object_release(of); object_release(oa);
}

void execute_operator(Stack *s, const char *op) {
    if (is_binary_op_symbol(op)) op_elementwise(s, op);
    else if (strcmp(op, "!") == 0) op_not(s);
    else if (strcmp(op, "$") == 0) op_select(s);
    else if (strcmp(op, "@") == 0) op_matmul(s);
    else if (strcmp(op, ".") == 0) op_dot(s);
    else if (strcmp(op, "c") == 0) op_conv2d(s);
    else if (strcmp(op, "r") == 0) op_reshape(s);
    else if (strcmp(op, "_") == 0) op_flatten(s);
    else if (strcmp(op, "#") == 0) op_shape(s);
    else if (strcmp(op, "?") == 0) op_random(s);
    else if (strcmp(op, "R") == 0) op_relu(s);
    else if (strcmp(op, "S") == 0) op_sum(s);
    else if (strcmp(op, "f") == 0) op_fill(s);
    else if (strcmp(op, "p") == 0) op_print(s);
    else if (strcmp(op, "d") == 0) op_dup(s);
    else if (strcmp(op, "s") == 0) op_swap(s);
    else if (strcmp(op, "o") == 0) op_over(s);
    else if (strcmp(op, "D") == 0) op_drop(s);
    else if (strcmp(op, "(") == 0) op_read_pgm(s);
    else if (strcmp(op, ")") == 0) op_write_pgm(s);
    else if (strcmp(op, "{") == 0) op_load_tensor(s);
    else if (strcmp(op, "}") == 0) op_store_tensor(s);
    else die("operatore sconosciuto");
}

// Nome: Abdelrahman
// Cognome: Abdelrahman
// Matricola: SM3201670

#include "scrivibin.h"
#include "supporto.h"
#include "checks.h"

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/mman.h>
#endif
#include <sys/stat.h>
#include <unistd.h>

static void skip_pgm_spaces_and_comments(FILE *fp) {
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (isspace(c)) continue;
        if (c == '#') {
            while ((c = fgetc(fp)) != EOF && c != '\n') {}
            continue;
        }
        ungetc(c, fp);
        return;
    }
}

/* Legge un'immagine PGM binaria P5 e normalizza i pixel in [0,1]. */
Object *read_pgm_as_tensor(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die("impossibile aprire PGM in lettura");

    char magic[3] = {0};
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) die("PGM non P5");

    skip_pgm_spaces_and_comments(fp);
    int width, height, maxval;
    if (fscanf(fp, "%d", &width) != 1) die("larghezza PGM non valida");
    skip_pgm_spaces_and_comments(fp);
    if (fscanf(fp, "%d", &height) != 1) die("altezza PGM non valida");
    skip_pgm_spaces_and_comments(fp);
    if (fscanf(fp, "%d", &maxval) != 1) die("maxval PGM non valido");
    if (maxval <= 0 || maxval > 255) die("supportati solo PGM con maxval 1..255");
    fgetc(fp); // consuma un singolo whitespace dopo maxval

    size_t n = (size_t)width * (size_t)height;
    unsigned char *buf = xmalloc(n);
    if (fread(buf, 1, n, fp) != n) die("dati PGM insufficienti");
    fclose(fp);

    float *data = xmalloc(n * sizeof(float));
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) data[i] = (float)buf[i] / (float)maxval;
    free(buf);

    int32_t shape[MAX_DIM] = {height, width};
    return object_new_tensor_from_data(2, shape, data, 1);
}

/* Scrive un tensore 2D come PGM P5, clippando i valori in [0,1]. */
void write_tensor_as_pgm(const Tensor *t, const char *path) {
    require_2d(t);
    int height = t->shape[0];
    int width = t->shape[1];
    size_t n = t->size;

    unsigned char *buf = xmalloc(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        float x = t->data[i];
        if (x < 0.0f) x = 0.0f;
        if (x > 1.0f) x = 1.0f;
        buf[i] = (unsigned char)lrintf(x * 255.0f);
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) die("impossibile aprire PGM in scrittura");
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    if (fwrite(buf, 1, n, fp) != n) die("errore scrittura dati PGM");
    fclose(fp);
    free(buf);
}

/* Legge un tensore nel formato binario richiesto usando mmap e senza copiare i dati. */
/* Legge un tensore nel formato binario richiesto.
   Su Linux/Ubuntu usa mmap senza copia, come richiesto dal progetto.
   Su Windows/MSYS2 UCRT64 usa un fallback con lettura in memoria per permettere il test locale. */
Object *mmap_tensor_from_file(const char *path) {
#ifndef _WIN32
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("impossibile aprire tensor binario");

    struct stat st;
    if (fstat(fd, &st) != 0) die("fstat fallita");
    if ((size_t)st.st_size < sizeof(on_disk_tensor)) die("file tensor troppo piccolo");

    void *addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) die("mmap fallita");

    on_disk_tensor *hdr = (on_disk_tensor *)addr;
    if (hdr->ndim < 1 || hdr->ndim > MAX_DIM) die("ndim su disco non valido");
    if (hdr->data_offset < (int64_t)sizeof(on_disk_tensor) || hdr->data_offset >= st.st_size) die("data_offset non valido");

    int32_t shape[MAX_DIM] = {hdr->shape[0], hdr->shape[1]};
    size_t n = tensor_numel(hdr->ndim, shape);
    size_t needed = (size_t)hdr->data_offset + n * sizeof(float);
    if (needed > (size_t)st.st_size) die("file tensor troncato");

    Object *obj = object_new_tensor_from_data(hdr->ndim, shape, (float *)((char *)addr + hdr->data_offset), 0);
    obj->as.tensor.mmap_addr = addr;
    obj->as.tensor.mmap_len = (size_t)st.st_size;
    return obj;
#else
    FILE *fp = fopen(path, "rb");
    if (!fp) die("impossibile aprire tensor binario");
    if (fseek(fp, 0, SEEK_END) != 0) die("fseek fallita");
    long len_long = ftell(fp);
    if (len_long < 0) die("ftell fallita");
    rewind(fp);

    size_t len = (size_t)len_long;
    if (len < sizeof(on_disk_tensor)) die("file tensor troppo piccolo");

    unsigned char *buffer = xmalloc(len);
    if (fread(buffer, 1, len, fp) != len) die("lettura tensor binario fallita");
    fclose(fp);

    on_disk_tensor *hdr = (on_disk_tensor *)buffer;
    if (hdr->ndim < 1 || hdr->ndim > MAX_DIM) die("ndim su disco non valido");
    if (hdr->data_offset < (int64_t)sizeof(on_disk_tensor) || (size_t)hdr->data_offset >= len) die("data_offset non valido");

    int32_t shape[MAX_DIM] = {hdr->shape[0], hdr->shape[1]};
    size_t n = tensor_numel(hdr->ndim, shape);
    size_t needed = (size_t)hdr->data_offset + n * sizeof(float);
    if (needed > len) die("file tensor troncato");

    Object *obj = object_new_tensor_from_data(hdr->ndim, shape, (float *)(buffer + hdr->data_offset), 0);
    obj->as.tensor.mmap_addr = buffer;
    obj->as.tensor.mmap_len = len;
    return obj;
#endif
}

/* Scrive un tensore nel formato binario richiesto. I dati iniziano a offset 64. */
void write_tensor_to_file(const Tensor *t, const char *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) die("impossibile aprire tensor binario in scrittura");

    on_disk_tensor hdr;
    hdr.shape[0] = t->shape[0];
    hdr.shape[1] = (t->ndim == 2) ? t->shape[1] : 1;
    hdr.ndim = t->ndim;
    hdr.data_offset = 64;

    unsigned char padding[64] = {0};
    memcpy(padding, &hdr, sizeof(hdr));
    if (fwrite(padding, 1, 64, fp) != 64) die("errore scrittura header tensor");
    if (fwrite(t->data, sizeof(float), t->size, fp) != t->size) die("errore scrittura dati tensor");
    fclose(fp);
}

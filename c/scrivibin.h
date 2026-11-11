// Nome: Abdelrahman
// Cognome: Abdelrahman
// Matricola: SM3201670
#ifndef SCRIVIBIN_H
#define SCRIVIBIN_H

#include "strutture.h"

Object *read_pgm_as_tensor(const char *path);
void write_tensor_as_pgm(const Tensor *t, const char *path);
Object *mmap_tensor_from_file(const char *path);
void write_tensor_to_file(const Tensor *t, const char *path);

#endif

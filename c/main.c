// Nome: Abdelrahman
// Cognome: Abdelrahman
// Matricola: SM3201670

#include "operandi.h"
#include "stack.h"
#include "supporto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Interprete TensorForth.
 * Uso: ./tensorforth programma.tf
 * Legge token da sinistra a destra:
 * - "[" apre un tensore testuale 1D
 * - "nomefile" inserisce una stringa sullo stack
 * - gli altri token sono operatori TensorForth
 */
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Uso corretto: tensorforth <nome file sorgente>\n");
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "Errore: impossibile aprire il file sorgente %s\n", argv[1]);
        return 1;
    }

    srand((unsigned)time(NULL));

    Stack stack;
    stack_init(&stack);

    char token[256];
    while (fscanf(fp, "%255s", token) == 1) {
        if (strcmp(token, "[") == 0) {
            stack_push(&stack, parse_tensor(fp));
        } else if (token[0] == '"') {
            stack_push(&stack, parse_string_token(token));
        } else {
            execute_operator(&stack, token);
        }

       
    }
        stack_print(&stack);
    fclose(fp);
    stack_free(&stack);
    return 0;
}

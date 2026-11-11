# TensorForth - Progetto C

## Compilazione

```bash
make clean
make
```

## Esecuzione

```bash
./tensorforth test.tf
```

Su Windows/MSYS2 UCRT64 l'eseguibile può apparire come:

```bash
./tensorforth.exe test.tf
```

## Test minimo

Nel file `test.tf`:

```text
[ 5 5 ] d + p
```

Output atteso:

```text
Tensor(shape=[2], data=[10 10])
```

## Nota su stampa

L'operatore richiesto dal progetto è:

```text
p ( a -- )
```

quindi stampa il tensore in cima allo stack e lo consuma.
Nel codice è anche presente `stack_print` come funzione di debug per stampare tutto lo stack, ma non è un operatore TensorForth aggiuntivo.

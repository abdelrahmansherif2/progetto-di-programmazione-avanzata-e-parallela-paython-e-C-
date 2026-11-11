# Nome: ABDELRAHMAN 
# Cognome: ABDELRAHMAN
# Matricola: SM3201670

import sys
import json
import pathlib
import numpy as n
from PIL import Image


# Uso corretto:
# python main.py <palette.json> <scene.json> <tiles.bin> <sprites.bin> <output.png>


# ============================================================
# CLASSI PER LA GESTIONE DEGLI ERRORI
# ============================================================

class ErroreEsecuzione(Exception):
    """Anomalia generale durante il processo di rendering."""
    pass


class FileErrato(Exception):
    """Problema rilevato nel contenuto o nella struttura interna di un file."""
    pass


class TipoFileErrato(Exception):
    """Problema nell'estensione di uno dei file indicati da terminale."""
    def __init__(self, idx, atteso, ricevuto):
        self.idx = idx
        self.atteso = atteso
        self.ricevuto = ricevuto
        super().__init__(f"Parametro {idx}: formato richiesto {atteso}, formato trovato {ricevuto}")


class FileAssente(Exception):
    """File di ingresso non trovato."""
    def __init__(self, path):
        self.path = path
        super().__init__(f"File non disponibile: {path}")


# ============================================================
# CLASSE PALETTE
# ============================================================

class Palette:
    """
    Carica e valida una palette JSON di 16 colori RGB.
    Ogni colore deve essere una terna [R, G, B] con valori tra 0 e 255.
    """

    def __init__(self, path):
        self.path = path
        self.colori = None

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fil:
                dati = json.load(fil)
        except Exception as e:
            raise FileErrato(f"Caricamento palette non riuscito ({self.path}): {e}")

        if not isinstance(dati, list) or len(dati) != 16:
            raise FileErrato("La palette deve essere composta da 16 colori")

        for colore in dati:
            if not isinstance(colore, list) or len(colore) != 3:
                raise FileErrato("Ogni elemento della palette deve avere tre componenti RGB")

            for valore in colore:
                if not isinstance(valore, int) or not (0 <= valore <= 255):
                    raise FileErrato("Le componenti RGB devono essere interi compresi tra 0 e 255")

        self.colori = n.array(dati, dtype=n.uint8)

    def risolvi(self, indici):
        """
        Converte indici di palette in valori RGB.
        Input: array con valori 0..15.
        Output: array RGB.
        """
        return self.colori[indici]


# ============================================================
# CLASSE VIRTUAL VRAM
# ============================================================

class VirtualVRAM:
    """
    Legge un file binario packed 4 bit per pixel.
    Ogni byte contiene due pixel:
    - nibble alto = primo pixel
    - nibble basso = secondo pixel

    Può caricare:
    - tiles.bin: 64 tile da 32x32
    - sprites.bin: 16 sprite da 64x64
    """

    DIM_FILE = 32768

    def __init__(self, path, tipo):
        self.path = path
        self.tipo = tipo      # "tiles" oppure "sprites"
        self.pixels = None

    def leggi_bin(self):
        try:
            byte2in1 = n.fromfile(self.path, dtype=n.uint8)
        except Exception as e:
            raise FileErrato(f"Lettura del file binario non riuscita ({self.path}): {e}")

        if byte2in1.size != self.DIM_FILE:
            raise FileErrato(
                f"Dimensione non valida per {self.path}: richiesti 32768 byte, trovati {byte2in1.size}"
            )

        pixelvect = n.empty(byte2in1.size * 2, dtype=n.uint8)

        # Primo pixel: 4 bit alti
        pixelvect[0::2] = (byte2in1 >> 4) & 0x0F

        # Secondo pixel: 4 bit bassi
        pixelvect[1::2] = byte2in1 & 0x0F

        matrice = pixelvect.reshape(256, 256)

        if self.tipo == "tiles":
            # Sheet 256x256 diviso in griglia 8x8 di tile 32x32
            self.pixels = (
                matrice
                .reshape(8, 32, 8, 32)
                .transpose(0, 2, 1, 3)
                .reshape(64, 32, 32)
            )

        elif self.tipo == "sprites":
            # Sheet 256x256 diviso in griglia 4x4 di sprite 64x64
            self.pixels = (
                matrice
                .reshape(4, 64, 4, 64)
                .transpose(0, 2, 1, 3)
                .reshape(16, 64, 64)
            )

        else:
            raise ErroreEsecuzione("Tipo di memoria video virtuale non riconosciuto")

    def ritorna(self):
        return self.pixels


# ============================================================
# CLASSE SPRITE REF
# ============================================================

class SpriteRef:
    """
    Rappresenta uno sprite letto dal file scene.json.
    Contiene id, posizione, flip orizzontale/verticale e rotazione.
    """

    def __init__(self, dati):
        campi = ["id", "x", "y", "flip_h", "flip_v", "rotation"]

        for campo in campi:
            if campo not in dati:
                raise FileErrato(f"Nel file scena manca un attributo dello sprite: {campo}")

        self.id = dati["id"]
        self.x = dati["x"]
        self.y = dati["y"]
        self.flip_h = dati["flip_h"]
        self.flip_v = dati["flip_v"]
        self.rotation = dati["rotation"]

        self._valida()

    def _valida(self):
        if not isinstance(self.id, int) or not (0 <= self.id <= 15):
            raise FileErrato("L'identificatore dello sprite deve essere un numero intero tra 0 e 15")

        if not isinstance(self.x, int) or not isinstance(self.y, int):
            raise FileErrato("Le coordinate dello sprite devono essere valori interi")

        if not isinstance(self.flip_h, bool) or not isinstance(self.flip_v, bool):
            raise FileErrato("I campi flip_h e flip_v devono contenere valori booleani")

        if self.rotation not in [0, 90, 180, 270]:
            raise FileErrato("La rotazione ammessa è soltanto 0, 90, 180 oppure 270 gradi")


# ============================================================
# CLASSE SCENE PARSER
# ============================================================

class SceneParser:
    """
    Legge e valida scene.json.
    Deve contenere:
    - transparent_index
    - tile_map 15x20
    - sprites
    """

    def __init__(self, path):
        self.path = path
        self.transparent_index = None
        self.tile_map = None
        self.sprites = []

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fil:
                dati = json.load(fil)
        except Exception as e:
            raise FileErrato(f"Caricamento della scena non riuscito ({self.path}): {e}")

        if "transparent_index" not in dati:
            raise FileErrato("Nel file scena non è presente transparent_index")

        if "tile_map" not in dati:
            raise FileErrato("Nel file scena non è presente tile_map")

        if "sprites" not in dati:
            raise FileErrato("Nel file scena non è presente sprites")

        self.transparent_index = dati["transparent_index"]
        self.tile_map = dati["tile_map"]

        self._valida_transparent_index()
        self._valida_tile_map()

        if not isinstance(dati["sprites"], list):
            raise FileErrato("Il campo sprites deve contenere una lista")

        for sprite in dati["sprites"]:
            self.sprites.append(SpriteRef(sprite))

    def _valida_transparent_index(self):
        if not isinstance(self.transparent_index, int) or not (0 <= self.transparent_index <= 15):
            raise FileErrato("transparent_index deve essere un indice intero compreso tra 0 e 15")

    def _valida_tile_map(self):
        if not isinstance(self.tile_map, list) or len(self.tile_map) != 15:
            raise FileErrato("La tile_map deve contenere 15 righe")

        for riga in self.tile_map:
            if not isinstance(riga, list) or len(riga) != 20:
                raise FileErrato("Ogni riga della tile_map deve contenere 20 elementi")

            for tile_id in riga:
                if not isinstance(tile_id, int) or not (0 <= tile_id <= 63):
                    raise FileErrato("Ogni identificatore di tile deve essere un intero da 0 a 63")


# ============================================================
# CLASSE BLITTER
# ============================================================

class Blitter:
    """
    Gestisce estrazione e trasformazione di tile/sprite.
    Per gli sprite applica:
    - flip orizzontale
    - flip verticale
    - rotazione 0/90/180/270
    """

    def __init__(self, tiles_vram, sprites_vram):
        self.tiles_vram = tiles_vram
        self.sprites_vram = sprites_vram

    def estrai_tile(self, tile_id):
        return self.tiles_vram.ritorna()[tile_id]

    def estrai_sprite_trasformato(self, sprite_ref):
        try:
            spr = self.sprites_vram.ritorna()[sprite_ref.id].copy()

            # Prima flip
            if sprite_ref.flip_h:
                spr = n.fliplr(spr)

            if sprite_ref.flip_v:
                spr = n.flipud(spr)

            # Poi rotazione
            k = sprite_ref.rotation // 90
            spr = n.rot90(spr, k=k)

            return spr

        except Exception as e:
            raise ErroreEsecuzione(f"Trasformazione non riuscita per lo sprite {sprite_ref.id}: {e}")

    def disegna_sprite(self, framebuffer, sprite_indici, x, y, transparent_index, palette):
        """
        Disegna uno sprite sul framebuffer RGB.
        Gestisce anche sprite parzialmente fuori schermo.
        I pixel con transparent_index non vengono copiati.
        """

        altezza_schermo, larghezza_schermo, _ = framebuffer.shape
        h, w = sprite_indici.shape

        x0 = x
        y0 = y
        x1 = x0 + w
        y1 = y0 + h

        # Rettangolo destinazione dentro lo schermo
        dst_x0 = max(x0, 0)
        dst_y0 = max(y0, 0)
        dst_x1 = min(x1, larghezza_schermo)
        dst_y1 = min(y1, altezza_schermo)

        # Sprite completamente fuori schermo
        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            return

        # Rettangolo sorgente dentro lo sprite
        src_x0 = dst_x0 - x0
        src_y0 = dst_y0 - y0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        sprite_cut = sprite_indici[src_y0:src_y1, src_x0:src_x1]
        sprite_rgb = palette.risolvi(sprite_cut)

        mask = sprite_cut != transparent_index

        framebuffer[dst_y0:dst_y1, dst_x0:dst_x1][mask] = sprite_rgb[mask]


# ============================================================
# CLASSE RENDERING PIPELINE
# ============================================================

class RenderingPipeline:
    """
    Esegue il rendering completo:
    1. crea framebuffer 640x480 RGB
    2. disegna il fondale usando tile_map
    3. disegna gli sprite sopra il fondale
    4. salva output PNG
    """

    LARGHEZZA = 640
    ALTEZZA = 480
    TILE_SIZE = 32

    def __init__(self, palette, scena, blitter, output_path):
        self.palette = palette
        self.scena = scena
        self.blitter = blitter
        self.output_path = output_path

        self.framebuffer = n.zeros(
            (self.ALTEZZA, self.LARGHEZZA, 3),
            dtype=n.uint8
        )

    def renderizza(self):
        self._disegna_fondale()
        self._disegna_sprites()
        self._salva_png()

    def _disegna_fondale(self):
        for y_tile in range(15):
            for x_tile in range(20):
                tile_id = self.scena.tile_map[y_tile][x_tile]

                tile_indici = self.blitter.estrai_tile(tile_id)
                tile_rgb = self.palette.risolvi(tile_indici)

                x0 = x_tile * self.TILE_SIZE
                y0 = y_tile * self.TILE_SIZE
                x1 = x0 + self.TILE_SIZE
                y1 = y0 + self.TILE_SIZE

                self.framebuffer[y0:y1, x0:x1] = tile_rgb

    def _disegna_sprites(self):
        for sprite_ref in self.scena.sprites:
            sprite_indici = self.blitter.estrai_sprite_trasformato(sprite_ref)

            self.blitter.disegna_sprite(
                framebuffer=self.framebuffer,
                sprite_indici=sprite_indici,
                x=sprite_ref.x,
                y=sprite_ref.y,
                transparent_index=self.scena.transparent_index,
                palette=self.palette
            )

    def _salva_png(self):
        try:
            img = Image.fromarray(self.framebuffer)
            img.save(self.output_path)
            print(f"Rendering completato correttamente.")
            print(f"File PNG generato: {self.output_path}")
        except Exception as e:
            raise ErroreEsecuzione(f"Salvataggio del PNG non riuscito ({self.output_path}): {e}")


# ============================================================
# CONTROLLO ARGOMENTI E FILE
# ============================================================

def controlla_argomenti():
    """
    Controlla che il programma sia chiamato con:
    python main.py palette.json scene.json tiles.bin sprites.bin output.png

    I file di input devono esistere.
    Il file output.png non deve per forza esistere.
    """

    if len(sys.argv) != 6:
        raise FileErrato(
            "Formato comando: python main.py <palette.json> <scene.json> <tiles.bin> <sprites.bin> <output.png>"
        )

    estensioni_attese = [".json", ".json", ".bin", ".bin", ".png"]

    for i in range(1, 6):
        path = pathlib.Path(sys.argv[i])
        estensione = path.suffix.lower()

        if estensione != estensioni_attese[i - 1]:
            raise TipoFileErrato(i, estensioni_attese[i - 1], estensione)

        # output.png non deve esistere prima
        if i != 5 and not path.exists():
            raise FileAssente(sys.argv[i])

    output_path = pathlib.Path(sys.argv[5])

    if output_path.parent != pathlib.Path(".") and not output_path.parent.exists():
        raise FileErrato(f"Directory di destinazione non trovata: {output_path.parent}")


# ============================================================
# CORPO PRINCIPALE
# ============================================================

def main():
    try:
        controlla_argomenti()

        # 1. Palette
        palette = Palette(sys.argv[1])
        palette.load()

        # 2. Scena
        scena = SceneParser(sys.argv[2])
        scena.load()

        # 3. Tiles
        tiles = VirtualVRAM(sys.argv[3], "tiles")
        tiles.leggi_bin()

        # 4. Sprites
        sprites = VirtualVRAM(sys.argv[4], "sprites")
        sprites.leggi_bin()

        # 5. Blitter
        blitter = Blitter(tiles, sprites)

        # 6. Rendering finale
        renderer = RenderingPipeline(
            palette=palette,
            scena=scena,
            blitter=blitter,
            output_path=sys.argv[5]
        )

        renderer.renderizza()

    except (FileErrato, FileAssente, TipoFileErrato, ErroreEsecuzione) as err:
        print(f"Esecuzione interrotta: {err}")
        print("Formato comando: python main.py <palette.json> <scene.json> <tiles.bin> <sprites.bin> <output.png>")
        sys.exit(1)

    except Exception as err:
        print(f"Interruzione imprevista del programma: {err}")
        print("Formato comando: python main.py <palette.json> <scene.json> <tiles.bin> <sprites.bin> <output.png>")
        sys.exit(1)


if __name__ == "__main__":
    main()
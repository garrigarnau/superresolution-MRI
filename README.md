# Superresolució MRI

Aquest repositori implementa un projecte de superresolució d'imatges de
ressonància magnètica cerebral a partir del dataset OASIS. El projecte està
organitzat en tres blocs experimentals:

1. Pipeline 2D amb models preentrenats per superresolució x4.
2. Fine-tuning de Swin2SR sobre el domini MRI.
3. Pipeline 3D volumètric amb superresolució x2 i coherència entre slices.

La idea general és començar amb baselines simples i comparables, adaptar
després un model al domini mèdic, i finalment explorar una solució volumètrica
que treballa directament amb patches 3D.

## Objectiu

L'objectiu és reconstruir imatges MRI d'alta resolució a partir d'entrades de
baixa resolució i comparar diferents estratègies:

- En 2D es treballa amb slices independents. Cada slice HR té mida `256x256` i
  la versió LR té mida `64x64`, per tant el factor de superresolució és x4.
- En fine-tuning s'adapta Swin2SR, inicialment preentrenat per imatges naturals,
  al domini de MRI en escala de grisos.
- En 3D es treballa amb volums complets i patches cúbics. Els volums HR tenen
  mida `(176, 208, 176)` i els LR `(88, 104, 88)`, per tant el factor és x2.

## Estructura del projecte

```text
.
|-- data/                       # Pairs 2D HR/LR preprocessats en PNG
|   |-- train/
|   |-- val/
|   `-- test/
|-- disc1/                      # Volums originals OASIS en format Analyze
|-- docs/                       # Documentació visual del pipeline 2D i resultats
|-- fine-tunning/               # Fine-tuning de Swin2SR
|   |-- checkpoints/            # Checkpoints entrenats, si existeixen
|   |-- dataset.py
|   |-- infer_finetuned.py
|   |-- losses.py
|   |-- model.py
|   |-- train.py
|   `-- results/
|-- 3dmodel/                    # Pipeline 3D x2 amb MedicalNet
|   |-- checkpoints/            # Checkpoints 3D, si existeixen
|   |-- dataset.py
|   |-- infer.py
|   |-- model.py
|   |-- preprocess.py
|   |-- train.py
|   |-- evaluate.py
|   `-- results/
|-- results/                    # Resultats del benchmark 2D sense fine-tuning
|-- scripts/                    # Preprocessat, benchmark i avaluació 2D
|-- weights/                    # Pesos descarregats, per exemple Real-ESRGAN
`-- requirements.txt
```

Nota: la carpeta `fine-tunning/` manté aquest nom perquè ja forma part de les
rutes del projecte.

## Instal·lació

Crear i activar un entorn virtual:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Instal·lar dependències:

```powershell
pip install -r requirements.txt
```

En Linux o macOS, l'activació equivalent és:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

El projecte utilitza PyTorch, Transformers, scikit-image, nibabel, Pillow i
Real-ESRGAN. Per entrenar amb CUDA cal una instal·lació de PyTorch compatible
amb la GPU de la màquina.

## Dataset

El dataset utilitzat és OASIS-1, disponible a:

```text
https://sites.wustl.edu/oasisbrains/home/oasis-1/
```

El fitxer descarregat per aquest projecte és:

```text
oasis_cross-sectional_disc1.tar.gz
```

El projecte espera els volums OASIS dins de `disc1/`, amb una estructura com:

```text
disc1/
`-- OAS1_XXXX_MR1/
    `-- PROCESSED/
        `-- MPRAGE/
            `-- T88_111/
                `-- *_masked_gfc.hdr
```

S'utilitzen els fitxers `*_masked_gfc.hdr`, és a dir, volums ja processats,
alineats i amb màscara cerebral.

OASIS-1 conté ressonàncies T1 del cervell. En aquest projecte s'utilitzen els
volums skull-stripped, corregits de bias field i registrats a l'espai Talairach
T88, amb resolució isotrópica d'1 mm i mida original `176x208x176`.

## Pipeline 2D: benchmark x4

El primer bloc converteix els volums 3D en slices 2D i avalua models
preentrenats sense entrenament addicional.

### 1. Preprocessar OASIS a imatges 2D

```powershell
python scripts/preprocess.py
```

Aquest script:

- Carrega els volums OASIS des de `disc1/`.
- Extreu slices axials, segons `SLICE_AXIS = 2`.
- Descarta slices amb massa poc contingut cerebral.
- Normalitza cada slice a escala de grisos.
- Genera una imatge HR de `256x256`.
- Genera una imatge LR de `64x64` fent downsample bicúbic.
- Divideix les dades en `train`, `val` i `test`.

La sortida queda a:

```text
data/
|-- train/
|   |-- HR/
|   `-- LR/
|-- val/
|   |-- HR/
|   `-- LR/
`-- test/
    |-- HR/
    `-- LR/
```

### 2. Executar els models preentrenats

```powershell
python scripts/benchmark.py --device auto
```

Aquest benchmark executa:

- `Swin2SR`: model `caidas/swin2SR-classical-sr-x4-64`.
- `Real-ESRGAN`: model x4 amb pesos guardats o descarregats a `weights/`.

Els resultats es guarden a:

```text
results/
|-- swin2sr/
|-- real_esrgan/
`-- timings.json
```

El Swin2SR del benchmark és la versió canònica alineada. El codi aplica
l'alineació sobre la sortida perquè el foreground de la SR coincideixi millor
amb la petjada de la LR.

### 3. Avaluar el benchmark 2D

```powershell
python scripts/evaluate.py
```

Aquest script compara cada sortida SR amb la seva imatge HR corresponent i
calcula:

- PSNR.
- SSIM.
- Temps mitjà per imatge, si existeix a `timings.json`.

També genera una graella visual a:

```text
results/visual_comparison.png
```

## Fine-tuning de Swin2SR

El segon bloc adapta Swin2SR al domini MRI. Swin2SR és un model preentrenat per
superresolució x4 i espera imatges RGB de 3 canals. Com que les MRI del projecte
són imatges en escala de grisos, el pas necessari és fer compatible aquesta
entrada d'1 canal amb el format que espera el model.

En el benchmark 2D preentrenat, això es fa convertint cada slice LR de grisos a
RGB abans d'entrar a Swin2SR. Conceptualment, és una rèplica del mateix canal en
R, G i B; no afegeix informació nova, només adapta el format d'entrada al model.

El resultat principal d'aquest bloc és `swin2sr_finetuned`, és a dir, Swin2SR
ajustat amb les imatges MRI del projecte.

En l'entrenament fine-tuned, el model es carrega des de
`caidas/swin2SR-classical-sr-x4-64` i s'adapta al domini MRI. La diferència
important respecte al benchmark és que aquí els pesos del model s'actualitzen
amb les parelles MRI `64x64 -> 256x256`, en comptes d'utilitzar el model
preentrenat directament.

### 1. Entrenar Swin2SR sobre MRI

```powershell
python fine-tunning/train.py
```

Si només es vol provar en CPU:

```powershell
python fine-tunning/train.py --cpu
```

El training utilitza:

- Imatges LR/HR de `data/train` i `data/val`.
- Factor de superresolució x4.
- Imatges MRI en escala de grisos adaptades al format d'entrada de Swin2SR.
- Loss combinada: `1.0 * L1 + 0.1 * perceptual`.
- Optimitzador AdamW.
- Scheduler `ReduceLROnPlateau`.
- Early stopping segons PSNR de validació.

La loss L1 mesura l'error píxel a píxel i afavoreix valors alts de PSNR. La
loss perceptual utilitza VGG19 congelat per comparar features visuals de les
imatges SR i HR. Com que VGG19 espera RGB, les imatges MRI d'un canal es
repeteixen internament a 3 canals només per calcular aquesta loss perceptual.

Configuració principal del fine-tuning:

| Paràmetre | Valor |
|---|---:|
| Model base | `caidas/swin2SR-classical-sr-x4-64` |
| Factor | x4 |
| Batch size | 2 |
| Èpoques màximes | 30 |
| Learning rate | `2e-5` |
| Weight decay | `1e-4` |
| Scheduler patience | 3 |
| Early stopping patience | 8 |
| Seed | 42 |

La taxa d'aprenentatge és baixa per adaptar progressivament el model
preentrenat al domini MRI sense destruir completament la inicialització.

Els checkpoints es guarden a:

```text
fine-tunning/checkpoints/
|-- best_model.pth
`-- last_model.pth
```

### 2. Fer inferència amb el model fine-tuned

```powershell
python fine-tunning/infer_finetuned.py --device auto
```

Per defecte, aquest script carrega el checkpoint fine-tuned:

```text
fine-tunning/checkpoints/best_model.pth
```

i guarda les sortides a:

```text
fine-tunning/results/swin2sr_finetuned/
```

### 3. Avaluar els resultats del bloc fine-tuning

```powershell
python scripts/evaluate.py --results-dir fine-tunning/results
```

Aquest pas compara les carpetes de resultats que existeixin dins de
`fine-tunning/results/`. En l'estat actual del repositori, les carpetes són:

- `swin2sr_finetuned/`
- `swin2sr/`
- `real_esrgan/`

La sortida principal és:

```text
fine-tunning/results/metrics.json
fine-tunning/results/visual_comparison.png
```

El script `fine-tunning/test_pipeline.py` serveix per validar el bloc abans
d'entrenar: comprova càrrega de dataset, forma dels tensors, forward pass,
loss, backward pass i guardat/carrega de checkpoints.

## Pipeline 3D: superresolució volumètrica x2

El tercer bloc treballa amb volums 3D complets. La motivació és preservar la
coherència anatòmica entre slices, cosa que el pipeline 2D no pot garantir
perquè processa cada slice independentment.

En aquest cas el factor és x2:

- HR: `(176, 208, 176)`
- LR: `(88, 104, 88)`
- Patch HR: `64x64x64`
- Patch LR: `32x32x32`

La diferència entre les dimensions del volum complet i les dimensions d'entrada
del model és intencionada. El volum complet no s'entrena d'una sola vegada
perquè les convolucions 3D i les activacions intermèdies consumirien massa
memòria. Per això el preprocessat talla cada volum en patches: el model aprèn
la transformació local `32x32x32 -> 64x64x64`, i durant la inferència aquests
patches SR es tornen a unir amb sliding window per reconstruir el volum complet
`176x208x176`.

### 1. Preprocessar volums i extreure patches 3D

```powershell
python 3dmodel/preprocess.py
```

Aquest script:

- Carrega els volums OASIS des de `disc1/`.
- Normalitza cada volum a `[0, 1]`.
- Genera el volum LR amb Gaussian blur i downsample x2.
- Extreu parelles de patches HR/LR.
- Divideix els subjectes en train, validation i test.
- Guarda també volums complets de test per a la inferència final.

En aquest context, un **subjecte** és un volum MRI complet d'un cas OASIS
(`OAS1_XXXX_MR1`). La divisió es fa a nivell de subjecte, no a nivell de patch:
tots els patches extrets d'un mateix volum van al mateix split. Això evita que
el model vegi parts del mateix cervell durant l'entrenament i després sigui
avaluat sobre altres patches del mateix subjecte, cosa que faria la mètrica
massa optimista.

La sortida queda a:

```text
3dmodel/patches/
|-- train/
|   |-- HR/
|   `-- LR/
|-- val/
|   |-- HR/
|   `-- LR/
|-- test/
|   |-- HR/
|   `-- LR/
`-- test_volumes/
    |-- *_hr.npy
    `-- *_lr.npy
```

La carpeta `test/` conté patches de test i queda com a conjunt auxiliar per
fer proves a nivell de patch. El pipeline final 3D, però, utilitza
`test_volumes/`: `infer.py` carrega els volums LR complets, reconstrueix els
volums SR amb sliding window, i `evaluate.py` compara aquests volums SR contra
els HR complets.

### 2. Entrenar el model 3D

```powershell
python 3dmodel/train.py
```

Si es vol forçar CPU:

```powershell
python 3dmodel/train.py --cpu
```

El model 3D està format per:

- Encoder MedicalNet ResNet3D-50 preentrenat.
- Encoder congelat durant l'entrenament.
- Decoder 3D entrenable amb convolucions transposades.
- Baseline trilineal x2 dins del forward.
- Residual acotat que s'afegeix sobre la interpolació trilineal.

MedicalNet s'utilitza perquè ja està preentrenat sobre imatges mèdiques 3D i
pot extreure features anatòmiques volumètriques. L'encoder es congela perquè el
nombre de subjectes és limitat; entrenar tots els paràmetres del ResNet3D-50
augmentaria molt el risc de sobreajust. Per això l'optimizer actualitza només
el decoder.

Durant l'entrenament, el flux és:

1. S'agafa un patch LR i el seu patch HR corresponent.
2. El patch LR es reescala amb interpolació trilineal:
   `LR 32x32x32 -> baseline 64x64x64`.
3. El mateix patch LR passa per l'encoder MedicalNet congelat:
   `LR 32x32x32 -> features 3D`.
4. El decoder entrenable transforma aquestes features en un residual:
   `features 3D -> residual 64x64x64`.
5. La sortida final és la suma:
   `SR = baseline trilineal + residual`.
6. Es compara `SR` amb el patch `HR` real mitjançant la loss L1.
7. La backpropagation actualitza només els pesos del decoder.

Per tant, la interpolació trilineal no s'aprèn: és una base fixa. El que aprèn
el model és el residual que corregeix aquesta base.

Configuració principal del model 3D:

| Paràmetre | Valor |
|---|---:|
| Encoder | `nwirandx/medicalnet-resnet3d50` |
| Factor | x2 |
| Loss | L1 |
| Batch size | 4 |
| Èpoques màximes | 50 |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Scheduler patience | 5 |
| Early stopping patience | 12 |
| Patch LR | `32x32x32` |
| Patch HR | `64x64x64` |

En 3D s'utilitza només L1 perquè no hi ha un equivalent estàndard i preentrenat
de VGG19 per volums mèdics 3D. L1 és estable, simple i adequada per optimitzar
fidelitat numèrica.

L'entrada del model és un patch LR de mida:

```text
(B, 1, 32, 32, 32)
```

i la sortida és un patch SR de mida:

```text
(B, 1, 64, 64, 64)
```

Els checkpoints es guarden a:

```text
3dmodel/checkpoints/
|-- best_model.pth
`-- last_model.pth
```

### 3. Inferència 3D amb sliding window

```powershell
python 3dmodel/infer.py
```

Per defecte carrega:

```text
3dmodel/checkpoints/best_model.pth
```

També es pot indicar un checkpoint concret:

```powershell
python 3dmodel/infer.py --checkpoint 3dmodel/checkpoints/best_model.pth
```

Durant la inferència:

- Es carreguen els volums LR complets de `3dmodel/patches/test_volumes/`.
- Es divideixen en finestres 3D de `32x32x32`.
- Cada patch passa pel model.
- Cada sortida `64x64x64` es recol·loca en coordenades x2.
- Les zones solapades es mitjanen amb un `weight_map`.
- També es genera una baseline trilineal x2.

Les sortides es guarden a:

```text
3dmodel/results/
|-- model_sr/
|   `-- *_sr.npy
`-- trilinear/
    `-- *_trilinear.npy
```

### 4. Avaluació 3D

```powershell
python 3dmodel/evaluate.py
```

Aquest script compara els volums SR contra els volums HR originals i calcula:

- PSNR 3D.
- SSIM 3D.
- Comparació contra la baseline trilineal.

Els resultats es guarden a:

```text
3dmodel/results/metrics_3d.json
```

## Resultats actuals

Benchmark 2D sense fine-tuning:

| Mètode | PSNR (dB) | SSIM | Temps (s/img) |
|---|---:|---:|---:|
| Swin2SR preentrenat alineat | 22.41 +/- 9.01 | 0.7619 +/- 0.2278 | 0.1083 |
| Real-ESRGAN preentrenat | 23.66 +/- 4.48 | 0.8463 +/- 0.0793 | 0.6360 |

Comparació amb fine-tuning:

| Mètode | PSNR (dB) | SSIM | Temps (s/img) |
|---|---:|---:|---:|
| Swin2SR preentrenat alineat | 22.41 +/- 9.01 | 0.7619 +/- 0.2278 | 0.1083 |
| Real-ESRGAN preentrenat | 23.66 +/- 4.48 | 0.8463 +/- 0.0793 | 0.6360 |
| Swin2SR fine-tuned | 37.04 +/- 2.69 | 0.9764 +/- 0.0131 | 0.0817 |

Resultats 3D:

| Mètode | Factor | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| MedicalNet SR 3D | x2 | 33.08 +/- 1.00 | 0.9164 +/- 0.0181 |
| Baseline trilineal 3D | x2 | 33.10 +/- 0.76 | 0.9411 +/- 0.0044 |

Els resultats 2D x4 i 3D x2 no són directament comparables, perquè resolen
tasques diferents. El x4 2D augmenta cada slice de `64x64` a `256x256`, mentre
que el x2 3D augmenta tot el volum i multiplica per 8 el nombre de voxels.

Interpretació principal:

- El fine-tuning 2D és el millor resultat quantitatiu del projecte en la tasca
  x4 sobre slices.
- Real-ESRGAN funciona millor que Swin2SR preentrenat en zero-shot, però és més
  lent i no està adaptat al domini MRI.
- El model 3D no supera clarament la baseline trilineal en mitjana, però
  introdueix un pipeline volumètric coherent i evita tractar cada slice com una
  imatge independent.
- El 3D és x2 perquè un x4 volumètric implicaria `4^3 = 64` vegades més voxels,
  cosa molt més costosa en memòria i temps.

## Ordre recomanat d'execució

Per reproduir el projecte complet:

```powershell
# 1. Preparar dades 2D
python scripts/preprocess.py

# 2. Benchmark 2D preentrenat
python scripts/benchmark.py --device auto
python scripts/evaluate.py

# 3. Fine-tuning Swin2SR
python fine-tunning/train.py
python fine-tunning/infer_finetuned.py --device auto
python scripts/evaluate.py --results-dir fine-tunning/results

# 4. Pipeline 3D x2
python 3dmodel/preprocess.py
python 3dmodel/train.py
python 3dmodel/infer.py
python 3dmodel/evaluate.py
```

## Notes importants

- `results/` es reserva per al benchmark 2D sense fine-tuning.
- `fine-tunning/results/` es reserva per a la comparació del bloc fine-tuning:
  Swin2SR, Real-ESRGAN i Swin2SR fine-tuned.
- `3dmodel/results/` es reserva per als experiments volumètrics.
- Per l'entrega, les carpetes de dades i resultats poden contenir només un
  exemple per carpeta. Els fitxers `metrics.json`, `timings.json` i
  `metrics_3d.json` conserven els resultats de les execucions completes.
- Els fitxers grans, com dades originals completes, patches, volums generats i
  checkpoints, no s'haurien de versionar sencers.
- El script `scripts/align_swin2sr.py` és una utilitat de compatibilitat per a
  sortides antigues. El benchmark actual ja escriu Swin2SR alineat directament a
  `results/swin2sr/`.

## Limitacions i millores futures

- El split 2D actual és per slice, no per subjecte. És útil per benchmarking
  inicial, però un split per subjecte seria més estricte.
- El model 2D fine-tuned dona bons resultats per slice, però no imposa
  coherència anatòmica entre slices consecutives.
- El model 3D està limitat pel nombre reduït de subjectes i patches; amb més
  dades es podria provar descongelar parcialment l'encoder.
- En 3D només es fa x2 per restriccions de memòria. Una línia futura seria fer
  upsampling progressiu x2 -> x2 per aproximar un x4 volumètric.
- També es podria explorar una loss perceptual 3D si es disposa d'un extractor
  de features mèdic volumètric preentrenat.

## Resum metodològic

El projecte defensa una evolució progressiva:

1. Validar baselines 2D x4 preentrenades per tenir una referència inicial.
2. Adaptar Swin2SR al domini MRI amb fine-tuning supervisat.
3. Explorar superresolució 3D x2 per tenir coherència volumètrica i evitar que
   cada slice es tracti com una imatge independent.

Aquest disseny permet comparar rendiment, qualitat visual, mètriques
quantitatives i cost computacional entre aproximacions 2D i 3D.

# Document AI Classifier

Document AI Classifier je Streamlit aplikacija za klasifikaciju dokumenata u
pet klasa:

- `invoice`
- `cv`
- `contract`
- `email`
- `scientific`

Aplikacija uspoređuje tri komplementarna modela:

- **ResNet50** - vizualna klasifikacija stranica dokumenta
- **XLM-RoBERTa** - klasifikacija tekstualnih chunkova
- **LayoutLMv3** - multimodalna klasifikacija slike, OCR riječi i layouta

## Multi-page pipeline

Aktualni pipeline obrađuje PDF dokumente na više stranica. Za PDF do 12
stranica koriste se sve stranice. Za dulje dokumente odabiru se reprezentativne
stranice s početka, sredine i kraja.

`scripts/build_multipage_dataset.py` radi sljedeće:

1. Izrađuje jedan autoritativni document manifest.
2. Radi group-aware train/validation/test split prema
   `parent_document_id` i `augmentation_group_id` prije stvaranja artefakata.
3. Stvara page-level slike za ResNet50.
4. Stvara page-level OCR riječi i bounding boxove za LayoutLMv3.
5. Stvara tokenizer overflow chunkove za XLM-RoBERTa.
6. Čuva sve stranice koje se mogu renderirati, a OCR-nečitljive stranice
   preskače samo za LayoutLMv3 i zapisuje ih u failure izvještaje.

Svi modeli agregiraju page/chunk predikcije na razini dokumenta. Jedan dokument
se u evaluaciji broji jednom. Trening skripte ne rade vlastiti nasumični split i
prekidaju prije treninga ako autoritativni manifest otkrije split leakage.

Izrada ili nastavak multi-page artefakata:

```powershell
python scripts\build_multipage_dataset.py --skip-existing
```

## Instalacija

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Na Windowsu OCR koristi Tesseract ako postoji na:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Za Streamlit Community Cloud sistemske ovisnosti definirane su u
`packages.txt`: Tesseract OCR i LibreOffice.

## Trening

Prije treninga mora biti završen multi-page build. Puni treninzi pokreću se
ručno:

```powershell
python src\train_resnet.py --epochs 8 --batch-size 16
python src\train_text_model.py --epochs 4 --batch-size 8
python src\train_layoutlm.py --epochs 4 --batch-size 2 --gradient-accumulation-steps 4
```

Novi modeli spremaju se u:

```text
models/resnet50_multipage/
models/xlm_roberta_multipage/
models/layoutlmv3_multipage/
```

Trening skripte nakon treninga automatski rade document-level evaluaciju na
internom testnom splitu i spremaju rezultate u odgovarajuće `_multipage`
foldere unutar `results/`.

## Evaluacija

Provjera i ispis već spremljenih internih rezultata:

```powershell
python src\evaluate.py
```

Evaluacija na vanjskom testnom skupu:

```powershell
python src\evaluate_external_test.py
```

Završna usporedba internih i vanjskih rezultata:

```powershell
python scripts\create_final_comparison.py
```

Vanjski dokumenti očekuju se u `data/external_test/raw/<label>/`. Evaluacija
provjerava SHA-256 duplikate prema internom datasetu i ne koristi naziv filea,
folder ili stvarnu labelu kao ulaz modelu.

## Streamlit aplikacija

```powershell
streamlit run app.py
```

Aplikacija podržava pojedinačne i usporedne predikcije ResNet50,
XLM-RoBERTa i LayoutLMv3 modela. Za PDF prikazuje ukupan broj stranica,
analizirane stranice, broj tekstualnih chunkova i konačnu document-level
predikciju.

## Modeli i Hugging Face

Model weights se ne spremaju na GitHub. Lokalni `models/` i cijeli `data/`
ignorirani su kroz `.gitignore`. Za lokalni upload istreniranih multi-page
modela na Hugging Face koristi se:

```powershell
$env:HF_TOKEN="hf_..."
python scripts\upload_models_to_hf.py
```

Skripta ništa ne uploada bez `HF_TOKEN`. Repo ID-jevi mogu se zadati kroz:

```text
HF_RESNET50_REPO_ID
HF_XLM_ROBERTA_REPO_ID
HF_LAYOUTLMV3_REPO_ID
```

Streamlit aplikacija prvo koristi lokalne multi-page modele. Ako nisu dostupni,
pokušava ih preuzeti s konfiguriranih Hugging Face repozitorija.

## Testovi

```powershell
python -m pytest tests
```

Ako `pytest` nije instaliran, svi postojeći testovi kompatibilni su i sa
standardnim `unittest` runnerom:

```powershell
python -m unittest discover -s tests -v
```

## GitHub sadržaj

GitHub repozitorij sadrži kod, testove, konfiguraciju i male rezultate potrebne
za Streamlit prikaz. Ne sadrži dataset, lokalne modele, backup foldere ni velike
binarne artefakte. Povijesne, nereferencirane datoteke sačuvane su u
`legacy/` i nisu dio aktualnog pipelinea.

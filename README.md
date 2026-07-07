# Document AI Classifier

Streamlit aplikacija za klasifikaciju dokumenata pomocu tri modela:

- ResNet50
- XLM-RoBERTa
- LayoutLMv3

Veliki model fileovi nisu u GitHub repozitoriju. Lokalno i na Streamlit Community Cloudu aplikacija ih ocekuje u `models/` folderu. Ako modeli ne postoje lokalno, aplikacija ih pokusava preuzeti s Hugging Face Huba.

## Lokalno pokretanje

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Ako `models/` folder postoji lokalno, aplikacija koristi lokalne modele i ne preuzima nista.

## Ocekivana struktura modela

```text
models/
  resnet50/
    best_model.pth
    label_mapping.json
  xlm_roberta/
    label_mapping.json
    best_model/
      config.json
      model.safetensors ili pytorch_model.bin
      tokenizer.json / tokenizer.model / sentencepiece.bpe.model
  layoutlmv3/
    best_model/
      config.json
      model.safetensors ili pytorch_model.bin
      label_mapping.json
      tokenizer/processor fileovi
```

## Upload modela na Hugging Face

Prvo napravi Hugging Face access token. Ako su repo-i private, token mora imati pravo pisanja i citanja.

Predlozeni repo-i:

- `M0nsterki11/document-ai-resnet50`
- `M0nsterki11/document-ai-xlm-roberta`
- `M0nsterki11/document-ai-layoutlmv3`

Upload pokreni lokalno iz root foldera projekta:

```powershell
$env:HF_TOKEN="hf_..."
python scripts/upload_models_to_hf.py
```

Skripta uploada:

- `models/resnet50/` u ResNet50 repo
- `models/xlm_roberta/` u XLM-RoBERTa repo
- `models/layoutlmv3/best_model/` u LayoutLMv3 repo

Ne uploadaju se `data/`, `results/`, `venv/` ni slicni folderi.

## Streamlit Community Cloud secrets

U Streamlit Cloud aplikaciji otvori **Settings -> Secrets** i dodaj:

```toml
HF_RESNET50_REPO_ID = "M0nsterki11/document-ai-resnet50"
HF_XLM_ROBERTA_REPO_ID = "M0nsterki11/document-ai-xlm-roberta"
HF_LAYOUTLMV3_REPO_ID = "M0nsterki11/document-ai-layoutlmv3"
```

Ako su Hugging Face repo-i private, dodaj i:

```toml
HF_TOKEN = "hf_..."
```

Ako su repo-i public, `HF_TOKEN` nije potreban.

## Kako deploy radi online

GitHub sadrzi samo kod, requirements, skripte, README i male result fileove. Streamlit Cloud starta aplikaciju iz GitHuba. Kada korisnik odabere model i uploada dokument, aplikacija provjeri postoji li model lokalno. Ako model nedostaje, pokusa ga preuzeti s Hugging Face Huba, spremi ga u `models/` i tek tada ucita odabrani model za live predikciju.

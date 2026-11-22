# Dijagnostika Document AI Classifiera

Datum: 15. rujna 2026. Projekt: `document_ai_classifier`. Grana: `diagnostika/modeli-20260915`, početni commit `d1bcf62b4997d9ceb5ee2c4675155ff297ad43e1`. Početni worktree bio je čist. Promjene nisu commitane, objavljene ni uploadane. Brojevi redaka koda odnose se na završne lokalne datoteke, osim gdje izričito piše „prije”. Putanje su relativne korijenu projekta.

## 1. Najvažniji nalazi

1. **Lokalna aplikacija stvarno učitava spremljene multipage modele**, a ne osnovne klasifikatore. U memoriji je provjereno svih 320 ResNet, 201 XLM i 216 LayoutLM state-dict tenzora: svi su jednaki odgovarajućim spremljenim težinama, bez nedostajućih/neiskorištenih ključeva. Svi modeli su u `eval()` načinu. Datoteke odgovaraju najboljim dostupnim checkpointima treninga iz 3. rujna.
2. **Trenutačne javne HF težine također su byte-for-byte jednake lokalnima.** Online Streamlit proces nije bio dostupan: ovaj nalaz ne dokazuje koje datoteke ili objekte taj proces trenutačno koristi.
3. Reproducirane su i lokalno popravljene četiri rubne greške: zadržavanje starog modela u cacheu nakon zamjene datoteka, nepotpuna multipage mapa koja zasjeni ispravan fallback, izgubljena ResNet agregacijska konfiguracija pri downloadu te pogrešne koordinate riječi u rotiranim PDF-ovima.
4. Na istom unaprijed odabranom uzorku prije/poslije nije se promijenila nijedna predikcija ni sigurnost. Popravci stoga **nisu dokaz poboljšanja točnosti**, niti je ustanovljeno da su uzrok profesorovih primjera.
5. Postoje stvarna ograničenja podataka: normalizirani tekst preklapa se između podjela; u sedam grupa jednaki su i spremljeni odabrani XLM tokeni. Između train i test potvrđen je par `contract_0100` / `contract_0010`. Interni rezultat nije čista mjera generalizacije na nove izvore/predloške.
6. Originalni Europass CV, mail i HT račun iz opisa nisu pouzdano identificirani. Nisu izmišljeni rezultati za njih; screenshotovi nisu tretirani kao originalni dokumenti.

Nije pokrenuto treniranje, ni jedan optimizacijski korak. Nisu promijenjeni postojeći modeli, manifesti, splitovi ili stari rezultati. Novi artefakti nalaze se samo u `reports/`; pripremljena eksperimentalna opcija nije pokrenuta.

## 2. Što je stvarno provjereno

Lokalno: Windows, Python iz `../venv`, PyTorch 2.11.0+cu128, RTX 5060 s približno 8 GB VRAM-a. Pozvani su stvarni Streamlit getteri iz `app.py`, a predikcije preko `app.run_prepared_model_prediction`, ne zamjenski klasifikator. To je stvarno izvođenje aplikacijskih Python funkcija u lokalnom procesu, ali nije pregled browser UI-ja ni cloud procesa. Upozorenje `missing ScriptRunContext` očekivano je pri tom načinu pozivanja.

Dokazi su odvojeni u:

- `baseline_20260915/`: početni odabir, izvorni rezultati, hashovi, testovi i pripremljeni ulazi prije popravaka;
- `baseline_20260915/corrected_harness/`: ponovljena usporedba spremljenih ulaza nakon ispravka same dijagnostičke skripte;
- `after_20260915/`: nova priprema istih sirovih dokumenata i evaluacija nakon popravaka;
- [metrics_summary.json](metrics_summary.json), [before_after_predictions.csv](before_after_predictions.csv), [per_class_metrics.csv](per_class_metrics.csv), [errors_and_misclassifications.csv](errors_and_misclassifications.csv).

Važna korekcija mjerenja: prvotna dijagnostička funkcija `stored_prepared` nije LayoutLM-u dodala riječi/boxove iz OCR JSON-a. Zato je izvorni `baseline_20260915/layoutlmv3` imao 100 pogrešaka samo u putu `app_stored`. **To nije greška produkcijskog koda. Taj podrezultat je nevaljan**, označen datotekom `INVALID_APP_STORED.txt` i nije korišten u zaključcima. Popravak dijagnostičke skripte je na `scripts/diagnose_models.py:177`; ponavljanje koristi sačuvane ulaze prije popravka i bilježi stvarne hashove izvršenog koda u `execution_code.json`. Izvorni `app_fresh` i `evaluation_stored` rezultati ostaju valjani. Inference funkcije, težine i postojeći evaluacijski Dataset kod nisu mijenjani.

Završni dijagnostički runner priprema dokumente serijski: PyMuPDF ne podržava višedretveno izvođenje u istom procesu ([službena dokumentacija](https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html)). Početne dvije pripreme koristile su dvije dretve, pa je dodatno napravljena serijska priprema cijelog istog uzorka u `serial_verification_20260915/`. Usporedba tekstualnih hashova, svih renderiranih piksela, riječi/boxova, odabranih stranica i grešaka bilježi se u `final_verification.json`. Ta dodatna provjera ne trenira modele niti ponovno bira uzorak. Završni test log je `serial_verification_20260915/tests.txt`.

**Završna provjera prošla je: 125/125 pripremljenih dokumenata potpuno se podudara s prethodnom pripremom nakon popravaka.** Sva tri manifesta i sve tri datoteke težina ostali su nepromijenjeni; 57/57 testova prolazi. Dokaz: [final_verification.json](final_verification.json). MuPDF je pri serijskoj pripremi ispisao neblokirajuća upozorenja `No common ancestor in structure tree`; nisu uzrokovala iznimku/odbačen dokument i nisu promijenila uspoređene izlaze. Ni ta upozorenja nisu dokaz uzroka profesorovih primjera.

## 3. Modeli, checkpointi i podrijetlo

### Stvarno učitane lokalne težine

| Model | Razriješena putanja | Najbolji dostupni trening | Usporedba memorija/datoteka | Agregacija |
|---|---|---|---|---|
| ResNet50 | `models/resnet50_multipage/best_model.pth` | 3. 9.; epoha 8, val macro-F1 0.875908 | 320/320 jednakih tenzora | `mean`, top_k=3 (ne utječe na mean) |
| XLM-RoBERTa | `models/xlm_roberta_multipage/best_model/model.safetensors` | 3. 9.; najbolja epoha 3 od 4 prema povijesti/val konfiguraciji | 201/201 jednakih tenzora | `top_k_mean`, 3 |
| LayoutLMv3 | `models/layoutlmv3_multipage/best_model/model.safetensors` | 3. 9.; najbolja epoha 4 od 4 prema povijesti/val konfiguraciji | 216/216 jednakih tenzora | `top_k_mean`, 3 |

SHA-256, redom:

```text
ResNet50    1c691fa0d61d8575dbea7d243f7bee6b204125c0bf2b73a8afd89bf6b9c70385
XLM-RoBERTa c84289d1e3a111e5d6b26d213e17147a7e4e951ec11e7b36a3181d7defecf34c
LayoutLMv3  9bf8314a872d03d7fabe288942a00213f76e11443b0e7946792b0fe7b39ee669
```

Veličine: 94.392.401 / 1.112.214.236 / 503.711.980 bajtova. Dokazi: `baseline_20260915/<model>/runtime_model.json`, odgovarajući `after_20260915/<model>/runtime_model.json`, te [provenance_remote_verified.json](provenance_remote_verified.json). SHA-256 težina isti je prije i poslije rada.

ResNet checkpoint izričito sadrži epohu i validacijski rezultat. XLM/Layout nemaju jednako bogat training receipt: njihov „najbolji trening” zaključak proizlazi iz spremljene povijesti, agregacijske konfiguracije, putanje i vremena datoteka. Nema kriptografski vezanog tadašnjeg manifesta i pune konfiguracije treninga. Ne može se dokazati svaka povijesna naredba ni autorstvo samo iz težina. Sva tri stara `results/*_multipage/test_predictions.csv` ipak sadrže upravo sadašnjih 475 test ID-jeva i njihove oznake, bez nepodudaranja.

### Jesu li to modeli trenirani od nule?

**Nijedan od ova tri treninga ne počinje cijelom mrežom nasumičnih težina.**

ResNet, `src/train_resnet.py:132`:

```python
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
for parameter in model.parameters():
    parameter.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
```

Početak je torchvision ImageNet checkpoint, u lokalnoj verziji `IMAGENET1K_V2` / `resnet50-11ad3fa6.pth`. Nova glava ima pet izlaza. Optimizator na `src/train_resnet.py:357` uzima samo trainable parametre: trenirano je 10.245 parametara glave od ukupno 23.518.277. Bitna nijansa: `model.train()` na retku 152 nije zamrznuo BatchNorm running statistike. Usporedba s lokalnim izvornim checkpointom potvrđuje 159 jednakih backbone parametarskih tenzora i 159 promijenjenih BN buffera. Zato „zamrznut backbone” ovdje znači zamrznute parametre, ne potpuno nepromjenjivo stanje mreže.

XLM, `src/train_text_model.py:39`, `:359`, `:365`:

```python
MODEL_NAME = "xlm-roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, ...)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
```

Nema eksplicitnog zamrzavanja encodera; `loss.backward()` je na retku 177. U spremljenom modelu svih 197 zajedničkih encoder tenzora razlikuje se od lokalnog osnovnog checkpointa; četiri klasifikatorska tenzora postoje u spremljenom klasifikatoru, ne u osnovnom jezičnom checkpointu. To podupire full fine-tuning, ne korištenje gotovog petoklasnog klasifikatora bez učenja.

LayoutLMv3: `MODEL_NAME = "microsoft/layoutlmv3-base"` na `src/train_layoutlm.py:42`; učitavanje na `:391`/`:396`, `AdamW(model.parameters(), ...)` na `:432`, backward preko scaler-a na `:217`. Nema eksplicitnog zamrzavanja; 209 od 212 zajedničkih tenzora promijenjeno je, tri relativna pozicijska bias tenzora ostala su ista. Nije točno tvrditi da se baš svaki tenzor promijenio; nije nađeno da su ta tri posebno zamrznuta u skripti. Nova klasifikacijska glava ima pet izlaza.

Lokalno dostupne osnovne HF revizije su XLM `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` i Layout `cfbbbff0762e6aab37086fdd4739ad14fe7d5db4`. **To su revizije u današnjem cacheu, ne dokaz povijesno korištene revizije**: trening skripte ne pinaju `revision=`. Detaljni tensor popisi su u `provenance_remote_verified.json`.

### Učitavanje, downloader i cloud

Put izvođenja: `app.py:258` → provjera dostupnosti (`app.py:216`) → getter (`app.py:110`, `:114`, `:118`) → lokalni loader (`src/predict_resnet.py:101`, `src/predict_text_model.py:100`, `src/predict_layoutlm.py:78`).

Prioritet je potpuna multipage mapa, pa potpuna legacy mapa `models/resnet50`, `models/xlm_roberta/best_model`, odnosno `models/layoutlmv3/best_model`. Ako postoji upotrebljiv lokalni model, `src/model_downloader.py:29` odmah vraća uspjeh: **nema automatske usporedbe s novim HF uploadom ni nadogradnje lokalnog legacy modela**.

Repo ID dolazi iz `HF_RESNET50_REPO_ID`, `HF_XLM_ROBERTA_REPO_ID`, `HF_LAYOUTLMV3_REPO_ID`: prvo environment, zatim Streamlit secrets (`src/model_downloader.py:237`). U pregledanom lokalnom procesu nije bio konfiguriran repo ID. Produkcijski downloader nema hardkodirani repo fallback. Donji repo ID-jevi su zadane vrijednosti zasebne upload skripte `scripts/upload_models_to_hf.py`, a ne očitani cloud secrets.

Download preko `snapshot_download` na `src/model_downloader.py:201` nema pinanu reviziju. ResNet/XLM traže prvo `models/<legacy_name>` unutar snapshota, zatim `<legacy_name>`, zatim root; XLM zatim `best_model` ili odgovarajuću model mapu. Layout redom traži `models/layoutlmv3/best_model`, `layoutlmv3/best_model`, `best_model`, root. Kopiraju u lokalne multipage ciljeve. Objavljeni upload format iz projekta uklapa se u te putove. Download nije izvršavan nad stvarnim modelima tijekom ove dijagnostike.

Javni HF API provjeren je bez tokena i bez slanja dokumenata/težina. Nakon početne mrežne zabrane uspio je odobren read-only dohvat:

| Repo iz upload skripte | Aktualna javna revizija | SHA težina jednak lokalnom |
|---|---|---|
| `M0nsterkill/document-ai-resnet50` | `6bbb0b94d393655bde1bb53de95ccc77d3663831` | Da |
| `M0nsterkill/document-ai-xlm-roberta` | `b72b75b2ec34cf301b6adc73d3d5d6abc96e1941` | Da |
| `M0nsterkill/document-ai-layoutlmv3` | `7718b3f84613a9dfc0dc2d0816f49f0524848197` | Da |

Sva tri remote ažuriranja su iz 3. rujna, oko 20:16–20:18 UTC. Izvor: spremljen API nalaz `provenance_remote_verified.json`; krajnje točke su `https://huggingface.co/api/models/<repo>?blobs=true`. Stari `debug_output/diagnostic_report.txt` iz 2. rujna odnosi se na druge, tadašnje revizije. Ni stara ni nova usporedba repozitorija nije dokaz stanja online memorijskog cachea.

## 4. Potvrđene greške i popravci

### A. Cache nije reagirao na novu lokalnu verziju

Prije: `@st.cache_resource` iznad gettera bez argumenata (`HEAD:app.py:85`). U kontroliranom testu loader je pozvan samo jednom i nakon zamjene testne datoteke.

Poslije: javni getter izračunava identitet odabranih težina i pripadajućih config/tokenizer/mapping datoteka; interni cache prima taj identitet (`app.py:86`, `src/model_runtime.py:33`). Isti fajl vraća isti objekt; zamjena daje novi objekt, dva poziva loadera. Identitet uključuje apsolutnu putanju, veličinu, mtime i ctime u nanosekundama. Cache je ograničen na jednu verziju po modelu. Namjerno očuvanje svih tih metapodataka nije detektirano; nakon deploya i dalje je razumno ponovno pokrenuti proces. Repo promjena bez zamjene lokalnih datoteka ne pokreće download.

### B. Nepotpuna mapa zasjenila je ispravan fallback

Prije: transformeri biraju multipage čim postoji `config.json`; dostupnost je mogla istodobno biti True zahvaljujući potpunoj legacy mapi. Test s nepotpunim multipage/kompletnim legacy direktorijem reproducirao je taj nesklad.

Poslije: loaderi i provjera dostupnosti koriste `is_local_model_complete` (`src/model_downloader.py:184`; `src/predict_text_model.py:61`, `src/predict_layoutlm.py:49`, `src/predict_resnet.py:30`). Provjera postojanja nije kriptografska provjera integriteta: oštećena prisutna datoteka i dalje može izazvati jasnu grešku pri učitavanju. Nije nađeno da je trenutačna lokalna mapa nepotpuna.

### C. ResNet download gubio je agregacijsku konfiguraciju

Prije: kopirani su mapping i težine, ne `aggregation_config.json`. Kontrolirani download lažnog checkpointa nije očuvao konfiguraciju.

Poslije: i ta se datoteka kopira (`src/model_downloader.py:78`). Sadašnji stvarni ResNet checkpoint u sebi već sadrži `mean`, pa ova greška **ne objašnjava sadašnju lokalnu ResNet predikciju**. Važna je za raw-state-dict izvoze koji konfiguraciju imaju samo izvan težina.

### D. Rotirani PDF: riječ i slika nisu bile poravnate

Prije: `_clean_pdf_words` skalira izvorni bbox bez rotacije. Za sintetičku riječ pri rotaciji 90° vraćao je `[90,117,339,172]`, a na renderiranoj slici trebalo je `[1028,90,1083,339]`.

Poslije: prije skaliranja primjenjuje se `fitz.Rect(item[:4]) * page.rotation_matrix` (`src/multipage_preprocess.py:129`). Testovi 0/90/180/270° sada svi prolaze. PyMuPDF vraća tekstualne koordinate u nerotiranom prostoru, dok render i `page.rect` uvažavaju rotaciju; to je opisano u [službenoj dokumentaciji](https://pymupdf.readthedocs.io/en/latest/page.html).

Od 2.059 sirovih PDF-ova u internom skupu tri dokumenta imaju rotirane stranice; svi su u treningu, ne u odabranom testu. Njihovi postojeći artefakti **nisu regenerirani**. Opseg stvarno pogođenih ranije odabranih stranica treba pregledati prije eventualne nove verzije dataset artefakata. Nijedan profesorov dokument nije identificiran kao rotiran.

Dokazi prije/poslije: `baseline_20260915/regressions.json` i `baseline_20260915/regressions_after.json`. Reprodukcije koriste sintetičke dokumente i privremene lažne modele, ne prepisuju stvarne modele. Početna zbirka: 56/56 testova; nakon dodatnog regresijskog testa: 57/57. Završni `git diff --check` nema whitespace pogrešaka.

## 5. Usklađenost treninga/evaluacije i aplikacije

### Potvrđeno na kodu i stvarnom izvođenju

Redoslijed svugdje: `0 invoice`, `1 cv`, `2 contract`, `3 email`, `4 scientific`. Provjerene su spremljene konfiguracije, mapiranja i state-dict glave; nema dokaza zamijenjenih oznaka ili slučajno učitane nove glave. `weights=None` u ResNet inference loaderu samo konstruira arhitekturu prije strogog `load_state_dict`; ne znači treniranje od nule.

ResNet: trening `src/train_resnet.py:90` i inference `src/predict_resnet.py:88` koriste RGB, Resize na 224×224, ToTensor i ImageNet mean `[.485,.456,.406]` / std `[.229,.224,.225]`. Nema center-cropa; stranica se deformira na kvadrat. Za svih 100 spremljenih dokumenata stvarni evaluacijski i aplikacijski tenzori te logiti jednaki su.

XLM: spremljeni chunkovi, max_length=512, stride=64, najviše 12 reprezentativnih chunkova (`src/multipage.py:113`; `src/predict_text_model.py:224`). Ne koristi samo prvih 512 tokena cijelog dokumenta, ali dio vrlo dugog teksta ipak izostaje izborom 12 chunkova. Tokenizer/padding ulazi jednaki su u svih 98 usporedivih testnih dokumenata; dva aplikacija odbija prije modela. Usporedbom se uklanja samo neaktivni trailing padding, ne aktivni tokeni. Na jednakim ulazima najveća razlika logita je nula.

Layout: trening `src/train_layoutlm.py:116`, inference `src/predict_layoutlm.py:120`; processor `apply_ocr=False` (`:84`), riječi i boxovi dolaze izvana. Koordinate se jednom normaliziraju u 0–1000 (`src/multipage.py:67`), riječ i bbox idu zajedno; svaka stranica se tokenizira uz truncation/padding do 512. Nema dodatnih tekstualnih chunkova unutar jedne Layout stranice. Trening Dataset i aplikacija imaju jednake kanonske ulaze za svih 100 spremljenih dokumenata. Najveća razlika logita 0.009422 odgovara različitom numeričkom putu: evaluacijska funkcija koristi CUDA autocast (`src/train_layoutlm.py:249`), aplikacija FP32. Top-1 rezultat ostao je isti u ovoj usporedbi; nije otkriven problem s redoslijedom boxova ili dvostrukom normalizacijom na ovim ulazima.

Agregacija je ista u treningu/evaluaciji i aplikaciji: `src/multipage.py:163`. `mean`, `max`, ili top-k logiti po klasi po stranicama/chunkovima; zatim jedan softmax nad klasama (`dim=0`, redak 188). `top_k_mean` odabire najbolje tri vrijednosti **zasebno za svaku klasu**, ne tri stranice s najvišom ukupnom sigurnošću. Per-page/per-chunk softmax ima `dim=1`. Ne agregiraju se već softmaxirane vrijednosti pa ponovno softmaxiraju. UI ih samo množi sa 100 pri prikazu (`app.py:413`). Prag UI-ja ostao je 0.50 (`app.py:67`), nije ugađan ovom evaluacijom. Predikcija 36,51% ne znači 36,51% ukupne točnosti; u trenutačnom UI-ju ne prolazi taj prag.

Stranice: do 12, reprezentativni početak/sredina/kraj (`src/multipage.py:38`). Dokument s dvije ili tri stranice zadržava sve stranice ako priprema uspije. ResNet prima i prazne renderirane stranice; Layout ih bez riječi preskače (`src/multipage_training.py:88`, `src/multipage_preprocess.py:44`). To je usklađeno s postojećim treningom, ali prazna ili semantički drukčija dodatna stranica može utjecati na dokumentnu predikciju. Nije mijenjano bez dokaza na konkretnim primjerima.

Cache ulaza je u pojedinačnom `prepared` objektu (`app.py:237`), a svaki upload dobiva novu privremenu mapu i novi objekt (`app.py:892`, `src/document_adapter.py:405`). Nije nađen globalni cache predikcije koji bi vraćao rezultat drugog dokumenta.

### Potvrđene razlike svježe pripreme prema starim artefaktima

| Model | Dokumenti s jednakim svježim i spremljenim tenzorima | Napomena |
|---|---:|---|
| ResNet | 92/100 | svih osam razlika odnosi se na TXT renderiranje |
| XLM | 70/98 usporedivih | 28 razlika u svježe izvučenom/OCR tekstu; još dva odbijena kratka ulaza |
| Layout | 87/100 | osam TXT rendera i pet PNG OCR/layout ulaza |

TXT u aplikaciji crta `src/document_adapter.py:100`; stara priprema koristi `src/preprocess.py:267`. Font, raspored i način dobivanja boxova nisu ista specifikacija. PNG u aplikaciji ponovno prolazi OCR, dok spremljeni artefakti mogu sadržavati ranije generiran tekst i boxove. To je stvarna razlika ulaza, **nije dokaz da je svježa verzija netočna niti je mijenjana radi rezultata testa**. Model dobiva isti rezultat kad dobije iste tenzore.

OCR lokalno radi: Tesseract 5.4.0.20240606, instalirani jezici samo `eng` i `osd` (`baseline_20260915/ocr_runtime.json`). `src/preprocess.py:141` ne zadaje `lang`, dakle ne traži hrvatski OCR. Ne može se obećati jednaka kvaliteta za hrvatske skenove. Jezik nije promijenjen niti instaliran novi paket.

Ostala ograničenja koja treba razlikovati od potvrđenog uzroka:

- PDF tekstualni put koristi ukupni embedded tekst ako ima barem 20 znakova; inače OCR (`src/document_adapter.py:294`, `src/predict_text_model.py:126`). U miješanom tekstualnom/skeniranom PDF-u to može izostaviti tekst skeniranih stranica. Nije potvrđeno da se to dogodilo profesorovim primjerima.
- Layout bira embedded riječi ako ih stranica ima, inače OCR; stranica s malim embedded dijelom i velikim skeniranim dijelom ostaje potencijalan rubni slučaj (`src/multipage_preprocess.py:139`).
- DOCX tekstualni put čita odlomke, ne tablice (`src/document_adapter.py:355`); vizualni put ovisi o LibreOffice pretvorbi. Nije provedena stvarna DOCX konverzija u ovoj evaluaciji. Test za nedostupan converter prolazi; to nije dokaz instaliranog convertera.
- U svim spremljenim tekstovima 34 dokumenta imaju manje od 20 znakova. Builder chunkova nema isti MIN_TEXT guard (`scripts/build_multipage_dataset.py:303`), dok aplikacija ima (`src/document_adapter.py:61`, `src/predict_text_model.py:214`). Zato evaluacija spremljenih tokena i dostupnost tekstualne predikcije nisu isti pokazatelj. Sigurnosni guard nije snižen radi povećanja pokrivenosti.

## 6. Ograničena evaluacija

Odabir: `random.Random(20260915)`, sortirani test ID-jevi, uzorak 20 po klasi prije gledanja predikcija (`scripts/diagnose_models.py:73`). Svih 100 ID-jeva i 25 vanjskih putanja u `baseline_20260915/selection.json`; ista datoteka korištena nakon popravaka. Ground truth internog skupa dolazi iz manifesta, vanjskog iz postojećih mapa.

Vanjski skup `data/external_test/raw` nema identičnih raw SHA-256 datoteka u internom skupu. To **ne dokazuje potpunu neovisnost izvora/predložaka**. Oznake nisu iznova stručno adjudicirane. Neki CV PDF-ovi nazivom označavaju zbirke primjera, a ne nužno jedan CV. Ovo je postojeći mali vanjski test, ne nov, reprezentativan, naslijepo prikupljen test. Nije pomiješan s internim rezultatima.

Rezultati pravog aplikacijskog puta sa svježom pripremom, prije = poslije:

| Skup | Model | Accuracy, uz greške obrade | Macro-F1 | Greške obrade |
|---|---|---:|---:|---:|
| Interni 100 | ResNet50 | 87/100 = 87% | 0.868163 | 0 |
| Interni 100 | XLM-RoBERTa | 94/100 = 94% | 0.949132 | 2 |
| Interni 100 | LayoutLMv3 | 99/100 = 99% | 0.989994 | 0 |
| Vanjski 25 | ResNet50 | 14/25 = 56% | 0.505165 | 0 |
| Vanjski 25 | XLM-RoBERTa | 24/25 = 96% | 0.959596 | 0 |
| Vanjski 25 | LayoutLMv3 | 25/25 = 100% | 1.000000 | 0 |

XLM odbijeni interni dokumenti: `invoice_extra_0226` (14 znakova) i `email_extra_0072` (12 znakova). Oba ostaju u nazivniku accuracy i false-negative brojanju svoje stvarne klase; nisu tiho izbačeni. `PROCESSING_ERROR` je izvještajni stupac, ne šesta naučena klasa. Macro-F1 je prosjek pet stvarnih klasa.

F1 po klasi (preciznost, odziv i support u [per_class_metrics.csv](per_class_metrics.csv)):

| Model/skup | invoice | cv | contract | email | scientific |
|---|---:|---:|---:|---:|---:|
| ResNet interni | .8780 | .8372 | .9302 | .8235 | .8718 |
| XLM interni | .9500 | .9500 | .9524 | .9189 | .9744 |
| Layout interni | 1.0000 | .9744 | 1.0000 | .9756 | 1.0000 |
| ResNet vanjski | .7500 | .4615 | .0000 | .6000 | .7143 |
| XLM vanjski | .8889 | 1.0000 | .9091 | 1.0000 | 1.0000 |
| Layout vanjski | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Sve matrice zabune: [matrice_zabune.txt](matrice_zabune.txt); strojno u `after_20260915/<model>/metrics.json`. ResNet nije pogodio nijedan od pet vanjskih ugovora. To je koristan signal ograničene generalizacije na tom skupu, ne dokaz uzroka konkretnog HT računa.

Analiza fiksnog praga sigurnosti 80%, bez ugađanja:

| Model | Interni: prolazi / ukupno | Pogrešnih među prolazima | Vanjski: prolazi / ukupno | Pogrešnih među prolazima |
|---|---:|---:|---:|---:|
| ResNet | 46/100 | 0 | 1/25 | 0 |
| XLM | 96/100 | 2 | 25/25 | 1 |
| Layout | 100/100 | 1 | 25/25 | 0 |

Dakle veća sigurnost nije jamstvo ispravnosti; ResNetov prag odbacio bi gotovo cijeli mali vanjski skup. Nema obećanja „80% točnosti” iz ovih brojki.

Kontrolni put na spremljenim artefaktima: ResNet evaluacija i app 86/100; XLM evaluacija 95/100 bez greške obrade, app 95/100 uz dva odbijanja (pogreške i odbijanja nisu isti ishodi); Layout oba 99/100. Razlike od svježe pripreme nisu skrivene. Po stranici/chunku sačuvani su logiti, ulazni hashovi, dimenzije, predikcije i dokumentna agregacija u `items.jsonl`; sadržaj OCR teksta nije ispisan u te rezultate.

Stari spremljeni rezultati za svih 475 test dokumenata: ResNet accuracy 0.873684 / macro-F1 0.873169; XLM 0.968421 / 0.968531; Layout 0.993684 / 0.993704. To su **postojeći rezultati**, ne novo izvršen puni test. Ne izjednačavati ih s novim uzorkom 100.

## 7. Podaci, preklapanja i trening

Aktualne tri skripte pozivaju `load_multipage_training_rows` (`src/multipage_training.py:53`), koji koristi `data/multipage/document_manifest.csv` i odgovarajuće page/chunk artefakte. Ne koriste legacy splitove. Provjerena je usklađenost ID-jeva, labela, splitova i augmentation/parent grupa.

| Klasa | Train | Validation | Test | Ukupno |
|---|---:|---:|---:|---:|
| invoice | 715 | 137 | 92 | 944 |
| cv | 734 | 142 | 96 | 972 |
| contract | 742 | 142 | 96 | 980 |
| email | 739 | 142 | 96 | 977 |
| scientific | 736 | 142 | 95 | 973 |
| Ukupno | 3666 | 705 | 475 | 4846 |

„Oko 1000 po klasi” odgovara ukupnom broju, **ne broju trening dokumenata po klasi**. Broj artefakata train/validation/test: ResNet stranice 8286/1547/1093; Layout valjane stranice 8274/1546/1091; XLM chunkovi 10510/1928/1329. Svaki model ima artefakte za svih 4846 dokumenata. Više stranica/chunkova nisu dodatni neovisni dokumenti.

SHA dokumentnog manifesta: `de66909f232402152779f6f00f66fb8bec172e85076a75a9c89d5351571c146a`. Detalji/hashovi page i chunk manifesta: `baseline_20260915/dataset_audit.json`.

Legacy `data/splits/*.csv` imaju iste ukupne ID-jeve/brojnosti, ali za 1931 dokument dodjela se razlikuje; cijeli popis je `baseline_20260915/split_disagreements.csv`. Multipage builder radi vlastitu group-aware podjelu (`src/multipage_manifest.py:135`), ne uvozi stare splitove. Kasniji importer čuva postojeće dodjele i dodaje samo train (`scripts/import_second_training_data.py:767`, `:804`, `:815`, `:822`). Dostupni import izvještaji navode dodatnih 121 + 12 trening dokumenata. Nije bilo promjene podjela u ovoj dijagnostici. Usporedba legacy i multipage metrika zato nije strogo usporedba na istom holdoutu.

### Duplikati i slični predlošci

- Izračunati stvarni SHA-256 svih 4846 raw datoteka: nema identičnih raw duplikata, nema nepodudaranja s manifestom ni read grešaka. Provjera deklariranih parent/augmentation grupa također prolazi. To **nije** dovoljno za dokaz izostanka sadržajnog preklapanja.
- Hash svih 10926 spremljenih page slika: tri grupe jednakih stranica između različitih dokumenata, jedna grupa preko splitova. Ta cross-split grupa je potpuno bijela prazna stranica, ne zajednički informativni sadržaj. Jedna informativna jednaka stranica nađena je unutar train skupa.
- Normalizirani tekst (casefold, riječi, razmaci): 208 grupa ponavljanja, 14 preko splitova. Nakon uklanjanja znamenki: 210 grupa, 15 preko splitova. Nije svaki takav par byte-identičan original niti svaki normalizirani tekst jednaki model-input.
- Stroža provjera stvarnih spremljenih `input_ids` svih odabranih chunkova: **7 cross-split grupa**. Primjeri `contract_0008` validation / `contract_0095` train; `contract_0010` test / `contract_0100` train. To je konkretno preklapanje tekstualnih ulaza, ne samo sumnja po nazivu dokumenta.
- Potonji par izričito je naveden na `data/multipage/document_manifest.csv:411` i `:501`: raw PDF `data/raw/contract/pdf/contract_0019.pdf` nasuprot TXT datoteci `data/raw/contract/text/contract_0019.txt`. Različiti formati imaju različite raw hashove, ali jednake odabrane tokene. Zbog toga sama provjera raw hashova taj par nije mogla prepoznati.
- Dijagnostički TF-IDF nearest-train pregled, prag cosine >= .98: 18 validation i 14 test kandidata vrlo sličnog teksta. To su kandidati za ručni pregled, ne automatski proglašeni duplikati. TF-IDF služi isključivo auditu; nije korišten za treniranje klasifikatora ili odabir njihovih hiperparametara.

Dokazi: `baseline_20260915/duplicate_groups.json`, `data_followup_20260915/identical_token_groups.json`, `identical_page_inspection.json`, `identical_normalized_text_details.json`, `near_text_candidates.csv`. Podjele nisu očišćene niti je naknadno odabran „povoljniji” testni uzorak.

### Izvori i zastupljenost

`data/dataset_sources_extra.csv` pokriva 3846 aktualnih ID-jeva; za prvih 1000 nedostaje taj provenance zapis. Za 1000 redaka izričito piše `hr`, za preostalih 2846 iz tog CSV-a jezik nije naveden. Ne može se iz toga potvrditi točan broj engleskih dokumenata. Među zapisanim izvorima su RVL-CDIP, CORD, SROIE, raw PDF resumes, CUAD, ContractNLI, SEC, AESLC, arXiv te ručni importi. „Vlastiti skup u projektu” nije isto što i dokumenti koje je korisnik osobno izradio ili prikupio bez javnih datasetova.

800 zapisa označeno je sintetičkima: po 200 hrvatskih invoice/CV/email/scientific; dodatnih 200 hrvatskih ugovora označeno je izvorom javne nabave. Ukupno 688 zapisa označeno je augmentiranima (to se može preklapati sa sintetičkim zapisima). Raspodjela izvora po klasi/splitu nalazi se u `data_followup_20260915/source_by_split.csv`.

Traženje pojmova u spremljenom tekstu nalazi „europass” u 26 dokumenata, „gmail” u 90, „outlook” u 39; nije našlo „uplatnica” ni „hrvatski telekom”. To su samo tekstualni indikatori: OCR može propustiti pojam, a riječ nije dokaz autentičnog predloška. `scripts/add_croatian_dataset.py:477` ima varijantu nazvanu „Europass”, ali sintetički crtež te oznake nije dokaz dobre zastupljenosti stvarnih Europass dokumenata. Nije potvrđen broj autentičnih HT računa s uplatnicama ni broj različitih mail klijenata/predložaka.

Samostalna uplatnica nije trenirana zasebna klasa. Račun s priloženom uplatnicom je dokument s različitim tipovima stranica; to nije isto što i samostalna uplatnica. Ne uvodi se nova klasa niti se profesorovi primjeri dodaju treningu.

### Krivulje i zaključci o učenju

Iz `results/*_multipage/training_history.csv`:

- ResNet, 8 epoha: train loss 1.2880 → .5462; validation loss 1.0741 → .5366; val macro-F1 .7967 → .8759. Nema klasičnog jasnog rastućeg validation-loss dokaza overfittinga. Poboljšanje do zadnje epohe i slabi vanjski rezultat opravdavaju istraživanje ograničenja zamrznutih značajki, ali ne dokazuju da je dodatni trening dovoljan.
- XLM, 4 epohe: train loss .2098 → .0337; train accuracy .9690 → .9959; val F1 oko .9716 → .9758, nakon treće epohe praktično plato. To je signal saturacije i train/validation razlike, ne dokaz da „treba više epoha”. Najbolji spremljeni izbor je epoha 3.
- Layout, 4 epohe: train accuracy .9587 → .9957; validation F1 .9660 → .9773. Validation loss .0866 u trećoj → .0951 u četvrtoj, uz bolji F1. Jedan pomak nije dovoljan za čvrst zaključak o preučenosti. Sadržajno preklapanje i raspodjela izvora otežavaju interpretaciju visokog internog rezultata.

Trening metrika koristi document-balanced sampler s ponavljanjem (`src/multipage_training.py:127`) i drukčiji način prikupljanja od potpune val evaluacije; train/validation accuracy nisu savršeno simetrične usporedbe. Postojeća UI rečenica o „jakom template/source biasu” (`app.py:848`) sama po sebi nije mjerenje ni dokaz uzroka profesorovih pogrešaka.

## 8. Što je pripremljeno, a nije izvršeno

`configs/resnet50_layer4_proposal.json` i `scripts/train_resnet_layer4.py` dodaju zaseban, opt-in eksperiment:

- počinje od točno hashiranog postojećeg lokalnog ResNet checkpointa;
- trainable samo `layer4.*` i `fc.*`: 14.974.981 parametar;
- backbone LR 1e-5, glava 1e-4;
- BN running statistike svih blokova ostaju zamrznute; BN affine parametri u layer4 jesu trainable;
- najbolji checkpoint prema validation macro-F1, patience=2, fiksna postojeća ResNet `mean` agregacija;
- novi, unaprijed nepostojeći direktorij `models/resnet50_layer4_experiment_20260915`; odbija postojeći izlazni direktorij i drukčiji hash ulaznog checkpointa;
- sprema konfiguraciju, seed, kod i manifest hashove; test ne koristi za odabir niti ga evaluira.

Dokaz samo preflight provjere: [resnet_layer4_preflight_final.json](resnet_layer4_preflight_final.json): `training_started=false`, BN svi eval, test forward passes=0. Eksperimentalna mapa nije kreirana. **Ni jedan batch nije treniran.** Ova konfiguracija je pripremljena mogućnost, ne preporuka da se odmah pokrene na podjelama s poznatim preklapanjima.

Za XLM/Layout nije dodana nova arhitektura ili novi trening. Najprije treba popraviti kvalitetu i neovisnost evaluacije, potvrditi OCR na stvarnim problematičnim dokumentima te cloud identitet. Ako budući odvojeni validacijski skup pokaže sustavne rupe, tek tada planirati ciljano prikupljanje novih train dokumenata i dodatno treniranje, bez korištenja prijavljenih testnih primjera za tuning.

## 9. Popis promijenjenih datoteka

Postojeći produkcijski kod: `app.py`, `src/model_downloader.py`, `src/multipage_preprocess.py`, `src/predict_resnet.py`, `src/predict_text_model.py`, `src/predict_layoutlm.py`.

Nove datoteke koda/konfiguracije: `src/model_runtime.py`, `configs/resnet50_layer4_proposal.json`, `scripts/diagnose_models.py`, `scripts/check_diagnostic_regressions.py`, `scripts/inspect_model_provenance.py`, `scripts/inspect_data_followup.py`, `scripts/train_resnet_layer4.py`, `scripts/log_model_runtime.py`, `scripts/summarize_diagnostics.py`, `scripts/verify_diagnostic_artifacts.py`, `tests/test_diagnostic_regressions.py`.

Novi izvještaji i rezultati: `reports/` s gore navedenim JSON/CSV/TXT datotekama, baseline/after/data_followup mapama i ovim dokumentom. `reports/.gitignore` isključuje pripremljene kopije originala, renderirane stranice i OCR u `**/prepared/` iz Git praćenja. Ti lokalni sadržaji nisu poslani na mrežu. Izvještaji sadrže ID-jeve i putanje; prije eventualnog javnog objavljivanja također ih treba pregledati. Nije napravljen commit ili push.

## 10. Nepotvrđeno i prioritetni sljedeći koraci

**Nije utvrđen uzrok triju profesorovih konkretnih predikcija.** Nisu dostupni pouzdano identificirani originali, online runtime log, verzija online koda, tadašnja priprema/tokens/boxovi niti svi izlazi sva tri modela. Mogući čimbenici su druga/stara datoteka u cloudu, OCR/predložak/jezik, limit tokena, domenska razlika ili utjecaj druge vrste stranice. Nijedan od njih nije pripisan konkretnom primjeru bez dokaza.

Cloud verzije biblioteka također nisu potvrđene; `requirements.txt:1` nadalje navodi pakete bez pinanja verzija. Lokalno ponovljivo izvođenje zato nije dokaz identičnog cloud programskog okruženja.

1. Najprije u online procesu uključiti donju dijagnostiku i usporediti SHA-256 sa sekcijom 3. Novi server log ispisuje kod/source hashove, stvarnu lokalnu datoteku, hash, class mapping i eval/aggregation stanje, bez tokena i sadržaja dokumenta. Prvi download s uključenim flagom logira i razriješenu snapshot reviziju. Za već postojeće kopije nema izmišljene povijesne revizije. Ponovno pokretanje procesa prazni memorijski cache, ali **ne preuzima automatski novu težinu ako lokalna već postoji**.
2. Dobaviti točne originalne CV/mail/HT dokumente. Pokrenuti svaki kroz sva tri modela i spremiti per-page/chunk izlaze. Ne koristiti PDF sa screenshotovima kao zamjenu. Ako imaju osobne podatke, zadržati lokalno ili pribaviti dopuštenje za slanje.
3. Ručno pregledati cross-split sadržajne parove i vrlo slične predloške; pripremiti zasebnu, verzioniranu, po izvoru/predlošku odvojenu validaciju i zaključani test. Ne mijenjati ovu postojeću podjelu usput niti predstavljati stare i nove test brojke kao isti benchmark.
4. Prije novog treninga definirati zajedničku TXT/render/OCR specifikaciju i politiku kratkih/miješanih tekst-sken ulaza. Eventualna obnova rotiranih artefakata treba novu verziju i hashove, ne tiho prepisivanje stare evaluacije.
5. Tek zatim razmotriti pripremljeni ResNet layer4 eksperiment. Postojeći rezultati ne daju razlog za automatsko povećavanje epoha XLM/Layout modela.

### Točne lokalne PowerShell naredbe

```powershell
Set-Location -LiteralPath 'C:\Users\borna\Desktop\AI zavrsni\AI modeli\document_ai_classifier'
$diagPython = '..\venv\Scripts\python.exe'

# Read-only provjera stvarno učitanog modela, bez dokumenta i bez downloada:
& $diagPython -B scripts/log_model_runtime.py --load --model resnet50
& $diagPython -B scripts/log_model_runtime.py --load --model xlm_roberta
& $diagPython -B scripts/log_model_runtime.py --load --model layoutlmv3

# Uključivanje logova pri učitavanju u aplikaciji:
$env:DOCUMENT_AI_MODEL_DIAGNOSTICS = '1'
& $diagPython -m streamlit run app.py

# Testovi:
& $diagPython -B -m unittest discover -s tests -v

# Ponovljiva evaluacija istog uzorka; novi naziv mape, postojeći se odbija:
& $diagPython -B scripts/diagnose_models.py prepare --output reports/recheck_01 --selection reports/baseline_20260915/selection.json
& $diagPython -B scripts/diagnose_models.py evaluate --output reports/recheck_01
& $diagPython -B scripts/diagnose_models.py audit --output reports/recheck_01

# Samo preflight, NE treniranje:
& $diagPython -B scripts/train_resnet_layer4.py

# SAMO ako kasnije izričito odlučiš trenirati; ova naredba OVDJE NIJE izvršena:
# & $diagPython -B scripts/train_resnet_layer4.py --train
```

Na Linux/Streamlit serveru postaviti environment `DOCUMENT_AI_MODEL_DIAGNOSTICS=1`, restartati aplikaciju i učitati model; tražiti `DOCUMENT_AI_MODEL` / `DOCUMENT_AI_DOWNLOAD` u serverskom logu. Ako imaš shell, `python scripts/log_model_runtime.py --load --model layoutlmv3` provjerava novi lokalni Python proces, **ne automatski već postojeći Streamlit proces**. Za postojeći proces potreban je njegov vlastiti log.

### Jednostavno, tehnički točno objašnjenje

Modeli nisu trenirani od nule. ResNet50 započeo je ImageNet težinama, uz učenje nove petoklasne glave i promjenu BatchNorm statistika. XLM-RoBERTa i LayoutLMv3 započeli su javnim osnovnim checkpointima pa su dodatno trenirani bez eksplicitnog zamrzavanja mreže na projektnom skupu. Aplikacija lokalno učitava te spremljene prilagođene modele; ne koristi samo netaknute osnovne klasifikatore. Podrijetlo pretreniranja i dodatni projektni trening dvije su različite faze. Takav način treniranja sam po sebi ne dokazuje dobru točnost ni generalizaciju; to treba provjeravati na neovisnim, kvalitetno označenim dokumentima uz navedena ograničenja.

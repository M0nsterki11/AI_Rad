import json
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict_resnet import (  # noqa: E402
    load_model as load_resnet_model,
    predict_images as predict_resnet_images,
)
from src.predict_text_model import (  # noqa: E402
    load_text_model,
    predict_text,
)
from src.predict_layoutlm import (  # noqa: E402
    load_layoutlm_model,
    predict_layout_pages,
)
from src.model_downloader import (  # noqa: E402
    ensure_model_available,
    is_model_available,
)
from src.document_adapter import prepare_document_for_models  # noqa: E402


def preferred_results_dir(multipage_name, legacy_name):
    multipage_dir = PROJECT_ROOT / "results" / multipage_name
    if (multipage_dir / "test_metrics.json").is_file():
        return multipage_dir
    return PROJECT_ROOT / "results" / legacy_name


RESNET_RESULTS_DIR = preferred_results_dir("resnet50_multipage", "resnet50")
XLM_RESULTS_DIR = preferred_results_dir("xlm_roberta_multipage", "xlm_roberta")
LAYOUT_RESULTS_DIR = preferred_results_dir("layoutlmv3_multipage", "layoutlmv3")
EXTERNAL_RESULTS_DIR = PROJECT_ROOT / "results" / "external_test"
FINAL_COMPARISON_PATH = PROJECT_ROOT / "results" / "final_comparison.csv"
FINAL_COMPARISON_CHART_PATH = PROJECT_ROOT / "results" / "final_comparison.png"

RESULT_DIRS = {
    "ResNet50": RESNET_RESULTS_DIR,
    "XLM-RoBERTa": XLM_RESULTS_DIR,
    "LayoutLMv3": LAYOUT_RESULTS_DIR,
}
EXTERNAL_RESULT_DIRS = {
    "ResNet50": EXTERNAL_RESULTS_DIR / "resnet50",
    "XLM-RoBERTa": EXTERNAL_RESULTS_DIR / "xlm_roberta",
    "LayoutLMv3": EXTERNAL_RESULTS_DIR / "layoutlmv3",
}
PREDICTION_MODEL_LABELS = {
    "resnet50": "ResNet50",
    "xlm_roberta": "XLM-RoBERTa",
    "layoutlmv3": "LayoutLMv3",
}
CLASS_NAMES = ["invoice", "cv", "contract", "email", "scientific"]
PREDICTION_CONFIDENCE_THRESHOLD = 0.50
ALL_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "txt", "docx"]

MODEL_OPTIONS = [
    "ResNet50 – vizualni model",
    "XLM-RoBERTa – tekstualni model",
    "LayoutLMv3 – multimodalni model",
    "Usporedi ResNet50 i XLM-RoBERTa",
    "Usporedi sva 3 modela",
]
MODEL_KEYS_BY_OPTION = {
    MODEL_OPTIONS[0]: ["resnet50"],
    MODEL_OPTIONS[1]: ["xlm_roberta"],
    MODEL_OPTIONS[2]: ["layoutlmv3"],
    MODEL_OPTIONS[3]: ["resnet50", "xlm_roberta"],
    MODEL_OPTIONS[4]: ["resnet50", "xlm_roberta", "layoutlmv3"],
}


@st.cache_resource
def get_resnet_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, checkpoint_structure = load_resnet_model(device)
    return model, class_names, device, checkpoint_structure


@st.cache_resource
def get_text_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, class_names, label_to_index, device = load_text_model(device)
    return model, tokenizer, class_names, label_to_index, device


@st.cache_resource
def get_layoutlm_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, class_names, label_to_index, device = load_layoutlm_model(device)
    return model, processor, class_names, label_to_index, device


def show_model_status_sidebar():
    st.sidebar.subheader("Status modela")
    for model_key, model_label in PREDICTION_MODEL_LABELS.items():
        status = "dostupno" if is_model_available(model_key) else "nedostaje"
        st.sidebar.write(f"{model_label}: {status}")


def is_recognized_prediction(result):
    confidence = float(result.get("confidence", 0.0) or 0.0)
    return confidence >= PREDICTION_CONFIDENCE_THRESHOLD


def preparation_error_reason(prepared, categories):
    matching = [
        message
        for message in prepared.get("errors", [])
        if any(message.startswith(category) for category in categories)
    ]
    return " ".join(matching)


def prepared_model_status(prepared, model_key):
    image_path = prepared.get("image_path")
    text_path = prepared.get("text_path")
    ocr_path = prepared.get("ocr_path")
    page_artifacts = prepared.get("page_artifacts") or []
    layout_page_artifacts = prepared.get("layout_page_artifacts") or []

    if model_key == "resnet50" and page_artifacts:
        if all(Path(page["image_path"]).is_file() for page in page_artifacts):
            return True, ""

    if model_key == "layoutlmv3" and layout_page_artifacts:
        try:
            for page in layout_page_artifacts:
                if not Path(page["image_path"]).is_file() or not Path(page["ocr_path"]).is_file():
                    raise ValueError(f"Nedostaje artefakt stranice {int(page['page_index']) + 1}.")
                words = page.get("words", [])
                boxes = page.get("boxes", [])
                if not words or len(words) != len(boxes):
                    raise ValueError(
                        f"OCR riječi i boxovi nisu valjani za stranicu "
                        f"{int(page['page_index']) + 1}."
                    )
        except Exception as error:
            return False, f"OCR/layout input nije moguće učitati: {error}"
        return True, ""

    if model_key == "resnet50":
        if image_path and Path(image_path).is_file():
            return True, ""
        reason = preparation_error_reason(prepared, ["Vizualni input"])
        return False, reason or "Nije moguće pripremiti sliku za ResNet50."

    if model_key == "xlm_roberta":
        if text_path and Path(text_path).is_file():
            return True, ""
        reason = preparation_error_reason(prepared, ["Tekstualni input"])
        return False, reason or "Nije moguće pripremiti tekst za XLM-RoBERTa."

    if model_key == "layoutlmv3":
        missing = []
        if not image_path or not Path(image_path).is_file():
            missing.append("slika")
        if not text_path or not Path(text_path).is_file():
            missing.append("tekst")
        if not ocr_path or not Path(ocr_path).is_file():
            missing.append("OCR/bounding boxovi")
        if missing:
            reason = preparation_error_reason(
                prepared,
                ["Vizualni input", "Tekstualni input", "OCR/layout input"],
            )
            fallback = f"Nije moguće pripremiti: {', '.join(missing)}."
            return False, reason or fallback

        try:
            payload = json.loads(Path(ocr_path).read_text(encoding="utf-8"))
            words = payload.get("words", [])
            boxes = payload.get("boxes", [])
            if not words or len(words) != len(boxes):
                raise ValueError("OCR riječi i bounding boxovi nisu valjani.")
        except Exception as error:
            return False, f"OCR/layout input nije moguće učitati: {error}"
        return True, ""

    return False, f"Nepoznat model: {model_key}"


def ensure_model_for_live_prediction(model_key):
    if is_model_available(model_key):
        return True, ""

    model_label = PREDICTION_MODEL_LABELS.get(model_key, model_key)
    with st.spinner(f"Preuzimam model {model_label}..."):
        available, message = ensure_model_available(model_key)
    return bool(available), str(message or "Model nije dostupan.")


def skipped_outcome(model_key, reason):
    return {
        "model_key": model_key,
        "model": PREDICTION_MODEL_LABELS.get(model_key, model_key),
        "status": "Preskočeno",
        "reason": reason,
        "result": None,
        "debug": "",
    }


def prepared_page_inputs(prepared, *, layout=False):
    cache_key = "_loaded_layout_page_inputs" if layout else "_loaded_page_inputs"
    cached = prepared.get(cache_key)
    if cached is not None:
        return cached
    artifact_key = "layout_page_artifacts" if layout else "page_artifacts"
    pages = prepared.get(artifact_key) or []
    images = []
    for page in pages:
        with Image.open(page["image_path"]) as source:
            images.append(source.convert("RGB"))
    cached = {
        "images": images,
        "words": [list(page.get("words", [])) for page in pages],
        "boxes": [list(page.get("boxes", [])) for page in pages],
        "page_indices": [int(page["page_index"]) for page in pages],
    }
    prepared[cache_key] = cached
    return cached


def run_prepared_model_prediction(prepared, model_key):
    ready, reason = prepared_model_status(prepared, model_key)
    if not ready:
        return skipped_outcome(model_key, reason)

    available, message = ensure_model_for_live_prediction(model_key)
    if not available:
        return skipped_outcome(model_key, message)

    try:
        if model_key == "resnet50":
            model, class_names, device, _ = get_resnet_model()
            page_inputs = prepared_page_inputs(prepared)
            result = predict_resnet_images(
                page_inputs["images"],
                page_indices=page_inputs["page_indices"],
                total_pages=prepared.get("total_pages"),
                model=model,
                class_names=class_names,
                device=device,
            )
        elif model_key == "xlm_roberta":
            text = Path(prepared["text_path"]).read_text(encoding="utf-8", errors="ignore")
            model, tokenizer, class_names, _, device = get_text_model()
            result = predict_text(
                text,
                model=model,
                tokenizer=tokenizer,
                class_names=class_names,
                device=device,
            )
        elif model_key == "layoutlmv3":
            page_inputs = prepared_page_inputs(prepared, layout=True)
            model, processor, class_names, _, device = get_layoutlm_model()
            result = predict_layout_pages(
                page_inputs["images"],
                page_inputs["words"],
                page_inputs["boxes"],
                page_indices=page_inputs["page_indices"],
                total_pages=prepared.get("total_pages"),
                model=model,
                processor=processor,
                class_names=class_names,
                device=device,
            )
        else:
            return skipped_outcome(model_key, f"Nepoznat model: {model_key}")
    except Exception as error:
        return {
            "model_key": model_key,
            "model": PREDICTION_MODEL_LABELS.get(model_key, model_key),
            "status": "FAIL",
            "reason": f"Predikcija nije uspjela: {error}",
            "result": None,
            "debug": traceback.format_exc(),
        }

    return {
        "model_key": model_key,
        "model": PREDICTION_MODEL_LABELS.get(model_key, model_key),
        "status": "Uspješno",
        "reason": "",
        "result": result,
        "debug": "",
    }


def outcome_probability_dict(outcome):
    result = outcome.get("result") or {}
    probabilities = result.get("probabilities", {})
    if isinstance(probabilities, dict):
        return {label: safe_float(value) for label, value in probabilities.items()}
    return {
        row.get("class"): safe_float(row.get("probability"))
        for row in probabilities
        if row.get("class")
    }


def show_prepared_document_preview(prepared, model_keys):
    suffix = prepared["suffix"]
    if suffix == ".docx":
        st.info(
            "DOCX se automatski pretvara u tekst za XLM-RoBERTa i u PDF/sliku "
            "za vizualne modele."
        )
    elif suffix == ".txt":
        st.info(
            "TXT se direktno koristi za XLM-RoBERTa, a za vizualne modele se "
            "renderira u sliku."
        )

    image_path = prepared.get("image_path")
    if image_path and Path(image_path).is_file():
        st.image(
            str(image_path),
            caption="Preview prve odabrane stranice",
            width="stretch",
        )

    total_pages = int(prepared.get("total_pages") or 0)
    selected_indices = [int(index) for index in prepared.get("analyzed_page_indices", [])]
    if total_pages:
        page_col, analyzed_col = st.columns(2)
        page_col.metric("Ukupan broj stranica", total_pages)
        analyzed_col.metric("Analizirano stranica", len(selected_indices))
        selected_display = ", ".join(str(index + 1) for index in selected_indices)
        st.caption(f"Odabrane stranice: {selected_display}")
        layout_indices = [int(index) for index in prepared.get("layout_page_indices", [])]
        if len(layout_indices) != len(selected_indices):
            st.caption(
                f"LayoutLMv3 valjane OCR stranice: {len(layout_indices)} / {len(selected_indices)}"
            )

    text_path = prepared.get("text_path")
    if text_path and Path(text_path).is_file():
        text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
        with st.expander("Preview pripremljenog teksta", expanded=False):
            st.text(text[:1500])

    preparation_rows = []
    for model_key in model_keys:
        ready, reason = prepared_model_status(prepared, model_key)
        preparation_rows.append(
            {
                "Model": PREDICTION_MODEL_LABELS.get(model_key, model_key),
                "Status pripreme": "Spremno" if ready else "Nepodržano / nije moguće pripremiti",
                "Razlog": reason,
            }
        )
    st.dataframe(pd.DataFrame(preparation_rows), hide_index=True, width="stretch")

    if prepared.get("errors"):
        with st.expander("Detalji pripreme dokumenta", expanded=False):
            for message in prepared["errors"]:
                st.write(f"- {message}")


def show_live_prediction_results(outcomes):
    summary_rows = []
    for outcome in outcomes:
        result = outcome.get("result") or {}
        recognized = (
            is_recognized_prediction(result)
            if outcome["status"] == "Uspješno"
            else None
        )
        summary_rows.append(
            {
                "Model": outcome["model"],
                "Status": outcome["status"],
                "Predikcija": result.get("predicted_class", "—"),
                "Prepoznato": recognized if recognized is not None else "—",
                "Sigurnost": (
                    f"{safe_float(result.get('confidence')) * 100:.2f}%"
                    if result
                    else "—"
                ),
                "Vrijeme": (
                    f"{safe_float(result.get('prediction_time_seconds')):.4f} s"
                    if result
                    else "—"
                ),
                "Analizirano": (
                    f"{result.get('pages_analyzed')} stranica"
                    if result.get("pages_analyzed") is not None
                    else (
                        f"{result.get('chunks_analyzed')} chunkova"
                        if result.get("chunks_analyzed") is not None
                        else "-"
                    )
                ),
                "Razlog": outcome.get("reason", ""),
            }
        )

    st.subheader("Rezultati live predikcije")
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    for outcome in outcomes:
        result = outcome.get("result") or {}
        page_rows = result.get("page_predictions") or []
        chunk_rows = result.get("chunk_predictions") or []
        if not page_rows and not chunk_rows:
            continue
        with st.expander(f"Detalji: {outcome['model']}", expanded=False):
            if page_rows:
                detail = pd.DataFrame(page_rows).rename(
                    columns={
                        "page_index": "Stranica (indeks)",
                        "predicted_class": "Predikcija",
                        "confidence": "Sigurnost",
                        "ocr_word_count": "OCR rijeci",
                    }
                )
                detail["Stranica"] = detail["Stranica (indeks)"] + 1
                detail["Sigurnost"] = detail["Sigurnost"].map(
                    lambda value: f"{safe_float(value) * 100:.2f}%"
                )
                visible = [
                    column
                    for column in ["Stranica", "Predikcija", "Sigurnost", "OCR rijeci"]
                    if column in detail.columns
                ]
                st.dataframe(detail[visible], hide_index=True, width="stretch")
            if chunk_rows:
                detail = pd.DataFrame(chunk_rows).rename(
                    columns={
                        "chunk_index": "Chunk (indeks)",
                        "predicted_class": "Predikcija",
                        "confidence": "Sigurnost",
                        "token_count": "Tokena",
                    }
                )
                detail["Sigurnost"] = detail["Sigurnost"].map(
                    lambda value: f"{safe_float(value) * 100:.2f}%"
                )
                st.write(
                    f"Ukupno chunkova: {result.get('total_chunks', len(chunk_rows))}; "
                    f"analizirano: {result.get('chunks_analyzed', len(chunk_rows))}."
                )
                st.dataframe(detail, hide_index=True, width="stretch")

    successful = [outcome for outcome in outcomes if outcome["status"] == "Uspješno"]
    for outcome in outcomes:
        if outcome["status"] == "Preskočeno":
            st.warning(f"{outcome['model']}: Preskočeno — {outcome['reason']}")
        elif outcome["status"] == "FAIL":
            st.error(f"{outcome['model']}: FAIL — predikcija nije uspjela.")

    if successful:
        probability_rows = []
        probability_maps = {
            outcome["model"]: outcome_probability_dict(outcome)
            for outcome in successful
        }
        for label in CLASS_NAMES:
            row = {"Klasa": label}
            row.update(
                {
                    model_label: probabilities.get(label, 0.0)
                    for model_label, probabilities in probability_maps.items()
                }
            )
            probability_rows.append(row)

        probability_df = pd.DataFrame(probability_rows).set_index("Klasa")
        st.subheader("Usporedba vjerojatnosti")
        st.dataframe(
            (probability_df * 100).map(lambda value: f"{value:.2f}%"),
            width="stretch",
        )
        st.bar_chart(probability_df * 100)

    failed_debug = [outcome for outcome in outcomes if outcome.get("debug")]
    if failed_debug:
        with st.expander("Debug detalji", expanded=False):
            for outcome in failed_debug:
                st.subheader(outcome["model"])
                st.code(outcome["debug"], language="text")


def show_live_prediction(prepared, model_keys):
    show_prepared_document_preview(prepared, model_keys)
    if not st.button("Klasificiraj dokument", type="primary"):
        return

    outcomes = []
    for model_key in model_keys:
        model_label = PREDICTION_MODEL_LABELS.get(model_key, model_key)
        with st.spinner(f"Pokrećem {model_label}..."):
            outcomes.append(run_prepared_model_prediction(prepared, model_key))
    show_live_prediction_results(outcomes)


def load_metrics(results_dir):
    metrics_path = results_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_percent(value):
    return f"{safe_float(value) * 100:.2f}%"


def format_seconds(value, digits=4):
    return f"{safe_float(value):.{digits}f} s"


def load_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def style_best_row(df, best_index):
    def apply_style(row):
        if row.name == best_index:
            return ["background-color: #e8f5e9; font-weight: 600"] * len(row)
        return [""] * len(row)

    return df.style.apply(apply_style, axis=1)


def show_metric_cards(metrics):
    col_acc, col_precision, col_recall, col_f1, col_total_time, col_doc_time = st.columns(6)
    col_acc.metric("Accuracy", format_percent(metrics.get("accuracy", 0.0)))
    col_precision.metric("Macro precision", format_percent(metrics.get("macro_precision", 0.0)))
    col_recall.metric("Macro recall", format_percent(metrics.get("macro_recall", 0.0)))
    col_f1.metric("Macro F1", format_percent(metrics.get("macro_f1", 0.0)))
    col_total_time.metric("Ukupno vrijeme", format_seconds(metrics.get("prediction_time_seconds", 0.0)))
    col_doc_time.metric("Vrijeme/doc", format_seconds(metrics.get("seconds_per_document", 0.0), digits=4))


def internal_metrics_table():
    rows = []
    for model_name, results_dir in RESULT_DIRS.items():
        metrics = load_metrics(results_dir)
        if metrics is None:
            rows.append(
                {
                    "Model": model_name,
                    "Accuracy": "Nedostaje",
                    "Macro precision": "Nedostaje",
                    "Macro recall": "Nedostaje",
                    "Macro F1": "Nedostaje",
                    "Vrijeme po dokumentu": "Nedostaje",
                }
            )
            continue

        rows.append(
            {
                "Model": model_name,
                "Accuracy": format_percent(metrics.get("accuracy", 0.0)),
                "Macro precision": format_percent(metrics.get("macro_precision", 0.0)),
                "Macro recall": format_percent(metrics.get("macro_recall", 0.0)),
                "Macro F1": format_percent(metrics.get("macro_f1", 0.0)),
                "Vrijeme po dokumentu": format_seconds(metrics.get("seconds_per_document", 0.0), digits=4),
            }
        )
    return pd.DataFrame(rows)


def show_internal_test_tab():
    st.dataframe(internal_metrics_table(), hide_index=True, width="stretch")

    selected_model = st.selectbox("Detalji modela", list(RESULT_DIRS), key="internal_model")
    results_dir = RESULT_DIRS[selected_model]

    if selected_model == "XLM-RoBERTa":
        st.caption(
            "Provjera splitova pronašla je 22 vrlo slična para dokumenata između splitova, "
            "pa rezultate treba interpretirati uz oprez."
        )
    elif selected_model == "LayoutLMv3":
        st.warning(
            "LayoutLMv3 ostvario je 100% na internom testnom skupu, ali provjera je pronašla "
            "velik broj vizualno vrlo sličnih dokumenata i mogući source/template bias. "
            "Rezultat treba potvrditi na dokumentima iz drugih izvora."
        )

    metrics = load_metrics(results_dir)
    if metrics is None:
        st.warning(f"Nedostaje {results_dir / 'test_metrics.json'}")
        return

    show_metric_cards(metrics)

    per_class = metrics.get("per_class", {})
    if per_class:
        rows = []
        for label, values in per_class.items():
            rows.append(
                {
                    "Klasa": label,
                    "Precision": format_percent(values.get("precision", 0.0)),
                    "Recall": format_percent(values.get("recall", 0.0)),
                    "F1": format_percent(values.get("f1", 0.0)),
                    "Support": int(safe_float(values.get("support", 0))),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    report_path = results_dir / "classification_report.txt"
    if report_path.exists():
        with st.expander("Classification report", expanded=False):
            st.text(report_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Nedostaje {report_path}")


def show_external_test_tab():
    comparison_path = EXTERNAL_RESULTS_DIR / "comparison_metrics.csv"
    df = load_csv(comparison_path)
    if df is None:
        st.warning(f"Nedostaje {comparison_path}")
        return

    for column in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "seconds_per_document",
        "documents_processed",
        "documents_failed",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    best_index = df["macro_f1"].idxmax() if "macro_f1" in df.columns and not df.empty else None
    display_df = pd.DataFrame(
        {
            "Model": df.get("model", ""),
            "Accuracy": df["accuracy"].map(format_percent),
            "Macro precision": df["macro_precision"].map(format_percent),
            "Macro recall": df["macro_recall"].map(format_percent),
            "Macro F1": df["macro_f1"].map(format_percent),
            "Vrijeme po dokumentu": df["seconds_per_document"].map(lambda value: format_seconds(value, digits=4)),
            "Obrađeno dokumenata": df["documents_processed"].astype(int),
            "Neuspjelih dokumenata": df["documents_failed"].astype(int),
            "Oznaka": ["Najbolji vanjski macro F1" if index == best_index else "" for index in df.index],
        }
    )
    st.dataframe(display_df, hide_index=True, width="stretch")

    chart_df = df.set_index("model")[["accuracy", "macro_f1"]] * 100
    chart_df = chart_df.rename(columns={"accuracy": "Accuracy", "macro_f1": "Macro F1"})
    st.bar_chart(chart_df)

    time_df = df.set_index("model")[["seconds_per_document"]]
    st.caption("Prosječno vrijeme predikcije po modelu")
    st.bar_chart(time_df)

    st.info(
        "Vanjski test sadrži 25 dokumenata, odnosno 5 dokumenata po klasi. "
        "Zbog malog broja primjera rezultate treba promatrati kao dodatnu provjeru generalizacije."
    )


def confusion_csv_as_frame(path):
    df = pd.read_csv(path)
    first_column = df.columns[0]
    return df.set_index(first_column)


def show_confusion_matrix_from_dir(results_dir, model_name, test_name):
    png_path = results_dir / "confusion_matrix.png"
    csv_path = results_dir / "confusion_matrix.csv"

    if png_path.exists():
        st.image(str(png_path), caption=f"{model_name} - {test_name}", width="stretch")
    elif csv_path.exists():
        matrix_df = confusion_csv_as_frame(csv_path)
        st.dataframe(matrix_df.style.background_gradient(cmap="Blues"), width="stretch")
    else:
        st.warning(f"Nedostaje confusion matrix PNG/CSV u {results_dir}")


def show_confusion_matrices_tab():
    test_set = st.selectbox("Odaberi test", ["Interni test", "Vanjski test"], key="confusion_test_set")
    model_name = st.selectbox("Odaberi model", list(RESULT_DIRS), key="confusion_model")
    results_dir = RESULT_DIRS[model_name] if test_set == "Interni test" else EXTERNAL_RESULT_DIRS[model_name]
    show_confusion_matrix_from_dir(results_dir, model_name, test_set)


def external_prediction_error(row):
    if not row["_processing_success"]:
        message = row.get("error_message", "")
        return "Nije obrađeno" if pd.isna(message) or not str(message).strip() else str(message)
    if row["_is_correct"]:
        return ""

    predicted_label = row.get("predicted_label", "")
    if pd.isna(predicted_label) or not str(predicted_label).strip():
        predicted_label = "bez predikcije"
    return f"{row['true_label']} → {predicted_label}"


def show_external_predictions_tab():
    predictions_path = EXTERNAL_RESULTS_DIR / "all_predictions.csv"
    df = load_csv(predictions_path)
    if df is None:
        st.warning(f"Nedostaje {predictions_path}")
        return

    for column in ["confidence", "prediction_time_seconds"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Model"] = df["model"].map(PREDICTION_MODEL_LABELS).fillna(df["model"])
    df["Dokument"] = df["document_path"].astype(str)
    df["Stvarna klasa"] = df["true_label"].astype(str)
    df["Predviđena klasa"] = df["predicted_label"].fillna("").astype(str)
    df["Status"] = df["status"].fillna("").astype(str)
    df["_processing_success"] = df["status"].eq("success")
    df["_is_correct"] = (
        df["_processing_success"]
        & df["predicted_label"].fillna("").astype(str).eq(df["true_label"].astype(str))
    )
    df["_prediction_error"] = df.apply(external_prediction_error, axis=1)

    model_options = ["Svi", *sorted(df["Model"].dropna().unique())]
    label_options = ["Sve", *sorted(df["Stvarna klasa"].dropna().unique())]
    selected_model = st.selectbox("Model", model_options, key="pred_model")
    selected_label = st.selectbox("Stvarna klasa", label_options, key="pred_label")
    only_wrong = st.checkbox("Samo pogrešne predikcije", value=False)
    only_success = st.checkbox("Samo uspješno obrađeni dokumenti", value=True)

    filtered = df.copy()
    if selected_model != "Svi":
        filtered = filtered[filtered["Model"] == selected_model]
    if selected_label != "Sve":
        filtered = filtered[filtered["Stvarna klasa"] == selected_label]
    if only_success:
        filtered = filtered[filtered["_processing_success"]]
    if only_wrong:
        filtered = filtered[~filtered["_is_correct"]]

    display_df = pd.DataFrame(
        {
            "Model": filtered["Model"],
            "Dokument": filtered["Dokument"],
            "Stvarna klasa": filtered["Stvarna klasa"],
            "Predviđena klasa": filtered["Predviđena klasa"],
            "Status obrade": filtered["_processing_success"].map(
                {True: "Obrađeno", False: "FAIL"}
            ),
            "Točnost": filtered["_is_correct"],
            "Greška": filtered["_prediction_error"],
            "Confidence": filtered["confidence"].map(
                lambda value: "" if pd.isna(value) else format_percent(value)
            ),
            "Vrijeme predikcije": filtered["prediction_time_seconds"].map(
                lambda value: "" if pd.isna(value) else format_seconds(value, digits=4)
            ),
        }
    )
    st.dataframe(display_df, hide_index=True, width="stretch")


def final_comparison_summary(df):
    best_internal = df.loc[df["internal_macro_f1"].idxmax()]
    best_external = df.loc[df["external_macro_f1"].idxmax()]
    smallest_drop = df.loc[df["macro_f1_drop"].idxmin()]
    largest_drops = df.sort_values("macro_f1_drop", ascending=False)["model"].head(2).tolist()

    drop_text = " i ".join(largest_drops)
    return (
        f"Na internom testnom skupu najbolji rezultat ostvario je {best_internal['model']}. "
        f"Na vanjskom testnom skupu najbolji rezultat ostvario je {best_external['model']}. "
        f"Najmanji pad macro F1 rezultata ima {smallest_drop['model']}. "
        f"Razlika između internog i vanjskog testa pokazuje da su {drop_text} osjetljiviji "
        "na promjenu izgleda i izvora dokumenata."
    )


def show_final_comparison_tab():
    df = load_csv(FINAL_COMPARISON_PATH)
    if df is None:
        st.warning("Prvo pokrenite: python scripts/create_final_comparison.py")
        return

    numeric_columns = [
        "internal_accuracy",
        "external_accuracy",
        "accuracy_drop",
        "internal_macro_f1",
        "external_macro_f1",
        "macro_f1_drop",
        "internal_seconds_per_document",
        "external_seconds_per_document",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    best_internal_index = df["internal_macro_f1"].idxmax()
    best_external_index = df["external_macro_f1"].idxmax()
    smallest_drop_index = df["macro_f1_drop"].idxmin()
    tags = []
    for index, _ in df.iterrows():
        row_tags = []
        if index == best_internal_index:
            row_tags.append("Najbolji interni")
        if index == best_external_index:
            row_tags.append("Najbolji vanjski")
        if index == smallest_drop_index:
            row_tags.append("Najmanji pad")
        tags.append(", ".join(row_tags))

    display_df = pd.DataFrame(
        {
            "Model": df["model"],
            "Interni accuracy": df["internal_accuracy"].map(format_percent),
            "Vanjski accuracy": df["external_accuracy"].map(format_percent),
            "Pad accuracy": df["accuracy_drop"].map(format_percent),
            "Interni macro F1": df["internal_macro_f1"].map(format_percent),
            "Vanjski macro F1": df["external_macro_f1"].map(format_percent),
            "Pad macro F1": df["macro_f1_drop"].map(format_percent),
            "Interno vrijeme po dokumentu": df["internal_seconds_per_document"].map(
                lambda value: format_seconds(value, digits=4)
            ),
            "Vanjsko vrijeme po dokumentu": df["external_seconds_per_document"].map(
                lambda value: format_seconds(value, digits=4)
            ),
            "Oznake": tags,
        }
    )
    st.dataframe(display_df, hide_index=True, width="stretch")

    chart_df = df.set_index("model")[["internal_macro_f1", "external_macro_f1"]] * 100
    chart_df = chart_df.rename(
        columns={
            "internal_macro_f1": "Interni macro F1",
            "external_macro_f1": "Vanjski macro F1",
        }
    )
    st.bar_chart(chart_df)

    if FINAL_COMPARISON_CHART_PATH.exists():
        with st.expander("Graf accuracy i macro F1", expanded=False):
            st.image(str(FINAL_COMPARISON_CHART_PATH), width="stretch")

    st.write(final_comparison_summary(df))
    st.markdown(
        """
- Interni test koristi 150 dokumenata iz istih izvora kao skup za treniranje.
- Vanjski test koristi 25 novih dokumenata iz drugih izvora.
- Kod XLM-RoBERTa pronađen je manji broj tekstualno vrlo sličnih dokumenata između internih splitova.
- Kod LayoutLMv3 pronađen je jak template/source bias u internom skupu.
- Zbog samo 5 vanjskih dokumenata po klasi vanjski rezultat ima veću statističku nesigurnost.
"""
    )


def show_results_dashboard():
    st.header("Evaluacija modela")
    st.write(
        "Interni test koristi dokumente iz istih izvora kao trening skup, dok vanjski test koristi "
        "nove dokumente koji nisu korišteni u treniranju."
    )
    tabs = st.tabs(
        [
            "Interni test",
            "Vanjski test",
            "Confusion matrice",
            "Pojedinačne predikcije vanjskog testa",
        ]
    )
    with tabs[0]:
        show_internal_test_tab()
    with tabs[1]:
        show_external_test_tab()
    with tabs[2]:
        show_confusion_matrices_tab()
    with tabs[3]:
        show_external_predictions_tab()

    st.info(
        "Interni test prikazuje rezultate na dokumentima iz istih izvora kao trening skup. "
        "Vanjski test prikazuje rezultate na novim dokumentima iz drugih izvora. Velika razlika "
        "između internih i vanjskih rezultata pokazuje da modeli mogu naučiti obilježja izvora "
        "i predloška dokumenta, a ne samo stvarnu semantičku klasu dokumenta."
    )


def main():
    st.set_page_config(page_title="Document AI Classifier", layout="wide")
    st.title("Document AI Classifier")

    show_model_status_sidebar()
    st.header("Predikcija jednog dokumenta")
    st.write(
        "Ovdje se učitava jedan dokument i prikazuje što svaki model predviđa za taj konkretni dokument."
    )

    selected_mode = st.selectbox("Model", MODEL_OPTIONS)
    uploaded_file = st.file_uploader(
        "Dokument",
        type=ALL_UPLOAD_TYPES,
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        required_models = MODEL_KEYS_BY_OPTION.get(selected_mode, [])
        with tempfile.TemporaryDirectory(prefix="document_adapter_") as temporary_dir:
            try:
                with st.spinner("Pripremam dokument za odabrane modele..."):
                    prepared = prepare_document_for_models(
                        uploaded_file,
                        Path(temporary_dir),
                    )
            except Exception:
                st.warning(
                    "Dokument nije moguće pripremiti za predikciju. Provjerite format "
                    "i sadržaj dokumenta."
                )
                with st.expander("Debug detalji", expanded=False):
                    st.code(traceback.format_exc(), language="text")
            else:
                show_live_prediction(prepared, required_models)

    st.divider()
    show_results_dashboard()


if __name__ == "__main__":
    main()

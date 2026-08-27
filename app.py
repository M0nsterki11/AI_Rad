import json
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict_resnet import (  # noqa: E402
    load_document_image as load_resnet_document_image,
    load_model as load_resnet_model,
    predict_file as predict_resnet_file,
)
from src.predict_text_model import (  # noqa: E402
    extract_text_from_file,
    load_text_model,
    predict_file as predict_text_file,
    predict_text,
)
from src.predict_layoutlm import (  # noqa: E402
    load_image_and_ocr,
    load_layoutlm_model,
    predict_layoutlm,
)
from src.model_downloader import (  # noqa: E402
    ensure_model_available,
    is_model_available,
)
from src.document_conversion import (  # noqa: E402
    DocumentConversionError,
    convert_docx_to_pdf,
    convert_pdf_first_page_to_image,
)
from src.document_adapter import prepare_document_for_models  # noqa: E402
from src.preprocess import OCRProcessingError  # noqa: E402


RESNET_RESULTS_DIR = PROJECT_ROOT / "results" / "resnet50"
XLM_RESULTS_DIR = PROJECT_ROOT / "results" / "xlm_roberta"
LAYOUT_RESULTS_DIR = PROJECT_ROOT / "results" / "layoutlmv3"
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
DOCX_CONVERSION_WARNING = (
    "Automatska konverzija DOCX-a nije uspjela. DOCX nije moguće pretvoriti "
    "u sliku na ovom serveru. Spremite dokument kao PDF i pokušajte ponovno."
)
ALL_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "txt", "docx"]
ADAPTABLE_DOCUMENT_EXTENSIONS = {f".{extension}" for extension in ALL_UPLOAD_TYPES}
MODEL_SUPPORTED_EXTENSIONS = {
    "resnet50": ADAPTABLE_DOCUMENT_EXTENSIONS,
    "xlm_roberta": ADAPTABLE_DOCUMENT_EXTENSIONS,
    "layoutlmv3": ADAPTABLE_DOCUMENT_EXTENSIONS,
}

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


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


@contextmanager
def prepare_resnet_input(document_path, model_keys):
    document_path = Path(document_path)
    if "resnet50" not in model_keys or document_path.suffix.lower() != ".docx":
        yield document_path
        return

    st.info(
        "ResNet50 je vizualni model. DOCX se automatski pokušava pretvoriti "
        "u PDF/sliku prije predikcije."
    )
    with tempfile.TemporaryDirectory(prefix="resnet_docx_") as temporary_dir:
        output_dir = Path(temporary_dir)
        with st.spinner("Pretvaram DOCX u sliku za vizualni model..."):
            pdf_path = convert_docx_to_pdf(document_path, output_dir)
            image_path = convert_pdf_first_page_to_image(pdf_path, output_dir)
        yield image_path


def show_model_status_sidebar():
    st.sidebar.subheader("Status modela")
    for model_key, model_label in PREDICTION_MODEL_LABELS.items():
        status = "dostupno" if is_model_available(model_key) else "nedostaje"
        st.sidebar.write(f"{model_label}: {status}")


def ensure_models_for_prediction(model_keys):
    for model_key in model_keys:
        if is_model_available(model_key):
            continue

        model_label = PREDICTION_MODEL_LABELS.get(model_key, model_key)
        with st.spinner("Preuzimam model..."):
            available, message = ensure_model_available(model_key)

        if available:
            continue

        if "Hugging Face repo ID" in message:
            st.warning(message)
        else:
            st.error(f"{model_label}: {message}")
        return False

    return True


def resnet_probability_frame(probabilities):
    rows = [
        {
            "klasa": label,
            "vjerojatnost": probability,
            "postotak": f"{probability * 100:.2f}%",
        }
        for label, probability in probabilities.items()
    ]
    rows.sort(key=lambda row: row["vjerojatnost"], reverse=True)
    return pd.DataFrame(rows)


def text_probability_frame(probabilities):
    return pd.DataFrame(
        [
            {
                "klasa": row["class"],
                "vjerojatnost": row["probability"],
                "postotak": f"{row['probability'] * 100:.2f}%",
            }
            for row in probabilities
        ]
    )


def ranked_probability_frame(probabilities):
    return text_probability_frame(probabilities)


def probability_dict_from_ranked(probabilities):
    return {row["class"]: row["probability"] for row in probabilities}


def show_probability_outputs(probability_df):
    st.dataframe(probability_df, hide_index=True, use_container_width=True)
    chart_df = probability_df.set_index("klasa")[["vjerojatnost"]]
    st.bar_chart(chart_df)


def is_recognized_prediction(result):
    confidence = float(result.get("confidence", 0.0) or 0.0)
    return confidence >= PREDICTION_CONFIDENCE_THRESHOLD


def show_prediction_status(result):
    recognized = is_recognized_prediction(result)
    if recognized:
        st.info(
            "Prepoznato: True. Model je iznad minimalnog praga sigurnosti "
            f"od {PREDICTION_CONFIDENCE_THRESHOLD * 100:.0f}%."
        )
    else:
        st.warning(
            "Predikcija je završena, ali model nije dovoljno siguran za pouzdanu "
            f"klasifikaciju (prag {PREDICTION_CONFIDENCE_THRESHOLD * 100:.0f}%)."
        )
    return recognized


def display_class(result, recognized):
    return result["predicted_class"] if recognized else "nepoznato"


def split_compatible_models(model_keys, extension):
    compatible = [
        model_key
        for model_key in model_keys
        if extension in MODEL_SUPPORTED_EXTENSIONS.get(model_key, set())
    ]
    incompatible = [model_key for model_key in model_keys if model_key not in compatible]
    return compatible, incompatible


def incompatible_model_reason(model_key, extension):
    if model_key == "resnet50" and extension == ".txt":
        return "ResNet50 je vizualni model i ne podržava TXT bez pretvaranja u sliku."
    return f"Model ne podržava {extension or 'ovu vrstu datoteke'} kao ulaz."


def show_incompatible_models(incompatible_model_keys, extension):
    if not incompatible_model_keys:
        return

    status_df = pd.DataFrame(
        [
            {
                "Model": PREDICTION_MODEL_LABELS.get(model_key, model_key),
                "Status obrade": "Nepodržano / nije moguće pripremiti",
                "Prepoznato": False,
                "Razlog": incompatible_model_reason(model_key, extension),
            }
            for model_key in incompatible_model_keys
        ]
    )
    if incompatible_model_keys == ["resnet50"] and extension == ".txt":
        st.warning(incompatible_model_reason("resnet50", extension))
    else:
        st.warning(
            "Dokument je učitan, ali ga svi odabrani modeli ne mogu obraditi. "
            "Tekstualne datoteke izravno podržava XLM-RoBERTa."
        )
    st.dataframe(status_df, hide_index=True, use_container_width=True)


def show_resnet_prediction(temp_path):
    model, class_names, device, _ = get_resnet_model()
    preview = load_resnet_document_image(temp_path)
    st.image(preview, caption="Prva stranica", use_container_width=True)

    if st.button("Klasificiraj dokument", type="primary"):
        result = predict_resnet_file(temp_path, model=model, class_names=class_names, device=device)
        probability_df = resnet_probability_frame(result["probabilities"])
        recognized = show_prediction_status(result)

        st.subheader("ResNet50 predikcija")
        col_status, col_class, col_confidence, col_time = st.columns(4)
        col_status.metric("Prepoznato", str(recognized))
        col_class.metric("Klasa", display_class(result, recognized))
        col_confidence.metric("Sigurnost", f"{result['confidence'] * 100:.2f}%")
        col_time.metric("Vrijeme", f"{result['prediction_time_seconds']:.4f} s")
        show_probability_outputs(probability_df)


def show_text_prediction(temp_path):
    model, tokenizer, class_names, _, device = get_text_model()
    text = extract_text_from_file(temp_path)

    with st.expander("Preview izdvojenog teksta", expanded=False):
        st.text(text[:1000])

    if st.button("Klasificiraj dokument", type="primary"):
        result = predict_text(
            text,
            model=model,
            tokenizer=tokenizer,
            class_names=class_names,
            device=device,
        )
        probability_df = text_probability_frame(result["probabilities"])
        recognized = show_prediction_status(result)

        st.subheader("XLM-RoBERTa predikcija")
        col_status, col_class, col_confidence, col_time = st.columns(4)
        col_status.metric("Prepoznato", str(recognized))
        col_class.metric("Klasa", display_class(result, recognized))
        col_confidence.metric("Sigurnost", f"{result['confidence'] * 100:.2f}%")
        col_time.metric("Vrijeme", f"{result['prediction_time_seconds']:.4f} s")
        show_probability_outputs(probability_df)


def show_layoutlm_prediction(temp_path):
    model, processor, class_names, _, device = get_layoutlm_model()
    image, words, boxes, ocr_text = load_image_and_ocr(temp_path)
    st.image(image, caption="Prva stranica", use_container_width=True)

    with st.expander("Preview OCR teksta", expanded=False):
        st.text(ocr_text[:1000])

    if st.button("Klasificiraj dokument", type="primary"):
        result = predict_layoutlm(
            image,
            words,
            boxes,
            model=model,
            processor=processor,
            class_names=class_names,
            device=device,
        )
        probability_df = ranked_probability_frame(result["probabilities"])
        recognized = show_prediction_status(result)

        st.subheader("LayoutLMv3 predikcija")
        col_status, col_class, col_confidence, col_time = st.columns(4)
        col_status.metric("Prepoznato", str(recognized))
        col_class.metric("Klasa", display_class(result, recognized))
        col_confidence.metric("Sigurnost", f"{result['confidence'] * 100:.2f}%")
        col_time.metric("Vrijeme", f"{result['prediction_time_seconds']:.4f} s")
        show_probability_outputs(probability_df)


def show_comparison(temp_path, resnet_input_path=None):
    resnet_input_path = Path(resnet_input_path or temp_path)
    resnet_model, resnet_classes, resnet_device, _ = get_resnet_model()
    text_model, tokenizer, text_classes, _, text_device = get_text_model()

    preview = load_resnet_document_image(resnet_input_path)
    st.image(preview, caption="Prva stranica za ResNet50", use_container_width=True)

    extracted_text = extract_text_from_file(temp_path)
    with st.expander("Preview izdvojenog teksta za XLM-RoBERTa", expanded=False):
        st.text(extracted_text[:1000])

    if st.button("Klasificiraj dokument", type="primary"):
        resnet_result = predict_resnet_file(
            resnet_input_path,
            model=resnet_model,
            class_names=resnet_classes,
            device=resnet_device,
        )
        text_result = predict_text(
            extracted_text,
            model=text_model,
            tokenizer=tokenizer,
            class_names=text_classes,
            device=text_device,
        )
        resnet_recognized = is_recognized_prediction(resnet_result)
        text_recognized = is_recognized_prediction(text_result)

        left, right = st.columns(2)
        with left:
            st.subheader("ResNet50")
            st.metric("Prepoznato", str(resnet_recognized))
            st.metric("Predikcija", display_class(resnet_result, resnet_recognized))
            st.metric("Sigurnost", f"{resnet_result['confidence'] * 100:.2f}%")
            st.metric("Vrijeme", f"{resnet_result['prediction_time_seconds']:.4f} s")

        with right:
            st.subheader("XLM-RoBERTa")
            st.metric("Prepoznato", str(text_recognized))
            st.metric("Predikcija", display_class(text_result, text_recognized))
            st.metric("Sigurnost", f"{text_result['confidence'] * 100:.2f}%")
            st.metric("Vrijeme", f"{text_result['prediction_time_seconds']:.4f} s")

        resnet_probs = resnet_result["probabilities"]
        text_probs = probability_dict_from_ranked(text_result["probabilities"])
        labels = resnet_classes if resnet_classes == text_classes else sorted(set(resnet_probs) | set(text_probs))
        comparison_df = pd.DataFrame(
            [
                {
                    "Klasa": label,
                    "ResNet50": resnet_probs.get(label, 0.0),
                    "XLM-RoBERTa": text_probs.get(label, 0.0),
                }
                for label in labels
            ]
        )
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)

        if not (resnet_recognized and text_recognized):
            st.warning("Najmanje jedan model nije dovoljno siguran: Prepoznato = False.")
        elif resnet_result["predicted_class"] != text_result["predicted_class"]:
            st.info("Modeli se ne slažu u predikciji.")


def show_all_models_comparison(temp_path, resnet_input_path=None):
    resnet_input_path = Path(resnet_input_path or temp_path)
    resnet_model, resnet_classes, resnet_device, _ = get_resnet_model()
    text_model, tokenizer, text_classes, _, text_device = get_text_model()
    layout_model, layout_processor, layout_classes, _, layout_device = get_layoutlm_model()

    image, words, boxes, ocr_text = load_image_and_ocr(temp_path)
    st.image(image, caption="Prva stranica", use_container_width=True)

    extracted_text = extract_text_from_file(temp_path)
    with st.expander("Preview izdvojenog teksta", expanded=False):
        st.text(extracted_text[:1000])
    with st.expander("Preview OCR teksta za LayoutLMv3", expanded=False):
        st.text(ocr_text[:1000])

    if st.button("Klasificiraj dokument", type="primary"):
        resnet_result = predict_resnet_file(
            resnet_input_path,
            model=resnet_model,
            class_names=resnet_classes,
            device=resnet_device,
        )
        text_result = predict_text(
            extracted_text,
            model=text_model,
            tokenizer=tokenizer,
            class_names=text_classes,
            device=text_device,
        )
        layout_result = predict_layoutlm(
            image,
            words,
            boxes,
            model=layout_model,
            processor=layout_processor,
            class_names=layout_classes,
            device=layout_device,
        )
        resnet_recognized = is_recognized_prediction(resnet_result)
        text_recognized = is_recognized_prediction(text_result)
        layout_recognized = is_recognized_prediction(layout_result)

        summary_df = pd.DataFrame(
            [
                {
                    "Model": "ResNet50",
                    "Predviđena klasa": display_class(resnet_result, resnet_recognized),
                    "Prepoznato": resnet_recognized,
                    "Sigurnost": f"{resnet_result['confidence'] * 100:.2f}%",
                    "Vrijeme predikcije": f"{resnet_result['prediction_time_seconds']:.4f} s",
                },
                {
                    "Model": "XLM-RoBERTa",
                    "Predviđena klasa": display_class(text_result, text_recognized),
                    "Prepoznato": text_recognized,
                    "Sigurnost": f"{text_result['confidence'] * 100:.2f}%",
                    "Vrijeme predikcije": f"{text_result['prediction_time_seconds']:.4f} s",
                },
                {
                    "Model": "LayoutLMv3",
                    "Predviđena klasa": display_class(layout_result, layout_recognized),
                    "Prepoznato": layout_recognized,
                    "Sigurnost": f"{layout_result['confidence'] * 100:.2f}%",
                    "Vrijeme predikcije": f"{layout_result['prediction_time_seconds']:.4f} s",
                },
            ]
        )
        st.subheader("Sažetak predikcije")
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        resnet_probs = resnet_result["probabilities"]
        text_probs = probability_dict_from_ranked(text_result["probabilities"])
        layout_probs = probability_dict_from_ranked(layout_result["probabilities"])

        comparison_df = pd.DataFrame(
            [
                {
                    "Klasa": label,
                    "ResNet50": format_percent(resnet_probs.get(label, 0.0)),
                    "XLM-RoBERTa": format_percent(text_probs.get(label, 0.0)),
                    "LayoutLMv3": format_percent(layout_probs.get(label, 0.0)),
                }
                for label in CLASS_NAMES
            ]
        )
        st.subheader("Vjerojatnosti po klasama")
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)

        chart_df = pd.DataFrame(
            [
                {
                    "Klasa": label,
                    "ResNet50": safe_float(resnet_probs.get(label, 0.0)) * 100,
                    "XLM-RoBERTa": safe_float(text_probs.get(label, 0.0)) * 100,
                    "LayoutLMv3": safe_float(layout_probs.get(label, 0.0)) * 100,
                }
                for label in CLASS_NAMES
            ]
        ).set_index("Klasa")
        st.bar_chart(chart_df)

        all_recognized = resnet_recognized and text_recognized and layout_recognized
        predicted_classes = {
            display_class(resnet_result, resnet_recognized),
            display_class(text_result, text_recognized),
            display_class(layout_result, layout_recognized),
        }
        if not all_recognized:
            st.warning("Najmanje jedan model nije dovoljno siguran: Prepoznato = False.")
        elif len(predicted_classes) > 1:
            st.warning("Modeli se ne slažu u predikciji za ovaj dokument.")
        else:
            st.info("Sva tri modela predviđaju istu klasu za ovaj dokument.")


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
            result = predict_resnet_file(
                prepared["image_path"],
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
            payload = json.loads(Path(prepared["ocr_path"]).read_text(encoding="utf-8"))
            with Image.open(prepared["image_path"]) as source_image:
                image = source_image.convert("RGB")
            model, processor, class_names, _, device = get_layoutlm_model()
            result = predict_layoutlm(
                image,
                payload.get("words", []),
                payload.get("boxes", []),
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
        st.image(str(image_path), caption="Pripremljeni vizualni input", use_container_width=True)

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
    st.dataframe(pd.DataFrame(preparation_rows), hide_index=True, use_container_width=True)

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
                "Razlog": outcome.get("reason", ""),
            }
        )

    st.subheader("Rezultati live predikcije")
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

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
            use_container_width=True,
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
    st.dataframe(internal_metrics_table(), hide_index=True, use_container_width=True)

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
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

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
    st.dataframe(display_df, hide_index=True, use_container_width=True)

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
        st.image(str(png_path), caption=f"{model_name} - {test_name}", use_container_width=True)
    elif csv_path.exists():
        matrix_df = confusion_csv_as_frame(csv_path)
        st.dataframe(matrix_df.style.background_gradient(cmap="Blues"), use_container_width=True)
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
    st.dataframe(display_df, hide_index=True, use_container_width=True)


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
    st.dataframe(display_df, hide_index=True, use_container_width=True)

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
            st.image(str(FINAL_COMPARISON_CHART_PATH), use_container_width=True)

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
            "Interni vs. vanjski test",
            "Confusion matrice",
            "Pojedinačne predikcije vanjskog testa",
        ]
    )
    with tabs[0]:
        show_internal_test_tab()
    with tabs[1]:
        show_external_test_tab()
    with tabs[2]:
        show_final_comparison_tab()
    with tabs[3]:
        show_confusion_matrices_tab()
    with tabs[4]:
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

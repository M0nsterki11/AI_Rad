import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch


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


def show_resnet_prediction(temp_path):
    model, class_names, device, _ = get_resnet_model()
    preview = load_resnet_document_image(temp_path)
    st.image(preview, caption="Prva stranica", use_container_width=True)

    if st.button("Klasificiraj dokument", type="primary"):
        result = predict_resnet_file(temp_path, model=model, class_names=class_names, device=device)
        probability_df = resnet_probability_frame(result["probabilities"])

        st.subheader("ResNet50 predikcija")
        col_class, col_confidence, col_time = st.columns(3)
        col_class.metric("Klasa", result["predicted_class"])
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

        st.subheader("XLM-RoBERTa predikcija")
        col_class, col_confidence, col_time = st.columns(3)
        col_class.metric("Klasa", result["predicted_class"])
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

        st.subheader("LayoutLMv3 predikcija")
        col_class, col_confidence, col_time = st.columns(3)
        col_class.metric("Klasa", result["predicted_class"])
        col_confidence.metric("Sigurnost", f"{result['confidence'] * 100:.2f}%")
        col_time.metric("Vrijeme", f"{result['prediction_time_seconds']:.4f} s")
        show_probability_outputs(probability_df)


def show_comparison(temp_path):
    resnet_model, resnet_classes, resnet_device, _ = get_resnet_model()
    text_model, tokenizer, text_classes, _, text_device = get_text_model()

    preview = load_resnet_document_image(temp_path)
    st.image(preview, caption="Prva stranica za ResNet50", use_container_width=True)

    extracted_text = extract_text_from_file(temp_path)
    with st.expander("Preview izdvojenog teksta za XLM-RoBERTa", expanded=False):
        st.text(extracted_text[:1000])

    if st.button("Klasificiraj dokument", type="primary"):
        resnet_result = predict_resnet_file(
            temp_path,
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

        left, right = st.columns(2)
        with left:
            st.subheader("ResNet50")
            st.metric("Predikcija", resnet_result["predicted_class"])
            st.metric("Sigurnost", f"{resnet_result['confidence'] * 100:.2f}%")
            st.metric("Vrijeme", f"{resnet_result['prediction_time_seconds']:.4f} s")

        with right:
            st.subheader("XLM-RoBERTa")
            st.metric("Predikcija", text_result["predicted_class"])
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

        if resnet_result["predicted_class"] != text_result["predicted_class"]:
            st.info("Modeli se ne slažu u predikciji.")


def show_all_models_comparison(temp_path):
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
            temp_path,
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

        summary_df = pd.DataFrame(
            [
                {
                    "Model": "ResNet50",
                    "Predviđena klasa": resnet_result["predicted_class"],
                    "Sigurnost": f"{resnet_result['confidence'] * 100:.2f}%",
                    "Vrijeme predikcije": f"{resnet_result['prediction_time_seconds']:.4f} s",
                },
                {
                    "Model": "XLM-RoBERTa",
                    "Predviđena klasa": text_result["predicted_class"],
                    "Sigurnost": f"{text_result['confidence'] * 100:.2f}%",
                    "Vrijeme predikcije": f"{text_result['prediction_time_seconds']:.4f} s",
                },
                {
                    "Model": "LayoutLMv3",
                    "Predviđena klasa": layout_result["predicted_class"],
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

        predicted_classes = {
            resnet_result["predicted_class"],
            text_result["predicted_class"],
            layout_result["predicted_class"],
        }
        if len(predicted_classes) > 1:
            st.warning("Modeli se ne slažu u predikciji za ovaj dokument.")
        else:
            st.success("Sva tri modela predviđaju istu klasu za ovaj dokument.")


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
        filtered = filtered[filtered["Status"] == "success"]
    if only_wrong:
        filtered = filtered[filtered["Stvarna klasa"] != filtered["Predviđena klasa"]]

    display_df = pd.DataFrame(
        {
            "Model": filtered["Model"],
            "Dokument": filtered["Dokument"],
            "Stvarna klasa": filtered["Stvarna klasa"],
            "Predviđena klasa": filtered["Predviđena klasa"],
            "Confidence": filtered["confidence"].map(
                lambda value: "" if pd.isna(value) else format_percent(value)
            ),
            "Vrijeme predikcije": filtered["prediction_time_seconds"].map(
                lambda value: "" if pd.isna(value) else format_seconds(value, digits=4)
            ),
            "Status": filtered["Status"],
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
    image_document_types = ["pdf", "png", "jpg", "jpeg"]
    text_document_types = ["pdf", "png", "jpg", "jpeg", "txt", "html", "htm", "docx"]
    upload_types = text_document_types if selected_mode == MODEL_OPTIONS[1] else image_document_types
    uploaded_file = st.file_uploader(
        "Dokument",
        type=upload_types,
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        temp_path = save_uploaded_file(uploaded_file)
        try:
            required_models = MODEL_KEYS_BY_OPTION.get(selected_mode, [])
            if ensure_models_for_prediction(required_models):
                if selected_mode == MODEL_OPTIONS[0]:
                    show_resnet_prediction(temp_path)
                elif selected_mode == MODEL_OPTIONS[1]:
                    show_text_prediction(temp_path)
                elif selected_mode == MODEL_OPTIONS[2]:
                    show_layoutlm_prediction(temp_path)
                elif selected_mode == MODEL_OPTIONS[3]:
                    show_comparison(temp_path)
                else:
                    show_all_models_comparison(temp_path)
        except Exception as error:
            st.error(str(error))
        finally:
            temp_path.unlink(missing_ok=True)

    st.divider()
    show_results_dashboard()


if __name__ == "__main__":
    main()

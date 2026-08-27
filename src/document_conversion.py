import shutil
import subprocess
from pathlib import Path

import fitz


LIBREOFFICE_TIMEOUT_SECONDS = 60
WINDOWS_LIBREOFFICE_PATHS = (
    Path(r"C:\Program Files\LibreOffice\program\soffice.com"),
    Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
    Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.com"),
    Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
)


class DocumentConversionError(RuntimeError):
    """Greška pri pripremi dokumenta za vizualni model."""


def find_libreoffice_executable():
    for command in ("libreoffice", "soffice"):
        executable = shutil.which(command)
        if executable:
            return Path(executable)

    for executable in WINDOWS_LIBREOFFICE_PATHS:
        if executable.exists():
            return executable

    return None


def convert_docx_to_pdf(docx_path: Path, output_dir: Path) -> Path:
    docx_path = Path(docx_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not docx_path.is_file() or docx_path.suffix.lower() != ".docx":
        raise DocumentConversionError(f"DOCX datoteka nije pronađena: {docx_path}")

    libreoffice = find_libreoffice_executable()
    if libreoffice is None:
        raise DocumentConversionError(
            "LibreOffice nije dostupan. Instalirajte LibreOffice ili spremite dokument kao PDF."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = output_dir / ".libreoffice_profile"
    command = [
        str(libreoffice),
        f"-env:UserInstallation={profile_dir.as_uri()}",
        "--headless",
        "--nologo",
        "--nodefault",
        "--nofirststartwizard",
        "--nolockcheck",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(docx_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=LIBREOFFICE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise DocumentConversionError(
            f"LibreOffice nije dovršio DOCX konverziju unutar {LIBREOFFICE_TIMEOUT_SECONDS} sekundi."
        ) from error
    except OSError as error:
        raise DocumentConversionError(
            f"LibreOffice nije moguće pokrenuti: {error}"
        ) from error

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "nepoznata LibreOffice greška").strip()
        raise DocumentConversionError(f"DOCX konverzija nije uspjela: {details}")

    expected_pdf = output_dir / f"{docx_path.stem}.pdf"
    if expected_pdf.is_file():
        return expected_pdf

    generated_pdfs = sorted(
        output_dir.glob("*.pdf"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if generated_pdfs:
        return generated_pdfs[0]

    raise DocumentConversionError(
        "LibreOffice je završio bez greške, ali generirani PDF nije pronađen."
    )


def convert_pdf_first_page_to_image(pdf_path: Path, output_dir: Path) -> Path:
    pdf_path = Path(pdf_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise DocumentConversionError(f"PDF datoteka nije pronađena: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}_first_page.png"

    try:
        document = fitz.open(str(pdf_path))
        try:
            if document.page_count < 1:
                raise DocumentConversionError("PDF nema nijednu stranicu za prikaz.")

            page = document.load_page(0)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            pixmap.save(str(output_path))
        finally:
            document.close()
    except DocumentConversionError:
        raise
    except Exception as error:
        raise DocumentConversionError(
            f"Prvu stranicu PDF-a nije moguće pretvoriti u sliku: {error}"
        ) from error

    if not output_path.is_file():
        raise DocumentConversionError("PNG prve stranice PDF-a nije generiran.")

    return output_path

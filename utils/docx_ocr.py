# utils/docx_ocr.py
from io import BytesIO
from typing import List
from PIL import Image

def extract_images_from_docx(docx_path_or_bytes) -> List[bytes]:
    """
    Returns list of image bytes from a DOCX.
    Works with docx bytes.
    """
    import zipfile

    if isinstance(docx_path_or_bytes, bytes):
        data = docx_path_or_bytes
        z = zipfile.ZipFile(BytesIO(data))
    else:
        z = zipfile.ZipFile(docx_path_or_bytes)

    image_bytes = []
    for name in z.namelist():
        if name.startswith("word/media/"):
            image_bytes.append(z.read(name))

    print(f"[DOCX_OCR] images found in docx: {len(image_bytes)}")
    return image_bytes

def ocr_images_easyocr(images: List[bytes]) -> str:
    import easyocr
    import numpy as np

    print(f"[DOCX_OCR] starting OCR on {len(images)} images")
    reader = easyocr.Reader(["en", "es"], gpu=False)
    lines = []

    for img_b in images:
        try:
            img = Image.open(BytesIO(img_b)).convert("RGB")

            # Upscale for better OCR on small tables
            w, h = img.size
            img = img.resize((w * 2, h * 2))

            img_np = np.array(img)  # EasyOCR wants numpy array / bytes / path
            text_lines = reader.readtext(img_np, detail=0, paragraph=True)

            if text_lines:
                if isinstance(text_lines, list):
                    lines.append("\n".join([str(x) for x in text_lines]))
                else:
                    lines.append(str(text_lines))

        except Exception as e:
            print(f"[DOCX_OCR] OCR failed on one image: {e}")
            continue

    final = "\n\n".join(lines).strip()
    print(f"[DOCX_OCR] OCR chars extracted: {len(final)}")
    return final

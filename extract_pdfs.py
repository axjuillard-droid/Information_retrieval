from pypdf import PdfReader

def extract(pdf_path, txt_path):
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully extracted {pdf_path} to {txt_path}")
    except Exception as e:
        print(f"Failed to extract {pdf_path}: {e}")

extract("projet1_TripAdvisor2026.pdf", "projet1_TripAdvisor2026.txt")
extract("Information Retrieval Instruction Project1 2026.pdf", "info_retrieval.txt")

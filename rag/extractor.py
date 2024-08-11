import requests
import io
import pymupdf
from typing import List


def get_text_from_pdfs(pdf_urls: List[str]):
    document_text = []
    for url in pdf_urls:
        try:
            # Download PDF content
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Open PDF from memory
            pdf_file = io.BytesIO(response.content)
            document = pymupdf.open(stream=pdf_file, filetype="pdf")

            text = ""
            for page in document:
                text += page.get_text()
            document_text.append(text)

        except requests.RequestException as e:
            print(f"Error downloading PDF from {url}: {e}")
        except Exception as e:
            print(f"Error extracting text from PDF {url}: {e}")
        finally:
            if 'document' in locals():
                document.close()

    return document_text

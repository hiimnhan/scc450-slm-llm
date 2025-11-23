import os
import json
from pathlib import Path
from typing import Dict, List, Any
import fitz
import pdfplumber
from PIL import Image
import io


class PDFExtractor:
    """Extract text, images, tables, and form data from PDF files."""

    def __init__(self, source_folder: str, output_folder: str = "converted"):
        self.source_folder = Path(source_folder)
        self.output_folder = Path(output_folder)

    def extract_from_folder(self):
        """Process all PDF files in the source folder and subfolders."""
        if not self.source_folder.exists():
            print(f"Error: Source folder '{self.source_folder}' does not exist.")
            return

        # Find all PDF files recursively
        pdf_files = list(self.source_folder.rglob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in '{self.source_folder}'")
            return

        print(f"Found {len(pdf_files)} PDF file(s) to process.\n")

        for pdf_path in pdf_files:
            self.process_pdf(pdf_path)

    def process_pdf(self, pdf_path: Path):
        """Extract all content from a single PDF file."""
        print(f"Processing: {pdf_path}")

        # Calculate relative path to preserve folder structure
        relative_path = pdf_path.relative_to(self.source_folder)
        output_dir = self.output_folder / relative_path.parent / relative_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract text
            text_content = self.extract_text(pdf_path)
            if text_content:
                text_file = output_dir / "text.txt"
                text_file.write_text(text_content, encoding='utf-8')
                print(f"Extracted text to {text_file}")

            # Extract images
            image_count = self.extract_images(pdf_path, output_dir)
            if image_count > 0:
                print(f"Extracted {image_count} image(s)")

            # Extract tables
            tables = self.extract_tables(pdf_path)
            if tables:
                tables_file = output_dir / "tables.json"
                with open(tables_file, 'w', encoding='utf-8') as f:
                    json.dump(tables, f, indent=2, ensure_ascii=False)
                print(f"Extracted {len(tables)} table(s) to {tables_file}")

            # Extract form data
            form_data = self.extract_form_data(pdf_path)
            if form_data:
                form_file = output_dir / "form_data.json"
                with open(form_file, 'w', encoding='utf-8') as f:
                    json.dump(form_data, f, indent=2, ensure_ascii=False)
                print(f"Extracted form data to {form_file}")

            print(f"Completed: {output_dir}\n")

        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}\n")

    def extract_text(self, pdf_path: Path) -> str:
        """Extract text content from PDF."""
        text_content = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_content.append(f"=== Page {page_num} ===\n{text}\n")
        except Exception as e:
            print(f"Warning: Error extracting text: {str(e)}")

        return "\n".join(text_content)

    def extract_images(self, pdf_path: Path, output_dir: Path) -> int:
        """Extract images from PDF."""
        image_count = 0
        images_dir = output_dir / "images"

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Save image
                    images_dir.mkdir(parents=True, exist_ok=True)
                    image_filename = images_dir / f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"

                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)

                    image_count += 1

            doc.close()

        except Exception as e:
            print(f"Warning: Error extracting images: {str(e)}")

        return image_count

    def extract_tables(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables from PDF."""
        all_tables = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()

                    for table_index, table in enumerate(tables):
                        if table:
                            table_data = {
                                "page": page_num,
                                "table_index": table_index + 1,
                                "data": table
                            }
                            all_tables.append(table_data)

        except Exception as e:
            print(f"Warning: Error extracting tables: {str(e)}")

        return all_tables

    def extract_form_data(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract fillable form data from PDF."""
        form_data = {
            "has_form": False,
            "fields": []
        }

        try:
            doc = fitz.open(pdf_path)

            # Check if PDF has form fields
            for page_num in range(len(doc)):
                page = doc[page_num]
                widgets = page.widgets()

                if widgets:
                    form_data["has_form"] = True

                    for widget in widgets:
                        field_info = {
                            "page": page_num + 1,
                            "field_name": widget.field_name,
                            "field_type": widget.field_type_string,
                            "field_value": widget.field_value,
                            "field_label": widget.field_label if hasattr(widget, 'field_label') else None
                        }

                        # Get additional info for specific field types
                        if widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                            field_info["is_checked"] = widget.field_value
                        elif widget.field_type == fitz.PDF_WIDGET_TYPE_COMBOBOX:
                            field_info["choices"] = widget.choice_values if hasattr(widget, 'choice_values') else []

                        form_data["fields"].append(field_info)

            doc.close()

        except Exception as e:
            print(f"Warning: Error extracting form data: {str(e)}")

        return form_data if form_data["has_form"] else None


def main():
    """Main function to run the PDF extraction."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract text, images, tables, and form data from PDF files.')
    parser.add_argument('--input', '-i', help='Source folder containing PDF files')
    parser.add_argument('--output', '-o', default='converted', help='Output folder (default: converted)')

    args = parser.parse_args()

    extractor = PDFExtractor(args.input, args.output)
    extractor.extract_from_folder()

    print("Extraction complete!")


if __name__ == "__main__":
    main()

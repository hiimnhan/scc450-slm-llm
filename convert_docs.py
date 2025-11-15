from pathlib import Path
import time
import argparse

from loguru import logger
import torch
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling_core.types.doc import ImageRefMode


def get_accelerator_device() -> AcceleratorDevice:
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using CUDA accelerator.")
        return AcceleratorDevice.CUDA
    elif torch.backends.mps.is_available():
        logger.info("MPS is available. Using MPS accelerator.")
        return AcceleratorDevice.MPS
    else:
        logger.info("No GPU accelerator available. Using CPU.")
        return AcceleratorDevice.CPU


def convert_documents(
    input_folder: str | Path,
    output_folder: str | Path,
    formats: list[str] = ["md", "json"],
) -> None:
    input_folder = Path(input_folder).resolve()
    output_folder = Path(output_folder).resolve()

    pdf_files = list(input_folder.rglob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in the input folder: {input_folder}")
        return

    num_pdfs = len(pdf_files)

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=get_accelerator_device(),
        ),
        ocr_batch_size=4,
        layout_batch_size=64,
        table_batch_size=4,
    )
    pipeline_options.do_ocr = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    success_count = 0
    fail_count = 0
    partial_count = 0

    start_time = time.time()
    doc_converter.initialize_pipeline(InputFormat.PDF)
    init_runtime = time.time() - start_time
    logger.info(f"Pipeline initialized in {init_runtime:.2f} seconds.")

    start_time = time.time()
    logger.info("Starting document conversion...")
    for pdf_file in pdf_files:
        logger.debug(f"Converting document: {pdf_file}")
        rel_path = pdf_file.relative_to(input_folder)
        output_dir = (output_folder / rel_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = pdf_file.stem

        try:
            res = doc_converter.convert(pdf_file, raises_on_error=False)
            logger.debug(res.status)
            if res.status == ConversionStatus.SUCCESS:
                doc = res.document
                if "json" in formats:
                    path = output_dir / f"{base_name}.json"
                    doc.save_as_json(path, image_mode=ImageRefMode.PLACEHOLDER)
                if "md" in formats:
                    path = output_dir / f"{base_name}.md"
                    doc.save_as_markdown(path, image_mode=ImageRefMode.PLACEHOLDER)
                if "txt" in formats:
                    path = output_dir / f"{base_name}.txt"
                    doc.save_as_markdown(
                        path, image_mode=ImageRefMode.PLACEHOLDER, strict_text=True
                    )
                if "html" in formats:
                    path = output_dir / f"{base_name}.html"
                    doc.save_as_html(path, image_mode=ImageRefMode.EMBEDDED)
                if "doctags" in formats:
                    path = output_dir / f"{base_name}.doctags.txt"
                    doc.save_as_doctags(path)
                success_count += 1
            elif res.status == ConversionStatus.PARTIAL_SUCCESS:
                logger.warning(
                    f"Partial success converting {res.input.file} with the following errors: "
                )
                for err in res.errors:
                    logger.debug(f" - {err.error_message}")
            else:
                logger.error(f"Failed to convert {res.input.file} with errors:")
                for err in res.errors:
                    logger.debug(f" - {err.error_message}")
                fail_count += 1
        except Exception as e:
            logger.error(f"Exception occurred while converting {pdf_file}: {e}")

    runtime = time.time() - start_time
    logger.info(f"Document conversion completed in {runtime:.2f} seconds.")
    logger.info(f"Total documents processed: {num_pdfs}")
    logger.info(f" - Successful conversions: {success_count}")
    logger.info(f" - Partial conversions: {partial_count}")
    logger.info(f" - Failed conversions: {fail_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDF documents to other formats."
    )
    parser.add_argument(
        "--input_folder",
        "-i",
        type=str,
        required=True,
        help="Path to the input folder containing PDF documents.",
    )
    parser.add_argument(
        "--formats",
        "-f",
        nargs="+",
        choices=["md", "json", "txt", "doctags"],
        default=["md", "json"],
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        required=True,
        help="Path to the output folder for converted documents.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_folder = Path(args.input_folder).resolve()
    output_folder = Path(args.output_folder).resolve()

    convert_documents(input_folder, output_folder, args.formats)

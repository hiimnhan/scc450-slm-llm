import json
from pathlib import Path
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling.utils.profiling import ProfilingItem
from loguru import logger
import torch
import time

print(torch.__version__)

# def get_accelerator_device() -> AcceleratorDevice:
#     if torch.cuda.is_available():
#         print("Using CUDA accelerator")
#         return AcceleratorDevice.CUDA
#     elif torch.backends.mps.is_available():
#         print("Using MPS accelerator")
#         return AcceleratorDevice.MPS
#     else:
#         print("Using CPU accelerator")
#         return AcceleratorDevice.CPU
#
#
# get_accelerator_device()

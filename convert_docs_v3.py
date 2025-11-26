import os
import dotenv
import argparse

from unstructured_ingest.pipeline.pipeline import Pipeline
from unstructured_ingest.interfaces import ProcessorConfig
from unstructured_ingest.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
    LocalUploaderConfig
)
from unstructured_ingest.processes.partitioner import PartitionerConfig
from unstructured_client.models import shared

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
        "--strategy",
        "-f",
        default=shared.Strategy.VLM,
        choices=[shared.Strategy.VLM, shared.Strategy.HI_RES]
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        required=True,
        default="converted",
        help="Path to the output folder for converted documents.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    dotenv.load_dotenv('.env')
    args = parse_args()
    Pipeline.from_configs(
        context=ProcessorConfig(
            num_processes=4,
            tqdm=True,
            # verbose=True
        ),
        indexer_config=LocalIndexerConfig(
            # input_path="TraffordCouncilPlanningApplicationsWA14/cil/",
            # input_path="TraffordCouncilPlanningApplicationsWA14/TraffordCouncil/108246-HHA-22/CIL_QUESTIONS-1077881.pdf",
            input_path=args.input_folder,
            recursive=True
        ),
        downloader_config=LocalDownloaderConfig(),
        source_connection_config=LocalConnectionConfig(),
        partitioner_config=PartitionerConfig(
            partition_by_api=True,
            api_key=os.getenv("UNSTRUCTURED_API_KEY"),
            partition_endpoint=os.getenv("UNSTRUCTURED_API_URL"),
            flatten_metadata=False,
            strategy=args.strategy,
            additional_partition_args={
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 15,
                "vlm_model": "gpt-4o",
                "vlm_model_provider": "openai",
                "extract_image_block_types": ['Image', 'Table']
                
            },

        ),
        uploader_config=LocalUploaderConfig(
            output_dir="converted",
            preserve_directory_structure=True
        )
    ).run()
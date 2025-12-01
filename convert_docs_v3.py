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
        "-s",
        default=shared.Strategy.HI_RES,
        choices=[shared.Strategy.VLM, shared.Strategy.HI_RES, shared.Strategy.FAST]
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        default="converted",
        help="Path to the output folder for converted documents.",
    )
    parser.add_argument(
        "--num_processes",
        "-p",
        type=int,
        default=8,
        help="Number of parallel processes (default: 8).",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use Unstructured API for faster processing (requires API key).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    dotenv.load_dotenv('.env')
    args = parse_args()

    print(f"Running input: {args.input_folder}")
    print(f"Strategy: {args.strategy}")
    print(f"Processes: {args.num_processes}")
    print(f"API mode: {'enabled' if args.use_api else 'disabled'}\n")

    # Prepare partition config
    partition_config = {
        "flatten_metadata": False,
        "strategy": args.strategy,
        "additional_partition_args": {
            "extract_image_block_types": ['Image', 'Table'],
            "extract_image_block_to_payload": True,
        }
    }

    # Add API configuration if requested
    if args.use_api:
        api_key = os.getenv("UNSTRUCTURED_API_KEY")
        api_url = os.getenv("UNSTRUCTURED_API_URL")

        
        partition_config["partition_by_api"] = True
        partition_config["api_key"] = api_key   
        partition_config["partition_endpoint"] = api_url

    # Enable PDF page splitting for parallelization if using API or VLM strategy
    if args.use_api or args.strategy == shared.Strategy.VLM:
        partition_config["additional_partition_args"]["split_pdf_page"] = True
        partition_config["additional_partition_args"]["split_pdf_allow_failed"] = True
        partition_config["additional_partition_args"]["split_pdf_concurrency_level"] = min(args.num_processes * 2, 20)

    Pipeline.from_configs(
        context=ProcessorConfig(
            num_processes=args.num_processes,
            tqdm=True,
            # verbose=True
        ),
        indexer_config=LocalIndexerConfig(
            input_path=args.input_folder,
            recursive=True
        ),
        downloader_config=LocalDownloaderConfig(),
        source_connection_config=LocalConnectionConfig(),
        partitioner_config=PartitionerConfig(**partition_config),
        uploader_config=LocalUploaderConfig(
            output_dir=args.output_folder,
            preserve_directory_structure=True
        )
    ).run()
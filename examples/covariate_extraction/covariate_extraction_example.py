#!/usr/bin/env python

# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Example demonstrating how to use the skip_to_covariates workflow 
followed by the CovariatesOnly indexing method.
"""

import asyncio
import logging
import os
from pathlib import Path

import pandas as pd

from graphrag.api.index import create_pipeline_context, register_workflow_function
from graphrag.callbacks.progress_workflow_callbacks import ProgressWorkflowCallbacks
from graphrag.callbacks.workflow_callbacks_manager import WorkflowCallbacksManager
from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.workflows.factory import PipelineFactory
from graphrag.logger.rich_progress import RichProgressLogger
from graphrag.storage.file_storage import FileStorage
from graphrag.utils.storage import write_table_to_storage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def skip_to_covariates(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """
    Load data from a CSV file and prepare it for covariate extraction,
    skipping all previous workflow steps.
    """
    # Path to your data file
    data_path = Path(config.root_dir) / "my_data.csv"
    
    logger.info(f"Loading data from {data_path}")
    
    # Load your data
    df = pd.read_csv(data_path)
    
    # Create documents table (might be needed by downstream processes)
    documents = pd.DataFrame({
        "id": [f"doc_{i}" for i in range(len(df))],
        "title": df.get("title", [f"Document {i}" for i in range(len(df))]),
        "text": df["text"],  # Assuming your CSV has a 'text' column
        "metadata": [{} for _ in range(len(df))]
    })
    
    # Create text_units from the documents
    text_units = []
    for i, doc in documents.iterrows():
        text_units.append({
            "id": f"tu_{i}",
            "document_id": doc["id"],
            "text": doc["text"],
            "chunk_index": 0,
            "metadata": doc["metadata"]
        })
    
    text_units_df = pd.DataFrame(text_units)
    
    # Log what we're doing
    context.callbacks.log(f"Prepared {len(text_units_df)} text units for covariate extraction")
    
    # Write both tables to storage
    await write_table_to_storage(documents, "documents", context.storage)
    await write_table_to_storage(text_units_df, "text_units", context.storage)
    
    return WorkflowFunctionOutput(result={"documents": documents, "text_units": text_units_df})


async def main():
    """Run the example."""
    # Create a simple data file for demonstration
    example_dir = Path("examples/covariate_extraction")
    example_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a sample CSV file with some text data
    sample_data = pd.DataFrame({
        "title": ["Example 1", "Example 2", "Example 3"],
        "text": [
            "Company A was fined for bid rigging in public tenders published by Government Agency B in 2022.",
            "Company C and Company D formed an alliance to develop new technologies in 2021.",
            "John Smith was appointed as the CEO of Corporation E in January 2023."
        ]
    })
    data_path = example_dir / "my_data.csv"
    sample_data.to_csv(data_path, index=False)
    logger.info(f"Created sample data file at {data_path}")
    
    # Register our custom workflow
    register_workflow_function("skip_to_covariates", skip_to_covariates)
    
    # Create a configuration
    config = create_graphrag_config(
        root_dir=str(example_dir),
        extract_claims={"enabled": True}
    )
    
    # 1. First run: Load data and prepare text units
    logger.info("Step 1: Running skip_to_covariates workflow")
    callbacks = WorkflowCallbacksManager()
    callbacks.register(ProgressWorkflowCallbacks(RichProgressLogger()))
    
    storage = FileStorage(os.path.join(example_dir, "output"))
    pipeline_context = create_pipeline_context(callbacks, storage)
    
    # Create a pipeline with just our custom workflow
    pipeline = PipelineFactory.create_pipeline(
        config=config, 
        method=IndexingMethod.Standard  # Not used since we're providing custom workflows
    )
    pipeline.workflows = [("skip_to_covariates", PipelineFactory.workflows["skip_to_covariates"])]
    
    # Run the pipeline
    results = await pipeline.run(config, pipeline_context)
    logger.info(f"Completed skip_to_covariates with results: {results}")
    
    # 2. Second run: Extract covariates only
    logger.info("Step 2: Running covariate extraction using CovariatesOnly method")
    
    # Create a new pipeline for covariate extraction only
    covariate_pipeline = PipelineFactory.create_pipeline(
        config=config,
        method=IndexingMethod.CovariatesOnly
    )
    
    # Run the covariate extraction
    covariate_results = await covariate_pipeline.run(config, pipeline_context)
    logger.info(f"Completed covariate extraction with results: {covariate_results}")
    
    # Let's display what we extracted
    covariates = await storage.load_dataframe("covariates")
    logger.info(f"Extracted {len(covariates)} covariates:")
    logger.info(covariates[["subject_id", "object_id", "type", "description"]].head())


if __name__ == "__main__":
    asyncio.run(main()) 
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing covariate_extraction_only workflow definition."""

from typing import Any
from uuid import uuid4

import pandas as pd

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import COVARIATES_FINAL_COLUMNS
from graphrag.index.operations.extract_covariates.extract_covariates import (
    extract_covariates as extractor,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage, storage_has_table


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """
    Performs only covariate extraction, assuming all prerequisites are in place.
    This workflow:
    1. Loads text_units from storage
    2. Extracts covariates
    3. Saves covariates to storage
    
    All other processing steps are skipped.
    """
    # Verify text_units exist
    if not await storage_has_table("text_units", context.storage):
        error_msg = "Text units not found in storage. Make sure previous steps have been completed."
        context.callbacks.error(error_msg)
        raise ValueError(error_msg)
    
    # Load text_units 
    text_units = await load_table_from_storage("text_units", context.storage)
    
    # Log the start of extraction
    context.callbacks.log(f"Starting covariate extraction on {len(text_units)} text units")
    
    # Configure extraction settings
    extract_claims_llm_settings = config.get_language_model_config(
        config.extract_claims.model_id
    )
    extraction_strategy = config.extract_claims.resolved_strategy(
        config.root_dir, extract_claims_llm_settings
    )

    async_mode = extract_claims_llm_settings.async_mode
    num_threads = extract_claims_llm_settings.concurrent_requests

    # Extract covariates
    output = await extract_covariates(
        text_units,
        context.callbacks,
        context.cache,
        "claim",
        extraction_strategy,
        async_mode=async_mode,
        entity_types=None,
        num_threads=num_threads,
    )

    # Save to storage
    await write_table_to_storage(output, "covariates", context.storage)
    
    # Log completion
    context.callbacks.log(f"Covariate extraction completed. Extracted {len(output)} covariates.")

    return WorkflowFunctionOutput(result=output)


async def extract_covariates(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    covariate_type: str,
    extraction_strategy: dict[str, Any] | None,
    async_mode: AsyncType = AsyncType.AsyncIO,
    entity_types: list[str] | None = None,
    num_threads: int = 4,
) -> pd.DataFrame:
    """All the steps to extract and format covariates."""
    # Reassign the id because it will be overwritten in the output by a covariate one
    # This also results in text_unit_id being copied to the output covariate table
    text_units["text_unit_id"] = text_units["id"]
    covariates = await extractor(
        input=text_units,
        callbacks=callbacks,
        cache=cache,
        column="text",
        covariate_type=covariate_type,
        strategy=extraction_strategy,
        async_mode=async_mode,
        entity_types=entity_types,
        num_threads=num_threads,
    )
    text_units.drop(columns=["text_unit_id"], inplace=True)  # Don't pollute the global
    covariates["id"] = covariates["covariate_type"].apply(lambda _x: str(uuid4()))
    covariates["human_readable_id"] = covariates.index + 1

    return covariates.loc[:, COVARIATES_FINAL_COLUMNS] 
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'GraphExtractionResult' and 'GraphExtractor' models."""

import logging  # noqa: I001
import re
import json
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx
import tiktoken

from graphrag.config.defaults import ENCODING_MODEL, graphrag_config_defaults
from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.index.utils.string import clean_str
from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.index.extract_graph import (
    CONTINUE_PROMPT_JSON,
    CONTINUE_PROMPT,
    ENTITY_CONTINUE_PROMPT,
    LOOP_PROMPT,
    GRAPH_EXTRACTION_PROMPT,
    GRAPH_EXTRACTION_ENTITY_PROMPT,
    GRAPH_EXTRACTION_RELATIONSHIP_PROMPT
)

from . import my_extraction_result

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]

log = logging.getLogger(__name__)


@dataclass
class GraphExtractionResult:
    """Unipartite graph extraction result class definition."""

    output: nx.Graph
    source_docs: dict[Any, Any]


class GraphExtractor:
    """Unipartite graph extractor class definition."""

    _model: ChatModel
    _join_descriptions: bool
    _tuple_delimiter_key: str
    _record_delimiter_key: str
    _entity_types_key: str
    _input_text_key: str
    _completion_delimiter_key: str
    _entity_name_key: str
    _input_descriptions_key: str
    _extraction_prompt: str
    _summarization_prompt: str
    _loop_args: dict[str, Any]
    _max_gleanings: int
    _on_error: ErrorHandlerFn

    def __init__(
        self,
        model_invoker: ChatModel,
        tuple_delimiter_key: str | None = None,
        record_delimiter_key: str | None = None,
        input_text_key: str | None = None,
        entity_types_key: str | None = None,
        completion_delimiter_key: str | None = None,
        prompt: str | None = None,
        join_descriptions=True,
        encoding_model: str | None = None,
        max_gleanings: int | None = None,
        on_error: ErrorHandlerFn | None = None,
    ):
        """Init method definition."""
        # TODO: streamline construction
        self._model = model_invoker
        self._join_descriptions = join_descriptions
        self._input_text_key = input_text_key or "input_text"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        self._completion_delimiter_key = (
            completion_delimiter_key or "completion_delimiter"
        )
        self._entity_types_key = entity_types_key or "entity_types"
        self._extraction_prompt = prompt or GRAPH_EXTRACTION_PROMPT
        # !! New Prompt
        self._extraction_entity_prompt = GRAPH_EXTRACTION_ENTITY_PROMPT
        self._extraction_relationship_prompt = GRAPH_EXTRACTION_RELATIONSHIP_PROMPT
        
        # !! New Prompt
        self._max_gleanings = (
            max_gleanings
            if max_gleanings is not None
            else graphrag_config_defaults.extract_graph.max_gleanings
        )
        self._on_error = on_error or (lambda _e, _s, _d: None)

        # Construct the looping arguments
        encoding = tiktoken.get_encoding(encoding_model or ENCODING_MODEL)
        yes = f"{encoding.encode('Y')[0]}"
        no = f"{encoding.encode('N')[0]}"
        self._loop_args = {"logit_bias": {yes: 100, no: 100}, "max_completion_tokens": 1}

    async def __call__(
        self, texts: list[str], prompt_variables: dict[str, Any] | None = None
    ) -> GraphExtractionResult:
        """Call method definition."""
        if prompt_variables is None:
            prompt_variables = {}
        all_records: dict[int, str] = {}
        source_doc_map: dict[int, str] = {}

        # Wire defaults into the prompt variables
        prompt_variables = {
            **prompt_variables,
            self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key)
            or DEFAULT_TUPLE_DELIMITER,
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
            or DEFAULT_RECORD_DELIMITER,
            self._completion_delimiter_key: prompt_variables.get(
                self._completion_delimiter_key
            )
            or DEFAULT_COMPLETION_DELIMITER,
            self._entity_types_key: ",".join(
                prompt_variables[self._entity_types_key] or DEFAULT_ENTITY_TYPES
            ),
        }

        # for doc_index, text in enumerate(texts):
        #     try:
        #         # Invoke the entity extraction
        #         result = await self._process_document(text, prompt_variables)
        #         source_doc_map[doc_index] = text
        #         all_records[doc_index] = result
        #     except Exception as e:
        #         log.exception("error extracting graph")
        #         self._on_error(
        #             e,
        #             traceback.format_exc(),
        #             {
        #                 "doc_index": doc_index,
        #                 "text": text,
        #             },
        #         )
        
        # output = await self._process_results_plain(
        #     all_records,
        # )
        
        
        
        all_extraction_results = {}
        for doc_index, text in enumerate(texts):
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
            
                try:
                    result = await self._process_document_chunk(text, retry_count)
                    all_extraction_results[doc_index] = result
                    break  # ! Success, exit retry loop
                except Exception as e:
                    retry_count += 1
                    log.warning(
                        f"Error extracting graph for document {doc_index}, retrying ({retry_count}/{max_retries})"
                    )
                    if retry_count >= max_retries:
                        log.exception("error extracting graph")
                        self._on_error(
                            e,
                            traceback.format_exc(),
                            {
                                "doc_index": doc_index,
                                "text": text,
                            },
                        )
                        break
        output = await self._process_extraction_result(
            all_extraction_results,
        )
        

        return GraphExtractionResult(
            output=output,
            source_docs=source_doc_map,
        )

    async def _process_document_chunk(
        self, text: str, retry_count: int
    ) -> my_extraction_result.ExtractResult:
        extract_entity_prompt_template = '''\
You are an expert at extracting high-level, chunk-aware entities from documents.  

1. Chunk definition  
A chunk is any coherent semantic unit—e.g. a paragraph separated by blank lines, a bullet/list item, an inline list (semicolons or commas), or a topic shift within a paragraph. Treat each chunk as one atomic description block.  

2. Extraction rules  
If a chunk describes an entity, copy it verbatim as that entity’s description.  
If the same entity spans multiple chunks, merge them by concatenating each chunk’s text (verbatim).  
If the merged description exceeds 300 words, output two entries: A “summary” entity (≤50 words) capturing the essence. One or more “detail” entities, each using a merged chunk verbatim.  

3. Output format (blank line between entities):  
<name>
<type>
<description>

Document:  
{input_text}'''
        
        extract_graph_prompt_template = '''\
Given a JSON array of entities, output all child entities and all relationships between original and those entities.
You need to generate relationships between each child entity and each parent entity if you generate child entities.
Output in the following format, with blank lines separating distinct entities and relationships.
"""
entities:
<name>
<type>
<description>

relationships:
<source_entity_name>
<source_entity_type>
<target_entity_name>
<target_entity_type>
<relationship_description>
"""

array of entities:
{entities}'''
        
        response = await self._model.achat(
            extract_entity_prompt_template.format(
                input_text=text,
            ),
            name=f"entity-chunk-{retry_count}",
        )
        
        entities = my_extraction_result.parse_entities(response.output.content).entities
        
        if not entities:
            log.warning("No entities extracted from the document chunk.")
        
        response = await self._model.achat(
            extract_graph_prompt_template.format(
                entities=[e.to_dict() for e in entities],
            ),
            name=f"graph-from-entity-chunk-{retry_count}",
        )
        
        graph = my_extraction_result.parse_graph(response.output.content)
        graph.entities.extend(entities)

        return graph


    async def _process_document_plain(
        self, text: str, prompt_variables: dict[str, str]
    ) -> str:
        response = await self._model.achat(
            self._extraction_prompt.format(**{
                **prompt_variables,
                self._input_text_key: text,
            }),
            # json=True,
        )
        results = response.output.content.replace("```", "").replace("\"\"\"", "") or ""
        results_entity = results.split("entities:")[1].split("relationships:")[0]
        results_relationship = results.split("relationships:")[1]

        # Repeat to ensure we maximize entity count
        for i in range(self._max_gleanings):
            
            new_response = await self._model.achat(
                CONTINUE_PROMPT.format(
                    previous_entities_and_relationships=results,
                    input_text=text,
                ),
                name=f"extract-continuation-{i}",
                # history=response.history,
                # json=True,
            )
            
            new_results = new_response.output.content.replace("```", "").replace("\"\"\"", "") or ""
            new_results_entity = new_results.split("entities:")[1].split("relationships:")[0]
            new_results_relationship = new_results.split("relationships:")[1]
            
            results_entity += "\n" + new_results_entity
            results_relationship += "\n" + new_results_relationship
            
        return f"entities: \n{results_entity}\nrelationships: \n{results_relationship}"
        return results

    async def _process_document_json(
        self, text: str, prompt_variables: dict[str, str]
    ) -> str:
        
        response = await self._model.achat(
            self._extraction_prompt.format(**{
                **prompt_variables,
                self._input_text_key: text,
            }),
            # json=True,
        )
        results = response.output.content.replace("```json", "").replace("```", "") or ""
        
        results = json.loads(results)

        # Repeat to ensure we maximize entity count
        for i in range(self._max_gleanings):
            response = await self._model.achat(
                CONTINUE_PROMPT_JSON.format(
                    previous_entities_and_relationships=results,
                    input_text=text,
                ),
                name=f"extract-continuation-{i}",
                # history=response.history,
                # json=True,
            )
            
            new_results = response.output.content.replace("```json", "").replace("```", "") or ""
            new_results = json.loads(new_results)
            print(new_results)
            
            results.get("entities", []).extend(
                new_results.get("entities", [])
            )
            results.get("relationships", []).extend(
                new_results.get("relationships", [])
            )
            # results += response.output.content or ""

            # if this is the final glean, don't bother updating the continuation flag
            if i >= self._max_gleanings - 1:
                break
            response = await self._model.achat(
                LOOP_PROMPT,
                name=f"extract-loopcheck-{i}",
                history=response.history,
                model_parameters=self._loop_args,
            )

            if response.output.content != "Y":
                break

    async def _process_results(
        self,
        results: dict[int, str],
        tuple_delimiter: str,
        record_delimiter: str,
    ) -> nx.Graph:
        """Parse the result string to create an undirected unipartite graph.

        Args:
            - results - dict of results from the extraction chain
            - tuple_delimiter - delimiter between tuples in an output record, default is '<|>'
            - record_delimiter - delimiter between records, default is '##'
        Returns:
            - output - unipartite graph in graphML format
        """
        graph = nx.Graph()
        for source_doc_id, extracted_data in results.items():
            records = [r.strip() for r in extracted_data.split(record_delimiter)]

            for record in records:
                record = re.sub(r"^\(|\)$", "", record.strip())
                record_attributes = record.split(tuple_delimiter)

                if record_attributes[0] in ['"entity"', "entity"] and len(record_attributes) >= 4:
                    # add this record as a node in the G
                    entity_name = clean_str(record_attributes[1].upper())
                    entity_type = clean_str(record_attributes[2].upper())
                    entity_description = clean_str(record_attributes[3])
                    entity_identifier = f"{entity_name}...{entity_type}"

                    if entity_identifier in graph.nodes():
                        node = graph.nodes[entity_identifier]
                        if self._join_descriptions:
                            node["description"] = "\n".join(
                                list({
                                    *_unpack_descriptions(node),
                                    entity_description,
                                })
                            )
                        else:
                            if len(entity_description) > len(node["description"]):
                                node["description"] = entity_description
                        node["source_id"] = ", ".join(
                            list({
                                *_unpack_source_ids(node),
                                str(source_doc_id),
                            })
                        )
                        node["type"] = (
                            entity_type if entity_type != "" else node["type"]
                        )
                    else:
                        graph.add_node(
                            entity_identifier,
                            type=entity_type,
                            description=entity_description,
                            source_id=str(source_doc_id),
                        )

                elif (
                    record_attributes[0] in ['"relationship"', "relationship"]
                    and len(record_attributes) >= 6
                ):
                    # add this record as edge
                    source = f"{clean_str(record_attributes[1].upper())}...{clean_str(record_attributes[2].upper())}"
                    target = f"{clean_str(record_attributes[3].upper())}...{clean_str(record_attributes[4].upper())}"
                    edge_description = clean_str(record_attributes[5])
                    edge_source_id = clean_str(str(source_doc_id))
                    try:
                        weight = float(record_attributes[-1])
                    except ValueError:
                        weight = 1.0

                    if source not in graph.nodes():
                        graph.add_node(
                            source,
                            type=clean_str(record_attributes[2].upper()),
                            description="",
                            source_id=edge_source_id,
                        )
                    if target not in graph.nodes():
                        graph.add_node(
                            target,
                            type=clean_str(record_attributes[4].upper()),
                            description="",
                            source_id=edge_source_id,
                        )
                    if graph.has_edge(source, target):
                        edge_data = graph.get_edge_data(source, target)
                        if edge_data is not None:
                            weight += edge_data["weight"]
                            if self._join_descriptions:
                                edge_description = "\n".join(
                                    list({
                                        *_unpack_descriptions(edge_data),
                                        edge_description,
                                    })
                                )
                            edge_source_id = ", ".join(
                                list({
                                    *_unpack_source_ids(edge_data),
                                    str(source_doc_id),
                                })
                            )
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=edge_description,
                        source_id=edge_source_id,
                    )
                
        return graph

    async def _process_results_json(
        self,
        results: dict[int, str],
    ) -> nx.Graph:
        """Parse the JSON result to create an undirected unipartite graph.

        Args:
            - results - dict of results from the extraction chain in JSON format
        Returns:
            - output - unipartite graph in graphML format
        """
        import json
        
        graph = nx.Graph()
        
        for source_doc_id, extracted_data in results.items():
            # Parse the JSON data
            # print("-"*20, extracted_data)
            # data = json.loads(extracted_data)
            data = extracted_data
            
            # Process entities
            if "entities" in data:
                for entity in data["entities"]:
                    entity_name = clean_str(entity.get("name", "").upper())
                    entity_type = clean_str(entity.get("type", "").upper())
                    entity_description = clean_str(entity.get("description", ""))
                    entity_identifier = f"{entity_name}...{entity_type}"
                    
                    if entity_identifier in graph.nodes():
                        node = graph.nodes[entity_identifier]
                        if self._join_descriptions:
                            node["description"] = "\n".join(
                                list({
                                    *_unpack_descriptions(node),
                                    entity_description,
                                })
                            )
                        else:
                            if len(entity_description) > len(node["description"]):
                                node["description"] = entity_description
                        node["source_id"] = ", ".join(
                            list({
                                *_unpack_source_ids(node),
                                str(source_doc_id),
                            })
                        )
                        node["type"] = (
                            entity_type if entity_type != "" else node["type"]
                        )
                    else:
                        graph.add_node(
                            entity_identifier,
                            type=entity_type,
                            description=entity_description,
                            source_id=str(source_doc_id),
                        )
            
            # Process relationships
            if "relationships" in data:
                for relationship in data["relationships"]:
                    source_name = clean_str(relationship.get("source_entity", "").upper())
                    source_type = clean_str(relationship.get("source_type", "").upper())
                    target_name = clean_str(relationship.get("target_entity", "").upper())
                    target_type = clean_str(relationship.get("target_type", "").upper())
                    edge_description = clean_str(relationship.get("description", ""))
                    
                    source = f"{source_name}...{source_type}"
                    target = f"{target_name}...{target_type}"
                    edge_source_id = clean_str(str(source_doc_id))
                    
                    # Handle weight if present, default to 1.0
                    try:
                        weight = float(relationship.get("weight", 1.0))
                    except (ValueError, TypeError):
                        weight = 1.0
                    
                    # Add source node if not exists
                    if source not in graph.nodes():
                        graph.add_node(
                            source,
                            type=source_type,
                            description="",
                            source_id=edge_source_id,
                        )
                    
                    # Add target node if not exists
                    if target not in graph.nodes():
                        graph.add_node(
                            target,
                            type=target_type,
                            description="",
                            source_id=edge_source_id,
                        )
                    
                    # Update existing edge or add new one
                    if graph.has_edge(source, target):
                        edge_data = graph.get_edge_data(source, target)
                        if edge_data is not None:
                            weight += edge_data["weight"]
                            if self._join_descriptions:
                                edge_description = "\n".join(
                                    list({
                                        *_unpack_descriptions(edge_data),
                                        edge_description,
                                    })
                                )
                            edge_source_id = ", ".join(
                                list({
                                    *_unpack_source_ids(edge_data),
                                    str(source_doc_id),
                                })
                            )
                    
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=edge_description,
                        source_id=edge_source_id,
                    )
        
        return graph
    
    async def _process_results_plain(
        self,
        results: dict[int, str],
    ) -> nx.Graph:
        """Parse the JSON result to create an undirected unipartite graph.

        Args:
            - results - dict of results from the extraction chain in JSON format
        Returns:
            - output - unipartite graph in graphML format
        """
        import json
        
        graph = nx.Graph()
        
        for source_doc_id, extracted_data in results.items():
            # Parse the JSON data
            # print("-"*20, extracted_data)
            # data = json.loads(extracted_data)
            data = extracted_data
            
            if "entities:" in data and "relationships:" in data:
                data = data.replace("```", "").replace("\"\"\"", "")
                entity_str = data.split("entities:")[1].split("relationships:")[0]
                relationship_str = data.split("relationships:")[1]
                entity_lines = [line.strip() for line in entity_str.split("\n") if line.strip()]
                relationship_lines = [line.strip() for line in relationship_str.split("\n") if line.strip()]
                
                try:
                    assert len(entity_lines) % 3 == 0, "Entity lines are not in multiples of 3"
                    assert len(relationship_lines) % 5 == 0, "Relationship lines are not in multiples of 5"
                except AssertionError as e:
                    log.error(f"Error in parsing entity or relationship lines: {e}")
                    continue
                
                # Process entities
                for i in range(0, len(entity_lines), 3):
                    entity_name = clean_str(entity_lines[i].upper())
                    entity_type = clean_str(entity_lines[i + 1].upper())
                    entity_description = clean_str(entity_lines[i + 2])
                    entity_identifier = f"{entity_name}...{entity_type}"
                    
                    if entity_identifier in graph.nodes():
                        node = graph.nodes[entity_identifier]
                        if self._join_descriptions:
                            node["description"] = "\n".join(
                                list({
                                    *_unpack_descriptions(node),
                                    entity_description,
                                })
                            )
                        else:
                            if len(entity_description) > len(node["description"]):
                                node["description"] = entity_description
                        node["source_id"] = ", ".join(
                            list({
                                *_unpack_source_ids(node),
                                str(source_doc_id),
                            })
                        )
                        node["type"] = (
                            entity_type if entity_type != "" else node["type"]
                        )
                    else:
                        graph.add_node(
                            entity_identifier,
                            type=entity_type,
                            description=entity_description,
                            source_id=str(source_doc_id),
                        )
            
            # Process relationships
                for i in range(0, len(relationship_lines), 5):
                    source_name = clean_str(relationship_lines[i].upper())
                    source_type = clean_str(relationship_lines[i + 1].upper())
                    target_name = clean_str(relationship_lines[i + 2].upper())
                    target_type = clean_str(relationship_lines[i + 3].upper())
                    edge_description = clean_str(relationship_lines[i + 4].upper())
                    
                    source = f"{source_name}...{source_type}"
                    target = f"{target_name}...{target_type}"
                    edge_source_id = clean_str(str(source_doc_id))
                    
                    # Handle weight if present, default to 1.0
                    weight = 1.0
                    
                    # Add source node if not exists
                    if source not in graph.nodes():
                        graph.add_node(
                            source,
                            type=source_type,
                            description="",
                            source_id=edge_source_id,
                        )
                    
                    # Add target node if not exists
                    if target not in graph.nodes():
                        graph.add_node(
                            target,
                            type=target_type,
                            description="",
                            source_id=edge_source_id,
                        )
                    
                    # Update existing edge or add new one
                    if graph.has_edge(source, target):
                        edge_data = graph.get_edge_data(source, target)
                        if edge_data is not None:
                            weight += edge_data["weight"]
                            if self._join_descriptions:
                                edge_description = "\n".join(
                                    list({
                                        *_unpack_descriptions(edge_data),
                                        edge_description,
                                    })
                                )
                            edge_source_id = ", ".join(
                                list({
                                    *_unpack_source_ids(edge_data),
                                    str(source_doc_id),
                                })
                            )
                    
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=edge_description,
                        source_id=edge_source_id,
                    )
        
        return graph

    async def _process_extraction_result(
        self,
        extraction_results: dict[int, my_extraction_result.ExtractResult]
    ) -> nx.Graph:
        """Process the extraction result to create a graph."""
        graph = nx.Graph()
        
        for doc_index, extraction_result in extraction_results.items():
            # Add entities to the graph
            for entity in extraction_result.entities:
                entity_identifier = f"{entity.name}...{entity.type}"
                if entity_identifier in graph.nodes():
                    node = graph.nodes[entity_identifier]

                    node["description"] = "\n".join(
                        list({
                            *_unpack_descriptions(node),
                            entity.description,
                        })
                    )
                    node["source_id"] = ", ".join(
                        list({
                            *_unpack_source_ids(node),
                            str(doc_index),
                        })
                    )
                    # ! Overwrite !
                    node["type"] = (
                        entity.type if entity.type != "" else node["type"]
                    )
                    
                else:
                    graph.add_node(
                        entity_identifier,
                        type=entity.type,
                        description=entity.description,
                        source_id=str(doc_index),
                    )

            # Add relationships to the graph
            for relationship in extraction_result.relationships:
                source = f"{relationship.source.name}...{relationship.source.type}"
                target = f"{relationship.target.name}...{relationship.target.type}"
                edge_description = relationship.description
                edge_source_id = str(doc_index)
                
                # Handle weight if present, default to 1.0
                weight = 1.0
                
                # ! Justification for bastard.
                # Add source node if not exists
                if source not in graph.nodes():
                    graph.add_node(
                        source,
                        type=relationship.source.type,
                        description="",
                        source_id=edge_source_id,
                    )
                
                # Add target node if not exists
                if target not in graph.nodes():
                    graph.add_node(
                        target,
                        type=relationship.target.type,
                        description="",
                        source_id=edge_source_id,
                    )
                
                # Update existing edge or add new one
                if graph.has_edge(source, target):
                    edge_data = graph.get_edge_data(source, target)
                    if edge_data is not None:
                        weight += edge_data["weight"]
                        if self._join_descriptions:
                            edge_description = "\n".join(
                                list({
                                    *_unpack_descriptions(edge_data),
                                    edge_description,
                                })
                            )
                        edge_source_id = ", ".join(
                            list({
                                *_unpack_source_ids(edge_data),
                                str(doc_index),
                            }))
                graph.add_edge(
                    source,
                    target,
                    weight=weight,
                    description=edge_description,
                    source_id=edge_source_id,
                )

        return graph

def _unpack_descriptions(data: Mapping) -> list[str]:
    value = data.get("description", None)
    return [] if value is None else value.split("\n")


def _unpack_source_ids(data: Mapping) -> list[str]:
    value = data.get("source_id", None)
    return [] if value is None else value.split(", ")

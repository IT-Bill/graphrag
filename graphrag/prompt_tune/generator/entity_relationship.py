# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Entity relationship example generation module."""

import asyncio

from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompt_tune.prompt.entity_relationship import (
    ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT,
    ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
    UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
)

MAX_EXAMPLES = 3
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"

async def generate_entity_relationship_examples(
    model: ChatModel,
    persona: str,
    entity_types: str | list[str] | None,
    docs: str | list[str],
    language: str,
    json_mode: bool = False,
) -> list[str]:
    """Generate a list of entity/relationships examples for use in generating an entity configuration.

    Will return entity/relationships examples as either JSON or in tuple_delimiter format depending
    on the json_mode parameter.
    """
    docs_list = [docs] if isinstance(docs, str) else docs
    history = [{"content": persona, "role": "user"}]

    if entity_types:
        entity_types_str = (
            entity_types
            if isinstance(entity_types, str)
            else ", ".join(map(str, entity_types))
        )
        # messages = [
        #     (
        #         ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
        #         if json_mode
        #         else ENTITY_RELATIONSHIPS_GENERATION_PROMPT
        #     ).format(entity_types=entity_types_str, input_text=doc, language=language)
        #     .format(
        #         tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
        #         record_delimiter=DEFAULT_RECORD_DELIMITER,
        #         completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
        #     )
        #     for doc in docs_list
        # ]
        messages = [
            (
                ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
            ).format(entity_types=entity_types_str, input_text=doc, language=language)
            for doc in docs_list
        ]
    else:
        messages = [
            UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT.format(
                input_text=doc, language=language
            )
            for doc in docs_list
        ]

    messages = messages[:MAX_EXAMPLES]

    tasks = [
        model.achat(message, history=history, json=json_mode) for message in messages
    ]

    responses = await asyncio.gather(*tasks)
    
    # save responses to json
    with open("responses.txt", "w") as f:  # noqa: ASYNC230
        for response in responses:
            f.write(response.output.content.replace("```json", "").replace("```", ""))
            f.write("----------------------------------------\n")

    with open("messages.txt", "w") as f:
        for message in messages:
            f.write(message)
            f.write("----------------------------------------\n")

    return [response.output.content.replace("```json", "").replace("```", "") for response in responses]

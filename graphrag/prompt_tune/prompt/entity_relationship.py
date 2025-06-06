# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Fine-tuning prompts for entity relationship generation."""

# ENTITY_RELATIONSHIPS_GENERATION_PROMPT = """
# -Goal-
# Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
 
# -Steps-
# 1. Identify all entities. For each identified entity, extract the following information:
# - entity_name: Name of the entity
# - entity_type: Example types: [{entity_types}]
# - entity_description: Comprehensive description of the entity's attributes and activities
# Format each entity as ("entity"{{tuple_delimiter}}<entity_name>{{tuple_delimiter}}<entity_type>{{tuple_delimiter}}<entity_description>)
 
# 2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
# For each pair of related entities, extract the following information:
# - source_entity_name: name of the source entity, as identified in step 1
# - source_entity_type: type of the source entity
# - target_entity_name: name of the target entity, as identified in step 1
# - target_entity_type: type of the target entity
# - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
# Format each relationship as ("relationship"{{tuple_delimiter}}<source_entity_name>{{tuple_delimiter}}<source_entity_type>{{tuple_delimiter}}<target_entity_name>{{tuple_delimiter}}<target_entity_type>{{tuple_delimiter}}<relationship_description>)

# 3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{{record_delimiter}}** as the list delimiter.
 
# 4. When finished, output {{completion_delimiter}}
 
# ######################
# -Examples-
# ######################
# Example 1:
# Entity_types: ORGANIZATION,PERSON
# Text:
# The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
# ######################
# Output:
# ("entity"{{tuple_delimiter}}CENTRAL INSTITUTION{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}MARTIN SMITH{{tuple_delimiter}}PERSON{{tuple_delimiter}}Martin Smith is the chair of the Central Institution)
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}MARKET STRATEGY COMMITTEE{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}MARTIN SMITH{{tuple_delimiter}}PERSON{{tuple_delimiter}}CENTRAL INSTITUTION{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}Martin Smith is the Chair of the Central Institution and will answer questions at a press conference)
# {{completion_delimiter}}

# ######################
# Example 2:
# Entity_types: ORGANIZATION
# Text:
# TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

# TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
# ######################
# Output:
# ("entity"{{tuple_delimiter}}TECHGLOBAL{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}VISION HOLDINGS{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}Vision Holdings is a firm that previously owned TechGlobal)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}TECHGLOBAL{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}VISION HOLDINGS{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}Vision Holdings formerly owned TechGlobal from 2014 until present)
# {{completion_delimiter}}

# ######################
# Example 3:
# Entity_types: ORGANIZATION,GEO,PERSON
# Text:
# Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

# The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

# The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

# They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

# The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
# ######################
# Output:
# ("entity"{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}Firuzabad held Aurelians as hostages)
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}AURELIA{{tuple_delimiter}}GEO{{tuple_delimiter}}Country seeking to release hostages)
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}QUINTARA{{tuple_delimiter}}GEO{{tuple_delimiter}}Country that negotiated a swap of money in exchange for hostages)
# {{record_delimiter}}
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}TIRUZIA{{tuple_delimiter}}GEO{{tuple_delimiter}}Capital of Firuzabad where the Aurelians were being held)
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}KROHAARA{{tuple_delimiter}}GEO{{tuple_delimiter}}Capital city in Quintara)
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}CASHION{{tuple_delimiter}}GEO{{tuple_delimiter}}Capital city in Aurelia)
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}SAMUEL NAMARA{{tuple_delimiter}}PERSON{{tuple_delimiter}}Aurelian who spent time in Tiruzia's Alhamia Prison)
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}ALHAMIA PRISON{{tuple_delimiter}}GEO{{tuple_delimiter}}Prison in Tiruzia)
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}DURKE BATAGLANI{{tuple_delimiter}}PERSON{{tuple_delimiter}}Aurelian journalist who was held hostage)
# {{record_delimiter}}
# ("entity"{{tuple_delimiter}}MEGGIE TAZBAH{{tuple_delimiter}}PERSON{{tuple_delimiter}}Bratinas national and environmentalist who was held hostage)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}AURELIA{{tuple_delimiter}}GEO{{tuple_delimiter}}Firuzabad negotiated a hostage exchange with Aurelia)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}QUINTARA{{tuple_delimiter}}GEO{{tuple_delimiter}}AURELIA{{tuple_delimiter}}GEO{{tuple_delimiter}}Quintara brokered the hostage exchange between Firuzabad and Aurelia)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}QUINTARA{{tuple_delimiter}}GEO{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}Quintara brokered the hostage exchange between Firuzabad and Aurelia)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}SAMUEL NAMARA{{tuple_delimiter}}PERSON{{tuple_delimiter}}ALHAMIA PRISON{{tuple_delimiter}}GEO{{tuple_delimiter}}Samuel Namara was a prisoner at Alhamia prison)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}SAMUEL NAMARA{{tuple_delimiter}}PERSON{{tuple_delimiter}}MEGGIE TAZBAH{{tuple_delimiter}}PERSON{{tuple_delimiter}}Samuel Namara and Meggie Tazbah were exchanged in the same hostage release)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}SAMUEL NAMARA{{tuple_delimiter}}PERSON{{tuple_delimiter}}DURKE BATAGLANI{{tuple_delimiter}}PERSON{{tuple_delimiter}}Samuel Namara and Durke Bataglani were exchanged in the same hostage release)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}MEGGIE TAZBAH{{tuple_delimiter}}PERSON{{tuple_delimiter}}DURKE BATAGLANI{{tuple_delimiter}}PERSON{{tuple_delimiter}}Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}SAMUEL NAMARA{{tuple_delimiter}}PERSON{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}Samuel Namara was a hostage in Firuzabad)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}MEGGIE TAZBAH{{tuple_delimiter}}PERSON{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}Meggie Tazbah was a hostage in Firuzabad)
# {{record_delimiter}}
# ("relationship"{{tuple_delimiter}}DURKE BATAGLANI{{tuple_delimiter}}PERSON{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}Durke Bataglani was a hostage in Firuzabad)
# {{completion_delimiter}}

# ######################
# -Real Data-
# ######################
# Entity_types: {entity_types}
# Text: {input_text}
# ######################
# Output:
# """

ENTITY_RELATIONSHIPS_GENERATION_PROMPT = '''
Goal
"""
Given a text document and a list of entity types, output all entities of those types from the text and all relationships among the identified entities.
"""

Steps
"""
1. Output all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity
- entity_type: Example types: {entity_types}
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, output all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity_name: name of the source entity, as identified in step 1
- source_entity_type: type of the source entity
- target_entity_name: name of the target entity, as identified in step 1
- target_entity_type: type of the target entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use {{record_delimiter}} as the list delimiter.

4. When finished, output {{completion_delimiter}}
"""

Output Format Example
"""
(entity{{tuple_delimiter}}<entity_name>{{tuple_delimiter}}<entity_type>{{tuple_delimiter}}<entity_description>)
{{record_delimiter}}
(relationship{{tuple_delimiter}}<source_entity_name>{{tuple_delimiter}}<source_entity_type>{{tuple_delimiter}}<target_entity_name>{{tuple_delimiter}}<target_entity_type>{{tuple_delimiter}}<relationship_description>)
{{record_delimiter}}
{{completion_delimiter}}
"""

Data
"""
Entity_types: {entity_types}
Text:
{input_text}
"""

Output:

'''


ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT = '''
Goal
"""
Given a text document, output all entities and their types.
Then output all relationships among the entities.
The response should be a single JSON object that can be parsed by json.loads.
"""

Examples
"""
{{
  "entities": [
    {{
      "name": <entity name>,
      "type": <entity type>,
      "description": <entity description>
    }},
  ],
  "relationships": [
    {{
      "source_entity": <source entity name>,
      "source_type": <source entity type>,
      "target_entity": <target entity name>,
      "target_type": <target entity type>,
      "description": <relationship description>
    }},
  ]
}}
"""


Document
"""
{input_text}
"""
output:
'''

UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT = '''
Goal
"""
Given a text document that is potentially relevant to this activity, first identify all entities needed from the text in order to capture the information and ideas in the text.
Next, report all relationships among the identified entities.
"""

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Suggest several labels or categories for the entity. The categories should not be specific, but should be as general as possible.
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{{tuple_delimiter}}<entity_name>{{tuple_delimiter}}<entity_type>{{tuple_delimiter}}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity_name: name of the source entity, as identified in step 1
- source_entity_type: type of the source entity
- target_entity_name: name of the target entity, as identified in step 1
- target_entity_type: type of the target entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
Format each relationship as ("relationship"{{tuple_delimiter}}<source_entity_name>{{tuple_delimiter}}<source_entity_type>{{tuple_delimiter}}<target_entity_name>{{tuple_delimiter}}<target_entity_type>{{tuple_delimiter}}<relationship_description>)


3. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{{record_delimiter}}** as the list delimiter.

4. If you have to translate into {language}, just translate the descriptions, nothing else!

5. When finished, output {{completion_delimiter}}.

######################
-Examples-
######################
Example 1:
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"{{tuple_delimiter}}CENTRAL INSTITUTION{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
{{record_delimiter}}
("entity"{{tuple_delimiter}}MARTIN SMITH{{tuple_delimiter}}PERSON{{tuple_delimiter}}Martin Smith is the chair of the Central Institution)
{{record_delimiter}}
("entity"{{tuple_delimiter}}MARKET STRATEGY COMMITTEE{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}MARTIN SMITH{{tuple_delimiter}}PERSON{{tuple_delimiter}}CENTRAL INSTITUTION{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}Martin Smith is the Chair of the Central Institution and will answer questions at a press conference)
{{completion_delimiter}}

######################
Example 2:
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"{{tuple_delimiter}}TECHGLOBAL{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
{{record_delimiter}}
("entity"{{tuple_delimiter}}VISION HOLDINGS{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}Vision Holdings is a firm that previously owned TechGlobal)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}TECHGLOBAL{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}VISION HOLDINGS{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}Vision Holdings formerly owned TechGlobal from 2014 until present)
{{completion_delimiter}}

######################
Example 3:
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
("entity"{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}Firuzabad held Aurelians as hostages)
{{record_delimiter}}
("entity"{{tuple_delimiter}}AURELIA{{tuple_delimiter}}GEO{{tuple_delimiter}}Country seeking to release hostages)
{{record_delimiter}}
("entity"{{tuple_delimiter}}QUINTARA{{tuple_delimiter}}GEO{{tuple_delimiter}}Country that negotiated a swap of money in exchange for hostages)
{{record_delimiter}}
{{record_delimiter}}
("entity"{{tuple_delimiter}}TIRUZIA{{tuple_delimiter}}GEO{{tuple_delimiter}}Capital of Firuzabad where the Aurelians were being held)
{{record_delimiter}}
("entity"{{tuple_delimiter}}KROHAARA{{tuple_delimiter}}GEO{{tuple_delimiter}}Capital city in Quintara)
{{record_delimiter}}
("entity"{{tuple_delimiter}}CASHION{{tuple_delimiter}}GEO{{tuple_delimiter}}Capital city in Aurelia)
{{record_delimiter}}
("entity"{{tuple_delimiter}}SAMUEL NAMARA{{tuple_delimiter}}PERSON{{tuple_delimiter}}Aurelian who spent time in Tiruzia's Alhamia Prison)
{{record_delimiter}}
("entity"{{tuple_delimiter}}ALHAMIA PRISON{{tuple_delimiter}}GEO{{tuple_delimiter}}Prison in Tiruzia)
{{record_delimiter}}
("entity"{{tuple_delimiter}}DURKE BATAGLANI{{tuple_delimiter}}PERSON{{tuple_delimiter}}Aurelian journalist who was held hostage)
{{record_delimiter}}
("entity"{{tuple_delimiter}}MEGGIE TAZBAH{{tuple_delimiter}}PERSON{{tuple_delimiter}}Bratinas national and environmentalist who was held hostage)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}AURELIA{{tuple_delimiter}}GEO{{tuple_delimiter}}Firuzabad negotiated a hostage exchange with Aurelia)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}QUINTARA{{tuple_delimiter}}GEO{{tuple_delimiter}}AURELIA{{tuple_delimiter}}GEO{{tuple_delimiter}}Quintara brokered the hostage exchange between Firuzabad and Aurelia)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}QUINTARA{{tuple_delimiter}}GEO{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}Quintara brokered the hostage exchange between Firuzabad and Aurelia)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}SAMUEL NAMARA{{tuple_delimiter}}PERSON{{tuple_delimiter}}ALHAMIA PRISON{{tuple_delimiter}}GEO{{tuple_delimiter}}Samuel Namara was a prisoner at Alhamia prison)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}SAMUEL NAMARA{{tuple_delimiter}}PERSON{{tuple_delimiter}}MEGGIE TAZBAH{{tuple_delimiter}}PERSON{{tuple_delimiter}}Samuel Namara and Meggie Tazbah were exchanged in the same hostage release)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}SAMUEL NAMARA{{tuple_delimiter}}PERSON{{tuple_delimiter}}DURKE BATAGLANI{{tuple_delimiter}}PERSON{{tuple_delimiter}}Samuel Namara and Durke Bataglani were exchanged in the same hostage release)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}MEGGIE TAZBAH{{tuple_delimiter}}PERSON{{tuple_delimiter}}DURKE BATAGLANI{{tuple_delimiter}}PERSON{{tuple_delimiter}}Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}SAMUEL NAMARA{{tuple_delimiter}}PERSON{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}Samuel Namara was a hostage in Firuzabad)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}MEGGIE TAZBAH{{tuple_delimiter}}PERSON{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}Meggie Tazbah was a hostage in Firuzabad)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}DURKE BATAGLANI{{tuple_delimiter}}PERSON{{tuple_delimiter}}FIRUZABAD{{tuple_delimiter}}GEO{{tuple_delimiter}}Durke Bataglani was a hostage in Firuzabad)
{{completion_delimiter}}

######################
-Real Data-
######################
Text: {input_text}
######################
Output:
'''

# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
 
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity
- entity_type: Example types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
 
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity_name: name of the source entity, as identified in step 1
- source_entity_type: type of the source entity
- target_entity_name: name of the target entity, as identified in step 1
- target_entity_type: type of the target entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
Format each relationship as ("relationship"{tuple_delimiter}<source_entity_name>{tuple_delimiter}<source_entity_type>{tuple_delimiter}<target_entity_name>{tuple_delimiter}<target_entity_type>{tuple_delimiter}<relationship_description>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use {record_delimiter} as the list delimiter.
 
4. When finished, output {completion_delimiter}
 
######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
{record_delimiter}
("entity"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}PERSON{tuple_delimiter}Martin Smith is the chair of the Central Institution)
{record_delimiter}
("entity"{tuple_delimiter}MARKET STRATEGY COMMITTEE{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
{record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}PERSON{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}ORGANIZATION{tuple_delimiter}Martin Smith is the Chair of the Central Institution and will answer questions at a press conference)
{completion_delimiter}

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}ORGANIZATION{tuple_delimiter}TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
{record_delimiter}
("entity"{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}ORGANIZATION{tuple_delimiter}Vision Holdings is a firm that previously owned TechGlobal)
{record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}ORGANIZATION{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}ORGANIZATION{tuple_delimiter}Vision Holdings formerly owned TechGlobal from 2014 until present)
{completion_delimiter}

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
("entity"{tuple_delimiter}FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}Firuzabad held Aurelians as hostages)
{record_delimiter}
("entity"{tuple_delimiter}AURELIA{tuple_delimiter}GEO{tuple_delimiter}Country seeking to release hostages)
{record_delimiter}
("entity"{tuple_delimiter}QUINTARA{tuple_delimiter}GEO{tuple_delimiter}Country that negotiated a swap of money in exchange for hostages)
{record_delimiter}
{record_delimiter}
("entity"{tuple_delimiter}TIRUZIA{tuple_delimiter}GEO{tuple_delimiter}Capital of Firuzabad where the Aurelians were being held)
{record_delimiter}
("entity"{tuple_delimiter}KROHAARA{tuple_delimiter}GEO{tuple_delimiter}Capital city in Quintara)
{record_delimiter}
("entity"{tuple_delimiter}CASHION{tuple_delimiter}GEO{tuple_delimiter}Capital city in Aurelia)
{record_delimiter}
("entity"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}PERSON{tuple_delimiter}Aurelian who spent time in Tiruzia's Alhamia Prison)
{record_delimiter}
("entity"{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}GEO{tuple_delimiter}Prison in Tiruzia)
{record_delimiter}
("entity"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}PERSON{tuple_delimiter}Aurelian journalist who was held hostage)
{record_delimiter}
("entity"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}PERSON{tuple_delimiter}Bratinas national and environmentalist who was held hostage)
{record_delimiter}
("relationship"{tuple_delimiter}FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}AURELIA{tuple_delimiter}GEO{tuple_delimiter}Firuzabad negotiated a hostage exchange with Aurelia)
{record_delimiter}
("relationship"{tuple_delimiter}QUINTARA{tuple_delimiter}GEO{tuple_delimiter}AURELIA{tuple_delimiter}GEO{tuple_delimiter}Quintara brokered the hostage exchange between Firuzabad and Aurelia)
{record_delimiter}
("relationship"{tuple_delimiter}QUINTARA{tuple_delimiter}GEO{tuple_delimiter}FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}Quintara brokered the hostage exchange between Firuzabad and Aurelia)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}PERSON{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}GEO{tuple_delimiter}Samuel Namara was a prisoner at Alhamia prison)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}PERSON{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}PERSON{tuple_delimiter}Samuel Namara and Meggie Tazbah were exchanged in the same hostage release)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}PERSON{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}PERSON{tuple_delimiter}Samuel Namara and Durke Bataglani were exchanged in the same hostage release)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}PERSON{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}PERSON{tuple_delimiter}Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}PERSON{tuple_delimiter}FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}Samuel Namara was a hostage in Firuzabad)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}PERSON{tuple_delimiter}FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}Meggie Tazbah was a hostage in Firuzabad)
{record_delimiter}
("relationship"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}PERSON{tuple_delimiter}FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}Durke Bataglani was a hostage in Firuzabad)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

# !!!!!!!!!!!!!!!!!!!!!!!
GRAPH_EXTRACTION_ENTITY_PROMPT = """

-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text.

-Steps-
1. Identify all entities (as many entities as possible). For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [scholarship, bursary, school, student, parent, department, course, activity, support service, facility, community]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Return output in English as a single list of all the entities and relationships identified in steps 1. Use {record_delimiter} as the list delimiter.

3. If you have to translate into English, just translate the descriptions, nothing else!

-Examples-
######################

Example 1:

entity_types: [scholarship, bursary, school, student, parent, department, course, activity, support service, facility, community]
text:
Scholarship & Bursaries

Millfield Prep

We award a number of Academic, Art, Drama, Music and Sports Scholarships each year for entry into Years 6, 7 and 8 for external applicants (into Years 6 and 7 only for internal applicants already attending Millfield Prep).

Additionally, our Millfield Prep Award recognises Prep pupils who are talented all-rounders (strong academics and talented in any two areas from Art, Drama, Music, Sport). Please note that this award is only for Years 6, 7 and 8 whilst at Millfield Prep School and will end on transfer to Year 9. Pupils who hold a Millfield Prep Award may apply for a new Scholarship for entry into Year 9 and these are currently available for Academic, Co-curricular and Sport.

Scholarships carry a maximum fee reduction of up to 10% (multiple awards may be applied for but the total fee reduction will be no more than 10%).  Alternatively, parents who do not require financial assistance may accept their child's Scholarship on an honorary basis. Parents can apply to have a Scholarship augmented by a bursary, subject to a means test. 

Bursaries may also be available for entry to Millfield Prep (Year 6 upwards) in cases where a pupil does not meet the requirements for a scholarship and where there is genuine financial need. Bursaries take the form of a fee reduction to a level dependent on the parents’ financial resources. 

Parents applying for a bursary will be asked to complete a confidential financial circumstances form, available from the Fees Department. All awards are subject to the school’s ability to fund them within the context of its overall budget, and therefore levels cannot be guaranteed.

Please do not hesitate to contact our Prep Admissions team on prepadmissions@millfieldschool.com should you have any queries. Pupils who live overseas should contact the Admissions department prior to submitting a scholarship application
------------------------
output:
("entity"{tuple_delimiter}MILLFIELD PREP{tuple_delimiter}SCHOOL{tuple_delimiter}Millfield Prep is a school that awards Academic, Art, Drama, Music, and Sports Scholarships for entry into Years 6, 7, and 8, and recognizes talented all-rounders with the Millfield Prep Award.)
{record_delimiter}
("entity"{tuple_delimiter}MILLFIELD PREP AWARD{tuple_delimiter}SCHOLARSHIP{tuple_delimiter}The Millfield Prep Award recognizes Prep pupils who are strong academically and talented in any two areas from Art, Drama, Music, or Sport, applicable for Years 6, 7, and 8.)
{record_delimiter}
("entity"{tuple_delimiter}SCHOLARSHIP{tuple_delimiter}SCHOLARSHIP{tuple_delimiter}Scholarships at Millfield Prep provide a maximum fee reduction of up to 10% for talented students in various disciplines.)
{record_delimiter}
("entity"{tuple_delimiter}BURSARY{tuple_delimiter}BURARY{tuple_delimiter}Bursaries are available for entry to Millfield Prep for students who do not meet scholarship requirements and have genuine financial need, providing fee reductions based on parents' financial resources.)
{record_delimiter}
("entity"{tuple_delimiter}FEES DEPARTMENT{tuple_delimiter}DEPARTMENT{tuple_delimiter}The Fees Department handles applications for bursaries and provides financial circumstances forms for parents applying for financial assistance.)
{record_delimiter}
("entity"{tuple_delimiter}PREP ADMISSIONS TEAM{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}The Prep Admissions team assists with queries related to scholarship applications and admissions at Millfield Prep.)
{record_delimiter}
#############################


Example 2:

entity_types: [scholarship, bursary, school, student, parent, department, course, activity, support service, facility, community]
text:
 10 and 11, more emphasis is placed on supporting students with individual GCSE subject choices and helping them develop skills to encourage and support independent learning.

Sixth Form LSC Support

In Sixth Form we continue to support AEN students in small groups; these sessions are timetabled in a student's private study periods. Identified students will attend the LSC at least once a week and receive information and advice on topics such as study skills and an introduction to the assistive technologies which are available in school for their use, to aid independent study. Students will have access to software such as Dragon Naturally Speaking, Read and Write 11 and Inspiration Mind mapping.

For AEN students who may only need occasional access to the LSC, we run lunchtime workshops, staffed by LSC Study Support staff.

Exam Access Arrangements

For students who meet the examination boards’ criteria we can provide a range of exam access arrangements including additional time, readers, scribes and the use of laptops. We have a team of experienced external invigilators who help to provide this service for end of year exams, controlled assessments, rehearsal exams and external GCSE and A level exams.

Individual Support

For those students who would like more specialist support, we are able to provide 1:1 lessons with either a Speech and Language Therapist or a peripatetic Teacher. There is an additional charge for these 1:1 sessions, please see Millfield Fees Sheet for more details.

Building self-esteem, easing the journey

The LSC aims to encourage students, build their self-esteem and set them on the road to independent learning. By having high expectations and developing positive working relationships we aim to maximise opportunities and success for our AEN students.
------------------------
output:
("entity"{tuple_delimiter}MILLFIELD{tuple_delimiter}SCHOOL{tuple_delimiter}Millfield is a school that supports students with individual GCSE subject choices and encourages independent learning through various support services.)
{record_delimiter}
("entity"{tuple_delimiter}LSC{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}The Learning Support Centre (LSC) provides support for AEN students, offering small group sessions and access to assistive technologies to aid independent study.)
{record_delimiter}
("entity"{tuple_delimiter}DRAGON NATURALLY SPEAKING{tuple_delimiter}FACILITY{tuple_delimiter}Dragon Naturally Speaking is a software available in the LSC to assist students with independent study through speech recognition.)
{record_delimiter}
("entity"{tuple_delimiter}READ AND WRITE 11{tuple_delimiter}FACILITY{tuple_delimiter}Read and Write 11 is a software tool provided in the LSC to support students in their independent learning.)
{record_delimiter}
("entity"{tuple_delimiter}INSPIRATION MIND MAPPING{tuple_delimiter}FACILITY{tuple_delimiter}Inspiration Mind Mapping is a software available in the LSC to help students organize their thoughts and ideas for independent study.)
{record_delimiter}
("entity"{tuple_delimiter}EXAM ACCESS ARRANGEMENTS{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}Exam Access Arrangements provide additional support for students during exams, including extra time and the use of assistive tools.)
{record_delimiter}
("entity"{tuple_delimiter}SPEECH AND LANGUAGE THERAPIST{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}A Speech and Language Therapist provides specialist 1:1 lessons for students needing additional support.)
{record_delimiter}
("entity"{tuple_delimiter}PERIPATETIC TEACHER{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}A peripatetic Teacher offers specialized 1:1 lessons for students requiring extra assistance in their studies.)
{record_delimiter}
#############################



-Real Data-
######################
entity_types: [scholarship, bursary, school, student, parent, department, course, activity, support service, facility, community]
text:
{input_text}
######################
output:
"""


# !!!!!!!!!!!!!!!!!!!!!!!
GRAPH_EXTRACTION_RELATIONSHIP_PROMPT = """

-Goal-
Given a text document that is potentially relevant to this activity and a list of identified entities from the previous step, identify all relationships among the identified entities.

-Steps-

1. From the entities identified, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
Each identified entity has been formated as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
For each pair of related entities, extract the following information:
- source_entity_name: name of the source entity, as identified in step 1
- source_entity_type: type of the source entity
- target_entity_name: name of the target entity, as identified in step 1
- target_entity_type: type of the target entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
Format each relationship as ("relationship"{tuple_delimiter}<source_entity_name>{tuple_delimiter}<source_entity_type>{tuple_delimiter}<target_entity_name>{tuple_delimiter}<target_entity_type>{tuple_delimiter}<relationship_description>)

2. Return output in English as a single list of all the entities and relationships identified in steps 1. Use {record_delimiter} as the list delimiter.

3. If you have to translate into English, just translate the descriptions, nothing else!

4. When finished, output {completion_delimiter}.

-Examples-
######################

Example 1:

text:
Scholarship & Bursaries

Millfield Prep

We award a number of Academic, Art, Drama, Music and Sports Scholarships each year for entry into Years 6, 7 and 8 for external applicants (into Years 6 and 7 only for internal applicants already attending Millfield Prep).

Additionally, our Millfield Prep Award recognises Prep pupils who are talented all-rounders (strong academics and talented in any two areas from Art, Drama, Music, Sport). Please note that this award is only for Years 6, 7 and 8 whilst at Millfield Prep School and will end on transfer to Year 9. Pupils who hold a Millfield Prep Award may apply for a new Scholarship for entry into Year 9 and these are currently available for Academic, Co-curricular and Sport.

Scholarships carry a maximum fee reduction of up to 10% (multiple awards may be applied for but the total fee reduction will be no more than 10%).  Alternatively, parents who do not require financial assistance may accept their child's Scholarship on an honorary basis. Parents can apply to have a Scholarship augmented by a bursary, subject to a means test. 

Bursaries may also be available for entry to Millfield Prep (Year 6 upwards) in cases where a pupil does not meet the requirements for a scholarship and where there is genuine financial need. Bursaries take the form of a fee reduction to a level dependent on the parents’ financial resources. 

Parents applying for a bursary will be asked to complete a confidential financial circumstances form, available from the Fees Department. All awards are subject to the school’s ability to fund them within the context of its overall budget, and therefore levels cannot be guaranteed.

Please do not hesitate to contact our Prep Admissions team on prepadmissions@millfieldschool.com should you have any queries. Pupils who live overseas should contact the Admissions department prior to submitting a scholarship application

identified entities:
("entity"{tuple_delimiter}MILLFIELD PREP{tuple_delimiter}SCHOOL{tuple_delimiter}Millfield Prep is a school that awards Academic, Art, Drama, Music, and Sports Scholarships for entry into Years 6, 7, and 8, and recognizes talented all-rounders with the Millfield Prep Award.)
{record_delimiter}
("entity"{tuple_delimiter}MILLFIELD PREP AWARD{tuple_delimiter}SCHOLARSHIP{tuple_delimiter}The Millfield Prep Award recognizes Prep pupils who are strong academically and talented in any two areas from Art, Drama, Music, or Sport, applicable for Years 6, 7, and 8.)
{record_delimiter}
("entity"{tuple_delimiter}SCHOLARSHIP{tuple_delimiter}SCHOLARSHIP{tuple_delimiter}Scholarships at Millfield Prep provide a maximum fee reduction of up to 10% for talented students in various disciplines.)
{record_delimiter}
("entity"{tuple_delimiter}BURSARY{tuple_delimiter}BURARY{tuple_delimiter}Bursaries are available for entry to Millfield Prep for students who do not meet scholarship requirements and have genuine financial need, providing fee reductions based on parents' financial resources.)
{record_delimiter}
("entity"{tuple_delimiter}FEES DEPARTMENT{tuple_delimiter}DEPARTMENT{tuple_delimiter}The Fees Department handles applications for bursaries and provides financial circumstances forms for parents applying for financial assistance.)
{record_delimiter}
("entity"{tuple_delimiter}PREP ADMISSIONS TEAM{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}The Prep Admissions team assists with queries related to scholarship applications and admissions at Millfield Prep.)
{record_delimiter}
------------------------
output:
("relationship"{tuple_delimiter}MILLFIELD PREP{tuple_delimiter}SCHOOL{tuple_delimiter}MILLFIELD PREP AWARD{tuple_delimiter}SCHOLARSHIP{tuple_delimiter}The Millfield Prep Award is a recognition given by Millfield Prep to talented all-rounders among its pupils)
{record_delimiter}
("relationship"{tuple_delimiter}MILLFIELD PREP{tuple_delimiter}SCHOOL{tuple_delimiter}SCHOLARSHIP{tuple_delimiter}SCHOLARSHIP{tuple_delimiter}Millfield Prep awards scholarships to students for various disciplines, including Academic, Art, Drama, Music, and Sports)
{record_delimiter}
("relationship"{tuple_delimiter}MILLFIELD PREP{tuple_delimiter}SCHOOL{tuple_delimiter}BURSARY{tuple_delimiter}BURARY{tuple_delimiter}Millfield Prep offers bursaries to students who do not qualify for scholarships and demonstrate financial need)
{record_delimiter}
("relationship"{tuple_delimiter}BURSARY{tuple_delimiter}BURARY{tuple_delimiter}FEES DEPARTMENT{tuple_delimiter}DEPARTMENT{tuple_delimiter}The Fees Department manages the bursary applications and financial circumstances forms for parents)
{record_delimiter}
("relationship"{tuple_delimiter}MILLFIELD PREP{tuple_delimiter}SCHOOL{tuple_delimiter}PREP ADMISSIONS TEAM{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}The Prep Admissions team is part of Millfield Prep and assists with scholarship application queries)
{completion_delimiter}
#############################


Example 2:

text:
 10 and 11, more emphasis is placed on supporting students with individual GCSE subject choices and helping them develop skills to encourage and support independent learning.

Sixth Form LSC Support

In Sixth Form we continue to support AEN students in small groups; these sessions are timetabled in a student's private study periods. Identified students will attend the LSC at least once a week and receive information and advice on topics such as study skills and an introduction to the assistive technologies which are available in school for their use, to aid independent study. Students will have access to software such as Dragon Naturally Speaking, Read and Write 11 and Inspiration Mind mapping.

For AEN students who may only need occasional access to the LSC, we run lunchtime workshops, staffed by LSC Study Support staff.

Exam Access Arrangements

For students who meet the examination boards’ criteria we can provide a range of exam access arrangements including additional time, readers, scribes and the use of laptops. We have a team of experienced external invigilators who help to provide this service for end of year exams, controlled assessments, rehearsal exams and external GCSE and A level exams.

Individual Support

For those students who would like more specialist support, we are able to provide 1:1 lessons with either a Speech and Language Therapist or a peripatetic Teacher. There is an additional charge for these 1:1 sessions, please see Millfield Fees Sheet for more details.

Building self-esteem, easing the journey

The LSC aims to encourage students, build their self-esteem and set them on the road to independent learning. By having high expectations and developing positive working relationships we aim to maximise opportunities and success for our AEN students.

identified entities:
("entity"{tuple_delimiter}MILLFIELD{tuple_delimiter}SCHOOL{tuple_delimiter}Millfield is a school that supports students with individual GCSE subject choices and encourages independent learning through various support services.)
{record_delimiter}
("entity"{tuple_delimiter}LSC{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}The Learning Support Centre (LSC) provides support for AEN students, offering small group sessions and access to assistive technologies to aid independent study.)
{record_delimiter}
("entity"{tuple_delimiter}DRAGON NATURALLY SPEAKING{tuple_delimiter}FACILITY{tuple_delimiter}Dragon Naturally Speaking is a software available in the LSC to assist students with independent study through speech recognition.)
{record_delimiter}
("entity"{tuple_delimiter}READ AND WRITE 11{tuple_delimiter}FACILITY{tuple_delimiter}Read and Write 11 is a software tool provided in the LSC to support students in their independent learning.)
{record_delimiter}
("entity"{tuple_delimiter}INSPIRATION MIND MAPPING{tuple_delimiter}FACILITY{tuple_delimiter}Inspiration Mind Mapping is a software available in the LSC to help students organize their thoughts and ideas for independent study.)
{record_delimiter}
("entity"{tuple_delimiter}EXAM ACCESS ARRANGEMENTS{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}Exam Access Arrangements provide additional support for students during exams, including extra time and the use of assistive tools.)
{record_delimiter}
("entity"{tuple_delimiter}SPEECH AND LANGUAGE THERAPIST{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}A Speech and Language Therapist provides specialist 1:1 lessons for students needing additional support.)
{record_delimiter}
("entity"{tuple_delimiter}PERIPATETIC TEACHER{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}A peripatetic Teacher offers specialized 1:1 lessons for students requiring extra assistance in their studies.)
{record_delimiter}
------------------------
output:
("relationship"{tuple_delimiter}MILLFIELD{tuple_delimiter}SCHOOL{tuple_delimiter}LSC{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}The LSC is a support service within Millfield that aids AEN students in their learning journey)
{record_delimiter}
("relationship"{tuple_delimiter}LSC{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}DRAGON NATURALLY SPEAKING{tuple_delimiter}FACILITY{tuple_delimiter}Dragon Naturally Speaking is a facility used by the LSC to assist students with their studies)
{record_delimiter}
("relationship"{tuple_delimiter}LSC{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}READ AND WRITE 11{tuple_delimiter}FACILITY{tuple_delimiter}Read and Write 11 is a facility utilized by the LSC to support students in their independent learning)
{record_delimiter}
("relationship"{tuple_delimiter}LSC{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}INSPIRATION MIND MAPPING{tuple_delimiter}FACILITY{tuple_delimiter}Inspiration Mind Mapping is a facility provided by the LSC to help students organize their thoughts)
{record_delimiter}
("relationship"{tuple_delimiter}MILLFIELD{tuple_delimiter}SCHOOL{tuple_delimiter}EXAM ACCESS ARRANGEMENTS{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}Millfield provides Exam Access Arrangements to support students during their examinations)
{record_delimiter}
("relationship"{tuple_delimiter}LSC{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}SPEECH AND LANGUAGE THERAPIST{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}The LSC offers 1:1 lessons with a Speech and Language Therapist for students needing specialized support)
{record_delimiter}
("relationship"{tuple_delimiter}LSC{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}PERIPATETIC TEACHER{tuple_delimiter}SUPPORT SERVICE{tuple_delimiter}The LSC provides access to a peripatetic Teacher for specialized 1:1 lessons for students)
{completion_delimiter}
#############################



-Real Data-
######################
text: 
{input_text}

identified entities:
{identified_entities}
######################
output:


"""


CONTINUE_PROMPT_JSON = '''
Goal
"""
Given a document, output all entities with their types and descriptions.
And output all relationships between those entities.
Some entities and relationships were missed in the last extraction.
Remember to only emit entities and relationships that match any of the previously entities and relationships.
Output the results in JSON format.

"""
Previous entities and relationships
"""
{previous_entities_and_relationships}
"""

Document
"""
{input_text}
"""

Output:
'''

CONTINUE_PROMPT = '''
Goal
"""
Given a document, output all entities with their types and descriptions.
And output all relationships between those entities.
Some entities and relationships were missed in the last extraction.
Remember to only output entities and relationships that match any of the previously entities and relationships.
Output in the desired format, with blank lines separating distinct entities and relationships.
"""

Desired format:
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

Previous entities and relationships
"""
{previous_entities_and_relationships}
"""

Document
"""
{input_text}
"""

Output:
'''

ENTITY_CONTINUE_PROMPT = "MANY entities were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed.  Answer Y or N if there are still entities or relationships that need to be added.\n"

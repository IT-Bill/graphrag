from dataclasses import dataclass

@dataclass(frozen=True)
class Entity:
    name: str
    type: str
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialise the entity for JSON output."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
        }

@dataclass(frozen=True)
class Relationship:
    source: Entity
    target: Entity
    description: str

    def to_dict(self) -> dict[str, str]:
        """Serialise the relationship for JSON output."""
        return {
            "source": self.source.name,
            "source_type": self.source.type,
            "target": self.target.name,
            "target_type": self.target.type,
            "description": self.description,
        }

@dataclass
class ExtractResult:
    entities: list[Entity]
    relationships: list[Relationship]

    def fitness(self) -> float:  # TODO – replace with a real metric
        if not self.entities:
            return float("inf")
        return 1.0 / len(self.entities)

    def to_dict(self) -> dict[str, list[dict]]:
        """Serialise the extraction result for JSON output."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
        }


def parse_one_entity(raw: str) -> Entity:
    raw = raw.replace("```", "").strip()
    lines = raw.split("\n")
    if len(lines) < 3:
        raise ValueError("Entity description must have at least 3 lines.")
    name = lines[0].strip()
    type = lines[1].strip()
    description = "\n".join(lines[2:]).strip()
    return Entity(name=name, type=type, description=description)

def parse_entities(raw: str) -> ExtractResult:
    raw = raw.replace("```", "").strip()
    entity_str_list = raw.split("\n\n")
    entities = []
    for entity_str in entity_str_list:
        entity_str = entity_str.strip()
        name = entity_str.split("\n")[0].strip()
        type = entity_str.split("\n")[1].strip()
        description = "\n".join(entity_str.split("\n")[2:]).strip()
        entities.append(Entity(name=name, type=type, description=description))
    return ExtractResult(entities, [])

def parse_one_relationship(raw: str) -> Relationship:
    raw = raw.replace("```", "").strip()
    lines = raw.split("\n")
    if len(lines) < 5:
        raise ValueError("Relationship description must have at least 5 lines.")
    sname = lines[0].strip()
    stype = lines[1].strip()
    tname = lines[2].strip()
    ttype = lines[3].strip()
    description = "\n".join(lines[4:]).strip()
    return Relationship(
        source=Entity(name=sname, type=stype),
        target=Entity(name=tname, type=ttype),
        description=description,
    )

def parse_relationships(raw: str) -> list[Relationship]:
    raw = raw.replace("```", "").strip()
    relationship_str_list = raw.split("\n\n")
    relationships = []
    for relationship_str in relationship_str_list:
        relationship_str = relationship_str.strip()
        if not relationship_str:
            continue
        sname = relationship_str.split("\n")[0].strip()
        stype = relationship_str.split("\n")[1].strip()
        tname = relationship_str.split("\n")[2].strip()
        ttype = relationship_str.split("\n")[3].strip()
        description = "\n".join(relationship_str.split("\n")[4:]).strip()
        relationships.append(
            Relationship(
                source=Entity(name=sname, type=stype),
                target=Entity(name=tname, type=ttype),
                description=description,
            )
        )
    return relationships

def parse_graph(raw: str) -> ExtractResult:
    
    def _single_keyword(raw: str) -> str:
        raw = raw.replace("```", "")
        try:
            entities_str = raw.split("entities:")[1].split("relationships:")[0]
            rels_str = raw.split("relationships:")[1]
        except IndexError as e:
            raise ValueError("Missing 'entities:' or 'relationships:' section") from e

        entities = parse_entities(entities_str)
        relationships = parse_relationships(rels_str)
        return ExtractResult(entities=entities.entities, relationships=relationships)

    def _multi_keyword(raw: str) -> ExtractResult:
        raw = raw.replace("```", "")
        
        entities_str_list = []
        relationships_str_list = []
        
        for part in raw.split("\n\n"):
            part = part.strip()
            if not part:
                continue
            if part.startswith("entities:"):
                entities_str_list.append(part.split("entities:")[1].strip())
            elif part.startswith("relationships:"):
                relationships_str_list.append(part.split("relationships:")[1].strip())
        if not entities_str_list or not relationships_str_list:
            raise ValueError("Missing 'entities:' or 'relationships:' section")
        
        entities_str = "\n\n".join(entities_str_list)
        relationships_str = "\n\n".join(relationships_str_list)
        
        entities = parse_entities(entities_str)
        relationships = parse_relationships(relationships_str)
        
        return ExtractResult(entities=entities.entities, relationships=relationships)

    
    if raw.count("entities:") == 1 and raw.count("relationships:") == 1:
        return _single_keyword(raw)
    elif raw.count("entities:") > 1 or raw.count("relationships:") > 1:
        return _multi_keyword(raw)
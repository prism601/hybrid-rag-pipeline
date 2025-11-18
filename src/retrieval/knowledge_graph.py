"""
Knowledge Graph extraction and retrieval
"""
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import logging
import re

import networkx as nx
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from src.core.chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    text: str
    entity_type: str
    mentions: List[str]
    metadata: Dict[str, Any]


@dataclass
class Relation:
    """Represents a relation between entities"""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_chunk_id: str


class KnowledgeGraph:
    """Knowledge graph for structured information retrieval"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.chunk_entities: Dict[str, List[str]] = {}  # chunk_id -> entity_ids

    def add_entity(self, entity: Entity) -> None:
        """Add entity to graph"""
        entity_id = self._normalize_entity(entity.text)

        if entity_id not in self.entities:
            self.entities[entity_id] = entity
            self.graph.add_node(
                entity_id,
                text=entity.text,
                entity_type=entity.entity_type,
                metadata=entity.metadata
            )
        else:
            # Update mentions
            existing = self.entities[entity_id]
            existing.mentions.extend(entity.mentions)

    def add_relation(self, relation: Relation) -> None:
        """Add relation to graph"""
        subject_id = self._normalize_entity(relation.subject)
        object_id = self._normalize_entity(relation.object)

        # Ensure nodes exist
        if subject_id not in self.graph:
            self.add_entity(Entity(
                text=relation.subject,
                entity_type="UNKNOWN",
                mentions=[relation.subject],
                metadata={}
            ))

        if object_id not in self.graph:
            self.add_entity(Entity(
                text=relation.object,
                entity_type="UNKNOWN",
                mentions=[relation.object],
                metadata={}
            ))

        # Add edge
        self.graph.add_edge(
            subject_id,
            object_id,
            predicate=relation.predicate,
            confidence=relation.confidence,
            source_chunk_id=relation.source_chunk_id
        )

        self.relations.append(relation)

    def link_chunk_to_entities(self, chunk_id: str, entity_ids: List[str]) -> None:
        """Link a chunk to its entities"""
        self.chunk_entities[chunk_id] = entity_ids

    def get_related_entities(self, entity_text: str, max_hops: int = 2) -> List[str]:
        """Get entities related to given entity within max_hops"""
        entity_id = self._normalize_entity(entity_text)

        if entity_id not in self.graph:
            return []

        # BFS to find related entities
        related = set()
        visited = {entity_id}
        queue = [(entity_id, 0)]

        while queue:
            current, hops = queue.pop(0)

            if hops >= max_hops:
                continue

            # Get neighbors
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    related.add(neighbor)
                    visited.add(neighbor)
                    queue.append((neighbor, hops + 1))

            # Also check predecessors
            for predecessor in self.graph.predecessors(current):
                if predecessor not in visited:
                    related.add(predecessor)
                    visited.add(predecessor)
                    queue.append((predecessor, hops + 1))

        return list(related)

    def get_entity_chunks(self, entity_text: str) -> List[str]:
        """Get chunks containing the entity"""
        entity_id = self._normalize_entity(entity_text)
        chunk_ids = []

        for chunk_id, entities in self.chunk_entities.items():
            if entity_id in entities:
                chunk_ids.append(chunk_id)

        return chunk_ids

    def get_subgraph(self, entity_texts: List[str]) -> nx.MultiDiGraph:
        """Extract subgraph containing specified entities"""
        entity_ids = [self._normalize_entity(e) for e in entity_texts]
        existing_ids = [e for e in entity_ids if e in self.graph]

        if not existing_ids:
            return nx.MultiDiGraph()

        return self.graph.subgraph(existing_ids).copy()

    @staticmethod
    def _normalize_entity(text: str) -> str:
        """Normalize entity text to ID"""
        return text.lower().strip()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize knowledge graph"""
        return {
            'entities': {
                entity_id: {
                    'text': entity.text,
                    'type': entity.entity_type,
                    'mentions': entity.mentions,
                    'metadata': entity.metadata
                }
                for entity_id, entity in self.entities.items()
            },
            'relations': [
                {
                    'subject': r.subject,
                    'predicate': r.predicate,
                    'object': r.object,
                    'confidence': r.confidence,
                    'source_chunk_id': r.source_chunk_id
                }
                for r in self.relations
            ],
            'chunk_entities': self.chunk_entities
        }


class KnowledgeGraphExtractor:
    """Extract entities and relations from text"""

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy and SPACY_AVAILABLE

        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for NER")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self.use_spacy = False

    def extract_from_chunks(self, chunks: List[Chunk]) -> KnowledgeGraph:
        """Extract knowledge graph from chunks"""
        kg = KnowledgeGraph()

        for chunk in chunks:
            entities, relations = self.extract_from_text(chunk.text, chunk.chunk_id)

            # Add entities
            for entity in entities:
                kg.add_entity(entity)

            # Add relations
            for relation in relations:
                kg.add_relation(relation)

            # Link chunk to entities
            entity_ids = [kg._normalize_entity(e.text) for e in entities]
            kg.link_chunk_to_entities(chunk.chunk_id, entity_ids)

        logger.info(
            f"Extracted knowledge graph: {len(kg.entities)} entities, "
            f"{len(kg.relations)} relations"
        )

        return kg

    def extract_from_text(
        self,
        text: str,
        source_id: str = ""
    ) -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from text"""
        if self.use_spacy:
            return self._extract_with_spacy(text, source_id)
        else:
            return self._extract_with_patterns(text, source_id)

    def _extract_with_spacy(
        self,
        text: str,
        source_id: str
    ) -> Tuple[List[Entity], List[Relation]]:
        """Extract using spaCy NER"""
        doc = self.nlp(text)

        # Extract entities
        entities = []
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                entity_type=ent.label_,
                mentions=[ent.text],
                metadata={
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'source_id': source_id
                }
            )
            entities.append(entity)

        # Extract simple relations (subject-verb-object)
        relations = []
        for sent in doc.sents:
            sent_relations = self._extract_relations_from_sentence(sent, source_id)
            relations.extend(sent_relations)

        return entities, relations

    def _extract_relations_from_sentence(
        self,
        sent,
        source_id: str
    ) -> List[Relation]:
        """Extract relations from sentence using dependency parsing"""
        relations = []

        for token in sent:
            if token.pos_ == "VERB":
                # Find subject and object
                subject = None
                obj = None

                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child
                    elif child.dep_ in ("dobj", "pobj", "attr"):
                        obj = child

                if subject and obj:
                    relation = Relation(
                        subject=subject.text,
                        predicate=token.text,
                        object=obj.text,
                        confidence=0.7,
                        source_chunk_id=source_id
                    )
                    relations.append(relation)

        return relations

    def _extract_with_patterns(
        self,
        text: str,
        source_id: str
    ) -> Tuple[List[Entity], List[Relation]]:
        """Simple pattern-based extraction (fallback)"""
        entities = []
        relations = []

        # Extract capitalized phrases as potential entities
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)

        for match in matches:
            entity = Entity(
                text=match,
                entity_type="ENTITY",
                mentions=[match],
                metadata={'source_id': source_id}
            )
            entities.append(entity)

        # Extract simple patterns like "X is Y" or "X has Y"
        relation_patterns = [
            r'(\w+)\s+is\s+(?:a|an|the)\s+(\w+)',
            r'(\w+)\s+has\s+(?:a|an|the)?\s*(\w+)',
            r'(\w+)\s+contains\s+(\w+)',
        ]

        for pattern in relation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    relation = Relation(
                        subject=match[0],
                        predicate="related_to",
                        object=match[1],
                        confidence=0.5,
                        source_chunk_id=source_id
                    )
                    relations.append(relation)

        return entities, relations


class KnowledgeGraphRetriever:
    """Retrieve relevant chunks using knowledge graph"""

    def __init__(self, knowledge_graph: KnowledgeGraph, chunk_store: Dict[str, Chunk]):
        self.kg = knowledge_graph
        self.chunk_store = chunk_store

    def search(
        self,
        query: str,
        top_k: int = 10,
        max_hops: int = 2,
    ) -> List[Tuple[Chunk, float]]:
        """Search using knowledge graph"""
        # Extract entities from query
        extractor = KnowledgeGraphExtractor()
        query_entities, _ = extractor.extract_from_text(query)

        if not query_entities:
            return []

        # Find related entities
        all_related_entities = set()
        for entity in query_entities:
            related = self.kg.get_related_entities(entity.text, max_hops=max_hops)
            all_related_entities.update(related)

        # Get chunks containing these entities
        chunk_scores: Dict[str, float] = {}

        for entity_id in all_related_entities:
            chunk_ids = self.kg.get_entity_chunks(entity_id)

            for chunk_id in chunk_ids:
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id] += 1.0
                else:
                    chunk_scores[chunk_id] = 1.0

        # Sort by score
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Return chunks with scores
        results = []
        for chunk_id, score in sorted_chunks:
            if chunk_id in self.chunk_store:
                chunk = self.chunk_store[chunk_id]
                # Normalize score
                normalized_score = score / len(all_related_entities) if all_related_entities else 0
                results.append((chunk, normalized_score))

        return results

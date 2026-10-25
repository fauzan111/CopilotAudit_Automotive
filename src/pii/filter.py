"""PII anonymization filter for vehicle telemetry and technician notes.

Strips VIN, GPS coordinates, and driver identifiers before anything reaches
storage. Uses Presidio's PatternRecognizer/AnonymizerEngine directly rather
than Presidio's full AnalyzerEngine orchestration, because AnalyzerEngine
bootstraps a spaCy NLP pipeline for NER, and the available spaCy wheels on
this environment (Python 3.12, Windows) hit a binary-compatibility bug in
spacy.tokenizer.Tokenizer._reset_cache (OverflowError: Python int too large
to convert to C ssize_t) on every model, including a blank, data-free
pipeline. That's a lower-level Cython/ABI issue unrelated to this project's
code, not something worth working around for a regex-pattern use case.

All the fields this filter protects, VIN, GPS, driver ID, and name/email in
free text, are structurally regular enough to match with plain patterns, so
nothing here actually needed contextual NER in the first place.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from presidio_analyzer import Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# --- Pattern recognizers, run directly without an NLP pipeline ---

VIN_RECOGNIZER = PatternRecognizer(
    supported_entity="VEHICLE_VIN",
    patterns=[Pattern(name="vin_pattern", regex=r"\b[A-HJ-NPR-Z0-9]{17}\b", score=0.85)],
    context=["vin", "vehicle", "chassis"],
)

GPS_RECOGNIZER = PatternRecognizer(
    supported_entity="GPS_COORDINATES",
    patterns=[
        Pattern(
            name="gps_pattern",
            regex=r"-?\d{1,3}\.\d{3,8}\s*,\s*-?\d{1,3}\.\d{3,8}",
            score=0.9,
        )
    ],
    context=["gps", "location", "coordinates", "lat", "lon"],
)

DRIVER_ID_RECOGNIZER = PatternRecognizer(
    supported_entity="DRIVER_ID",
    patterns=[Pattern(name="driver_id_pattern", regex=r"\bDRV-\d{4,8}\b", score=0.9)],
    context=["driver", "operator", "technician"],
)

EMAIL_RECOGNIZER = PatternRecognizer(
    supported_entity="EMAIL_ADDRESS",
    patterns=[
        Pattern(
            name="email_pattern",
            regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            score=0.9,
        )
    ],
)

# Name detection without NER: matches two consecutive capitalized words.
# This is a deliberate, acknowledged trade-off, a regex can't disambiguate
# "Marco Rossi" (a name) from "Turin Italy" (not one) as reliably as true
# NER would. Good enough for this project's structured technician-log
# format; swapping in a working NLP engine later only touches this one
# recognizer, not the rest of the filter.
PERSON_RECOGNIZER = PatternRecognizer(
    supported_entity="PERSON",
    patterns=[
        Pattern(
            name="person_pattern",
            regex=r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b",
            score=0.4,
        )
    ],
    context=["technician", "driver", "name"],
    # PatternRecognizer defaults to case-insensitive matching (I|M|S), which
    # defeats a capitalization-based heuristic. Drop IGNORECASE explicitly.
    global_regex_flags=re.MULTILINE | re.DOTALL,
)

_RECOGNIZERS = [
    VIN_RECOGNIZER,
    GPS_RECOGNIZER,
    DRIVER_ID_RECOGNIZER,
    EMAIL_RECOGNIZER,
    PERSON_RECOGNIZER,
]


@dataclass
class FilterResult:
    """What the filter found and what it produced."""

    anonymized_text: str
    entities_found: list[str]
    vehicle_hash: str | None  # irreversible hash standing in for the real VIN


class TelemetryPIIFilter:
    """Strips VIN, GPS, driver-ID, email, and name fields; replaces VIN with
    a stable hash so records for the same vehicle can still be correlated
    without storing the real identifier anywhere.
    """

    def __init__(self) -> None:
        self._anonymizer = AnonymizerEngine()

    @staticmethod
    def hash_vin(vin: str) -> str:
        """One-way hash so the same vehicle can be tracked across readings
        without the real VIN ever being stored. Salted via a fixed app-level
        secret in production; left unsalted here since this is a portfolio
        demo, not a real deployment.
        """
        return hashlib.sha256(vin.encode("utf-8")).hexdigest()[:16]

    def _analyze(self, text: str) -> list:
        results = []
        for recognizer in _RECOGNIZERS:
            results.extend(
                recognizer.analyze(text, entities=[recognizer.supported_entities[0]], nlp_artifacts=None)
            )
        # Drop overlapping lower-score matches (e.g. PERSON regex catching
        # part of an email-adjacent capitalized word).
        results.sort(key=lambda r: (-r.score, r.start))
        kept, covered = [], set()
        for r in results:
            span = set(range(r.start, r.end))
            if span & covered:
                continue
            kept.append(r)
            covered |= span
        return kept

    def scan(self, text: str) -> list[str]:
        """Return the entity types found in text, without modifying it."""
        return sorted({r.entity_type for r in self._analyze(text)})

    def filter(self, text: str) -> FilterResult:
        """Anonymize text, extracting the VIN (if present) as a stable hash
        before it's replaced, so correlation by vehicle is still possible
        downstream.
        """
        results = self._analyze(text)

        vehicle_hash = None
        for r in results:
            if r.entity_type == "VEHICLE_VIN":
                raw_vin = text[r.start : r.end]
                vehicle_hash = self.hash_vin(raw_vin)
                break

        operators = {
            "VEHICLE_VIN": OperatorConfig("replace", {"new_value": "[VIN_REDACTED]"}),
            "GPS_COORDINATES": OperatorConfig("replace", {"new_value": "[GPS_REDACTED]"}),
            "DRIVER_ID": OperatorConfig("replace", {"new_value": "[DRIVER_REDACTED]"}),
            "PERSON": OperatorConfig("replace", {"new_value": "[NAME_REDACTED]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL_REDACTED]"}),
        }

        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)

        return FilterResult(
            anonymized_text=anonymized.text,
            entities_found=sorted({r.entity_type for r in results}),
            vehicle_hash=vehicle_hash,
        )

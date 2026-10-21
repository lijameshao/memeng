from enum import StrEnum


class Category(StrEnum):
    BUSINESS_LOGIC = "business_logic"
    ARCHITECTURE = "architecture"
    INTEGRATION = "integration"
    DATA_MODEL = "data_model"
    PROCESS = "process"
    CONSTRAINT = "constraint"


class SupersessionStatus(StrEnum):
    UNRESOLVED = "unresolved"
    SUPERSEDED = "superseded"
    RETAINED = "retained"


class CaptureOutcome(StrEnum):
    GATE_NO = "gate_no"
    NOTHING_EXTRACTED = "nothing_extracted"
    ALL_DUPLICATE = "all_duplicate"
    STORED = "stored"
    FAILED = "failed"


class VerdictTarget(StrEnum):
    FACT = "fact"
    CAPTURE = "capture"
    HIT = "hit"


class ConflictSource(StrEnum):
    EXTRACTOR = "extractor"
    COSINE = "cosine"

from .schemas import RoivDecisionMaker_v2
from ..logger import get_logger

log = get_logger(__name__)


def _normalize_name(name: str) -> str:
    return " ".join(name.lower().split())


def merge_persons(persons: list[RoivDecisionMaker_v2]) -> list[RoivDecisionMaker_v2]:
    """Deduplicate by person_full_name.

    First record wins for non-None fields; subsequent records with the same name
    only fill in fields that are still None.
    """
    groups: dict[str, dict] = {}
    order: list[str] = []

    for person in persons:
        key = _normalize_name(person.person_full_name)

        if key not in groups:
            groups[key] = person.model_dump()
            order.append(key)
        else:
            existing = groups[key]
            for field, value in person.model_dump().items():
                if existing.get(field) is None and value is not None:
                    existing[field] = value

    result = []
    for key in order:
        try:
            result.append(RoivDecisionMaker_v2.model_validate(groups[key]))
        except Exception as e:
            log.warning("Skipping '%s' after merge: %s", key, e)

    return result

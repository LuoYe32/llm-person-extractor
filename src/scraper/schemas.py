import re
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator, ValidationInfo


def _precise_pad(phone: str, area_code: str) -> str:
    """Prepend area code to short (local) phone numbers."""
    if not phone or not area_code:
        return phone
    digits = re.sub(r"\D", "", phone)
    if len(digits) <= 7:
        return f"{area_code}{phone}"
    return phone


def _validate_full_name(name: str) -> str:
    """Reject abbreviated or non-standard FIO formats.

    Rejects:
    - Single initials as separate words: «А. И. Фролов» (split → «А.» «И.» «Фролов»)
    - Compact multi-initial words: «Фролов А.И.» (split → «Фролов» «А.И.»)
    - Names with only 1 word.

    Accepts full names in any word order (we don't enforce Surname First).
    """
    stripped = name.strip()
    if not stripped:
        raise ValueError("person_full_name is empty")

    words = stripped.split()
    if len(words) < 2:
        raise ValueError(f"person_full_name too short (only 1 word): {stripped!r}")

    _single_initial = re.compile(r"^[А-ЯЁа-яёA-Za-z]\.?$")
    _compact_initials = re.compile(r"^([А-ЯЁа-яёA-Za-z]\.)+[А-ЯЁа-яёA-Za-z]?$")

    for word in words:
        w = word.strip(".")
        if _single_initial.match(word):
            raise ValueError(
                f"person_full_name looks like an abbreviated name (initials): {stripped!r}"
            )
        if _compact_initials.match(word):
            raise ValueError(
                f"person_full_name looks like an abbreviated name (initials): {stripped!r}"
            )

    return stripped


class RoivDecisionMaker_v2(BaseModel):
    """
    Сведения о конкретном должностном лице управленческого уровня,
    относящемся к региональному органу исполнительной власти (РОИВ)
    или его внутреннему структурному подразделению.
    """

    model_config = ConfigDict(revalidate_instances="always")

    person_full_name: str = Field(..., description="Полное ФИО: Фамилия Имя Отчество")
    roiv_full_name: str = Field(..., description="Наименование РОИВ без географической привязки субъекта РФ")
    position: str = Field(..., description="Управленческая должность, обрезанная до базовой роли без тематической части")
    person_email: Optional[str] = Field(None)
    person_phone: Optional[str] = Field(None)
    address: Optional[str] = Field(None)
    organization_email: Optional[str] = Field(None)
    organization_phone: Optional[str] = Field(None)
    division_name: Optional[str] = Field(None, description="Внутреннее подразделение РОИВ")
    photo_url: Optional[str] = Field(None)
    person_bio: Optional[str] = Field(None)
    date_birth: Optional[str] = Field(None, description="YYYY-MM-DD или YYYY, только при явном биографическом маркере")
    parsing_url: Optional[str] = Field(None, description="URL страницы, с которой извлечена запись")

    @model_validator(mode="before")
    @classmethod
    def check_full_name(cls, data):
        name = data.get("person_full_name", "") if isinstance(data, dict) else ""
        if name:
            _validate_full_name(name)
        return data

    @model_validator(mode="after")
    def fill_phones(self, context: ValidationInfo):
        if context.context and context.context.get("phone_area_code"):
            area = context.context["phone_area_code"]
            if self.person_phone is not None:
                self.person_phone = _precise_pad(self.person_phone, area)
            if self.organization_phone is not None:
                self.organization_phone = _precise_pad(self.organization_phone, area)
        return self


class PersonsOnPage(BaseModel):
    persons: list[RoivDecisionMaker_v2] = Field(default_factory=list)

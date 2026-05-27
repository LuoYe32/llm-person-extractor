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

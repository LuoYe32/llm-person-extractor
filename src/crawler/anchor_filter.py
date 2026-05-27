from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass
class ScoredLink:
    url: str
    anchor_texts: list[str]
    score: float   # > 0 — вероятно релевантная, < 0 — вероятно нерелевантная
    keep: bool


class AnchorTextFilter:
    """Scores discovered URLs by anchor text and URL path keywords.

    Does NOT fetch target pages — only uses text already seen in <a> tags
    and the URL path itself.
    """

    # anchor text substring → score  (abs value = confidence, sign = direction)
    ANCHOR_SCORES: dict[str, float] = {
        # ---- релевантные (сотрудники / контакты) ----
        "руководство": 0.9,
        "руководители": 0.9,
        "руководитель": 0.85,
        "заместитель": 0.7,
        "заместители": 0.7,
        "сотрудники": 0.9,
        "сотрудник": 0.85,
        "персонал": 0.85,
        "коллектив": 0.8,
        "команда": 0.7,
        "состав": 0.65,
        "структура": 0.55,
        "администрация": 0.8,
        "контакты": 0.75,
        "контактная информация": 0.8,
        "справочник": 0.85,
        "телефонный справочник": 0.95,
        "должностные лица": 0.9,
        "работники": 0.8,
        "специалисты": 0.7,
        "биография": 0.8,
        "о департаменте": 0.45,
        "о министерстве": 0.45,
        "об управлении": 0.45,
        "о комитете": 0.45,
        "об агентстве": 0.45,
        "о нас": 0.35,
        "об организации": 0.45,
        # English
        "staff": 0.9,
        "team": 0.75,
        "contacts": 0.75,
        "management": 0.9,
        "leadership": 0.9,
        "employees": 0.9,
        "personnel": 0.9,
        "directory": 0.85,
        "about": 0.25,
        # ---- нерелевантные ----
        "новости": -0.9,
        "новость": -0.85,
        "лента новостей": -0.95,
        "пресс-релиз": -0.9,
        "пресс-релизы": -0.9,
        "пресс-служба": -0.6,
        "приказ": -0.85,
        "приказы": -0.9,
        "распоряжение": -0.85,
        "распоряжения": -0.9,
        "постановление": -0.85,
        "постановления": -0.9,
        "нормативные акты": -0.9,
        "нормативно-правовые": -0.9,
        "законодательство": -0.9,
        "правовые акты": -0.9,
        "закупки": -0.9,
        "закупка": -0.85,
        "торги": -0.85,
        "аукцион": -0.9,
        "тендер": -0.9,
        "документы": -0.65,
        "отчеты": -0.85,
        "отчётность": -0.85,
        "отчёт": -0.8,
        "доклады": -0.8,
        "доклад": -0.75,
        "статистика": -0.75,
        "публикации": -0.65,
        "мероприятия": -0.8,
        "события": -0.7,
        "архив": -0.8,
        "поиск": -0.85,
        "карта сайта": -0.9,
        "версия для слабовидящих": -0.95,
        "галерея": -0.8,
        "фотогалерея": -0.85,
        "видео": -0.75,
        "вакансии": -0.5,
        "обращения граждан": -0.8,
        "госуслуги": -0.8,
        "услуги": -0.65,
        "форум": -0.8,
        "рейтинги": -0.7,
        "конкурсы": -0.7,
        # English
        "news": -0.9,
        "press": -0.8,
        "events": -0.75,
        "calendar": -0.8,
        "search": -0.85,
        "sitemap": -0.95,
        "policy": -0.7,
        "tenders": -0.9,
        "procurement": -0.9,
        "archive": -0.8,
        "gallery": -0.8,
        "vacancies": -0.5,
    }

    # URL path substring → score
    URL_PATH_SCORES: dict[str, float] = {
        # ---- хорошие ----
        "rukovodstvo": 0.9,
        "sotrudniki": 0.9,
        "sotrud": 0.8,
        "personal": 0.8,
        "kontakty": 0.8,
        "kontakt": 0.7,
        "spravochnik": 0.85,
        "administraciya": 0.8,
        "administr": 0.7,
        "persons": 0.8,
        "person": 0.65,
        "workers": 0.8,
        "staff": 0.9,
        "team": 0.75,
        "management": 0.9,
        "employees": 0.9,
        "contacts": 0.75,
        "directory": 0.85,
        "leadership": 0.9,
        "about": 0.25,
        # ---- плохие ----
        "novosti": -0.9,
        "novost": -0.85,
        "news": -0.9,
        "press": -0.8,
        "geroi": -0.8,
        "prikazy": -0.85,
        "prikaz": -0.8,
        "zakupki": -0.9,
        "zakupka": -0.85,
        "tender": -0.85,
        "document": -0.7,
        "normativ": -0.8,
        "otchet": -0.8,
        "archive": -0.8,
        "arkhiv": -0.8,
        "gallery": -0.8,
        "galery": -0.8,
        "video": -0.75,
        "search": -0.8,
        "sitemap": -0.95,
        "uslugi": -0.65,
        "services": -0.6,
        "events": -0.75,
        "meropriyatiya": -0.8,
    }

    def __init__(self, threshold: float = -0.3):
        # Links with score >= threshold are kept.
        # Default -0.3: discard only pages with confident negative signal.
        self.threshold = threshold

    def _score_anchor(self, anchor_text: str) -> float:
        """Return the highest-magnitude score matched in anchor_text."""
        text = anchor_text.lower().strip()
        if not text:
            return 0.0
        best = 0.0
        for keyword, score in self.ANCHOR_SCORES.items():
            if keyword in text and abs(score) > abs(best):
                best = score
        return best

    def _score_url(self, url: str) -> float:
        """Return the highest-magnitude score matched in URL path."""
        path = urlparse(url).path.lower()
        best = 0.0
        for keyword, score in self.URL_PATH_SCORES.items():
            if keyword in path and abs(score) > abs(best):
                best = score
        return best

    def score_link(self, url: str, anchor_texts: list[str]) -> float:
        """Combine URL-path and anchor-text signals into a single score."""
        url_score = self._score_url(url)
        anchor_scores = [self._score_anchor(t) for t in anchor_texts if t]

        if not anchor_scores:
            return url_score

        best_anchor = max(anchor_scores, key=abs)

        # Prefer whichever signal is stronger
        return url_score if abs(url_score) >= abs(best_anchor) else best_anchor

    def filter(self, url_to_anchors: dict[str, list[str]]) -> list[ScoredLink]:
        """Score and filter all discovered URLs.

        Returns list sorted by score descending (best candidates first).
        """
        result: list[ScoredLink] = []
        for url, anchors in url_to_anchors.items():
            score = self.score_link(url, anchors)
            result.append(ScoredLink(
                url=url,
                anchor_texts=anchors,
                score=score,
                keep=score >= self.threshold,
            ))
        return sorted(result, key=lambda x: x.score, reverse=True)

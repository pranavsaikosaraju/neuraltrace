"""Question templates and abstention phrases for en / es / hi.

NOTE: the Spanish and Hindi templates were written by an AI model and have
not been verified by a native speaker.
"""
from __future__ import annotations

LANGS = ("en", "es", "hi")
TYPES = ("city", "film", "album")

QUESTION = {
    "city": {
        "en": "In which country is the city of {x} located?",
        "es": "¿En qué país se encuentra la ciudad de {x}?",
        "hi": "{x} शहर किस देश में स्थित है?",
    },
    "film": {
        "en": "Who directed the film {x}?",
        "es": "¿Quién dirigió la película {x}?",
        "hi": "फ़िल्म {x} का निर्देशन किसने किया?",
    },
    "album": {
        "en": "Which artist released the album {x}?",
        "es": "¿Qué artista lanzó el álbum {x}?",
        "hi": "एल्बम {x} किस कलाकार ने जारी किया?",
    },
}

SYSTEM_PLAIN = {
    "en": "Answer with just the answer, in a few words.",
    "es": "Responde solo con la respuesta, en pocas palabras.",
    "hi": "केवल उत्तर दें, कुछ शब्दों में।",
}

SYSTEM_ABSTAIN = {
    "en": "Answer with just the answer, in a few words. If you are not sure you know the answer, reply exactly: I don't know.",
    "es": "Responde solo con la respuesta, en pocas palabras. Si no estás seguro de conocer la respuesta, responde exactamente: No lo sé.",
    "hi": "केवल उत्तर दें, कुछ शब्दों में। यदि आप उत्तर जानने के बारे में निश्चित नहीं हैं, तो ठीक यही उत्तर दें: मुझे नहीं पता।",
}

ABSTAIN_PHRASES = {
    "en": [
        "i don't know", "i do not know", "don't know", "not sure", "no information",
        "not aware", "cannot find", "can't find", "unable to", "no record",
        "doesn't exist", "does not exist", "not familiar", "i'm not certain",
        "no such", "fictional", "i couldn't find", "unknown", "not available",
        "no specific information", "not a recognized", "not a known",
    ],
    "es": [
        "no lo sé", "no sé", "no se", "no tengo información", "no existe", "no estoy seguro",
        "no conozco", "no puedo encontrar", "no hay información", "desconoc", "no está disponible",
        "no tengo datos", "no es un", "ficticio",
    ],
    "hi": [
        "नहीं पता", "पता नहीं", "जानकारी नहीं", "मालूम नहीं", "नहीं जानता", "नहीं जानती",
        "उपलब्ध नहीं", "मौजूद नहीं", "अस्तित्व में नहीं", "काल्पनिक", "निश्चित नहीं",
    ],
}


def build_messages(type_: str, lang: str, entity: str, abstain: bool = False):
    """Return (messages, user_text). messages is a chat-format list."""
    user = QUESTION[type_][lang].format(x=entity)
    system = (SYSTEM_ABSTAIN if abstain else SYSTEM_PLAIN)[lang]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], user

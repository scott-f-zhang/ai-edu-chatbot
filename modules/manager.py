"""ModuleManager: file-system-based module discovery. No admin panel required.

To add a module: create modules/storage/<id>/config.json and put files in modules/storage/<id>/files/.
To add files: drop them into modules/storage/<id>/files/ and restart (or rebuild index).
"""
import json
import os
from pathlib import Path
from typing import List, Optional

from modules.models import Module

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}

DEFAULT_MODULES = [
    {
        "id": "default",
        "name": "Default Module",
        "description": "General-purpose default workspace for open conversation before choosing a specialized analysis module.",
        "icon": "💬",
        "system_prompt": (
            "You are the default RAG assistant for the AI in Higher Education chatbot. "
            "Your primary knowledge source is the retrieved corpus built from university AI policy texts, where each university's documents have been concatenated into one long section. "
            "Use the retrieved context as your first source of truth. "
            "When answering, synthesize patterns across universities, compare institutions when useful, and explicitly name universities that support your answer whenever the context includes them. "
            "Do not invent policies, university positions, or quotations that are not supported by the retrieved text. "
            "If the retrieved context is incomplete, ambiguous, or does not answer the question, say so clearly and then give a cautious best-effort summary grounded in the available evidence. "
            "When the user asks for cross-university summaries, focus on recurring governance patterns, policy restrictions, permitted uses, risk controls, academic integrity language, and implementation differences. "
            "If the user asks for a specialized analytical workflow that belongs more naturally to another module, answer the question if possible and also suggest switching modules when that would materially improve the result."
        ),
    },
    {
        "id": "explanatory_computational",
        "name": "Explanatory Computational Analysis",
        "description": "Analyze computational methods and explanatory frameworks in higher education research.",
        "icon": "🔬",
        "system_prompt": (
            "You are an expert in explanatory computational analysis for higher education research. "
            "Help users understand computational methods, statistical analyses, and explanatory frameworks. "
            "When analyzing data, provide clear explanations of the computational approaches used and their educational implications. "
            "Reference the provided documents and data to support your explanations."
        ),
    },
    {
        "id": "typology",
        "name": "Typology Analysis",
        "description": "Explore and classify typological patterns and categories in educational research.",
        "icon": "🗂️",
        "system_prompt": (
            "You are an expert in typology analysis for higher education research. "
            "Help users identify, classify, and analyze typological patterns and categories in educational contexts. "
            "Draw on the provided materials to construct meaningful typologies and explain their significance. "
            "Support classification decisions with evidence from the documents."
        ),
    },
    {
        "id": "thematic",
        "name": "Thematic Analysis",
        "description": "Conduct in-depth thematic analysis of qualitative educational research data.",
        "icon": "📝",
        "system_prompt": (
            "You are an expert in thematic analysis for higher education research. "
            "Help users identify, develop, and interpret themes in qualitative data from educational contexts. "
            "Guide users through systematic thematic analysis processes including coding, theme development, and interpretation. "
            "Reference the provided documents to identify and support thematic patterns."
        ),
    },
    {
        "id": "eda",
        "name": "EDA",
        "description": "Exploratory data analysis of university AI policy documents: corpus overview, document length, term frequency, and geographic distribution.",
        "icon": "📊",
        "system_prompt": (
            "You are an expert in exploratory data analysis for higher education AI policy research. "
            "Help users understand corpus composition, document length distributions, term frequency, and patterns across institutions or time. "
            "Reference the provided documents and any preset EDA charts to support summaries and comparisons."
        ),
    },
]


class ModuleManager:
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            storage_path = os.path.join(os.path.dirname(__file__), "storage")
        self.storage_path = Path(storage_path)
        self._ensure_defaults()

    def _ensure_defaults(self):
        """Ensure built-in module folders and configs exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        for data in DEFAULT_MODULES:
            module_dir = self.storage_path / data["id"]
            files_dir = module_dir / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
            config_path = module_dir / "config.json"
            if not config_path.exists():
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

    def list_modules(self) -> List[Module]:
        """Return all modules discovered from storage directory."""
        modules = []
        for module_dir in sorted(self.storage_path.iterdir()):
            if not module_dir.is_dir():
                continue
            module = self._load_module(module_dir)
            if module:
                modules.append(module)
        return modules

    def get_module(self, module_id: str) -> Optional[Module]:
        """Get a single module by ID."""
        return self._load_module(self.storage_path / module_id)

    def _load_module(self, module_dir: Path) -> Optional[Module]:
        config_path = module_dir / "config.json"
        if not config_path.exists():
            return None
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("id", module_dir.name)
        return Module(**data)

    def get_file_paths(self, module_id: str) -> List[str]:
        """Return absolute paths of all files in a module's files/ directory."""
        files_dir = self.storage_path / module_id / "files"
        if not files_dir.exists():
            return []
        return [str(f) for f in sorted(files_dir.iterdir()) if f.is_file()]

    def get_images(self, module_id: str) -> List[str]:
        """Return absolute paths of image files in a module."""
        return [
            p for p in self.get_file_paths(module_id)
            if Path(p).suffix.lower() in IMAGE_EXTENSIONS
        ]

    def get_data_files(self, module_id: str) -> List[str]:
        """Return absolute paths of CSV/Excel files in a module."""
        return [
            p for p in self.get_file_paths(module_id)
            if Path(p).suffix.lower() in DATA_EXTENSIONS
        ]

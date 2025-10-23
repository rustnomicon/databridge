import json
from typing import Optional, Dict, Any

from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit, QTextEdit, QHBoxLayout, QPushButton


class FilterDialog(QDialog):
    def __init__(self, initial: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Фильтры колонки")
        # Инициализируем rules с новыми полями по умолчанию
        self.rules = initial.copy() if initial else {
            "trim": True,
            "to_string": False,
            "to_integer": False,
            "digits_only": False,
            "normalize_phone": False,
            "lower": False,
            "upper": False,
            "format_date": False,
            "remove_chars": [],
            "regex_remove": [],
            "regex_replace": []
        }
        lay = QFormLayout(self)
        self.ed_remove_chars = QLineEdit(",".join(self.rules.get("remove_chars", [])))
        self.ed_regex_remove = QTextEdit("\n".join(self.rules.get("regex_remove", [])))
        self.ed_regex_replace = QTextEdit(json.dumps(self.rules.get("regex_replace", []), ensure_ascii=False, indent=2))
        self.chk_trim = self._check("trim", True)
        self.chk_to_string = self._check("to_string", False)
        self.chk_to_int = self._check("to_integer", False)
        self.chk_digits = self._check("digits_only", False)
        self.chk_normalize_phone = self._check("normalize_phone", False)
        self.chk_lower = self._check("lower", False)
        self.chk_upper = self._check("upper", False)
        self.chk_date = self._check("format_date", False)

        lay.addRow("Удалить символы (через запятую):", self.ed_remove_chars)
        lay.addRow("Regex удалить (по одному на строку):", self.ed_regex_remove)
        lay.addRow("Regex replace (JSON):", self.ed_regex_replace)
        lay.addRow("Trim", self.chk_trim)
        lay.addRow("К строке (str)", self.chk_to_string)
        lay.addRow("К числу (int) (Digits only auto)", self.chk_to_int)
        lay.addRow("Digits only", self.chk_digits)
        lay.addRow("Нормализация телефона (RU)", self.chk_normalize_phone)
        lay.addRow("lower()", self.chk_lower)
        lay.addRow("UPPER()", self.chk_upper)
        lay.addRow("Дата YYYY-MM-DD", self.chk_date)

        buttons = QHBoxLayout()
        ok = QPushButton("OK")
        cancel = QPushButton("Отмена")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(ok)
        buttons.addWidget(cancel)
        lay.addRow(buttons)

    def _check(self, key: str, default: bool):
        cb = QPushButton()
        cb.setCheckable(True)
        cb.setChecked(self.rules.get(key, default))
        cb.setText("✓" if cb.isChecked() else " ")
        cb.toggled.connect(lambda v, b=cb: b.setText("✓" if v else " "))
        cb.setFixedWidth(30)
        return cb

    def get_rules(self) -> Dict[str, Any]:
        rules = {
            "trim": self.chk_trim.isChecked(),
            "to_string": self.chk_to_string.isChecked(),
            "to_integer": self.chk_to_int.isChecked(),
            "digits_only": self.chk_digits.isChecked(),
            "normalize_phone": self.chk_normalize_phone.isChecked(),
            "lower": self.chk_lower.isChecked(),
            "upper": self.chk_upper.isChecked(),
            "format_date": self.chk_date.isChecked()
        }
        remove_chars = [x.strip() for x in self.ed_remove_chars.text().split(",") if x.strip()]
        if remove_chars:
            rules["remove_chars"] = remove_chars
        regex_remove = [ln.strip() for ln in self.ed_regex_remove.toPlainText().splitlines() if ln.strip()]
        if regex_remove:
            rules["regex_remove"] = regex_remove
        try:
            regex_replace = json.loads(self.ed_regex_replace.toPlainText() or "[]")
            if isinstance(regex_replace, list):
                rules["regex_replace"] = regex_replace
        except Exception:
            pass
        return rules

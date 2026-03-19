"""
Query decomposition — split compound queries into parts.

"What is gravity? Also, who discovered it?"
→ ["What is gravity", "who discovered it"]

Splits on natural language boundaries: question marks,
conjunctions followed by question words, semicolons.
"""
import re


class Decomposition:

    SPLIT_PATTERN = re.compile(
        r'\?\s*'
        r'|\.?\s*(?:Also|Then|Next|Additionally|Furthermore|Second|Finally|Now)[,\s]+'
        r'|(?<![DdMmSsJj][rRrsS])\.\s+(?=[A-Z])'
        r'|,?\s+and\s+(?:also\s+)?(?:what|who|how|why|where|when|explain|tell|name|list|write|translate|compare)'
        r'|,?\s+also\s+(?:what|who|how|why|where|when|explain|tell)'
        r'|,?\s+then\s+(?:what|who|how|why|explain|tell)'
        r'|;\s*',
        re.IGNORECASE
    )

    def split(self, query):
        """Split a compound query into sub-queries."""
        parts = self.SPLIT_PATTERN.split(query)
        clean = [p.strip().strip('.,;') for p in parts if len(p.strip()) > 5]
        return clean if len(clean) > 1 else [query]

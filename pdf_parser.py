"""PDF text extraction and multi-period financial label parsing."""
import re
import pdfplumber
import pandas as pd

LABELS = {
    "Revenue": [r"\btotal revenue\b", r"\btotal net revenue\b", r"\brevenue\b", r"\bnet sales\b", r"\bsales revenue\b", r"\bsales\b"],
    "Net Profit": [r"\bnet profit\b", r"\bnet income\b", r"\bprofit after tax\b", r"\bprofit for the year\b", r"\bnet loss\b"],
    "Operating Expenses": [r"\btotal operating expenses?\b", r"\boperating expenses?\b"],
    "Current Assets": [r"\btotal current assets\b", r"\bcurrent assets\b"],
    "Current Liabilities": [r"\btotal current liabilities\b", r"\bcurrent liabilities\b"],
    "Total Debt": [r"\btotal debt\b", r"\btotal borrowings\b", r"\bborrowings\b"],
    "Shareholders Equity": [r"\bshareholders'? equity\b", r"\bstockholders'? equity\b", r"\btotal equity\b"],
    "Total Assets": [r"\btotal assets\b"],
    "Total Liabilities": [r"\btotal liabilities\b"],
    "Operating Cash Flow": [r"\bnet cash (?:provided by|from) operating activities\b", r"\bcash flow from operating activities\b", r"\boperating cash flow\b"],
}

def extract_text(pdf_file) -> str:
    chunks = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            chunks.append(page.extract_text() or "")
    return "\n".join(chunks)

def _parse_number(raw: str):
    raw = raw.strip().replace(",", "").replace("₹", "").replace("$", "").replace("€", "")
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()")
    try:
        value = float(raw)
        return -value if negative else value
    except ValueError:
        return None

def parse_financial_values(text: str):
    """Parse label/value lines while retaining the most recent year/period heading."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = []
    current_period = "Latest"
    period_re = re.compile(r"(?:financial summary|financial results|year ended|year ending|for the year|fiscal year|annual report|period ended|summary)[^\d]{0,30}(20\d{2})", re.I)
    year_re = re.compile(r"\b(20\d{2})\b")

    for line in lines:
        period_match = period_re.search(line)
        if period_match:
            current_period = period_match.group(1)
        elif re.fullmatch(r"(?:FY\s*)?(20\d{2})(?:\s*(?:FY|annual))?", line, re.I):
            current_period = year_re.search(line).group(1)

        matched_metric = None
        for metric, patterns in LABELS.items():
            if any(re.search(pattern, line, re.I) for pattern in patterns):
                matched_metric = metric
                break
        if not matched_metric:
            continue

        # If a year is on a metric row, use the last year on that line.
        years_on_line = year_re.findall(line)
        row_period = years_on_line[-1] if years_on_line else current_period
        # Avoid treating years in headings as financial values.
        candidates = re.findall(r"\(?-?\d[\d,]*(?:\.\d+)?\)?", line)
        # Remove years from numeric candidates when they appear in the same line.
        candidates = [c for c in candidates if not (len(c.replace(",", "")) == 4 and c.replace(",", "").startswith("20"))]
        if not candidates:
            continue
        value = _parse_number(candidates[-1])
        if value is not None:
            rows.append({"Metric": matched_metric, "Value": value, "Period": row_period})

    if not rows:
        return pd.DataFrame(columns=["Metric", "Value", "Period"])
    result = pd.DataFrame(rows, columns=["Metric", "Value", "Period"])
    # If duplicate label-period pairs occur, keep the last occurrence but expose it for review.
    result = result.drop_duplicates(subset=["Metric", "Period"], keep="last").reset_index(drop=True)
    return result

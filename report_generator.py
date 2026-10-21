"""Generate a downloadable PDF analysis report."""
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def create_pdf_report(data, ratios, changes, insights):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story = [Paragraph("AI-Based Financial Report Analysis", styles["Title"]),
             Paragraph("Automated educational analysis — verify all extracted values against the source report.", styles["Italic"]),
             Spacer(1, 14), Paragraph("Key insights", styles["Heading2"])]
    for item in insights:
        story.append(Paragraph("• " + item, styles["BodyText"]))
        story.append(Spacer(1, 5))
    story.append(Spacer(1, 8))
    for title, frame in [("Extracted financial data", data), ("Financial ratios", ratios), ("Period-over-period changes", changes)]:
        story.append(Paragraph(title, styles["Heading2"]))
        if frame is None or frame.empty:
            story.append(Paragraph("No data available.", styles["BodyText"]))
        else:
            table_data = [list(map(str, frame.columns))]
            for _, row in frame.head(35).iterrows():
                table_data.append([("" if str(v) == "nan" else str(round(v, 3) if isinstance(v, (int, float)) else v)) for v in row.tolist()])
            widths = [max(55, (A4[0]-84)/len(table_data[0]))] * len(table_data[0])
            t = Table(table_data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#183153")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 7),
                ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(t)
        story.append(Spacer(1, 12))
    story.append(Paragraph("Disclaimer: This output is for educational purposes only and is not accounting, audit, legal, or investment advice.", styles["Small"]))
    doc.build(story)
    return buffer.getvalue()

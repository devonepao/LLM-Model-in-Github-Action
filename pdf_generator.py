"""
PDF generation module for creating reports with screenshots and summaries.
"""
import os
from datetime import datetime
from typing import List, Dict
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from PIL import Image as PILImage


class PDFReportGenerator:
    """Generate PDF reports with screenshots and text summaries."""
    
    def __init__(self, filename: str = "navigation_report.pdf"):
        self.filename = filename
        self.doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a73e8'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#5f6368'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # Step header style
        self.styles.add(ParagraphStyle(
            name='StepHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#202124'),
            spaceAfter=8,
            spaceBefore=16
        ))
    
    def add_title(self, title: str):
        """Add main title to the report."""
        self.story.append(Paragraph(title, self.styles['CustomTitle']))
        self.story.append(Spacer(1, 0.2 * inch))
    
    def add_subtitle(self, subtitle: str):
        """Add subtitle to the report."""
        self.story.append(Paragraph(subtitle, self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.1 * inch))
    
    def add_metadata(self, metadata: Dict):
        """Add metadata table to the report."""
        data = []
        for key, value in metadata.items():
            data.append([key, str(value)])
        
        table = Table(data, colWidths=[2 * inch, 4 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0fe')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.3 * inch))
    
    def add_step(self, step_number: int, step_data: Dict):
        """Add a step section to the report."""
        # Step header
        header_text = f"Step {step_number}: {step_data.get('action', 'Unknown').replace('_', ' ').title()}"
        self.story.append(Paragraph(header_text, self.styles['StepHeader']))
        
        # Step details
        details = []
        for key, value in step_data.items():
            if key not in ['action', 'timestamp'] and value:
                details.append(f"<b>{key.replace('_', ' ').title()}:</b> {value}")
        
        if details:
            details_text = "<br/>".join(details)
            self.story.append(Paragraph(details_text, self.styles['Normal']))
            self.story.append(Spacer(1, 0.1 * inch))
        
        # Timestamp
        if 'timestamp' in step_data:
            timestamp_text = f"<i>Timestamp: {step_data['timestamp']}</i>"
            self.story.append(Paragraph(timestamp_text, self.styles['Normal']))
            self.story.append(Spacer(1, 0.1 * inch))
    
    def add_screenshot(self, screenshot_bytes: bytes, caption: str = ""):
        """Add a screenshot to the report."""
        try:
            # Load image and resize if needed
            img = PILImage.open(BytesIO(screenshot_bytes))
            
            # Calculate dimensions to fit on page
            max_width = 6.5 * inch
            max_height = 8 * inch
            
            img_width, img_height = img.size
            aspect_ratio = img_width / img_height
            
            if img_width > max_width * 2:  # assuming 72 dpi
                new_width = max_width
                new_height = new_width / aspect_ratio
            else:
                new_width = img_width / 2
                new_height = img_height / 2
            
            # Limit height
            if new_height > max_height:
                new_height = max_height
                new_width = new_height * aspect_ratio
            
            # Create Image flowable
            img_buffer = BytesIO(screenshot_bytes)
            img_flowable = Image(img_buffer, width=new_width, height=new_height)
            self.story.append(img_flowable)
            
            # Add caption if provided
            if caption:
                caption_style = ParagraphStyle(
                    name='Caption',
                    parent=self.styles['Normal'],
                    fontSize=9,
                    textColor=colors.HexColor('#5f6368'),
                    alignment=TA_CENTER,
                    spaceAfter=12
                )
                self.story.append(Spacer(1, 0.05 * inch))
                self.story.append(Paragraph(caption, caption_style))
            
            self.story.append(Spacer(1, 0.2 * inch))
            
        except Exception as e:
            error_text = f"<i>Error loading screenshot: {str(e)}</i>"
            self.story.append(Paragraph(error_text, self.styles['Normal']))
            self.story.append(Spacer(1, 0.1 * inch))
    
    def add_text_section(self, heading: str, content: str):
        """Add a text section with heading."""
        self.story.append(Paragraph(heading, self.styles['StepHeader']))
        
        # Split long content into paragraphs
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Escape special characters for reportlab
                para_escaped = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                self.story.append(Paragraph(para_escaped, self.styles['Normal']))
                self.story.append(Spacer(1, 0.1 * inch))
    
    def add_page_break(self):
        """Add a page break."""
        self.story.append(PageBreak())
    
    def build(self):
        """Build and save the PDF."""
        try:
            self.doc.build(self.story)
            print(f"PDF report generated: {self.filename}")
            return self.filename
        except Exception as e:
            print(f"Error generating PDF: {e}")
            raise


def create_navigation_report(
    steps: List[Dict],
    screenshots: List[bytes],
    summary: str,
    original_prompt: str,
    output_file: str = "navigation_report.pdf"
) -> str:
    """
    Create a complete navigation report PDF.
    
    Args:
        steps: List of navigation steps
        screenshots: List of screenshot bytes
        summary: Text summary of the navigation
        original_prompt: The original user prompt
        output_file: Output PDF filename
    
    Returns:
        Path to generated PDF file
    """
    pdf = PDFReportGenerator(output_file)
    
    # Add title
    pdf.add_title("Web Navigation Report")
    
    # Add metadata
    metadata = {
        "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Original Prompt": original_prompt,
        "Total Steps": len(steps),
        "Screenshots": len(screenshots)
    }
    pdf.add_metadata(metadata)
    
    # Add steps with screenshots
    for idx, step in enumerate(steps, 1):
        pdf.add_step(idx, step)
        
        # Add corresponding screenshot if available
        if idx <= len(screenshots):
            caption = f"Screenshot from step {idx}"
            pdf.add_screenshot(screenshots[idx - 1], caption)
        
        # Add page break after each step except the last
        if idx < len(steps):
            pdf.add_page_break()
    
    # Add summary section
    if summary:
        pdf.add_page_break()
        pdf.add_text_section("Summary", summary)
    
    # Build PDF
    return pdf.build()

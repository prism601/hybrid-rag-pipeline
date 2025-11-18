"""
Multi-modal document loaders for PDF, HTML, Markdown, and other formats
"""
import io
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# PDF processing
import pypdf
import pdfplumber

# HTML processing
from bs4 import BeautifulSoup

# Markdown
import markdown

# Document format
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document with metadata"""
    content: str
    metadata: Dict[str, Any]
    source: str
    doc_type: str
    page_number: Optional[int] = None

    def __str__(self) -> str:
        return f"Document(source={self.source}, type={self.doc_type}, length={len(self.content)})"


class BaseDocumentLoader(ABC):
    """Base class for document loaders"""

    @abstractmethod
    def load(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[Document]:
        """Load documents from source"""
        pass

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Check if loader supports this file type"""
        pass


class PDFLoader(BaseDocumentLoader):
    """Advanced PDF loader with multiple extraction strategies"""

    def __init__(self, extract_images: bool = False, use_layout: bool = True):
        self.extract_images = extract_images
        self.use_layout = use_layout

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')

    def load(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[Document]:
        """Load PDF with enhanced text extraction"""
        documents = []

        try:
            # Use pdfplumber for better layout preservation
            if self.use_layout:
                documents = self._load_with_pdfplumber(source)
            else:
                documents = self._load_with_pypdf(source)

            logger.info(f"Loaded {len(documents)} pages from PDF")
            return documents

        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise

    def _load_with_pdfplumber(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[Document]:
        """Load using pdfplumber for better layout"""
        documents = []

        with pdfplumber.open(source) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                # Extract tables if present
                tables = page.extract_tables()
                table_text = ""
                if tables:
                    for table in tables:
                        table_text += "\n" + self._format_table(table)

                metadata = {
                    "page_number": page_num,
                    "total_pages": len(pdf.pages),
                    "has_tables": len(tables) > 0,
                    "width": page.width,
                    "height": page.height,
                }

                documents.append(Document(
                    content=text + table_text,
                    metadata=metadata,
                    source=str(source),
                    doc_type="pdf",
                    page_number=page_num
                ))

        return documents

    def _load_with_pypdf(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[Document]:
        """Fallback loader using pypdf"""
        documents = []

        with open(source, 'rb') as file:
            pdf = pypdf.PdfReader(file)

            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()

                metadata = {
                    "page_number": page_num,
                    "total_pages": len(pdf.pages),
                }

                documents.append(Document(
                    content=text,
                    metadata=metadata,
                    source=str(source),
                    doc_type="pdf",
                    page_number=page_num
                ))

        return documents

    @staticmethod
    def _format_table(table: List[List[Any]]) -> str:
        """Format table as markdown"""
        if not table:
            return ""

        lines = []
        for row in table:
            line = " | ".join(str(cell) if cell else "" for cell in row)
            lines.append(f"| {line} |")

        return "\n".join(lines)


class HTMLLoader(BaseDocumentLoader):
    """HTML document loader with cleaning"""

    def __init__(self, remove_scripts: bool = True, remove_styles: bool = True):
        self.remove_scripts = remove_scripts
        self.remove_styles = remove_styles

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.html', '.htm'))

    def load(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[Document]:
        """Load and clean HTML content"""
        try:
            if isinstance(source, (str, Path)):
                with open(source, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            else:
                html_content = source.read()
                if isinstance(html_content, bytes):
                    html_content = html_content.decode('utf-8')

            soup = BeautifulSoup(html_content, 'lxml')

            # Remove unwanted elements
            if self.remove_scripts:
                for script in soup.find_all('script'):
                    script.decompose()

            if self.remove_styles:
                for style in soup.find_all('style'):
                    style.decompose()

            # Extract text
            text = soup.get_text(separator='\n', strip=True)

            # Extract metadata
            title = soup.title.string if soup.title else ""
            meta_desc = ""
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag:
                meta_desc = meta_tag.get('content', '')

            metadata = {
                "title": title,
                "description": meta_desc,
                "num_links": len(soup.find_all('a')),
                "num_images": len(soup.find_all('img')),
            }

            return [Document(
                content=text,
                metadata=metadata,
                source=str(source),
                doc_type="html"
            )]

        except Exception as e:
            logger.error(f"Error loading HTML: {e}")
            raise


class MarkdownLoader(BaseDocumentLoader):
    """Markdown document loader"""

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.md', '.markdown'))

    def load(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[Document]:
        """Load markdown and optionally convert to HTML"""
        try:
            if isinstance(source, (str, Path)):
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = source.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')

            # Parse markdown to extract structure
            html_content = markdown.markdown(
                content,
                extensions=['meta', 'tables', 'fenced_code', 'toc']
            )

            # Extract plain text from HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)

            # Count headers
            num_headers = len([line for line in content.split('\n') if line.startswith('#')])
            num_code_blocks = content.count('```')

            metadata = {
                "num_headers": num_headers,
                "num_code_blocks": num_code_blocks // 2,
                "length": len(content),
            }

            return [Document(
                content=text,
                metadata=metadata,
                source=str(source),
                doc_type="markdown"
            )]

        except Exception as e:
            logger.error(f"Error loading Markdown: {e}")
            raise


class DocxLoader(BaseDocumentLoader):
    """Microsoft Word document loader"""

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith('.docx')

    def load(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[Document]:
        """Load DOCX file"""
        try:
            doc = DocxDocument(source)

            # Extract paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            content = '\n\n'.join(paragraphs)

            # Extract tables
            tables_text = []
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(' | '.join(row_data))
                tables_text.append('\n'.join(table_data))

            if tables_text:
                content += '\n\n' + '\n\n'.join(tables_text)

            metadata = {
                "num_paragraphs": len(paragraphs),
                "num_tables": len(doc.tables),
                "num_sections": len(doc.sections),
            }

            return [Document(
                content=content,
                metadata=metadata,
                source=str(source),
                doc_type="docx"
            )]

        except Exception as e:
            logger.error(f"Error loading DOCX: {e}")
            raise


class DocumentLoaderFactory:
    """Factory for creating appropriate document loaders"""

    def __init__(self):
        self.loaders: List[BaseDocumentLoader] = [
            PDFLoader(),
            HTMLLoader(),
            MarkdownLoader(),
            DocxLoader(),
        ]

    def get_loader(self, file_path: str) -> Optional[BaseDocumentLoader]:
        """Get appropriate loader for file type"""
        for loader in self.loaders:
            if loader.supports(file_path):
                return loader
        return None

    def load_document(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[Document]:
        """Load document using appropriate loader"""
        if isinstance(source, (str, Path)):
            file_path = str(source)
        else:
            raise ValueError("Must provide file path for loader selection")

        loader = self.get_loader(file_path)
        if loader is None:
            raise ValueError(f"No loader found for file: {file_path}")

        return loader.load(source)

    def load_documents(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Load multiple documents"""
        all_documents = []

        for source in sources:
            try:
                documents = self.load_document(source)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to load {source}: {e}")

        return all_documents

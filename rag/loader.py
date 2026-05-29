import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _extract_tables(file_path: str) -> dict:
    tables_by_page = {}
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            if not page_tables:
                continue
            blocks = []
            for table in page_tables:
                rows = []
                for j, row in enumerate(table):
                    cells = [str(c or "").strip() for c in row]
                    rows.append("| " + " | ".join(cells) + " |")
                    if j == 0:
                        rows.append("| " + " | ".join("---" for _ in cells) + " |")
                blocks.append("\n".join(rows))
            tables_by_page[i] = "\n\n".join(blocks)
    return tables_by_page


def load_and_split(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    tables_by_page = _extract_tables(file_path)

    for doc in documents:
        page_num = doc.metadata.get("page", 0)
        table_text = tables_by_page.get(page_num, "")
        if table_text:
            doc.page_content = doc.page_content + "\n\n" + table_text

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

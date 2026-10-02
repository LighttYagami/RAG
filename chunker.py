from ingest import documents
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_sections(html_content, source):
    sections = []

    current_h1 = None
    current_h2 = None
    current_h3 = None
    current_content = []

    def save_section():
        if not current_content:
            return

        # Build the hierarchy path
        path = []

        if current_h1:
            path.append(current_h1)

        if current_h2:
            path.append(current_h2)

        if current_h3:
            path.append(current_h3)

        sections.append(
            Document(
                page_content="\n".join(current_content),
                metadata={
                    "source": source,
                    "section_path": " > ".join(path),
                    "h1": current_h1,
                    "h2": current_h2,
                    "h3": current_h3,
                }
            )
        )

    for element in html_content.find_all(
        ["h1", "h2", "h3", "p", "li"]
    ):

        # New top-level section
        if element.name == "h1":

            save_section()

            current_h1 = element.get_text(
                " ",
                strip=True
            )

            current_h2 = None
            current_h3 = None
            current_content = []

        # New subsection
        elif element.name == "h2":

            save_section()

            current_h2 = element.get_text(
                " ",
                strip=True
            )

            current_h3 = None
            current_content = []

        # New sub-subsection
        elif element.name == "h3":

            save_section()

            current_h3 = element.get_text(
                " ",
                strip=True
            )

            current_content = []

        # Normal content
        else:

            text = element.get_text(
                " ",
                strip=True
            )

            if text:
                current_content.append(text)

    # Save final section
    save_section()

    return sections


# --------------------------------------------------
# Extract structured sections from all documents
# --------------------------------------------------

all_sections = []

for doc in documents:

    sections = extract_sections(
        doc["content"],
        doc["source"]
    )

    all_sections.extend(sections)


print(f"Created {len(all_sections)} structured sections")


# --------------------------------------------------
# Split oversized sections
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

final_chunks = []

for section in all_sections:

    if len(section.page_content) <= 1000:

        final_chunks.append(section)

    else:

        chunks = splitter.split_documents([section])
        final_chunks.extend(chunks)


print(f"Created {len(final_chunks)} final chunks")


# --------------------------------------------------
# Inspect final chunks
# --------------------------------------------------

for i, chunk in enumerate(final_chunks[:10]):

    print("\n" + "=" * 70)
    print(f"CHUNK {i + 1}")
    print("=" * 70)

    print("Source:", chunk.metadata["source"])
    print("Section:", chunk.metadata["section_path"])
    print("Characters:", len(chunk.page_content))

    print("\nContent:")
    print(chunk.page_content[:])
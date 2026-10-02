import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

URLS = [
    "https://handbook.gitlab.com/handbook/people-policies/",
    "https://handbook.gitlab.com/handbook/people-group/acceptable-use-policy/",
    "https://handbook.gitlab.com/handbook/people-group/contracts-probation-periods/",
    "https://handbook.gitlab.com/handbook/people-group/people-compliance/",
]

documents = []

for url in URLS:

    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract ONLY the main article content
    main = soup.find(
        "main",
        class_="col-12 col-md-9 col-xl-8 ps-md-5"
    )

    if main is None:
        print(f"Could not find main content: {url}")
        continue

    # Remove things that aren't useful for RAG
    for tag in main.find_all(["script", "style", "nav", "div"]):
        if tag.name == "div":
            if "taxonomy-terms-article" in tag.get("class", []):
                tag.decompose()

        elif tag.name in ["script", "style", "nav"]:
            tag.decompose()
            
    documents.append(
    {
        "content": main,
        "source": url
    }
)

print(f"Loaded {len(documents)} documents")

for i, doc in enumerate(documents):

    print("\n" + "=" * 70)
    print(f"DOCUMENT {i + 1}")
    print("=" * 70)

    print("Source:", doc["source"])
    print("Content tag:", doc["content"].name)
    print("Content class:", doc["content"].get("class"))
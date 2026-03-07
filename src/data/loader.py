from datasets import load_dataset
from typing import Optional

DEFAULT_SIMPLE_CONFIG = "20231101.simple"
REPO_ID = "wikimedia/wikipedia"

def load_wikipedia_simple(limit: Optional[int] = 1000, config: str = DEFAULT_SIMPLE_CONFIG):
    """
    Load Simple English Wikipedia.
    """
    ds = load_dataset(REPO_ID, config, split="train")

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return ds

if __name__ == "__main__":
    ds = load_wikipedia_simple(limit=5)
    for item in ds:
        print(f"Title: {item['title']}")
        print(f"Text: {item['text'][:200]}...")  # Print first 200 characters
        print("-" * 40)
        
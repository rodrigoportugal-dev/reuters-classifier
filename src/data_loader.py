import os
import glob
from bs4 import BeautifulSoup

def load_reuters_dataset(data_dir):
    """
    Reads the .sgm files from the Reuters-21578 corpus.
    Returns the texts and labels separated into Train and Test according to LEWISSPLIT.
    """
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    # Search for all .sgm files in your unzipped folder
    sgm_files = glob.glob(os.path.join(data_dir, "*.sgm"))
    
    if not sgm_files:
        raise FileNotFoundError(f"No .sgm files found in {data_dir}")

    for file_path in sgm_files:
        # The corpus usually has special characters, latin-1 prevents reading crashes
        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
            content = f.read()

        # Parse the SGML using BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all documents within the file
        documents = soup.find_all('reuters')

        for doc in documents:
            # --- 1. REQUIREMENT: Train / Test split ---
            lewis_split = doc.get('lewissplit', '').upper()
            if lewis_split not in ["TRAIN", "TEST"]:
                continue  # Ignore documents that are not Train or Test (e.g., "NOT-USED")

            # --- 2. REQUIREMENT: Only the first class ---
            topics_tag = doc.find('topics')
            if not topics_tag:
                continue
                
            # Look for <d> tags inside <topics>
            d_tags = topics_tag.find_all('d')
            if not d_tags:
                continue  # Ignore documents without a category
                
            # Strictly take the first element, e.g.: <d>money-fx</d> -> "money-fx"
            first_label = d_tags[0].text.strip()

            # 3. Text extraction 
            text_node = doc.find('text')
            if not text_node:
                continue

            # The useful text is usually divided between the title and the body
            title = text_node.find('title').text.strip() if text_node.find('title') else ""
            body = text_node.find('body').text.strip() if text_node.find('body') else ""
            
            full_text = f"{title} {body}".strip()
            if not full_text:
                continue

            # Assign to the corresponding lists
            if lewis_split == "TRAIN":
                train_texts.append(full_text)
                train_labels.append(first_label)
            elif lewis_split == "TEST":
                test_texts.append(full_text)
                test_labels.append(first_label)

    return train_texts, train_labels, test_texts, test_labels

# --- Usage example ---
if __name__ == "__main__":
    # Assuming you extracted the zip into a folder called 'reuters_data'
    data_path = "./reuters_data" 
    
    X_train, y_train, X_test, y_test = load_reuters_dataset(data_path)
    
    print(f"Training documents: {len(X_train)}")
    print(f"Test documents: {len(X_test)}")
    print(f"Example of extracted class: {y_train[0]}")
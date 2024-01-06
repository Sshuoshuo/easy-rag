import os
from typing import List
import re
import tqdm

from langchain.schema import Document
import spacy
import PyPDF2


def extract_page_text(filepath, max_len=256, overlap_len=100):
    page_content  = []
    # spliter = spacy.load("zh_core_web_sm")
    # chunks = []
    with open(filepath, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        page_count = 0
        # pattern = r'^\d{1,3}'
        for page in tqdm.tqdm(pdf_reader.pages):
            page_text = page.extract_text().strip()
            raw_text = [text.strip() for text in page_text.split('\n')]
            new_text = '\n'.join(raw_text)
            new_text = re.sub(r'\n\d{2,3}\s?', '\n', new_text)
            # new_text = re.sub(pattern, '', new_text).strip()
            if len(new_text)>10 and '..............' not in new_text:
                page_content.append(new_text)


    cleaned_chunks = []
    i = 0
    all_str = ''.join(page_content)
    all_str = all_str.replace('\n', '')
    while i<len(all_str):
        cur_s = all_str[i:i+max_len]
        if len(cur_s)>10:
            cleaned_chunks.append(Document(page_content=cur_s, metadata={'page':page_count+1}))
        i+=(max_len - overlap_len)

    return cleaned_chunks

import os
import re

# M.MEETARBHAN
# 7/23/2024
# RAG Chain for Large PDF Queries

# TODO : Implement auto check for PDF of > len(8000)




# FUNCTIONS ======================================================================

#split_text(text) chunks a text file by splitting with each new paragraph.
# TODO : Explore different chunking strategies. Chunking by paragraph is posisble but 
# inefficient with retrieval.

def split_text(text: str):
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i != ""]

chunked_text = split_text(text=pdf_text)






# ================================================================================
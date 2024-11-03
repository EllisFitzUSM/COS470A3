Sentence-Transformers Information Retrieval
=

Using SBERT Sentence-Transformers Bi-Encoder and Cross-Encoder, this project aims to retrieve
relevant answers to Travel Stack Exchange questions. 

Pretrained Models Used:
-
- **cross-encoder/ms-marco-MiniLM-L-6-v2**
- **sentence-transformers/multi-qa-MiniLM-L6-cos-v1**

How to run?:
-
- First, install dependency packages.


    C:\> pip install -r requirements.txt

- Then run the program with this pattern:


    C:\>python __main__.py path/to/answers/json path/to/qrel path/to/topics
**This gives us the documents to search over, the queries to administer, and the QREL for Train, Validation, and Testing**. *Topics file and QREL file MUST correlate*

- You can append MORE topics files AFTER these arguments


    C:\>python __main__.py answers qrel topics path/to/more/topics . . . path/to/topics/N

*ATM there is no loading fine-tuned models from memory despite the folders/files present. So unless you want to code the sanity check for the existence of the directory and the model, Happy Training!*

Enjoy
=




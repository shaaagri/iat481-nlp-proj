python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
rmdir chroma_db /s /q
.venv\Scripts\python app.py vectorize ..\data\BetterSleep_QnADataset.csv
pause

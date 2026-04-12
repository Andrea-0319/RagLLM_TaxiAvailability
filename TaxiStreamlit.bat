@echo off
set PATH=C:\Users\andre\AppData\Local\Programs\Python\Python311;%PATH%
cd /d "%~dp0"
python -m streamlit run llm_tool\StreamlitRania\app.py --server.folderWatchball false
pause
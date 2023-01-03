# run gradle for grobid
cd /mnt/d/Projects/cs420-SciBERTSUm/grobid
./gradlew run

# another terminal
conda activate zalo
cd /mnt/d/Projects/cs420-SciBERTSUm/src
streamlit run app.py

# BT4013 Project Team QuQu
**1. Create conda environment**
conda create -n bt4013_ququ_env python=3.7

**2. Activate environment**
activate bt4013_ququ_env

**3. Install dependencies**
pip install -r requirements.txt

**4. Run main**
python main.py

**Additional:**
Comment out clean() in main.py when downloading new tickerData
Uncomment clean() again to standardize the headers in tickerData. Sometimes it is ' CLOSE' vs 'CLOSE'
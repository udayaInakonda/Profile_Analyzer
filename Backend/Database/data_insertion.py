import os
import mysql.connector

# MySQL connection details
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Amzur@#123',
    'database': 'profile_analyzer'
}

folder_path = r'C:\Users\UdayaI\Desktop\OOF\Profile_Analyzer\Backend\resumes'  # Use raw string for Windows path

# Connect to MySQL
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Create table if not exists
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS resumes (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     path VARCHAR(1024) NOT NULL
# )
# ''')

# Insert file paths into the table
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        cursor.execute(
            'INSERT INTO resume (resume_address) VALUES (%s)',
            (file_path,)
        )

conn.commit()
cursor.close()
conn.close()

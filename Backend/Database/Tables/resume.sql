-- USE profile_analyzer;

CREATE TABLE IF NOT EXISTS resume (
    id INT AUTO_INCREMENT PRIMARY KEY,
    resume_address VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'applied' NOT NULL
);

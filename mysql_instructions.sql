-- MySQL helper: create DB & user (run in mysql shell) --
CREATE DATABASE IF NOT EXISTS dubai_analytics;
CREATE USER IF NOT EXISTS 'rag_user'@'localhost' IDENTIFIED BY 'your_password_here';
GRANT ALL PRIVILEGES ON dubai_analytics.* TO 'rag_user'@'localhost';
FLUSH PRIVILEGES;

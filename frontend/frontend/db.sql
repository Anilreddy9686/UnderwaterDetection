-- Safe, repeatable schema for the underwater detection application.
-- This file intentionally does not drop or erase an existing database.
CREATE DATABASE IF NOT EXISTS under_water
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE under_water;

CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(32) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS detections (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NULL,
    source_type VARCHAR(24) NOT NULL,
    source_file VARCHAR(255) NOT NULL,
    model_file VARCHAR(255) NOT NULL,
    detection_count INT NOT NULL DEFAULT 0,
    detections_json LONGTEXT NOT NULL,
    output_file VARCHAR(255) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_detections_user_date (user_id, created_at),
    CONSTRAINT fk_detections_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NULL,
    action VARCHAR(100) NOT NULL,
    details VARCHAR(1000) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_audit_user_date (user_id, created_at)
);

CREATE TABLE IF NOT EXISTS app_settings (
    setting_key VARCHAR(64) PRIMARY KEY,
    setting_value VARCHAR(64) NOT NULL,
    updated_by INT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- To grant administrator access, register the account first, then configure
-- ADMIN_EMAIL in the environment before starting the app. Startup promotes it.

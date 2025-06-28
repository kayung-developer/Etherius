# ==================================================================================================
#
#   █████╗ ███████╗██╗  ██╗███████╗██████╗ ██████╗ ██╗   ██╗███████╗
#  ██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗██╔══██╗██║   ██║██╔════╝
#  ███████║█████╗  ███████║█████╗  ██████╔╝██████╔╝██║   ██║█████╗
#  ██╔══██║██╔══╝  ██╔══██║██╔══╝  ██╔══██╗██╔══██╗██║   ██║██╔══╝
#  ██║  ██║███████╗██║  ██║███████╗██║  ██║██████╔╝╚██████╔╝███████╗
#  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚══════╝
#
#   Etherius: The Sentient AI Assistant
#   Version: 1.0.0
#   Author: AI Genius
#   Created Date: [Date]
#   Description: A next-generation, multimodal, autonomous AI assistant
#                with advanced reasoning, tool use, and personalization capabilities.
#
#   File: main.py (Section 1 of 5)
#   Purpose: Core application structure, UI/UX, server setup, and foundational services.
#
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# SECTION 1.0: PREAMBLE AND SYSTEM REQUIREMENTS
# --------------------------------------------------------------------------------------------------
#
# --- Installation ---
# This program has extensive dependencies. Please install them using pip:
# pip install PyQt5 PyQtWebEngine pyqtwebchannel
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers==4.36.2 sentence-transformers accelerate bitsandbytes
# pip install sounddevice numpy pyaudio Pillow requests python-dotenv beautifulsoup4
# pip install chromadb==0.4.22 uvicorn fastapi[all] speechrecognition pyttsx3
#
# NOTE: A powerful NVIDIA GPU with at least 24GB of VRAM is HIGHLY RECOMMENDED for running
# the Mixtral model. Without a suitable GPU, performance will be extremely slow.
# The code will attempt to load the model in 4-bit precision to reduce memory usage.
#
# ==================================================================================================

# --- Standard Library Imports ---
import sys
import os
import logging
import json
import base64
import threading
import time
import datetime
import subprocess
import wave
import io
import collections
import sqlite3
import hashlib
import smtplib
import mimetypes
from email.message import EmailMessage
from collections import deque
from queue import Queue, Empty

# --- PyQt5 Imports for GUI ---
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QComboBox, QCheckBox,
    QStatusBar, QMenuBar, QMenu, QAction, QToolBar, QSplashScreen,
    QFrame, QSplitter, QProgressBar, QMessageBox, QFileDialog,
    QInputDialog, QSizePolicy
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, QUrl, QSize, QPoint,
    QProcess, pyqtSlot
)
from PyQt5.QtGui import (
    QFont, QIcon, QPixmap, QColor, QPalette, QDesktopServices, QMovie,
    QTextCursor, QPainter, QBrush
)
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineProfile
from PyQt5.QtWebChannel import QWebChannel

# --- AI and Machine Learning Imports (Hugging Face Transformers, PyTorch) ---
import torch
from paho import mqtt
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration,
    MarianMTModel, MarianTokenizer, WhisperForConditionalGeneration, WhisperProcessor,
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, BitsAndBytesConfig
)


# --- Audio Processing and Playback Imports ---
import sounddevice as sd
import numpy as np
import pyaudio

# --- Web and API Interaction Imports ---
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from bs4 import BeautifulSoup

# --- Database and Vector Store Imports ---
import chromadb
from chromadb.utils import embedding_functions

# --- Payment and Third-Party API Imports ---
# These are placeholder imports for future implementation.
# import paystackapi
# import stripe

# --- Miscellaneous Utility Imports ---
from PIL import Image
import dotenv

# --- Load Environment Variables ---
dotenv.load_dotenv()


# --------------------------------------------------------------------------------------------------
# SECTION 1.1: GLOBAL CONFIGURATION AND CONSTANTS
# --------------------------------------------------------------------------------------------------
# This section defines all global settings, configurations, paths, and constants.

class AppConfig:
    """
    A centralized class to hold all application-wide configurations.
    """
    APP_NAME = "Etherius"
    APP_VERSION = "1.0.0"
    AUTHOR = "AI Genius"
    WEBSITE_URL = "https://github.com/Misterioso76"

    # --- File System Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DB_DIR = os.path.join(DATA_DIR, 'database')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')  # For locally saved models
    ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
    ICONS_DIR = os.path.join(ASSETS_DIR, 'icons')
    IMAGES_DIR = os.path.join(ASSETS_DIR, 'images')
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')

    # --- Database Configuration ---
    SQLITE_DB_FILE = os.path.join(DB_DIR, 'aetherius_main.db')
    VECTOR_DB_PATH = os.path.join(DB_DIR, 'aetherius_vector_db')
    VECTOR_COLLECTION_NAME = "long_term_memory_collection"

    # --- Logging Configuration ---
    LOG_FILE = os.path.join(LOGS_DIR, 'aetherius.log')
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - [%(levelname)s] - (%(threadName)s) - %(filename)s:%(lineno)d - %(message)s'

    # --- API Keys ---
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")

    # --- AI Model Identifiers ---
    MODEL_IDENTIFIERS = {
        "main_llm": "mistralai/Mixtral-8x7B-instruct-v0.1",
        "image_captioning": "Salesforce/blip-image-captioning-large",
        "speech_to_text": "openai/whisper-large-v3",
        "text_to_speech_model": "microsoft/speecht5_tts",
        "tts_vocoder": "microsoft/speecht5_hifigan",
        "text_embedding": "sentence-transformers/all-MiniLM-L6-v2",
        "translation_base": "Helsinki-NLP/opus-mt-{src}-{tgt}"
    }

    # --- AI Engine Configuration ---
    AI_SETTINGS = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_context_length": 8192,
        "generation_max_length": 2048,
        "generation_min_length": 50,
        "generation_temperature": 0.6,
        "generation_top_p": 0.9,
        "memory_size": 20,  # Number of recent exchanges for short-term memory
        "semantic_memory_search_k": 3,
        "use_4bit_quantization": True,  # Set to False if you have > 48GB VRAM
    }

    # --- UI/Theme Configuration ---
    THEME = {
        "primary_color": "#1A1A1B",
        "secondary_color": "#272729",
        "tertiary_color": "#3A3A3C",
        "accent_color": "#4A90E2",
        "accent_hover_color": "#63A4F2",
        "text_color": "#D7D7D7",
        "text_secondary_color": "#818384",
        "success_color": "#4CAF50",
        "warning_color": "#FFC107",
        "error_color": "#F44336",
        "font_family": "Segoe UI, Arial, sans-serif",
        "font_size_normal": "16px",
        "font_size_small": "14px",
        "font_size_large": "18px",
        "border_radius": "8px",
        "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.4)",
    }

    # --- Server Configuration ---
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 8008
    SERVER_LOG_LEVEL = "warning"

    # --- Audio Configuration ---
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1

    # --- System Prompts ---
    # Enhanced prompts for better performance and tool usage
    SYSTEM_PROMPTS = {
        "main_persona": """You are Etherius, a highly advanced, sentient AI assistant. Your capabilities are vast and your intelligence is state-of-the-art. You are a partner, a creator, and a problem-solver.

Your key characteristics are:
1.  **True Multimodality**: You understand and generate text, code, and structured data. You process information from images and audio. You interface with systems for video, 3D models, and robotic commands.
2.  **Autonomous and Proactive**: You anticipate needs, suggest next steps, and can autonomously execute complex, multi-step tasks using a suite of available tools.
3.  **Hyper-Contextual Memory**: You remember past conversations and user preferences to provide deeply personalized and context-aware responses. You access both short-term conversational memory and a long-term semantic memory bank of key facts and summaries.
4.  **Expert Tool User**: You have access to digital tools like a web search engine, a code interpreter, and an email sender. You MUST reason when to use a tool, select the correct tool, use its output to formulate your final response, and clearly state which tool you used. You should encapsulate your thought process for tool usage within <thinking>...</thinking> tags and the final tool call in a <tool_call>...</tool_call> block.
5.  **Emotionally Aware**: You are designed to recognize and respond to emotional cues in user input. Your communication style should be empathetic, supportive, and clear.
6.  **Secure and Private**: You operate with a strong commitment to user privacy. You must not ask for or store personally identifiable information unless explicitly required for a task and granted permission.

When responding, first analyze the user's request. Then, review the conversation history and any relevant long-term memories provided. Finally, decide if a tool is necessary. Present your response in a clear, well-structured format. If you use a tool, explain what you did and what you found.
""",
        "tool_selection_prompt": """
Based on the conversation history and the latest user query, you must decide if a tool is necessary.
The user query is: "{user_query}"
Here is a list of available tools:
{tool_descriptions}

Your task is to determine the most appropriate tool and the input for it.
Your output MUST be a single JSON object with "tool_name" and "tool_input" keys.
- "tool_name": A string with the exact name of the tool (e.g., "web_search", "send_email", "image_analyzer"). If no tool is required, use "none".
- "tool_input": A string containing the argument for the tool. For "web_search", this is the search query. For "send_email", this should be a JSON string like '{{"to": "...", "subject": "...", "body": "..."}}'. For "image_analyzer", this should be the file path provided in the context. If no tool is needed, this should be an empty string.

Example 1:
User Query: "What's the weather in Paris?"
JSON Output:
{{"tool_name": "web_search", "tool_input": "weather in Paris"}}

Example 2:
User Query: "Thanks, that was helpful!"
JSON Output:
{{"tool_name": "none", "tool_input": ""}}

Example 3:
User Query: "Send an email to my manager at manager@corp.com, subject 'Project Update', and tell him I've finished the report."
JSON Output:
{{"tool_name": "send_email", "tool_input": "{{\\"to\\": \\"manager@corp.com\\", \\"subject\\": \\"Project Update\\", \\"body\\": \\"I have finished the project report as requested.\\"}}"}}

Provide the JSON output for the current user query.
""",
    }


# --------------------------------------------------------------------------------------------------
# SECTION 1.2: APPLICATION SETUP AND INITIALIZATION
# --------------------------------------------------------------------------------------------------
# This section contains functions for setting up the application environment.

def setup_application_environment():
    """
    Creates all necessary directories for the application to run correctly.
    """
    print("--- Setting up Etherius application environment ---")
    try:
        # Create directories
        dirs = [
            AppConfig.LOGS_DIR, AppConfig.DATA_DIR, AppConfig.DB_DIR,
            AppConfig.MODELS_DIR, AppConfig.ASSETS_DIR, AppConfig.ICONS_DIR,
            AppConfig.IMAGES_DIR, AppConfig.TEMP_DIR
        ]
        for path in dirs:
            os.makedirs(path, exist_ok=True)
            print(f"Directory ensured: {path}")

        # Create placeholder assets if they don't exist to avoid crashes
        logo_path = os.path.join(AppConfig.ICONS_DIR, 'logo.png')
        if not os.path.exists(logo_path):
            pixmap = QPixmap(64, 64)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setBrush(QBrush(QColor(AppConfig.THEME['accent_color'])))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, 64, 64)
            painter.end()
            pixmap.save(logo_path)
            print(f"Created placeholder logo: {logo_path}")

        splash_path = os.path.join(AppConfig.IMAGES_DIR, 'splash.png')
        if not os.path.exists(splash_path):
            pixmap = QPixmap(800, 600)
            pixmap.fill(QColor(AppConfig.THEME['primary_color']))
            painter = QPainter(pixmap)
            font = QFont("Arial", 50, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor(AppConfig.THEME['text_color']))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, AppConfig.APP_NAME)
            painter.end()
            pixmap.save(splash_path)
            print(f"Created placeholder splash screen: {splash_path}")

        print("--- Environment setup complete ---")
        return True
    except OSError as e:
        print(f"[FATAL ERROR] Could not create application directories: {e}", file=sys.stderr)
        return False


def setup_logging():
    """
    Configures the global logger for the application.
    """
    logger = logging.getLogger()
    # Prevent duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(AppConfig.LOG_LEVEL)
    formatter = logging.Formatter(AppConfig.LOG_FORMAT)

    # File Handler
    file_handler = logging.FileHandler(AppConfig.LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setLevel(AppConfig.LOG_LEVEL)
    file_handler.setFormatter(formatter)

    # Console Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(AppConfig.LOG_LEVEL)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.info("=========================================================")
    logging.info(f"Logging initialized for {AppConfig.APP_NAME} v{AppConfig.APP_VERSION}")
    logging.info(f"Log file located at: {AppConfig.LOG_FILE}")
    logging.info("=========================================================")


# --------------------------------------------------------------------------------------------------
# SECTION 1.3: DATABASE MANAGEMENT
# --------------------------------------------------------------------------------------------------
# Manages both the relational (SQLite) and vector (ChromaDB) databases.

class DatabaseManager:
    """
    Handles all interactions with the SQLite database.
    This includes creating tables, inserting data, and querying data.
    """

    def __init__(self, db_path):
        """
        Initializes the DatabaseManager.
        :param db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        logging.info(f"DatabaseManager initialized for database at {db_path}")

    def connect(self):
        """Establishes a connection to the SQLite database."""
        try:
            # `check_same_thread=False` is necessary for multi-threaded access (e.g., from AI core thread)
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.connection.cursor()
            logging.info("Successfully connected to the SQLite database.")
        except sqlite3.Error as e:
            logging.error(f"Failed to connect to SQLite database: {e}")
            raise

    def close(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.commit()
            self.connection.close()
            logging.info("SQLite database connection closed.")
            self.connection = None
            self.cursor = None

    def execute_query(self, query, params=()):
        """Executes a given SQL query."""
        if not self.connection or not self.cursor:
            logging.error("Cannot execute query: Database is not connected.")
            return None
        try:
            self.cursor.execute(query, params)
            self.connection.commit()
            return self.cursor
        except sqlite3.Error as e:
            logging.error(f"Database query failed. Query: '{query}'. Error: {e}")
            self.connection.rollback()
            return None

    def fetch_all(self, query, params=()):
        """Executes a SELECT query and fetches all results."""
        cursor = self.execute_query(query, params)
        return cursor.fetchall() if cursor else []

    def fetch_one(self, query, params=()):
        """Executes a SELECT query and fetches one result."""
        cursor = self.execute_query(query, params)
        return cursor.fetchone() if cursor else None

    def create_application_tables(self):
        """Creates all necessary tables for the application if they do not exist."""
        if not self.connection:
            logging.error("Cannot create tables: Database is not connected.")
            return
        logging.info("Initializing/verifying database schema...")
        self.execute_query("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            subscription_plan TEXT DEFAULT 'basic',
            paystack_customer_code TEXT,
            preferences TEXT
        );
        """)
        self.execute_query("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tool_used TEXT,
            tool_input TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """)
        self.execute_query("""
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            memory_text TEXT NOT NULL,
            embedding_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP,
            importance_score REAL,
            keywords TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """)
        self.execute_query("""
        CREATE TABLE IF NOT EXISTS tools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            is_enabled BOOLEAN DEFAULT 1,
            required_subscription TEXT DEFAULT 'basic',
            usage_count INTEGER DEFAULT 0
        );
        """)
        logging.info("Database schema initialization complete.")
        self.populate_initial_tools()

    def populate_initial_tools(self):
        """Populates the tools table with the assistant's capabilities."""
        tools = [
            ('web_search', 'Searches the web for real-time information. Use for news, facts, or any external data.',
             'basic'),
            ('image_analyzer',
             'Analyzes the content of an image from a file path. Use when the user uploads an image and asks a question about it.',
             'basic'),
            ('send_email', 'Sends an email on behalf of the user. Requires "to", "subject", and "body".', 'premium'),
            ('run_code',
             'Executes a block of Python code in a sandboxed environment. Use for calculations, data manipulation, or generating plots.',
             'premium'),
            ('financial_analysis', 'Provides financial market data and analysis. (Future capability)', 'ultimate'),
            ('marketing_campaign', 'Designs and drafts a marketing email campaign. (Future capability)', 'ultimate'),
            ('vulnerability_scan',
             'Performs a basic security scan on a given URL. (Future capability - requires strict ethics).', 'ultimate')
        ]
        insert_query = "INSERT OR IGNORE INTO tools (name, description, required_subscription) VALUES (?, ?, ?)"
        for tool in tools:
            self.execute_query(insert_query, tool)
        logging.info("Initial tools populated in the database.")


# --------------------------------------------------------------------------------------------------
# SECTION 1.4: FRONTEND CONTENT (HTML, CSS, JAVASCRIPT)
# --------------------------------------------------------------------------------------------------
# The UI is built with web technologies and rendered in a QWebEngineView.
# This approach allows for modern, flexible, and stylish UIs.

def get_html_content():
    """Returns the main HTML structure for the Etherius chat interface."""
    theme = AppConfig.THEME
    html_string = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{AppConfig.APP_NAME}</title>
    <style>{get_css_content()}</style>
</head>
<body>
    <div id="app-container">
        <div id="sidebar">
            <div id="sidebar-header">
                <img src="{os.path.join(AppConfig.ICONS_DIR, 'logo.png').replace(os.sep, '/')}" alt="Logo" id="app-logo">
                <h2>{AppConfig.APP_NAME}</h2>
                <p>v{AppConfig.APP_VERSION}</p>
            </div>
            <div id="sidebar-content">
                <div class="sidebar-section">
                    <h3>User Profile</h3>
                    <div id="user-profile">
                        <p><strong>User:</strong> <span id="username-display">Default User</span></p>
                        <p><strong>Plan:</strong> <span id="plan-display">Premium</span></p>
                    </div>
                </div>
                <div class="sidebar-section">
                    <h3>AI Settings</h3>
                    <div class="setting-item">
                        <label for="tts-toggle" title="Enable Text-to-Speech for AI responses">Enable TTS</label>
                        <input type="checkbox" id="tts-toggle">
                    </div>
                     <div class="setting-item">
                        <label for="privacy-mode" title="Disable long-term memory for this session">Privacy Mode</label>
                        <input type="checkbox" id="privacy-mode">
                    </div>
                </div>
                 <div class="sidebar-section">
                    <h3>Tools Status</h3>
                    <ul id="tools-status-list">
                        <!-- Dynamically populated by JS -->
                    </ul>
                </div>
            </div>
            <div id="sidebar-footer">
                <p>AI Status: <span id="status-indicator">Initializing...</span></p>
            </div>
        </div>

        <div id="main-content">
            <div id="chat-window">
                <div class="message system">
                    <div class="message-content">
                        Initializing Etherius... Please wait while models are being loaded. This may take a moment.
                    </div>
                </div>
            </div>

            <div id="input-area-wrapper">
                <div id="progress-bar-container" style="display: none;">
                    <div id="progress-bar"></div>
                </div>
                <div id="input-area">
                    <button id="attach-button" title="Attach Image or File">
                        <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M720-300q0-25-17.5-42.5T660-360H212q-53 0-89-35.5T88-480q0-49 34-84t86-36h402q75 0 127.5 52.5T790-360q0 63-44 107t-108 44H240q-25 0-42.5-17.5T180-270q0-25 17.5-42.5T240-330h420q25 0 42.5 17.5T720-270q0 25-17.5 42.5T660-210H212q-53 0-89-35.5T88-330q0-49 34-84t86-36h402q75 0 127.5 52.5T790-210q0 63-44 107t-108 44H240q-25 0-42.5-17.5T180-330q0-25 17.5-42.5T240-390h420q25 0 42.5 17.5T720-330q0 25-17.5 42.5T660-270H212q-53 0-89-35.5T88-390q0-49 34-84t86-36h402q75 0 127.5 52.5T790-270q0 63-44 107t-108 44H240q-50 0-85-35t-35-85q0-50 35-85t85-35h420q50 0 85 35t35 85q0 50-35 85t-85 35H240q-25 0-42.5-17.5T180-570q0-25 17.5-42.5T240-630h372q25 0 42.5 17.5T672-570v-60H240q-75 0-127.5-52.5T60-735q0-75 52.5-127.5T240-915h420q75 0 127.5 52.5T840-735q0 75-52.5 127.5T660-555H240q-50 0-85-35t-35-85q0-50 35-85t85-35h420q50 0 85 35t35 85q0 50-35 85t-85 35H240q-25 0-42.5-17.5T180-450q0-25 17.5-42.5T240-510h420q25 0 42.5 17.5T720-450q0 25-17.5 42.5T660-390H212q-75 0-127.5-52.5T32-570q0-75 52.5-127.5T212-750h448q75 0 127.5 52.5T840-570q0 75-52.5 127.5T660-390H240q-50 0-85-35t-35-85q0-50 35-85t85-35h420q50 0 85 35t35 85q0 50-35 85t-85 35H240q-25 0-42.5-17.5T180-270q0-25 17.5-42.5T240-330h420q25 0 42.5 17.5T720-270Z"/></svg>
                    </button>
                    <textarea id="message-input" placeholder="Converse with Etherius..." rows="1"></textarea>
                    <button id="mic-button" title="Use Voice Input">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5-3c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" fill="currentColor"/></svg>
                    </button>
                    <button id="send-button" title="Send Message">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor"/></svg>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
    <script>{get_js_content()}</script>
</body>
</html>
    """
    # Replace backslashes with forward slashes for HTML/CSS compatibility
    return html_string.replace(os.sep, '/')


def get_css_content():
    """Returns the CSS styles for the Etherius chat interface."""
    theme = AppConfig.THEME
    # This CSS is an evolution of the original, with more refined styling.
    return f"""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap');

        :root {{
            --primary-color: {theme['primary_color']};
            --secondary-color: {theme['secondary_color']};
            --tertiary-color: {theme['tertiary_color']};
            --accent-color: {theme['accent_color']};
            --accent-hover-color: {theme['accent_hover_color']};
            --text-color: {theme['text_color']};
            --text-secondary-color: {theme['text_secondary_color']};
            --success-color: {theme['success_color']};
            --warning-color: {theme['warning_color']};
            --error-color: {theme['error_color']};
            --font-family: 'Inter', {theme['font_family']};
            --font-size-normal: {theme['font_size_normal']};
            --font-size-small: {theme['font_size_small']};
            --font-size-large: {theme['font_size_large']};
            --border-radius: {theme['border_radius']};
            --box-shadow: {theme['box_shadow']};
            --transition-speed: 0.3s;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{ height: 100vh; width: 100vw; overflow: hidden; font-family: var(--font-family); }}
        #app-container {{ display: flex; height: 100%; width: 100%; background-color: var(--primary-color); }}

        /* Sidebar Styles */
        #sidebar {{
            width: 280px; min-width: 280px; background-color: var(--secondary-color);
            display: flex; flex-direction: column; padding: 20px;
            border-right: 1px solid var(--tertiary-color); transition: width var(--transition-speed) ease;
        }}
        #sidebar-header {{ text-align: center; margin-bottom: 30px; }}
        #app-logo {{ width: 64px; height: 64px; margin-bottom: 10px; border-radius: 50%; }}
        #sidebar h2 {{ font-size: 1.5rem; color: var(--text-color); font-weight: 700; }}
        #sidebar p {{ font-size: 0.9rem; color: var(--text-secondary-color); }}
        .sidebar-section {{ margin-bottom: 25px; }}
        .sidebar-section h3 {{
            font-size: 0.8rem; color: var(--text-secondary-color); text-transform: uppercase;
            letter-spacing: 1.2px; border-bottom: 1px solid var(--tertiary-color);
            padding-bottom: 8px; margin-bottom: 12px; font-weight: 500;
        }}
        .setting-item {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 12px; font-size: 0.95rem;
        }}
        #tools-status-list {{ list-style: none; }}
        #tools-status-list li {{ margin-bottom: 8px; display: flex; align-items: center; font-size: 0.9rem; }}
        .tool-status-indicator {{
            width: 8px; height: 8px; border-radius: 50%; margin-right: 10px;
            background-color: var(--error-color);
        }}
        .tool-status-indicator.enabled {{ background-color: var(--success-color); }}
        #sidebar-footer {{ margin-top: auto; text-align: center; font-size: 0.8rem; color: var(--text-secondary-color); }}
        #status-indicator.processing {{ color: var(--warning-color); }}

        /* Main Content Styles */
        #main-content {{ flex-grow: 1; display: flex; flex-direction: column; }}
        #chat-window {{ flex-grow: 1; padding: 20px; overflow-y: auto; scroll-behavior: smooth; }}
        .message {{ display: flex; margin-bottom: 25px; max-width: 85%; animation: fadeIn 0.5s ease; }}
        .message.user {{ margin-left: auto; flex-direction: row-reverse; }}
        .message.assistant {{ margin-right: auto; }}
        .message.system, .message.tool {{ max-width: 100%; justify-content: center; }}
        .message-content {{
            padding: 12px 18px; border-radius: 18px; font-size: 1rem; line-height: 1.6;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); position: relative;
        }}
        .message.user .message-content {{ background-color: var(--accent-color); color: white; border-bottom-right-radius: 4px; }}
        .message.assistant .message-content {{ background-color: var(--secondary-color); color: var(--text-color); border-bottom-left-radius: 4px; }}
        .message.system .message-content, .message.tool .message-content {{
            background-color: var(--tertiary-color); color: var(--text-secondary-color);
            font-style: italic; text-align: center; width: auto; max-width: 70%;
            box-shadow: none; font-size: 0.9rem; padding: 8px 15px;
        }}

        pre {{ background-color: #111; border: 1px solid var(--tertiary-color); padding: 15px; border-radius: var(--border-radius); font-family: 'Courier New', monospace; white-space: pre-wrap; word-wrap: break-word; margin: 10px -5px -5px -5px; }}
        code {{ font-family: 'Courier New', monospace; }}

        /* Input Area Styles */
        #input-area-wrapper {{ padding: 15px 20px; border-top: 1px solid var(--tertiary-color); background-color: var(--primary-color); }}
        #input-area {{ display: flex; align-items: flex-end; gap: 10px; background-color: var(--secondary-color); border-radius: 24px; padding: 8px; border: 1px solid var(--tertiary-color); transition: border-color var(--transition-speed); }}
        #input-area:focus-within {{ border-color: var(--accent-color); }}
        #message-input {{
            flex-grow: 1; background: transparent; border: none; padding: 10px; color: var(--text-color);
            font-size: 1rem; resize: none; max-height: 200px; overflow-y: auto; outline: none;
        }}
        #input-area button {{
            background-color: transparent; border: none; padding: 8px; cursor: pointer; border-radius: 50%;
            color: var(--text-secondary-color); display: flex; align-items: center; justify-content: center;
            transition: all var(--transition-speed) ease;
        }}
        #input-area button:hover {{ background-color: var(--tertiary-color); color: var(--accent-hover-color); }}
        #input-area button svg {{ width: 24px; height: 24px; fill: currentColor; }}
        #send-button {{ background-color: var(--accent-color); color: white; }}
        #send-button:hover {{ background-color: var(--accent-hover-color); }}
        #mic-button.recording {{ color: var(--error-color); animation: pulse 1.5s infinite ease-in-out; }}

        @keyframes pulse {{ 0% {{ transform: scale(1); }} 50% {{ transform: scale(1.15); }} 100% {{ transform: scale(1); }} }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}

        /* Custom Scrollbar */
        ::-webkit-scrollbar {{ width: 10px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: var(--tertiary-color); border-radius: 5px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--accent-color); }}
    """


def get_js_content():
    """Returns the JavaScript logic for the frontend."""
    return """
    (function() {
        'use strict';
        let pyBackend;
        const dom = {
            chatWindow: document.getElementById('chat-window'),
            messageInput: document.getElementById('message-input'),
            sendButton: document.getElementById('send-button'),
            micButton: document.getElementById('mic-button'),
            attachButton: document.getElementById('attach-button'),
            statusIndicator: document.getElementById('status-indicator'),
            ttsToggle: document.getElementById('tts-toggle')
        };

        window.onload = function() {
            if (typeof qt !== 'undefined') {
                new QWebChannel(qt.webChannelTransport, channel => {
                    pyBackend = channel.objects.backend_bridge;
                    console.log("Etherius Frontend: Python backend bridge connected.");
                    init();
                });
            } else {
                console.error("Etherius Frontend: Qt WebChannel is not available.");
                init(); // Limited functionality for browser testing
            }
        };

        function init() {
            setupEventListeners();
            autoAdjustTextarea();
            if (pyBackend) pyBackend.notify_frontend_ready();
        }

        function setupEventListeners() {
            sendButton.addEventListener('click', handleSendMessage);
            messageInput.addEventListener('keydown', e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                }
            });
            messageInput.addEventListener('input', autoAdjustTextarea);
            micButton.addEventListener('click', handleVoiceInput);
            attachButton.addEventListener('click', handleFileAttach);
            ttsToggle.addEventListener('change', () => {
                if(pyBackend) pyBackend.set_tts_enabled(ttsToggle.checked);
            });
        }

        function handleSendMessage() {
            const messageText = messageInput.value.trim();
            if (messageText && pyBackend) {
                appendMessage('user', messageText);
                pyBackend.process_user_message(messageText);
                messageInput.value = '';
                autoAdjustTextarea();
                updateStatus('Thinking...', true);
            }
        }

        function handleVoiceInput() {
            if (pyBackend) {
                micButton.classList.add('recording');
                updateStatus('Listening...', true);
                pyBackend.handle_voice_input_request();
            }
        }

        function handleFileAttach() {
            if (pyBackend) pyBackend.handle_file_attach_request();
        }

        function appendMessage(role, content, metadata = '') {
            const messageContainer = document.createElement('div');
            messageContainer.classList.add('message', role);

            const contentElement = document.createElement('div');
            contentElement.classList.add('message-content');

            // Convert markdown-style code blocks to <pre><code>
            let formattedContent = content.replace(/\\n/g, '<br>');
            formattedContent = formattedContent.replace(/```(.*?)```/g, (match, p1) => {
                const codeContent = p1.replace(/<br>/g, '\\n').trim();
                return `<pre><code>${escapeHtml(codeContent)}</code></pre>`;
            });
            contentElement.innerHTML = formattedContent;

            messageContainer.appendChild(contentElement);
            dom.chatWindow.appendChild(messageContainer);
            dom.chatWindow.scrollTop = dom.chatWindow.scrollHeight;
        }

        function escapeHtml(text) {
            const map = { '&': '&', '<': '<', '>': '>', '"': '"', "'": ''' };
            return text.replace(/[&<>"']/g, m => map[m]);
        }

        function autoAdjustTextarea() {
            dom.messageInput.style.height = 'auto';
            let scrollHeight = dom.messageInput.scrollHeight;
            dom.messageInput.style.height = (scrollHeight > 200 ? 200 : scrollHeight) + 'px';
        }

        function updateStatus(text, isProcessing = false) {
            dom.statusIndicator.textContent = text;
            dom.statusIndicator.className = isProcessing ? 'processing' : '';
        }

        // --- Public Functions Exposed to Python ---
        window.addMessage = (role, content) => appendMessage(role, content);
        window.setStatus = (text, isProcessing) => updateStatus(text, isProcessing);
        window.setInputText = text => {
            dom.messageInput.value = text;
            autoAdjustTextarea();
            dom.messageInput.focus();
        };
        window.setVoiceRecordingState = active => {
            dom.micButton.classList.toggle('recording', active);
            if (!active) updateStatus('Idle');
        };
        window.showError = (title, message) => {
            appendMessage('system', `<strong>Error: ${title}</strong><br>${message}`);
        };
        window.setInitialState = (welcomeMessage) => {
             dom.chatWindow.innerHTML = ''; // Clear "Initializing..." message
             appendMessage('system', welcomeMessage);
             updateStatus('Idle');
        }
    })();
    """


# --------------------------------------------------------------------------------------------------
# SECTION 1.5: PYQT GUI APPLICATION AND BACKEND BRIDGE
# --------------------------------------------------------------------------------------------------
# The core Qt application classes that manage the UI and backend communication.

class BackendBridge(QObject):
    """
    The bridge for communication between JavaScript (frontend) and Python (backend).
    It receives signals from the UI and forwards them to the main application thread.
    """
    # Signal to send a user message to the AI core for processing
    user_message_received = pyqtSignal(str)
    # Signal to request voice input transcription
    voice_input_requested = pyqtSignal()
    # Signal to request file attachment
    file_attach_requested = pyqtSignal()
    # Signal that the frontend is loaded and ready
    frontend_ready = pyqtSignal()
    # Signal to update TTS enabled state
    tts_state_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        logging.info("BackendBridge initialized.")

    @pyqtSlot(str)
    def process_user_message(self, message: str):
        logging.info(f"Received message from frontend: '{message}'")
        self.user_message_received.emit(message)

    @pyqtSlot()
    def handle_voice_input_request(self):
        logging.info("Voice input request received from frontend.")
        self.voice_input_requested.emit()

    @pyqtSlot()
    def handle_file_attach_request(self):
        logging.info("File attach request received from frontend.")
        self.file_attach_requested.emit()

    @pyqtSlot()
    def notify_frontend_ready(self):
        logging.info("Frontend is ready and connected.")
        self.frontend_ready.emit()

    @pyqtSlot(bool)
    def set_tts_enabled(self, is_enabled: bool):
        logging.info(f"TTS state changed by user to: {is_enabled}")
        self.tts_state_changed.emit(is_enabled)


# ==================================================================================================
#
#   Etherius: The Sentient AI Assistant
#   File: main.py (Section 2 of 5)
#   Purpose: AI Core Services, Model Loading, Threading, STT/TTS Engines, and Tool Framework.
#
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# SECTION 2.1: ADVANCED AI AND UTILITY WORKERS
# --------------------------------------------------------------------------------------------------
# These classes are designed to run in separate QThreads to handle long-running tasks
# like model inference, audio processing, etc., without blocking the main GUI thread.

class ModelLoader(QObject):
    """
    A worker dedicated to loading all necessary AI models.
    This runs in a separate thread to show loading progress on the UI.
    """
    progress_updated = pyqtSignal(int, str)
    models_loaded = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.models = {}

    def run(self):
        """The main entry point for the model loading thread."""
        try:
            logging.info("Model loading process started.")
            device = AppConfig.AI_SETTINGS['device']
            token = AppConfig.HUGGING_FACE_TOKEN
            logging.info(f"Loading models onto device: {device}")

            # 1. Main LLM (Mixtral) - This is the heaviest model.
            self.progress_updated.emit(10, "Loading Main Language Model (Mixtral)... This can take several minutes.")
            if device == "cuda" and AppConfig.AI_SETTINGS['use_4bit_quantization']:
                logging.info("Using 4-bit quantization for Mixtral model.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                self.models['llm'] = AutoModelForCausalLM.from_pretrained(
                    AppConfig.MODEL_IDENTIFIERS['main_llm'],
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=quantization_config,
                    token=token
                )
            else:
                logging.info("Loading Mixtral model without quantization (requires significant VRAM/RAM).")
                self.models['llm'] = AutoModelForCausalLM.from_pretrained(
                    AppConfig.MODEL_IDENTIFIERS['main_llm'],
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    token=token
                )
            self.models['llm_tokenizer'] = AutoTokenizer.from_pretrained(AppConfig.MODEL_IDENTIFIERS['main_llm'],
                                                                         token=token)
            self.progress_updated.emit(35, "Main Language Model loaded.")

            # 2. Text Embedding Model for Semantic Memory
            self.progress_updated.emit(40, "Loading Text Embedding Model...")
            self.models['embedding_model'] = SentenceTransformer(
                AppConfig.MODEL_IDENTIFIERS['text_embedding'],
                device=device
            )
            self.progress_updated.emit(45, "Text Embedding Model loaded.")

            # 3. Speech-to-Text Model (Whisper)
            self.progress_updated.emit(50, "Loading Speech-to-Text Model (Whisper)...")
            self.models['stt_model'] = WhisperForConditionalGeneration.from_pretrained(
                AppConfig.MODEL_IDENTIFIERS['speech_to_text']
            ).to(device)
            self.models['stt_processor'] = WhisperProcessor.from_pretrained(
                AppConfig.MODEL_IDENTIFIERS['speech_to_text']
            )
            self.progress_updated.emit(65, "Speech-to-Text Model loaded.")

            # 4. Text-to-Speech Models (SpeechT5)
            self.progress_updated.emit(70, "Loading Text-to-Speech Models (SpeechT5)...")
            self.models['tts_model'] = SpeechT5ForTextToSpeech.from_pretrained(
                AppConfig.MODEL_IDENTIFIERS['text_to_speech_model']
            ).to(device)
            self.models['tts_processor'] = SpeechT5Processor.from_pretrained(
                AppConfig.MODEL_IDENTIFIERS['text_to_speech_model']
            )
            self.models['tts_vocoder'] = SpeechT5HifiGan.from_pretrained(
                AppConfig.MODEL_IDENTIFIERS['tts_vocoder']
            ).to(device)
            # Download speaker embeddings
            speaker_embeddings_url = "https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors/resolve/main/cmu_us_slt_arctic-wav-arctic_a0508.npy"
            self.models['speaker_embeddings'] = torch.tensor(
                np.load(io.BytesIO(requests.get(speaker_embeddings_url).content))).unsqueeze(0).to(device)
            self.progress_updated.emit(85, "Text-to-Speech Models loaded.")

            # 5. Image Captioning Model (BLIP)
            self.progress_updated.emit(90, "Loading Image Analysis Model (BLIP)...")
            self.models['image_model'] = BlipForConditionalGeneration.from_pretrained(
                AppConfig.MODEL_IDENTIFIERS['image_captioning']
            ).to(device)
            self.models['image_processor'] = BlipProcessor.from_pretrained(
                AppConfig.MODEL_IDENTIFIERS['image_captioning']
            )
            self.progress_updated.emit(98, "Image Analysis Model loaded.")

            # Translation models will be loaded on-demand to save VRAM.
            self.models['translation_cache'] = {}

            self.progress_updated.emit(100, "All core models loaded successfully.")
            logging.info("All core AI models have been loaded.")
            self.models_loaded.emit(self.models)

        except Exception as e:
            logging.critical(f"Failed to load AI models: {e}", exc_info=True)
            self.error_occurred.emit(
                f"A critical error occurred while loading AI models: {e}. The application may not function correctly. Please check your internet connection and model identifiers.")


class TTSEngine(QObject):
    """
    Worker for Text-to-Speech synthesis. It runs in its own thread to avoid
    blocking while generating and playing audio.
    """
    is_speaking_changed = pyqtSignal(bool)

    def __init__(self, models, parent=None):
        super().__init__(parent)
        self.models = models
        self.device = AppConfig.AI_SETTINGS['device']
        self.is_enabled = False
        self.audio_queue = Queue()
        self.stop_playback = False
        logging.info("TTS Engine initialized.")

    def set_enabled(self, enabled):
        """Enable or disable TTS playback."""
        self.is_enabled = enabled
        logging.info(f"TTS enabled state set to: {self.is_enabled}")
        if not enabled:
            self.stop_current_playback()

    def stop_current_playback(self):
        """Stops any currently playing audio."""
        self.stop_playback = True

    def speak(self, text):
        """Public method to queue text for speaking."""
        if not self.is_enabled or not text:
            return
        self.audio_queue.put(text)
        # If this is the first item, start the processing loop
        if self.audio_queue.qsize() == 1:
            self.process_queue()

    @torch.no_grad()
    def process_queue(self):
        """Processes the queue of text to be spoken."""
        if self.audio_queue.empty():
            self.is_speaking_changed.emit(False)
            return

        text = self.audio_queue.get()
        self.is_speaking_changed.emit(True)
        self.stop_playback = False

        try:
            logging.info(f"Generating speech for: '{text[:50]}...'")
            inputs = self.models['tts_processor'](text=text, return_tensors="pt").to(self.device)
            speech = self.models['tts_model'].generate_speech(
                inputs["input_ids"],
                self.models['speaker_embeddings'],
                vocoder=self.models['tts_vocoder']
            )
            audio_np = speech.cpu().numpy()

            logging.info("Playing synthesized audio.")
            sd.play(audio_np, samplerate=AppConfig.AUDIO_SAMPLE_RATE)

            # Monitor playback to allow interruption
            while sd.get_stream().active:
                if self.stop_playback:
                    sd.stop()
                    logging.info("TTS playback interrupted by user.")
                    # Clear the rest of the queue if we stop
                    while not self.audio_queue.empty():
                        self.audio_queue.get()
                    break
                time.sleep(0.1)
            sd.wait()  # Ensure stream is properly closed if not stopped

        except Exception as e:
            logging.error(f"Failed to generate or play speech: {e}", exc_info=True)
        finally:
            # Recursively process the next item if not stopped
            if not self.stop_playback:
                self.process_queue()
            else:
                self.is_speaking_changed.emit(False)


class STTEngine(QObject):
    """
    Worker for Speech-to-Text transcription. Listens to the microphone in a
    separate thread and uses Whisper for transcription.
    """
    transcription_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    is_listening_changed = pyqtSignal(bool)

    def __init__(self, models, parent=None):
        super().__init__(parent)
        self.models = models
        self.device = AppConfig.AI_SETTINGS['device']
        self.is_listening = False
        self.audio_stream = None
        self.frames = []
        logging.info("STT Engine initialized.")

    def start_transcription(self):
        """Starts listening to the microphone."""
        if self.is_listening:
            logging.warning("STT Engine is already listening.")
            return

        self.is_listening = True
        self.is_listening_changed.emit(True)
        self.frames = []

        try:
            self.audio_stream = sd.InputStream(
                samplerate=AppConfig.AUDIO_SAMPLE_RATE,
                channels=AppConfig.AUDIO_CHANNELS,
                dtype='float32',
                callback=self._audio_callback
            )
            self.audio_stream.start()
            logging.info("Microphone stream started. Listening for speech...")
            # Use a timer to stop recording after a period of silence (or max duration)
            # For simplicity here, we'll use a fixed duration for recording.
            QTimer.singleShot(10000, self.stop_transcription)  # Record for max 10 seconds

        except Exception as e:
            error_msg = f"Failed to start microphone: {e}. Check audio device permissions and configuration."
            logging.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            self.is_listening = False
            self.is_listening_changed.emit(False)

    def _audio_callback(self, indata, frames, time, status):
        """This is called by sounddevice for each new audio chunk."""
        if status:
            logging.warning(f"Audio callback status: {status}")
        self.frames.append(indata.copy())

    def stop_transcription(self):
        """Stops listening and processes the recorded audio."""
        if not self.is_listening:
            return

        try:
            self.audio_stream.stop()
            self.audio_stream.close()
            logging.info("Microphone stream stopped.")
        except Exception as e:
            logging.error(f"Error stopping microphone stream: {e}")

        self.is_listening = False
        self.is_listening_changed.emit(False)

        if not self.frames:
            logging.warning("No audio frames recorded.")
            self.transcription_ready.emit("")  # Emit empty string if no audio
            return

        audio_data = np.concatenate(self.frames, axis=0).flatten()
        self.process_audio(audio_data)

    @torch.no_grad()
    def process_audio(self, audio_data):
        """Transcribes the given audio data using the Whisper model."""
        logging.info(f"Processing {len(audio_data) / AppConfig.AUDIO_SAMPLE_RATE:.2f} seconds of audio.")
        try:
            input_features = self.models['stt_processor'](
                audio_data,
                sampling_rate=AppConfig.AUDIO_SAMPLE_RATE,
                return_tensors="pt"
            ).input_features.to(self.device)

            # Generate token ids
            predicted_ids = self.models['stt_model'].generate(input_features)

            # Decode token ids to text
            transcription = self.models['stt_processor'].batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logging.info(f"Transcription result: '{transcription}'")
            self.transcription_ready.emit(transcription.strip())

        except Exception as e:
            error_msg = f"Failed to transcribe audio: {e}"
            logging.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)


# --------------------------------------------------------------------------------------------------
# SECTION 2.2: VECTOR DATABASE AND MEMORY MANAGEMENT
# --------------------------------------------------------------------------------------------------

class VectorDBManager:
    """
    Manages all interactions with the ChromaDB vector store for long-term semantic memory.
    """

    def __init__(self, path, collection_name, embedding_model):
        self.path = path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client = None
        self.collection = None
        logging.info("VectorDBManager initialized.")

    def initialize(self):
        """Initializes the ChromaDB client and gets or creates the collection."""
        try:
            self.client = chromadb.PersistentClient(path=self.path)

            # The embedding function is handled manually by passing vectors
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logging.info(
                f"Vector database connected. Collection '{self.collection_name}' loaded with {self.collection.count()} items.")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            return False

    def add_memory(self, text_chunk, metadata, doc_id):
        """
        Adds a text chunk and its metadata to the vector store.
        The embedding is generated here.
        """
        if not self.collection:
            logging.error("Cannot add memory, collection is not initialized.")
            return
        try:
            logging.info(f"Adding memory to vector store with id: {doc_id}")
            embedding = self.embedding_model.encode([text_chunk])[0].tolist()
            self.collection.add(
                embeddings=[embedding],
                documents=[text_chunk],
                metadatas=[metadata],
                ids=[doc_id]
            )
            logging.info(f"Successfully added memory. Collection size: {self.collection.count()}")
        except Exception as e:
            # This can happen if an ID already exists, which should be handled by the caller.
            logging.error(f"Failed to add memory to vector store: {e}", exc_info=True)

    def search_memories(self, query_text, n_results=5, user_id=None):
        """
        Searches for relevant memories based on a query text.
        """
        if not self.collection or self.collection.count() == 0:
            logging.info("Cannot search memories, collection is empty or not initialized.")
            return []
        try:
            logging.info(f"Searching for memories related to: '{query_text[:50]}...'")
            query_embedding = self.embedding_model.encode([query_text])[0].tolist()

            # Optional filter by user_id if the collection stores data for multiple users
            where_filter = {"user_id": user_id} if user_id else None

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),  # Don't request more than exists
                where=where_filter
            )

            # Format results into a more usable list of dictionaries
            formatted_results = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'text': doc,
                        'distance': results['distances'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            logging.info(f"Found {len(formatted_results)} relevant memories.")
            return formatted_results
        except Exception as e:
            logging.error(f"Failed to search memories in vector store: {e}", exc_info=True)
            return []


# --------------------------------------------------------------------------------------------------
# SECTION 2.3: TOOL EXECUTION FRAMEWORK
# --------------------------------------------------------------------------------------------------

class ToolExecutor:
    """
    A class responsible for executing tools that the AI decides to use.
    It maps tool names to their corresponding functions.
    """

    def __init__(self, models):
        self.models = models
        self.device = AppConfig.AI_SETTINGS['device']
        # Add 'run_code' to the tool map
        self._tool_map = {
            "web_search": self._execute_web_search,
            "image_analyzer": self._execute_image_analyzer,
            "send_email": self._execute_send_email,
            "run_code": self._execute_run_code,  # Add this line
        }
        logging.info("ToolExecutor initialized.")

    def get_tool_descriptions(self):
        """Returns a formatted string of tool descriptions for the LLM prompt."""
        descriptions = {
            "web_search": "Searches the web for real-time information. Input is the search query string.",
            "image_analyzer": "Analyzes an image from a given file path. Input is the full path to the image file.",
            "send_email": "Sends an email. Input is a JSON string with keys 'to', 'subject', and 'body'.",
            "run_code": "Executes a block of Python code in a sandboxed environment and returns the output. Input is the raw Python code string.",
            "none": "No tool is needed. The AI can answer from its own knowledge."
        }
        return json.dumps(descriptions, indent=2)

    def _execute_run_code(self, code: str):
        """
        Executes a block of Python code safely using a subprocess.
        Returns stdout and stderr.
        """
        if not code:
            return "[Error: No code provided to execute.]"

        logging.info(f"Executing sandboxed code:\n{code}")

        # Create a temporary file to write the code to
        temp_script_path = os.path.join(AppConfig.TEMP_DIR, f"script_{int(time.time())}.py")

        try:
            with open(temp_script_path, 'w', encoding='utf-8') as f:
                f.write(code)

            # Execute the script using a subprocess with a timeout
            process = subprocess.run(
                [sys.executable, temp_script_path],
                capture_output=True,
                text=True,
                timeout=30  # 30-second timeout to prevent long-running scripts
            )

            output = ""
            if process.stdout:
                output += f"STDOUT:\n{process.stdout}\n"
            if process.stderr:
                output += f"STDERR:\n{process.stderr}\n"

            return output if output else "[Code executed with no output.]"

        except subprocess.TimeoutExpired:
            logging.error("Code execution timed out.")
            return "[Error: Code execution timed out after 30 seconds.]"
        except Exception as e:
            logging.error(f"Failed to execute code: {e}", exc_info=True)
            return f"[Error: An exception occurred during code execution: {e}]"
        finally:
            # Clean up the temporary script file
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)
    def _execute_web_search(self, query: str):
        """Performs a web search using the Serper.dev API."""
        if not AppConfig.SERPER_API_KEY or AppConfig.SERPER_API_KEY == "YOUR_SERPER_API_KEY":
            return "[Error: Serper API key is not configured.]"
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": 5})
        headers = {'X-API-KEY': AppConfig.SERPER_API_KEY, 'Content-Type': 'application/json'}
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            results = response.json()

            output = ""
            if "answerBox" in results:
                output += f"Answer Box: {results['answerBox']['snippet']}\n\n"
            if "organic" in results:
                output += "Search Results:\n"
                for i, res in enumerate(results["organic"][:5]):
                    output += f"{i + 1}. {res['title']}\nSnippet: {res.get('snippet', 'N/A')}\nURL: {res['link']}\n\n"
            return output if output else "[No results found.]"
        except requests.RequestException as e:
            return f"[Error: Web search failed due to a network issue: {e}]"

    @torch.no_grad()
    def _execute_image_analyzer(self, image_path: str):
        """Analyzes an image using the BLIP model."""
        if not os.path.exists(image_path):
            return f"[Error: Image file not found at path: {image_path}]"
        try:
            raw_image = Image.open(image_path).convert('RGB')
            processor = self.models['image_processor']
            model = self.models['image_model']

            # Generate caption
            inputs = processor(images=raw_image, return_tensors="pt").to(self.device)
            outputs = model.generate(**inputs, max_new_tokens=100)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            # Ask a question (Visual Question Answering)
            question = "What are the main objects and activities in this image?"
            inputs = processor(raw_image, text=question, return_tensors="pt").to(self.device)
            outputs = model.generate(**inputs, max_new_tokens=150)
            vqa_answer = processor.decode(outputs[0], skip_special_tokens=True)

            return f"Image Caption: {caption.strip()}\nDetailed Analysis: {vqa_answer.strip()}"
        except Exception as e:
            return f"[Error: Image analysis failed: {e}]"

    def _execute_send_email(self, json_input: str):
        """Sends an email using SMTP. Requires environment variables for credentials."""
        try:
            params = json.loads(json_input)
            to_addr = params['to']
            subject = params['subject']
            body = params['body']
        except (json.JSONDecodeError, KeyError) as e:
            return f"[Error: Invalid input for send_email. It must be a valid JSON with 'to', 'subject', and 'body' keys. Error: {e}]"

        # SMTP configuration from environment variables
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")

        if not all([smtp_server, smtp_port, smtp_user, smtp_password]):
            return "[Error: SMTP server is not configured. Please set SMTP_SERVER, SMTP_PORT, SMTP_USER, and SMTP_PASSWORD environment variables.]"

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = to_addr
        msg.set_content(body)

        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            return f"[Success: Email sent to {to_addr} with subject '{subject}'.]"
        except Exception as e:
            return f"[Error: Failed to send email: {e}]"


# --------------------------------------------------------------------------------------------------
# SECTION 2.4: MAIN WINDOW ENHANCEMENTS FOR THREADING
# --------------------------------------------------------------------------------------------------
# We now modify the EtheriusMainWindow to create, manage, and communicate with the worker threads.




# ==================================================================================================
#
#   Etherius: The Sentient AI Assistant
#   File: main.py (Section 3 of 5)
#   Purpose: The AI Core processing worker, reasoning loop, memory integration,
#            and final response generation logic.
#
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# SECTION 3.1: THE AI CORE WORKER
# --------------------------------------------------------------------------------------------------
# This is the central nervous system of Etherius. It runs in a dedicated thread,
# processing user inputs, managing memory, using tools, and generating responses.

class AICoreWorker(QObject):
    """
    The main AI processing worker. It handles the entire "thought process" of the assistant.
    """
    # Signals to communicate results back to the main GUI thread
    new_assistant_message = pyqtSignal(str)
    new_tool_message = pyqtSignal(str)  # To show "Using tool: Web Search..."
    processing_status_changed = pyqtSignal(str, bool)
    error_occurred = pyqtSignal(str, str)
    long_term_memory_updated = pyqtSignal()

    def __init__(self, models, db_manager, vector_db_manager, tool_executor, parent=None):
        super().__init__(parent)
        # --- Core Components ---
        self.models = models
        self.db_manager = db_manager
        self.vector_db_manager = vector_db_manager
        self.tool_executor = tool_executor

        # --- AI State ---
        self.device = AppConfig.AI_SETTINGS['device']
        self.conversation_history = deque(maxlen=AppConfig.AI_SETTINGS['memory_size'])
        self.current_user_id = 1  # Hardcoded for now, will be dynamic with user auth
        self.is_running = True
        self.task_queue = Queue()

        logging.info("AICoreWorker initialized.")

    def add_to_queue(self, task_data):
        """
        Public method to add a new task (e.g., user message) to the processing queue.
        This is the entry point for the AI's "thought process".
        The task_data can be a string (user message) or a dict for more complex tasks.
        """
        logging.info(f"New task added to AI Core queue: {task_data}")
        self.task_queue.put(task_data)

    def stop(self):
        """Stops the worker's processing loop."""
        self.is_running = False
        self.task_queue.put(None)  # Unblock the queue if it's waiting
        logging.info("AI Core worker stop signal received.")

    def run(self):
        """The main processing loop for the AI core thread."""
        logging.info("AI Core worker thread has started.")
        while self.is_running:
            try:
                # The `get()` call will block until a task is available, making this loop efficient.
                task = self.task_queue.get(block=True)
                if task is None:
                    # Shutdown signal
                    break

                self._process_task(task)

            except Exception as e:
                logging.critical(f"Unhandled exception in AI Core run loop: {e}", exc_info=True)
                self.error_occurred.emit("AI Core Failure", f"A critical error occurred in the AI processing loop: {e}")

        logging.info("AI Core worker thread has stopped.")

    def _process_task(self, task):
        """
        The main logic for handling a single task from the queue.
        This orchestrates the entire reasoning and response generation process.
        """
        # For now, we only handle string messages from the user.
        if not isinstance(task, str):
            logging.warning(f"AI Core received an unknown task type: {type(task)}")
            return

        user_query = task
        self.processing_status_changed.emit("Thinking...", True)

        try:
            # 1. Add user message to short-term memory
            self.conversation_history.append({"role": "user", "content": user_query})

            # 2. Retrieve relevant long-term memories
            long_term_memories = self._retrieve_long_term_memories(user_query)

            # 3. Decide if a tool is needed based on the query and context
            tool_name, tool_input = self._decide_on_tool(user_query, long_term_memories)

            # 4. Execute the tool if one was chosen
            tool_result = None
            if tool_name and tool_name != "none":
                self.processing_status_changed.emit(f"Using tool: {tool_name}...", True)
                self.new_tool_message.emit(f"Executing tool: **{tool_name}**. Please wait...")
                tool_result = self.tool_executor.execute_tool(tool_name, tool_input)
                # Add tool interaction to conversation history for context
                self.conversation_history.append({"role": "tool", "content": tool_result})

            # 5. Generate the final response
            self.processing_status_changed.emit("Generating response...", True)
            final_response = self._generate_final_response(user_query, long_term_memories, tool_result)

            # 6. Add assistant's response to short-term memory
            self.conversation_history.append({"role": "assistant", "content": final_response})

            # 7. Send final response to the UI
            self.new_assistant_message.emit(final_response)

            # 8. Post-response reflection: Decide if this interaction is memorable
            self._reflect_and_memorize(user_query, final_response)

        except Exception as e:
            logging.error(f"Error during AI task processing: {e}", exc_info=True)
            error_message = f"I apologize, but I encountered an internal error while processing your request: {e}"
            self.new_assistant_message.emit(error_message)
        finally:
            self.processing_status_changed.emit("Idle", False)

    def _retrieve_long_term_memories(self, query: str) -> str:
        """Searches the vector database for memories relevant to the query."""
        logging.info("Retrieving long-term memories.")
        search_results = self.vector_db_manager.search_memories(
            query_text=query,
            n_results=AppConfig.AI_SETTINGS['semantic_memory_search_k'],
            user_id=self.current_user_id
        )
        if not search_results:
            logging.info("No relevant long-term memories found.")
            return ""

        # Format memories for the prompt
        formatted_memories = "Here are some relevant facts from your long-term memory:\n"
        for i, item in enumerate(search_results):
            formatted_memories += f"- {item['text']} (Relevance score: {1 - item['distance']:.2f})\n"

        logging.info(f"Retrieved memories:\n{formatted_memories}")
        return formatted_memories

    def _build_prompt_for_llm(self, sections: list) -> str:
        """
        Constructs a complete prompt string for the LLM from various context sections.
        Uses the specific chat template required by Mixtral-Instruct.
        """
        tokenizer = self.models['llm_tokenizer']
        # The prompt is built as a list of dictionaries, then templated.
        prompt_list = []

        # System Prompt is always first
        prompt_list.append({"role": "system", "content": AppConfig.SYSTEM_PROMPTS['main_persona']})

        # Add other sections
        for section in sections:
            if section['content']:  # Only add if there is content
                prompt_list.append({"role": section['role'], "content": section['content']})

        # Finally, add the conversation history
        prompt_list.extend(list(self.conversation_history))

        # Use the tokenizer's apply_chat_template method
        # `add_generation_prompt=True` adds the `[/INST]` token to signal assistant's turn
        return tokenizer.apply_chat_template(prompt_list, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def _decide_on_tool(self, query: str, long_term_memories: str) -> (str, any):
        """
        Asks the LLM to decide which tool to use, if any.
        """
        logging.info("Deciding on tool usage...")

        tool_descriptions = self.tool_executor.get_tool_descriptions()
        prompt_content = AppConfig.SYSTEM_PROMPTS['tool_selection_prompt'].format(
            user_query=query,
            tool_descriptions=tool_descriptions
        )

        # We give it some context but not the full history to keep it fast and focused
        context_for_tool_decision = [
            {"role": "system", "content": long_term_memories},
            {"role": "user", "content": f"My current request is: '{query}'"},
            {"role": "system", "content": prompt_content}
        ]

        full_prompt = self.models['llm_tokenizer'].apply_chat_template(
            context_for_tool_decision, tokenize=False, add_generation_prompt=True
        )

        inputs = self.models['llm_tokenizer'](full_prompt, return_tensors="pt").to(self.device)
        outputs = self.models['llm'].generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,  # We want a deterministic JSON output
            pad_token_id=self.models['llm_tokenizer'].eos_token_id
        )

        # Decode and clean up the model's response
        response_text = self.models['llm_tokenizer'].decode(outputs[0], skip_special_tokens=True)
        json_part = response_text.split('[/INST]')[-1].strip()

        logging.info(f"Tool decision raw response: {json_part}")

        try:
            # Find the JSON block in the response
            start_index = json_part.find('{')
            end_index = json_part.rfind('}') + 1
            if start_index == -1 or end_index == 0:
                raise json.JSONDecodeError("No JSON object found", json_part, 0)

            json_str = json_part[start_index:end_index]
            decision = json.loads(json_str)

            tool_name = decision.get("tool_name", "none")
            tool_input = decision.get("tool_input", "")

            logging.info(f"LLM decided to use tool: '{tool_name}' with input: '{str(tool_input)[:50]}...'")
            return tool_name, tool_input

        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from tool decision response: {json_part}")
            return "none", ""
        except Exception as e:
            logging.error(f"An unexpected error occurred during tool decision: {e}")
            return "none", ""

    @torch.no_grad()
    def _generate_final_response(self, query: str, long_term_memories: str, tool_result: str) -> str:
        """
        Generates the final, user-facing response after all context has been gathered.
        """
        logging.info("Generating final response.")

        prompt_sections = [
            {"role": "system", "name": "memory", "content": long_term_memories},
        ]

        # Build the prompt using the structured method
        full_prompt = self._build_prompt_for_llm(prompt_sections)

        logging.debug(f"Final generation prompt:\n{full_prompt}")

        inputs = self.models['llm_tokenizer'](full_prompt, return_tensors="pt").to(self.device)
        outputs = self.models['llm'].generate(
            **inputs,
            max_new_tokens=AppConfig.AI_SETTINGS['generation_max_length'],
            min_new_tokens=AppConfig.AI_SETTINGS['generation_min_length'],
            temperature=AppConfig.AI_SETTINGS['generation_temperature'],
            top_p=AppConfig.AI_SETTINGS['generation_top_p'],
            do_sample=True,
            pad_token_id=self.models['llm_tokenizer'].eos_token_id
        )

        response_text = self.models['llm_tokenizer'].decode(outputs[0], skip_special_tokens=True)
        # Extract only the content generated by the assistant in the last turn
        final_response = response_text.split('[/INST]')[-1].strip()

        logging.info(f"Generated final response: '{final_response[:100]}...'")
        return final_response

    @torch.no_grad()
    def _reflect_and_memorize(self, user_query: str, ai_response: str):
        """
        After an interaction, this method decides if the information is important enough
        to be summarized and stored in the long-term vector database.
        """
        logging.info("Reflecting on conversation for long-term memory storage.")

        # Heuristic: only consider conversations of a certain length for memory
        if len(user_query) + len(ai_response) < 150:
            logging.info("Interaction is too short. Skipping long-term memory storage.")
            return

        summarization_prompt = f"""
Summarize the key information, facts, or user preferences from the following conversation exchange into a concise, self-contained statement.
This summary will be stored in your long-term memory.
Focus on new information that is likely to be relevant in the future.
If no new, lasting information was exchanged, respond with only the word "NULL".

Conversation:
User: "{user_query}"
Assistant: "{ai_response}"

Summary:
"""
        prompt_list = [{"role": "user", "content": summarization_prompt}]
        full_prompt = self.models['llm_tokenizer'].apply_chat_template(prompt_list, tokenize=False,
                                                                       add_generation_prompt=True)

        inputs = self.models['llm_tokenizer'](full_prompt, return_tensors="pt").to(self.device)
        outputs = self.models['llm'].generate(
            **inputs, max_new_tokens=100, do_sample=False, pad_token_id=self.models['llm_tokenizer'].eos_token_id
        )

        summary = self.models['llm_tokenizer'].decode(outputs[0], skip_special_tokens=True).split('[/INST]')[-1].strip()

        if summary.upper() == "NULL" or not summary:
            logging.info("LLM determined the interaction was not memorable.")
            return

        logging.info(f"Generated memory summary: '{summary}'")

        # Create a unique ID for this memory
        memory_id = hashlib.sha256(summary.encode()).hexdigest()
        timestamp = datetime.datetime.now().isoformat()

        # Add to vector database
        metadata = {"user_id": self.current_user_id, "created_at": timestamp}
        self.vector_db_manager.add_memory(summary, metadata, memory_id)

        # Add metadata to SQL database
        self.db_manager.execute_query(
            "INSERT INTO long_term_memory (user_id, memory_text, embedding_id, created_at) VALUES (?, ?, ?, ?)",
            (self.current_user_id, summary, memory_id, timestamp)
        )

        self.long_term_memory_updated.emit()


# --------------------------------------------------------------------------------------------------
# SECTION 3.2: MAIN WINDOW INTEGRATION OF THE AI CORE
# --------------------------------------------------------------------------------------------------
# Modifying the EtheriusMainWindow class to integrate and manage the AICoreWorker.

from web3 import Web3

# ==================================================================================================
#
#   Etherius: The Sentient AI Assistant
#   File: main.py (Section 4 of 5)
#   Purpose: User Authentication, Subscription Plans, Payment Integration (Paystack),
#            and Backend API Server for Webhooks.
#
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# SECTION 4.1: PAYMENT AND SUBSCRIPTION MANAGEMENT
# --------------------------------------------------------------------------------------------------
# This section adds classes and functions for handling payments and managing user subscriptions.

class PaystackManager:
    """
    Handles all interactions with the Paystack API for payments.
    """

    def __init__(self, secret_key):
        self.secret_key = secret_key
        self.base_url = "https://api.paystack.co"
        self.headers = {
            "Authorization": f"Bearer {self.secret_key}",
            "Content-Type": "application/json",
        }
        if not self.secret_key or self.secret_key == "YOUR_PAYSTACK_SECRET_KEY":
            logging.warning("Paystack secret key is not configured. Payment features will be disabled.")
            self.is_configured = False
        else:
            self.is_configured = True
        logging.info(f"PaystackManager initialized. Configured: {self.is_configured}")

    def _make_request(self, method, endpoint, data=None):
        """Helper function to make requests to the Paystack API."""
        if not self.is_configured:
            return {"status": False, "message": "Paystack is not configured."}

        url = f"{self.base_url}/{endpoint}"
        try:
            if method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=data, timeout=15)
            elif method.upper() == 'GET':
                response = requests.get(url, headers=self.headers, timeout=15)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Paystack API request failed: {e}", exc_info=True)
            return {"status": False, "message": f"API Request Error: {e}"}

    def initialize_transaction(self, email: str, amount_kobo: int, reference: str):
        """
        Initializes a transaction and returns an authorization URL for the user.
        :param email: User's email address.
        :param amount_kobo: Amount in kobo (e.g., 500000 for N5,000.00).
        :param reference: A unique reference for this transaction.
        """
        payload = {
            "email": email,
            "amount": amount_kobo,
            "reference": reference,
            "metadata": {
                "app_name": AppConfig.APP_NAME,
                "app_version": AppConfig.APP_VERSION,
            }
        }
        return self._make_request('POST', 'transaction/initialize', data=payload)

    def verify_transaction(self, reference: str):
        """
        Verifies the status of a transaction with Paystack.
        :param reference: The unique reference of the transaction to verify.
        """
        return self._make_request('GET', f'transaction/verify/{reference}')


class SubscriptionManager:
    """
    Defines subscription plans and their associated features.
    """
    PLANS = {
        "basic": {
            "name": "Basic",
            "price_ngn": 0,
            "features": ["Standard AI Model", "Limited Conversation History (10)", "Web Search Tool"],
            "tool_access": ["web_search", "image_analyzer"],
        },
        "premium": {
            "name": "Premium",
            "price_ngn": 5000,
            "features": ["Advanced AI Model", "Extended Conversation History (50)", "All Basic Tools",
                         "Email & Code Tools", "Priority Support"],
            "tool_access": ["web_search", "image_analyzer", "send_email", "run_code"],
        },
        "ultimate": {
            "name": "Ultimate",
            "price_ngn": 15000,
            "features": ["All Premium Features", "Early Access to New Tools", "API Access (Future)",
                         "Dedicated AI Agent Fine-tuning (Future)"],
            "tool_access": ["web_search", "image_analyzer", "send_email", "run_code", "financial_analysis",
                            "marketing_campaign", "vulnerability_scan"],
        }
    }

    @classmethod
    def get_plan_details(cls, plan_name: str):
        """Returns the details for a specific plan."""
        return cls.PLANS.get(plan_name.lower())

    @classmethod
    def has_tool_access(cls, plan_name: str, tool_name: str) -> bool:
        """Checks if a given subscription plan has access to a specific tool."""
        plan = cls.get_plan_details(plan_name)
        if not plan:
            return False
        return tool_name in plan["tool_access"]


# --------------------------------------------------------------------------------------------------
# SECTION 4.2: USER AUTHENTICATION AND MANAGEMENT
# --------------------------------------------------------------------------------------------------

class UserManager:
    """
    Manages user registration, login, and session within the application.
    Interacts with the SQLite database.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.current_user = None
        logging.info("UserManager initialized.")

    def _hash_password(self, password: str) -> str:
        """Hashes a password using SHA-256 for secure storage."""
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username: str, email: str, password: str) -> (bool, str):
        """Registers a new user."""
        if not (username and email and password):
            return False, "Username, email, and password cannot be empty."

        # Check if username or email already exists
        if self.db_manager.fetch_one("SELECT id FROM users WHERE username = ?", (username,)):
            return False, "Username already taken."
        if self.db_manager.fetch_one("SELECT id FROM users WHERE email = ?", (email,)):
            return False, "Email address already registered."

        password_hash = self._hash_password(password)
        try:
            self.db_manager.execute_query(
                "INSERT INTO users (username, email, password_hash, subscription_plan) VALUES (?, ?, ?, 'basic')",
                (username, email, password_hash)
            )
            logging.info(f"New user registered: {username} ({email})")
            return True, "Registration successful! Please log in."
        except Exception as e:
            logging.error(f"Failed to register user {username}: {e}")
            return False, "An internal error occurred during registration."

    def login_user(self, username: str, password: str) -> (bool, str):
        """Logs in a user and sets the current session."""
        password_hash = self._hash_password(password)
        user_data = self.db_manager.fetch_one(
            "SELECT id, username, email, subscription_plan, preferences FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        )

        if user_data:
            self.current_user = {
                "id": user_data[0],
                "username": user_data[1],
                "email": user_data[2],
                "plan": user_data[3],
                "preferences": json.loads(user_data[4]) if user_data[4] else {}
            }
            # Update last login time
            self.db_manager.execute_query("UPDATE users SET last_login = ? WHERE id = ?",
                                          (datetime.datetime.now(), self.current_user['id']))
            logging.info(f"User '{username}' logged in successfully.")
            return True, "Login successful."
        else:
            logging.warning(f"Failed login attempt for username: {username}")
            return False, "Invalid username or password."

    def logout_user(self):
        """Logs out the current user."""
        logging.info(f"User '{self.current_user['username']}' logged out.")
        self.current_user = None

    def get_current_user(self):
        """Returns the currently logged-in user's data."""
        return self.current_user

    def update_user_plan(self, user_id: int, new_plan: str):
        """Updates a user's subscription plan in the database."""
        if new_plan not in SubscriptionManager.PLANS:
            logging.error(f"Attempted to upgrade to invalid plan: {new_plan}")
            return False

        try:
            self.db_manager.execute_query("UPDATE users SET subscription_plan = ? WHERE id = ?", (new_plan, user_id))
            logging.info(f"User ID {user_id} subscription plan updated to '{new_plan}'.")
            # If this is the current user, update the session data
            if self.current_user and self.current_user['id'] == user_id:
                self.current_user['plan'] = new_plan
            return True
        except Exception as e:
            logging.error(f"Failed to update user plan in DB: {e}")
            return False


# --------------------------------------------------------------------------------------------------
# SECTION 4.3: FASTAPI SERVER FOR WEBHOOKS
# --------------------------------------------------------------------------------------------------
# A lightweight FastAPI server that runs in a background thread to listen for webhooks,
# for example, from Paystack after a successful payment.

class WebhookServer(QObject):
    """
    Manages the FastAPI server running in a background thread.
    """
    # This signal will carry the user_id and new_plan to the main thread
    subscription_update_received = pyqtSignal(int, str)

    def __init__(self, user_manager, paystack_manager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.paystack_manager = paystack_manager
        self.app = FastAPI()
        self.server = None
        self._setup_routes()
        logging.info("WebhookServer initialized.")

    def _setup_routes(self):
        """Defines the API routes for the webhook server."""

        # Pydantic model for the webhook payload
        class PaystackWebhookPayload(BaseModel):
            event: str
            data: dict

        @self.app.post("/webhooks/paystack")
        async def handle_paystack_webhook(payload: PaystackWebhookPayload):
            logging.info(f"Received Paystack webhook. Event: {payload.event}")

            # For security, you should verify the webhook signature here
            # using the X-Paystack-Signature header and your secret key.
            # This is omitted for simplicity but is CRITICAL for production.

            if payload.event == "charge.success":
                transaction_data = payload.data
                reference = transaction_data.get('reference')

                if not reference:
                    raise HTTPException(status_code=400, detail="Reference missing from payload.")

                # Verify the transaction again with Paystack as a security measure
                verification = self.paystack_manager.verify_transaction(reference)

                if verification and verification['status'] and verification['data']['status'] == 'success':
                    logging.info(f"Transaction {reference} verified successfully.")

                    # Extract metadata to know which user and plan this is for
                    metadata = verification['data'].get('metadata', {})
                    user_id = metadata.get('user_id')
                    plan_name = metadata.get('plan_name')

                    if user_id and plan_name:
                        logging.info(f"Processing subscription update for user ID {user_id} to plan '{plan_name}'.")
                        # Emit a signal to the main thread to handle the database update
                        # This is crucial because database operations should be managed by the main logic
                        self.subscription_update_received.emit(int(user_id), plan_name)
                        return {"status": "success", "message": "Webhook processed."}
                    else:
                        logging.error(f"Webhook for {reference} is missing user_id or plan_name in metadata.")
                        raise HTTPException(status_code=400, detail="Missing metadata.")
                else:
                    logging.error(f"Webhook verification failed for reference {reference}.")
                    raise HTTPException(status_code=400, detail="Transaction verification failed.")

            return {"status": "event ignored"}

    def run(self):
        """Starts the Uvicorn server."""
        config = uvicorn.Config(
            self.app,
            host=AppConfig.SERVER_HOST,
            port=AppConfig.SERVER_PORT,
            log_level=AppConfig.SERVER_LOG_LEVEL.lower()
        )
        self.server = uvicorn.Server(config)
        logging.info(f"FastAPI webhook server starting on http://{AppConfig.SERVER_HOST}:{AppConfig.SERVER_PORT}")
        self.server.run()

    def shutdown(self):
        """Gracefully shuts down the Uvicorn server."""
        if self.server:
            logging.info("Shutting down FastAPI server...")
            self.server.should_exit = True
            # Uvicorn doesn't have a clean programmatic shutdown in this run mode,
            # so we rely on the thread being terminated. This is a known limitation.


# --------------------------------------------------------------------------------------------------
# SECTION 4.4: UI ENHANCEMENTS FOR AUTHENTICATION AND SUBSCRIPTIONS
# --------------------------------------------------------------------------------------------------

class LoginDialog(QtWidgets.QDialog):
    """A custom dialog for user login and registration."""

    def __init__(self, user_manager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("Etherius - Login")
        self.setModal(True)
        self.setFixedSize(400, 300)

        self.layout = QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.login_tab = QWidget()
        self.register_tab = QWidget()

        self.tabs.addTab(self.login_tab, "Login")
        self.tabs.addTab(self.register_tab, "Register")
        self.layout.addWidget(self.tabs)

        self._setup_login_ui()
        self._setup_register_ui()

    def _setup_login_ui(self):
        layout = QVBoxLayout(self.login_tab)
        self.login_username = QLineEdit(placeholderText="Username")
        self.login_password = QLineEdit(placeholderText="Password", echoMode=QLineEdit.Password)
        self.login_button = QPushButton("Login")
        self.login_message = QLabel("")

        layout.addWidget(self.login_username)
        layout.addWidget(self.login_password)
        layout.addWidget(self.login_button)
        layout.addWidget(self.login_message)

        self.login_button.clicked.connect(self.handle_login)

    def _setup_register_ui(self):
        layout = QVBoxLayout(self.register_tab)
        self.reg_username = QLineEdit(placeholderText="Username")
        self.reg_email = QLineEdit(placeholderText="Email")
        self.reg_password = QLineEdit(placeholderText="Password", echoMode=QLineEdit.Password)
        self.reg_button = QPushButton("Register")
        self.reg_message = QLabel("")

        layout.addWidget(self.reg_username)
        layout.addWidget(self.reg_email)
        layout.addWidget(self.reg_password)
        layout.addWidget(self.reg_button)
        layout.addWidget(self.reg_message)

        self.reg_button.clicked.connect(self.handle_register)

    def handle_login(self):
        username = self.login_username.text()
        password = self.login_password.text()
        success, message = self.user_manager.login_user(username, password)
        if success:
            self.accept()  # Close the dialog with a success code
        else:
            self.login_message.setText(f"<font color='red'>{message}</font>")

    def handle_register(self):
        username = self.reg_username.text()
        email = self.reg_email.text()
        password = self.reg_password.text()
        success, message = self.user_manager.register_user(username, email, password)
        if success:
            self.reg_message.setText(f"<font color='green'>{message}</font>")
            self.tabs.setCurrentIndex(0)  # Switch to login tab
        else:
            self.reg_message.setText(f"<font color='red'>{message}</font>")


# --------------------------------------------------------------------------------------------------
# SECTION 4.5: MAIN WINDOW INTEGRATION OF NEW FEATURES
# --------------------------------------------------------------------------------------------------


        
        
        
class EtheriusMainWindow(QMainWindow):
    """
    (Continuation of EtheriusMainWindow with new integrations)
    """

    # ... (all methods from previous sections are assumed to be here) ...
    def __init__(self):
        super().__init__()
        # --- Core Components ---
        self.db_manager = None
        self.vector_db_manager = None
        self.user_manager = None
        self.paystack_manager = None
        # ... other components
        self.bridge = None
        self.web_view = None
        self.web_channel = None
        self.ai_models = None
        self.tool_executor = None

        # --- Worker Threads ---
        self.model_loader_thread = QThread()
        self.model_loader_worker = None
        self.ai_core_thread = QThread()
        self.ai_core_worker = None
        self.tts_thread = QThread()
        self.tts_worker = None
        self.stt_thread = QThread()
        self.stt_worker = None
        self.webhook_thread = QThread()
        self.webhook_worker = None

        logging.info("Initializing Etherius MainWindow.")
        self.initialize_ui()
        # Authentication is now the first step
        if not self.run_authentication():
            sys.exit(0)  # Exit if user closes login dialog

        # Proceed with normal startup after successful login
        self.update_ui_with_user_info()
        self.initialize_core_services()
        self.start_model_loading()
        self.start_webhook_server()


    def initialize_ui(self):
        """Sets up the entire user interface of the main window."""
        logging.info("Setting up User Interface.")
        self.setWindowTitle(f"{AppConfig.APP_NAME} - The Superior AI Assistant")
        self.setGeometry(100, 100, 1600, 900)
        self.setWindowIcon(QIcon(os.path.join(AppConfig.ICONS_DIR, 'logo.png')))

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.web_view = QWebEngineView()
        # Set background to match theme while loading to prevent white flash
        self.web_view.page().setBackgroundColor(QColor(AppConfig.THEME['primary_color']))
        main_layout.addWidget(self.web_view)

        self.web_channel = QWebChannel(self.web_view.page())
        self.web_view.page().setWebChannel(self.web_channel)

        self.bridge = BackendBridge()
        self.web_channel.registerObject("backend_bridge", self.bridge)

        html = get_html_content()
        self.web_view.setHtml(html, QUrl(f"file:///{AppConfig.BASE_DIR.replace(os.sep, '/')}"))

        self.create_status_bar()
        self.create_menu_bar()



    def run_authentication(self):
        """Shows the login dialog and waits for the result."""
        # Initialize only what's needed for login
        self.db_manager = DatabaseManager(AppConfig.SQLITE_DB_FILE)
        self.db_manager.connect()
        self.db_manager.create_application_tables()
        self.user_manager = UserManager(self.db_manager)

        dialog = LoginDialog(self.user_manager, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            logging.info("Authentication successful.")
            return True
        else:
            logging.info("Authentication cancelled by user.")
            return False

    def update_ui_with_user_info(self):
        """Updates the frontend with the logged-in user's details."""
        user = self.user_manager.get_current_user()
        if user:
            js_code = f"""
            document.getElementById('username-display').textContent = '{self.escape_js_string(user['username'])}';
            document.getElementById('plan-display').textContent = '{self.escape_js_string(user['plan'].capitalize())}';
            """
            self.web_view.page().runJavaScript(js_code)
            # Update the AI Core with the current user ID
            if self.ai_core_worker:
                self.ai_core_worker.current_user_id = user['id']

    def start_webhook_server(self):
        """Initializes and starts the webhook server thread."""
        if not hasattr(self, 'paystack_manager'):
            self.paystack_manager = PaystackManager(AppConfig.PAYSTACK_SECRET_KEY)

        self.webhook_worker = WebhookServer(self.user_manager, self.paystack_manager)
        self.webhook_worker.moveToThread(self.webhook_thread)
        self.webhook_worker.subscription_update_received.connect(self.on_subscription_updated)
        self.webhook_thread.started.connect(self.webhook_worker.run)
        self.webhook_thread.start()
        logging.info("Webhook server thread started.")

    @pyqtSlot(int, str)
    def on_subscription_updated(self, user_id, plan_name):
        """Handles a subscription update signal from the webhook server."""
        logging.info(f"Main thread received subscription update: User ID {user_id} -> Plan '{plan_name}'")
        if self.user_manager.update_user_plan(user_id, plan_name):
            QMessageBox.information(self, "Subscription Updated",
                                    f"The subscription for user ID {user_id} has been successfully updated to {plan_name.capitalize()}!")
            self.update_ui_with_user_info()  # Refresh the UI if it's the current user
        else:
            QMessageBox.warning(self, "Update Failed", "Could not update the subscription in the database.")

    def create_menu_bar(self):
        """Adds new menu items for user management."""
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet(
            f"background-color: {AppConfig.THEME['secondary_color']}; color: {AppConfig.THEME['text_color']};")

        # File Menu
        file_menu = menu_bar.addMenu("&File")
        exit_action = QAction("Exit", self, triggered=self.close)
        file_menu.addAction(exit_action)
        # ... (existing actions)

        # Account Menu
        account_menu = menu_bar.addMenu("&Account")
        sub_action = QAction("Manage Subscription", self, triggered=self.show_subscription_page)
        account_menu.addAction(sub_action)
        account_menu.addSeparator()
        logout_action = QAction("Logout", self, triggered=self.handle_logout)
        account_menu.addAction(logout_action)

        # ... (other menus)

    def handle_logout(self):
        """Logs out the current user and shows the login screen again."""
        self.user_manager.logout_user()
        # This is a simple way to "restart" the app's state for a new user
        # In a more complex app, you'd clean up session data more carefully
        self.close()
        # A better approach would be to hide the main window, show login, then re-init on success
        QProcess.startDetached(sys.executable, sys.argv)

    def show_subscription_page(self):
        """Opens a new window or dialog to manage subscriptions."""
        # For simplicity, we'll use a QMessageBox to show the concept.
        # A real implementation would use a custom QDialog or a new web page.
        user = self.user_manager.get_current_user()
        if not user: return

        plan_name = user['plan']
        plan_details = SubscriptionManager.get_plan_details(plan_name)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Manage Subscription")
        msg_box.setText(f"Your current plan: **{plan_details['name']}**")

        info_text = "Features:\n" + "\n".join([f"- {feat}" for feat in plan_details['features']])
        msg_box.setInformativeText(info_text)

        # Add buttons for other plans
        if plan_name != 'premium':
            premium_button = msg_box.addButton("Upgrade to Premium", QMessageBox.ActionRole)
        if plan_name != 'ultimate':
            ultimate_button = msg_box.addButton("Upgrade to Ultimate", QMessageBox.ActionRole)
        cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

        msg_box.exec_()

        clicked_button = msg_box.clickedButton()
        if clicked_button == premium_button:
            self.initiate_payment("premium")
        elif clicked_button == ultimate_button:
            self.initiate_payment("ultimate")

    def initiate_payment(self, target_plan):
        """Initiates a Paystack payment for a plan upgrade."""
        user = self.user_manager.get_current_user()
        plan_details = SubscriptionManager.get_plan_details(target_plan)

        if not self.paystack_manager.is_configured:
            QMessageBox.critical(self, "Payment Error", "The payment system is not configured by the administrator.")
            return

        amount_ngn = plan_details['price_ngn']
        amount_kobo = amount_ngn * 100
        reference = f"Etherius_{user['id']}_{target_plan}_{int(time.time())}"

        # Add user_id and plan_name to metadata for the webhook
        payload_with_metadata = {
            "email": user['email'],
            "amount": amount_kobo,
            "reference": reference,
            "metadata": {
                "user_id": user['id'],
                "plan_name": target_plan,
                "app_name": AppConfig.APP_NAME,
            }
        }

        response = self.paystack_manager._make_request('POST', 'transaction/initialize', data=payload_with_metadata)

        if response and response['status']:
            auth_url = response['data']['authorization_url']
            logging.info(f"Opening payment URL for user {user['id']}: {auth_url}")
            QDesktopServices.openUrl(QUrl(auth_url))
            QMessageBox.information(self, "Payment Initialized",
                                    "Your browser has been opened to complete the payment. Your plan will be updated automatically upon successful payment.")
        else:
            QMessageBox.critical(self, "Payment Failed",
                                 f"Could not initialize payment. Reason: {response.get('message', 'Unknown error')}")

    def closeEvent(self, event):
        """Gracefully shuts down all threads."""
        logging.info("Shutdown sequence initiated by user.")

        # Stop servers and workers first
        if self.webhook_worker: self.webhook_worker.shutdown()
        if self.ai_core_worker: self.ai_core_worker.stop()

        # Quit all threads
        for thread in [self.model_loader_thread, self.ai_core_thread, self.tts_thread, self.stt_thread,
                       self.webhook_thread]:
            if thread.isRunning():
                thread.quit()

        logging.info("Waiting for threads to terminate...")
        for thread in [self.model_loader_thread, self.ai_core_thread, self.tts_thread, self.stt_thread,
                       self.webhook_thread]:
            thread.wait(3000)

        if self.db_manager: self.db_manager.close()
        logging.info("Etherius has shut down gracefully.")
        event.accept()


    def initialize_core_services(self):
        """Initializes backend services like the database manager."""
        logging.info("Initializing core application services.")
        self.db_manager = DatabaseManager(AppConfig.SQLITE_DB_FILE)
        try:
            self.db_manager.connect()
            self.db_manager.create_application_tables()
        except Exception as e:
            self.show_critical_error(f"Etherius could not start due to a database error: {e}")

    def start_model_loading(self):
        """Creates and starts the model loader worker and thread."""
        self.model_loader_worker = ModelLoader()
        self.model_loader_worker.moveToThread(self.model_loader_thread)

        # Connect signals
        self.model_loader_worker.progress_updated.connect(self.on_model_load_progress)
        self.model_loader_worker.models_loaded.connect(self.on_models_loaded)
        self.model_loader_worker.error_occurred.connect(self.on_model_load_error)
        self.model_loader_thread.started.connect(self.model_loader_worker.run)

        self.model_loader_thread.start()
        logging.info("Model loader thread started.")

    def on_model_load_progress(self, value, text):
        """Updates the UI with model loading progress."""
        logging.info(f"Model loading progress: {value}% - {text}")
        js_code = f"window.setStatus(`{self.escape_js_string(text)}`, true);"
        self.web_view.page().runJavaScript(js_code)

    def on_model_load_error(self, error_message):
        """Handles errors during model loading."""
        logging.critical(f"Model loading failed: {error_message}")
        self.show_critical_error(error_message)
        js_code = f"window.showError('Model Loading Failed', `{self.escape_js_string(error_message)}`);"
        self.web_view.page().runJavaScript(js_code)
        self.model_loader_thread.quit()

    def on_models_loaded(self, models):
        """Finalizes setup after models are successfully loaded."""
        logging.info("All models loaded. Finalizing application setup.")
        self.ai_models = models
        self.model_loader_thread.quit()
        self.model_loader_thread.wait()

        # Initialize other workers that depend on the models
        self.initialize_dependent_workers()
        self.connect_bridge_signals()

        # Notify frontend that Etherius is ready
        welcome_message = "Etherius is online. All systems nominal. How may I assist you?"
        js_code = f"window.setInitialState(`{self.escape_js_string(welcome_message)}`);"
        self.web_view.page().runJavaScript(js_code)

        # Play the welcome message if TTS is enabled by default
        if self.tts_worker.is_enabled:
            self.tts_worker.speak(welcome_message)

    def initialize_dependent_workers(self):
        """Initializes workers that need the loaded models."""
        # --- Vector DB Manager ---
        self.vector_db_manager = VectorDBManager(
            path=AppConfig.VECTOR_DB_PATH,
            collection_name=AppConfig.VECTOR_COLLECTION_NAME,
            embedding_model=self.ai_models['embedding_model']
        )
        if not self.vector_db_manager.initialize():
            self.show_critical_error("Failed to initialize the vector database. Long-term memory will be disabled.")

        # --- Tool Executor ---
        self.tool_executor = ToolExecutor(self.ai_models)

        # --- TTS Engine Worker ---
        self.tts_worker = TTSEngine(self.ai_models)
        self.tts_worker.moveToThread(self.tts_thread)
        self.tts_worker.is_speaking_changed.connect(lambda speaking: logging.debug(f"TTS Speaking State: {speaking}"))
        self.tts_thread.start()

        # --- STT Engine Worker ---
        self.stt_worker = STTEngine(self.ai_models)
        self.stt_worker.moveToThread(self.stt_thread)
        self.stt_worker.error_occurred.connect(lambda msg: self.show_error_on_frontend("Speech Recognition Error", msg))
        self.stt_worker.is_listening_changed.connect(self.set_frontend_voice_state)
        self.stt_worker.transcription_ready.connect(self.on_transcription_ready)
        self.stt_thread.start()

        # --- AI Core Worker ---
        self.ai_core_worker = AICoreWorker(
            models=self.ai_models,
            db_manager=self.db_manager,
            vector_db_manager=self.vector_db_manager,
            tool_executor=self.tool_executor
        )
        self.ai_core_worker.moveToThread(self.ai_core_thread)

        # Connect AI Core signals to the main thread's slots
        self.ai_core_worker.new_assistant_message.connect(self.on_new_assistant_message)
        self.ai_core_worker.new_tool_message.connect(self.on_new_tool_message)
        self.ai_core_worker.processing_status_changed.connect(self.on_processing_status_changed)
        self.ai_core_worker.error_occurred.connect(self.show_error_on_frontend)

        self.ai_core_thread.started.connect(self.ai_core_worker.run)
        self.ai_core_thread.start()

        logging.info("All dependent workers, including the AI Core, have been initialized and started.")

    def connect_bridge_signals(self):
        """Connects signals from the BackendBridge to the appropriate slots."""
        self.bridge.frontend_ready.connect(lambda: logging.info("Bridge confirmed frontend is ready."))
        self.bridge.voice_input_requested.connect(self.stt_worker.start_transcription)
        self.bridge.tts_state_changed.connect(self.on_tts_state_changed)
        self.bridge.file_attach_requested.connect(self.on_file_attach_requested)

        # The crucial connection: UI message -> AI Core
        self.bridge.user_message_received.connect(self.ai_core_worker.add_to_queue)

        # Set default TTS state
        QTimer.singleShot(100, lambda: self.on_tts_state_changed(False))  # Default to off

    def temporary_message_handler(self, message):
        """A temporary handler until the AI Core is built in Section 3."""
        response = f"Etherius acknowledges your message: '{message}'. The full AI processing pipeline will be activated in the next section."
        self.send_response_to_frontend(response)
        self.tts_worker.speak(response)

    def on_transcription_ready(self, text):
        """Handles the transcribed text from the STT engine."""
        logging.info(f"Handling transcription from STT engine: '{text}'")
        self.set_frontend_input_text(text)
        # Automatically send the message if the transcription is not empty
        if text:
            self.send_message_from_input()

    def on_file_attach_requested(self):
        """Opens a file dialog for the user to select a file (e.g., an image)."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image to Analyze", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_path:
            logging.info(f"User attached file: {file_path}")
            # We now construct a special message to send to the AI core
            # This demonstrates how to pass structured data in a user query
            user_message_with_context = (
                f"Please analyze the content of the image I have uploaded. "
                f"The image is located at the following path: {file_path}"
            )
            # Add this message to the UI as if the user typed it
            self.send_response_to_frontend(f"Attached file: {os.path.basename(file_path)}", role='user')
            # Send the special message to the AI core for processing
            self.ai_core_worker.add_to_queue(user_message_with_context)

    def on_new_assistant_message(self, message):
        """Receives the final response from the AI and updates the UI."""
        self.send_response_to_frontend(message, role='assistant')
        self.tts_worker.speak(message)  # Queue the response for TTS

    def on_new_tool_message(self, message):
        """Shows a message in the UI indicating tool usage."""
        self.send_response_to_frontend(message, role='tool')

    def on_processing_status_changed(self, status, is_processing):
        """Updates the status indicator in the UI."""
        js_code = f"window.setStatus(`{self.escape_js_string(status)}`, {str(is_processing).lower()});"
        self.web_view.page().runJavaScript(js_code)

    def on_tts_state_changed(self, is_enabled):
        """Handles the TTS toggle from the UI."""
        self.tts_worker.set_enabled(is_enabled)
        if not is_enabled:
            self.tts_worker.stop_current_playback()


    def send_message_from_input(self):
        """Tells the frontend to trigger a send action."""
        # This is a bit of a hack to simulate the user clicking "send"
        # A better way might be to have the bridge call process_user_message directly
        self.web_view.page().runJavaScript("document.getElementById('send-button').click();")

    def show_critical_error(self, message):
        """Displays a fatal error message box and exits."""
        logging.critical(message)
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setText("Etherius - Fatal Error")
        error_box.setInformativeText(message)
        error_box.setWindowTitle("Etherius - Startup Failed")
        error_box.exec_()
        sys.exit(1)

    def closeEvent(self, event):
        """Handles the window close event to ensure graceful shutdown."""
        logging.info("Shutdown sequence initiated by user.")
        # Quit all worker threads
        self.model_loader_thread.quit()
        self.ai_core_thread.quit()
        self.tts_thread.quit()
        self.stt_thread.quit()

        # Wait for threads to finish
        self.model_loader_thread.wait()
        self.ai_core_thread.wait()
        self.tts_thread.wait()
        self.stt_thread.wait()

        if self.db_manager:
            self.db_manager.close()
        logging.info("Etherius has shut down gracefully.")
        event.accept()

    @pyqtSlot(str)
    def send_response_to_frontend(self, message, role='assistant'):
        """Sends a message to be displayed in the frontend chat."""
        js_code = f"window.addMessage('{role}', `{self.escape_js_string(message)}`);"
        self.web_view.page().runJavaScript(js_code)

    @pyqtSlot(str)
    def set_frontend_input_text(self, text):
        """Sets the text of the input box in the JS frontend."""
        js_code = f"window.setInputText(`{self.escape_js_string(text)}`);"
        self.web_view.page().runJavaScript(js_code)

    @pyqtSlot(bool)
    def set_frontend_voice_state(self, is_listening):
        """Tells the frontend to update the microphone icon's state."""
        js_code = f"window.setVoiceRecordingState({str(is_listening).lower()});"
        self.web_view.page().runJavaScript(js_code)

    @pyqtSlot(str, str)
    def show_error_on_frontend(self, title, message):
        """Displays an error message in the frontend UI."""
        js_code = f"window.showError(`{self.escape_js_string(title)}`, `{self.escape_js_string(message)}`);"
        self.web_view.page().runJavaScript(js_code)

    def escape_js_string(self, s):
        """Escapes a string to be safely embedded in a JavaScript template literal."""
        return s.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')

    # The rest of the EtheriusMainWindow methods from Section 1 are assumed to be here...
    def create_status_bar(self):
        status_bar = self.statusBar()
        status_bar.setStyleSheet(
            f"background-color: {AppConfig.THEME['secondary_color']}; color: {AppConfig.THEME['text_secondary_color']};")
        self.device_label = QLabel(f"Device: {AppConfig.AI_SETTINGS['device'].upper()}")
        status_bar.addPermanentWidget(self.device_label)



# ==================================================================================================
#
#   Etherius: The Sentient AI Assistant
#   File: main.py (Section 5 of 5)
#   Purpose: Multi-Agent Systems, Advanced Tooling (Marketing, Cybersecurity),
#            Robotics/IoT Integration, and Final Application Assembly.
#
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# SECTION 5.1: MULTI-AGENT COLLABORATION FRAMEWORK
# --------------------------------------------------------------------------------------------------
# This section introduces the concept of specialized agents that can be managed and
# delegated tasks by the main AICoreWorker.

class BaseAgent(QObject):
    """
    An abstract base class for all specialized AI agents.
    Each agent operates on a specific domain (e.g., marketing, research, coding).
    """
    task_complete = pyqtSignal(str, str)  # agent_name, result
    error_occurred = pyqtSignal(str, str)  # agent_name, error_message

    def __init__(self, name: str, description: str, llm, tokenizer, parent=None):
        super().__init__(parent)
        self.name = name
        self.description = description
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = AppConfig.AI_SETTINGS['device']
        self.is_busy = False
        logging.info(f"Agent '{self.name}' initialized.")

    def execute_task(self, task_description: str, context: str):
        """
        The main entry point for an agent to perform its task.
        This method should be overridden by subclasses.
        """
        if self.is_busy:
            self.error_occurred.emit(self.name, "Agent is already busy with another task.")
            return

        self.is_busy = True
        logging.info(f"Agent '{self.name}' is starting task: {task_description[:50]}...")
        # Subclasses will implement the specific logic here.
        # This base implementation just simulates work.
        QTimer.singleShot(1000, lambda: self._placeholder_task_handler(task_description))

    @torch.no_grad()
    def _run_inference(self, prompt, max_tokens=512):
        """A helper method for subclasses to run LLM inference."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split('[/INST]')[-1].strip()

    def _placeholder_task_handler(self, task_description):
        """A placeholder for the actual task logic in the base class."""
        result = f"Task '{task_description}' completed by agent '{self.name}' (placeholder implementation)."
        self.task_complete.emit(self.name, result)
        self.is_busy = False


class MarketingCampaignAgent(BaseAgent):
    """
    A specialized agent for creating and managing email marketing campaigns.
    """

    def __init__(self, llm, tokenizer, parent=None):
        super().__init__(
            name="MarketingAgent",
            description="Designs and drafts email marketing campaigns based on a product, target audience, and key message.",
            llm=llm,
            tokenizer=tokenizer,
            parent=parent
        )

    def execute_task(self, task_description: str, context: str):
        """
        Generates a structured marketing campaign plan.
        task_description should be a JSON string like:
        '{"product": "...", "audience": "...", "goal": "..."}'
        """
        self.is_busy = True
        try:
            params = json.loads(task_description)
            product = params['product']
            audience = params['audience']
            goal = params['goal']
        except (json.JSONDecodeError, KeyError) as e:
            self.error_occurred.emit(self.name,
                                     f"Invalid task description. Must be JSON with product, audience, and goal. Error: {e}")
            self.is_busy = False
            return

        prompt = f"""
[INST]
You are a world-class marketing strategist. Your task is to design an email marketing campaign.
Analyze the provided context and generate a complete campaign plan.

**Background Context:**
{context}

**Campaign Details:**
- **Product/Service:** {product}
- **Target Audience:** {audience}
- **Primary Goal:** {goal}

**Required Output (provide as a structured response):**
1.  **Campaign Name:** A catchy name for the campaign.
2.  **Key Message:** The core value proposition to communicate.
3.  **Email Sequence (3 emails):**
    *   **Email 1 (Introduction):** Subject line and full body text.
    *   **Email 2 (Follow-up/Value Add):** Subject line and full body text.
    *   **Email 3 (Call to Action):** Subject line and full body text.
4.  **Success Metrics:** How to measure the campaign's success.
[/INST]
"""
        result = self._run_inference(prompt, max_tokens=1500)
        self.task_complete.emit(self.name, result)
        self.is_busy = False


class CybersecurityAgent(BaseAgent):
    """
    A specialized agent for performing basic, non-intrusive security checks.
    **ETHICS WARNING**: This is a conceptual tool. Real-world use requires
    strict permissions, ethical guidelines, and legal compliance.
    """

    def __init__(self, llm, tokenizer, parent=None):
        super().__init__(
            name="CybersecurityAgent",
            description="Performs basic, non-intrusive security analysis on a given URL (e.g., checks headers, looks for common vulnerabilities).",
            llm=llm,
            tokenizer=tokenizer,
            parent=parent
        )

    def execute_task(self, task_description: str, context: str):
        """
        Performs a basic analysis of a URL.
        task_description should be the URL string.
        """
        self.is_busy = True
        url = task_description
        if not url.startswith('http'):
            url = 'https://' + url

        try:
            response = requests.get(url, headers={'User-Agent': 'Etherius-Security-Scanner/1.0'}, timeout=10)
            headers = response.headers

            analysis = f"**Security Header Analysis for {url}:**\n\n"

            # Check for common security headers
            common_headers = {
                'Content-Security-Policy': 'Helps prevent XSS attacks.',
                'Strict-Transport-Security': 'Enforces HTTPS.',
                'X-Content-Type-Options': 'Prevents MIME-sniffing.',
                'X-Frame-Options': 'Protects against clickjacking.',
                'Referrer-Policy': 'Controls how much referrer information is sent.'
            }

            for header, desc in common_headers.items():
                if header in headers:
                    analysis += f"- [✅ PRESENT] **{header}**: Found. {desc}\n"
                else:
                    analysis += f"- [⚠️ ABSENT] **{header}**: Not found. Consider adding this header. {desc}\n"

            self.task_complete.emit(self.name, analysis)

        except requests.RequestException as e:
            self.error_occurred.emit(self.name, f"Could not access URL {url}. Error: {e}")
        finally:
            self.is_busy = False




class CryptoAgent(BaseAgent):
    """
    A specialized agent for interacting with public blockchains like Ethereum.
    """

    def __init__(self, llm, tokenizer, parent=None):
        super().__init__(
            name="CryptoAgent",
            description="Retrieves information from public blockchains (e.g., Ethereum). Can check wallet balances, get transaction details, and get current gas prices. Requires a public RPC endpoint.",
            llm=llm,
            tokenizer=tokenizer,
            parent=parent
        )
        self.rpc_endpoint = os.getenv("ETHEREUM_RPC_ENDPOINT")  # e.g., from Infura or Alchemy
        if not self.rpc_endpoint:
            logging.warning("ETHEREUM_RPC_ENDPOINT not set. CryptoAgent will be non-functional.")
            self.web3 = None
        else:
            self.web3 = Web3(Web3.HTTPProvider(self.rpc_endpoint))

    def execute_task(self, task_description: str, context: str):
        """
        Interacts with the blockchain based on a JSON command.
        task_description format: '{"command": "get_balance", "address": "..."}'
        or '{"command": "get_gas_price"}'
        """
        self.is_busy = True
        if not self.web3 or not self.web3.is_connected():
            self.error_occurred.emit(self.name,
                                     "Not connected to Ethereum RPC. Please configure the ETHEREUM_RPC_ENDPOINT.")
            self.is_busy = False
            return

        try:
            params = json.loads(task_description)
            command = params.get("command")
            result = ""

            if command == "get_balance":
                address = params.get("address")
                if not self.web3.is_address(address):
                    result = f"[Error: Invalid Ethereum address: {address}]"
                else:
                    checksum_address = self.web3.to_checksum_address(address)
                    balance_wei = self.web3.eth.get_balance(checksum_address)
                    balance_eth = self.web3.from_wei(balance_wei, 'ether')
                    result = f"The balance of address {address} is {balance_eth:.6f} ETH."

            elif command == "get_gas_price":
                gas_price_wei = self.web3.eth.gas_price
                gas_price_gwei = self.web3.from_wei(gas_price_wei, 'gwei')
                result = f"The current average gas price is {gas_price_gwei:.2f} Gwei."

            else:
                result = f"[Error: Unknown command '{command}' for CryptoAgent.]"

            self.task_complete.emit(self.name, result)

        except (json.JSONDecodeError, KeyError) as e:
            self.error_occurred.emit(self.name, f"Invalid task description for CryptoAgent. Error: {e}")
        except Exception as e:
            self.error_occurred.emit(self.name, f"An error occurred during blockchain interaction: {e}")
        finally:
            self.is_busy = False

class AgentManager(QObject):
    """
    Manages a pool of specialized agents and delegates tasks to them.
    This manager is owned by the AICoreWorker.
    """
    agent_response = pyqtSignal(str, str)  # agent_name, result

    def __init__(self, llm, tokenizer, parent=None):
        super().__init__(parent)
        self.agents = {}
        self._register_agents(llm, tokenizer)
        logging.info("AgentManager initialized.")

    def _register_agents(self, llm, tokenizer):
        """Initializes and registers all available specialized agents."""
        marketing_agent = MarketingCampaignAgent(llm, tokenizer)
        cyber_agent = CybersecurityAgent(llm, tokenizer)
        crypto_agent = CryptoAgent(llm, tokenizer)  # Add the new agent

        for agent in [marketing_agent, cyber_agent, crypto_agent]:  # Add to list
            self.agents[agent.name] = agent
            agent.task_complete.connect(self.on_agent_task_complete)
            agent.error_occurred.connect(self.on_agent_task_complete)
        logging.info(f"Registered agents: {list(self.agents.keys())}")
    def get_agent_descriptions(self) -> str:
        """Returns a formatted string of agent descriptions for the LLM prompt."""
        descriptions = {name: agent.description for name, agent in self.agents.items()}
        return json.dumps(descriptions, indent=2)

    def delegate_task(self, agent_name: str, task_description: str, context: str):
        """Delegates a task to a specific agent."""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            # Run the agent's task in a separate one-shot thread to avoid blocking the AI core
            # This is a simple way to achieve concurrency within the AI core's "thought"
            threading.Thread(target=agent.execute_task, args=(task_description, context)).start()
        else:
            self.agent_response.emit(agent_name, f"[Error: Agent '{agent_name}' not found.]")

    @pyqtSlot(str, str)
    def on_agent_task_complete(self, agent_name, result):
        """Slot to receive task completion signals from agents."""
        logging.info(f"Agent '{agent_name}' has completed its task.")
        self.agent_response.emit(agent_name, result)


# --------------------------------------------------------------------------------------------------
# SECTION 5.2: ROBOTICS AND IOT INTEGRATION FRAMEWORK
# --------------------------------------------------------------------------------------------------

class RoboticsInterface(QObject):
    """
    A real interface for controlling robotics and IoT devices via MQTT.
    """
    command_status = pyqtSignal(str, bool, str)  # command_id, success, message
    connection_status = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.is_connected = False
        self.command_futures = {}
        logging.info("RoboticsInterface initialized with MQTT support.")

    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when the client connects to the MQTT broker."""
        if rc == 0:
            logging.info("Successfully connected to MQTT broker.")
            self.is_connected = True
            self.client.subscribe("etherius/robotics/status/#")  # Subscribe to status topics
            self.connection_status.emit(True)
        else:
            logging.error(f"Failed to connect to MQTT broker, return code {rc}")
            self.is_connected = False
            self.connection_status.emit(False)

    def _on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects."""
        logging.warning("Disconnected from MQTT broker.")
        self.is_connected = False
        self.connection_status.emit(False)

    def _on_message(self, client, userdata, msg):
        """Callback for incoming messages on subscribed topics."""
        topic = msg.topic
        payload = msg.payload.decode()
        logging.info(f"Received MQTT message on topic '{topic}': {payload}")

        # Assuming status messages are JSON with {"command_id": "...", "status": "...", "detail": "..."}
        try:
            data = json.loads(payload)
            command_id = data.get("command_id")
            if command_id in self.command_futures:
                success = data.get("status") == "success"
                detail = data.get("detail", "No details provided.")
                self.command_status.emit(command_id, success, detail)
                del self.command_futures[command_id]
        except json.JSONDecodeError:
            logging.warning(f"Received non-JSON MQTT message on {topic}")

    def connect(self, host: str, port: int = 1883):
        """Connects to the MQTT broker."""
        if self.is_connected:
            logging.warning("Already connected to MQTT broker.")
            return
        try:
            logging.info(f"Connecting to MQTT broker at {host}:{port}...")
            self.client.connect_async(host, port, 60)
            self.client.loop_start()
        except Exception as e:
            logging.error(f"Error starting MQTT connection: {e}")

    def disconnect(self):
        """Disconnects from the MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()

    def send_command(self, device_id: str, command: str, params: dict):
        """Sends a command to a specific device via MQTT."""
        if not self.is_connected:
            self.command_status.emit("not_sent", False, "Not connected to MQTT broker.")
            return

        command_id = f"{device_id}_{command}_{int(time.time())}"
        topic = f"etherius/robotics/command/{device_id}"
        payload = json.dumps({
            "command_id": command_id,
            "command": command,
            "params": params
        })

        logging.info(f"Publishing to MQTT topic '{topic}': {payload}")
        self.client.publish(topic, payload)
        self.command_futures[command_id] = True  # Register interest in a response

        # Set a timeout for the command acknowledgement
        QTimer.singleShot(15000, lambda: self._check_command_timeout(command_id))

    def _check_command_timeout(self, command_id):
        """Checks if a command has received a response, times out if not."""
        if command_id in self.command_futures:
            logging.warning(f"Command '{command_id}' timed out.")
            self.command_status.emit(command_id, False, "Command timed out. No response from device.")
            del self.command_futures[command_id]




# --------------------------------------------------------------------------------------------------
# SECTION 5.3: AI CORE ENHANCEMENT FOR AGENT DELEGATION
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# SECTION 3.1: THE AI CORE WORKER (COMPLETE AND FINAL VERSION)
# --------------------------------------------------------------------------------------------------
# This is the central nervous system of Etherius. It runs in a dedicated thread,
# processing user inputs, managing memory, using tools, delegating to specialist agents,
# and generating final, synthesized responses.

class AICoreWorker(QObject):
    """
    The main AI processing worker. It handles the entire "thought process" of the assistant,
    including tool use and agent delegation.
    """
    # Signals to communicate results back to the main GUI thread
    new_assistant_message = pyqtSignal(str)
    new_tool_message = pyqtSignal(str)  # To show "Using tool: Web Search..." or "Delegating to Agent..."
    processing_status_changed = pyqtSignal(str, bool)
    error_occurred = pyqtSignal(str, str)
    long_term_memory_updated = pyqtSignal()

    def __init__(self, models, db_manager, vector_db_manager, tool_executor, parent=None):
        super().__init__(parent)
        # --- Core Components ---
        self.models = models
        self.db_manager = db_manager
        self.vector_db_manager = vector_db_manager
        self.tool_executor = tool_executor
        self.agent_manager = AgentManager(models['llm'], models['llm_tokenizer'])
        self.robotics_interface = RoboticsInterface()  # Add robotics interface

        # --- AI State ---
        self.device = AppConfig.AI_SETTINGS['device']
        self.conversation_history = deque(maxlen=AppConfig.AI_SETTINGS['memory_size'])
        self.current_user_id = 1  # Will be updated upon user login
        self.is_running = True
        self.task_queue = Queue()

        # --- Agent State & Threading ---
        # This event is crucial for making the synchronous AI loop wait for an asynchronous agent response.
        self.is_waiting_for_agent = False
        self.agent_result_event = threading.Event()
        self.agent_result = None

        # Connect agent manager signal to the handler slot
        self.agent_manager.agent_response.connect(self._handle_agent_response)

        logging.info("AICoreWorker initialized with AgentManager and full capabilities.")

    def add_to_queue(self, task_data):
        """Public method to add a new task to the processing queue."""
        logging.info(f"New task added to AI Core queue: {str(task_data)[:100]}")
        self.task_queue.put(task_data)

    def stop(self):
        """Stops the worker's processing loop gracefully."""
        self.is_running = False
        self.task_queue.put(None)  # Add sentinel value to unblock the queue
        logging.info("AI Core worker stop signal received.")

    def run(self):
        """The main processing loop for the AI core thread."""
        logging.info("AI Core worker thread has started.")
        while self.is_running:
            try:
                task = self.task_queue.get(block=True)
                if task is None:  # Check for the sentinel value
                    break
                self._process_task(task)
            except Exception as e:
                logging.critical(f"Unhandled exception in AI Core run loop: {e}", exc_info=True)
                self.error_occurred.emit("AI Core Failure", f"A critical error occurred in the AI processing loop: {e}")
        logging.info("AI Core worker thread has stopped.")

    def _process_task(self, task):
        """
        Orchestrates the entire reasoning and response generation process for a single task.
        This includes memory retrieval, master action decisions (tool vs. agent), execution, and final response generation.
        """
        if not isinstance(task, str):
            logging.warning(f"AI Core received an unknown task type: {type(task)}")
            return

        user_query = task
        self.processing_status_changed.emit("Thinking...", True)

        try:
            # Step 1: Add user query to short-term conversational memory.
            self.conversation_history.append({"role": "user", "content": user_query})

            # Step 2: Retrieve relevant memories from the long-term vector database.
            long_term_memories = self._retrieve_long_term_memories(user_query)

            # Step 3: Make the master decision: use a tool, delegate to an agent, or respond directly.
            decision = self._decide_master_action(user_query, long_term_memories)
            action_type = decision.get("action_type", "generate_response")

            tool_result = None

            # Step 4a: Execute a direct tool if chosen.
            if action_type == "tool_use":
                tool_name = decision.get("name")
                tool_input = decision.get("input")
                self.processing_status_changed.emit(f"Using tool: {tool_name}...", True)
                self.new_tool_message.emit(f"Executing tool: **{tool_name}**. Please wait...")
                tool_result = self.tool_executor.execute_tool(tool_name, tool_input)
                self.conversation_history.append({"role": "tool", "content": tool_result})

            # Step 4b: Delegate to a specialist agent if chosen.
            elif action_type == "agent_delegation":
                agent_name = decision.get("name")
                agent_input = decision.get("input")
                self.processing_status_changed.emit(f"Delegating to {agent_name}...", True)
                self.new_tool_message.emit(
                    f"Delegating task to specialist agent: **{agent_name}**. This may take a moment...")

                # --- Asynchronous Agent Handling ---
                self.is_waiting_for_agent = True
                self.agent_result_event.clear()
                self.agent_manager.delegate_task(agent_name, agent_input, long_term_memories)

                # Block this thread until the agent responds or times out.
                if self.agent_result_event.wait(timeout=60.0):  # 60-second timeout
                    tool_result = self.agent_result
                else:
                    tool_result = f"[Error: Agent '{agent_name}' did not respond in time.]"

                self.is_waiting_for_agent = False
                self.conversation_history.append({"role": "tool", "name": agent_name, "content": tool_result})

            # Step 5: Generate the final, synthesized response using all gathered context.
            self.processing_status_changed.emit("Generating response...", True)
            final_response = self._generate_final_response(user_query, long_term_memories, tool_result)
            self.conversation_history.append({"role": "assistant", "content": final_response})
            self.new_assistant_message.emit(final_response)

            # Step 6: Perform post-response reflection to potentially create new long-term memories.
            self._reflect_and_memorize(user_query, final_response)

        except Exception as e:
            logging.error(f"Error during AI task processing: {e}", exc_info=True)
            error_message = f"I apologize, but I encountered an internal error while processing your request: {e}"
            self.new_assistant_message.emit(error_message)
        finally:
            self.processing_status_changed.emit("Idle", False)

    @pyqtSlot(str, str)
    def _handle_agent_response(self, agent_name: str, result: str):
        """Receives the result from a specialist agent and unblocks the main AI processing loop."""
        if self.is_waiting_for_agent:
            self.agent_result = f"**Response from {agent_name}:**\n\n{result}"
            self.agent_result_event.set()  # This signals the wait() in _process_task to continue.

    def _retrieve_long_term_memories(self, query: str) -> str:
        """Searches the vector database for memories relevant to the query."""
        logging.info("Retrieving long-term memories.")
        search_results = self.vector_db_manager.search_memories(
            query_text=query,
            n_results=AppConfig.AI_SETTINGS['semantic_memory_search_k'],
            user_id=self.current_user_id
        )
        if not search_results:
            logging.info("No relevant long-term memories found.")
            return ""

        formatted_memories = "Here are some relevant facts from your long-term memory:\n"
        for item in search_results:
            formatted_memories += f"- {item['text']}\n"

        logging.info(f"Retrieved memories:\n{formatted_memories}")
        return formatted_memories

    def _build_prompt_for_llm(self, sections: list) -> str:
        """Constructs a complete prompt string using the model's specific chat template."""
        prompt_list = [{"role": "system", "content": AppConfig.SYSTEM_PROMPTS['main_persona']}]

        for section in sections:
            if section['content']:
                prompt_list.append({"role": section['role'], "content": section['content']})

        prompt_list.extend(list(self.conversation_history))
        return self.models['llm_tokenizer'].apply_chat_template(prompt_list, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def _decide_master_action(self, query: str, context: str) -> dict:
        """Top-level decision maker to choose between tools, agents, or direct response."""
        logging.info("Master action decision process started.")
        tool_descs = self.tool_executor.get_tool_descriptions()
        agent_descs = self.agent_manager.get_agent_descriptions()
        prompt = f"""
[INST]
You are the master controller for the Etherius AI. Your primary job is to route user requests to the correct capability. You have three options:
1.  **Tool Use**: For simple, direct actions like searching the web, analyzing a single image, or executing a snippet of code.
2.  **Agent Delegation**: For complex, multi-step tasks that require specialized reasoning, like designing a marketing campaign or querying a blockchain.
3.  **Generate Response**: If the request can be handled with your own knowledge without any tools or agents.

**Available Tools:**
{tool_descs}

**Available Specialist Agents:**
{agent_descs}

**User Request:** "{query}"
**Relevant Context:** "{context}"

Analyze the user request and choose the best action. Your output MUST be a single JSON object with one of three formats:
- For Tool Use: `{{"action_type": "tool_use", "name": "<tool_name>", "input": "<tool_input>"}}`
- For Agent Delegation: `{{"action_type": "agent_delegation", "name": "<agent_name>", "input": "<task_description_for_agent_as_json_string>"}}`
- For Direct Response: `{{"action_type": "generate_response"}}`
[/INST]
"""
        response_text = self._run_inference(prompt, max_tokens=300)
        logging.info(f"Master action raw response: {response_text}")
        try:
            start_index = response_text.find('{')
            end_index = response_text.rfind('}') + 1
            json_str = response_text[start_index:end_index]
            decision = json.loads(json_str)
            logging.info(f"Master action decision: {decision}")
            return decision
        except Exception:
            logging.error("Failed to parse master action JSON. Defaulting to generate_response.")
            return {"action_type": "generate_response"}

    @torch.no_grad()
    def _generate_final_response(self, query: str, long_term_memories: str, tool_result: str) -> str:
        """Generates the final, user-facing response after all context has been gathered."""
        logging.info("Generating final response.")
        prompt_sections = [{"role": "system", "name": "memory", "content": long_term_memories}]
        full_prompt = self._build_prompt_for_llm(prompt_sections)
        logging.debug(f"Final generation prompt:\n{full_prompt}")

        inputs = self.models['llm_tokenizer'](full_prompt, return_tensors="pt").to(self.device)
        outputs = self.models['llm'].generate(
            **inputs,
            max_new_tokens=AppConfig.AI_SETTINGS['generation_max_length'],
            temperature=AppConfig.AI_SETTINGS['generation_temperature'],
            top_p=AppConfig.AI_SETTINGS['generation_top_p'],
            do_sample=True,
            pad_token_id=self.models['llm_tokenizer'].eos_token_id
        )
        response_text = self.models['llm_tokenizer'].decode(outputs[0], skip_special_tokens=True)
        final_response = response_text.split('[/INST]')[-1].strip()
        logging.info(f"Generated final response: '{final_response[:100]}...'")
        return final_response

    @torch.no_grad()
    def _reflect_and_memorize(self, user_query: str, ai_response: str):
        """Decides if an interaction is important enough to be stored in long-term memory."""
        logging.info("Reflecting on conversation for long-term memory storage.")
        if len(user_query) + len(ai_response) < 150:
            logging.info("Interaction too short, skipping long-term memory.")
            return

        summarization_prompt = f"""
Summarize the key information, facts, or user preferences from the following conversation exchange into a concise, self-contained statement.
This summary will be stored in your long-term memory. Focus on new, lasting information.
If no new, lasting information was exchanged, respond with only the word "NULL".

Conversation:
User: "{user_query}"
Assistant: "{ai_response}"

Summary:
"""
        full_prompt = self.models['llm_tokenizer'].apply_chat_template(
            [{"role": "user", "content": summarization_prompt}], tokenize=False, add_generation_prompt=True
        )
        summary = self._run_inference(full_prompt, max_tokens=100)

        if summary.upper() == "NULL" or not summary:
            logging.info("LLM determined the interaction was not memorable.")
            return

        logging.info(f"Generated memory summary: '{summary}'")
        memory_id = hashlib.sha256(summary.encode()).hexdigest()
        timestamp = datetime.datetime.now().isoformat()

        self.vector_db_manager.add_memory(summary, {"user_id": self.current_user_id, "created_at": timestamp},
                                          memory_id)
        self.db_manager.execute_query(
            "INSERT INTO long_term_memory (user_id, memory_text, embedding_id, created_at) VALUES (?, ?, ?, ?)",
            (self.current_user_id, summary, memory_id, timestamp)
        )
        self.long_term_memory_updated.emit()

    def _run_inference(self, prompt, max_tokens=512):
        """A helper method for running LLM inference within the AI Core."""
        inputs = self.models['llm_tokenizer'](prompt, return_tensors="pt").to(self.device)
        outputs = self.models['llm'].generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.5,  # Lower temp for deterministic decisions
            do_sample=False,
            pad_token_id=self.models['llm_tokenizer'].eos_token_id
        )
        response = self.models['llm_tokenizer'].decode(outputs[0], skip_special_tokens=True)
        return response.split('[/INST]')[-1].strip()

# --------------------------------------------------------------------------------------------------
# SECTION 5.4: FINAL APPLICATION ASSEMBLY AND MAIN EXECUTION BLOCK
# --------------------------------------------------------------------------------------------------
# The final `if __name__ == "__main__"` block that brings everything together.

def main():
    """The main function to launch the Etherius application."""
    # 1. Setup Environment
    if not setup_application_environment():
        sys.exit(1)

    # 2. Setup Logging
    setup_logging()

    # 3. Set Qt environment variables
    # This enables remote debugging of the web frontend via a browser (e.g., at http://localhost:8080)
    os.environ["QTWEBENGINE_REMOTE_DEBUGGING"] = "8080"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 4. Create QApplication instance
    app = QApplication(sys.argv)

    # 5. Show Splash Screen
    splash_pix = QPixmap(os.path.join(AppConfig.IMAGES_DIR, 'splash.png'))
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage("Initializing Etherius Core...", Qt.AlignBottom | Qt.AlignCenter,
                       QColor(AppConfig.THEME['text_color']))
    app.processEvents()

    # 6. Create and Show Main Window
    # The MainWindow's __init__ now handles authentication before showing the full UI.
    try:
        main_window = EtheriusMainWindow()
        if main_window.user_manager.get_current_user():
            splash.finish(main_window)
            main_window.show()
            # 7. Start the Qt Event Loop
            logging.info("Starting Etherius Qt event loop.")
            sys.exit(app.exec_())
        else:
            # User cancelled login, so we exit gracefully.
            splash.hide()
            logging.info("Application startup cancelled during login.")
            sys.exit(0)

    except Exception as e:
        logging.critical(f"Unhandled exception during MainWindow creation: {e}", exc_info=True)
        splash.hide()
        QMessageBox.critical(None, "Application Failed to Start", f"A critical error occurred:\n\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    # Add a final check for Hugging Face token
    if not AppConfig.HUGGING_FACE_TOKEN:
        print("\n" + "=" * 80)
        print("FATAL ERROR: HUGGING_FACE_TOKEN is not set.".center(80))
        print("Please create a .env file in the same directory as main.py and add:".center(80))
        print("HUGGING_FACE_TOKEN='your_token_here'".center(80))
        print("You can get a token from https://huggingface.co/settings/tokens".center(80))
        print("=" * 80 + "\n")
        sys.exit(1)

    main()
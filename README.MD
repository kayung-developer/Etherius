
## 🔥 Etherius

**Etherius** is not just another chatbot. It is a comprehensive, full-stack AI platform designed to be a true digital partner. It integrates state-of-the-art AI models for advanced reasoning, tool use, and long-term memory. It's built to be extensible, allowing for the integration of specialized AI agents, IoT devices, and external APIs.

This project demonstrates how to build a commercially-ready AI assistant from the ground up, combining a sophisticated local backend with a sleek, modern UI.

## ✨ Core Features

Etherius is packed with revolutionary features that set it apart:

🧠 **True Multimodal Intelligence**: Understands and generates text and code, and analyzes images.
*   **Core LLM**: Powered by `mistralai/Mixtral-8x7B-instruct-v0.1` for superior reasoning.
*   **Image Analysis**: Uses the `Salesforce/blip-image-captioning-large` model to understand images.
*   **Voice I/O**: High-quality Speech-to-Text (`openai/whisper-large-v3`) and Text-to-Speech (`microsoft/speecht5_tts`).

🛠️ **Autonomous Tool & Agent System**: Etherius can decide when to use tools or delegate complex tasks to specialized agents.
*   **Tools**: Web Search, Code Execution, Image Analysis, Email Sending.
*   **Agents**: Includes a `MarketingAgent`, `CybersecurityAgent`, and `CryptoAgent`.

📚 **Hyper-Contextual Memory**:
*   **Short-Term**: A rolling conversation buffer for immediate context.
*   **Long-Term**: A persistent **Vector Database** (ChromaDB) stores key memories, allowing the AI to recall facts and preferences from past conversations.

🔐 **User & Subscription Management**:
*   **Full Authentication**: Secure user registration and login system.
*   **Subscription Tiers**: Supports Basic, Premium, and Ultimate plans, unlocking different features.
*   **Payment Integration**: Fully integrated with **Paystack** for handling subscription payments, including a backend webhook server.

🤖 **Extensible & Future-Ready**:
*   **Robotics/IoT Interface**: Includes a ready-to-use MQTT interface for controlling real-world devices.
*   **Blockchain Integration**: The `CryptoAgent` can interact with the Ethereum blockchain via a public RPC.

## ⚙️ Tech Stack

*   **Backend**: Python, FastAPI (for webhooks)
*   **Desktop GUI**: PyQt5, QtWebEngine (rendering HTML/CSS/JS)
*   **AI Models**: PyTorch, Hugging Face (Transformers, Sentence-Transformers)
*   **Databases**: SQLite (for user data), ChromaDB (for vector memory)
*   **Audio**: SoundDevice, PyAudio
*   **APIs**: Paystack, Serper.dev

---

## 🚀 Getting Started

### Prerequisites

*   **Python 3.9+**
*   **NVIDIA GPU with >= 24GB VRAM** (Critically important for running the Mixtral model)
*   **NVIDIA CUDA Toolkit** (version compatible with PyTorch, e.g., 11.8)
*   A microphone for voice input.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Etherius.git
cd Etherius
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

You can install all required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
*(Note: The PyTorch installation within `requirements.txt` is CPU-only by default for compatibility. For GPU support, which is **required**, please install the correct version manually first.)*

**For GPU Support (Required for decent performance):**
Find your CUDA version and install the correct PyTorch package from the [official website](https://pytorch.org/get-started/locally/). For CUDA 11.8, the command is:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
<p align="center">
  <strong>A Next-Generation, Multimodal, Autonomous AI Assistant</strong><br/>
  Built with Python, PyQt5, and Hugging Face Transformers
</p>

### 4. Configure API Keys

Create a file named `.env` in the root of the project directory and populate it with your secret keys. A template is provided below.

```env
# --- Hugging Face (Required) ---
HUGGING_FACE_TOKEN="your_hugging_face_token_here"

# --- Web Search (Optional but Recommended) ---
SERPER_API_KEY="your_serper_dev_api_key_here"

# --- Payment Gateway (Optional) ---
PAYSTACK_SECRET_KEY="sk_test_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# --- Email Sending (Optional) ---
SMTP_SERVER="smtp.gmail.com"
SMTP_PORT="587"
SMTP_USER="your_email@gmail.com"
SMTP_PASSWORD="your_gmail_app_password_here"

# --- Crypto Agent (Optional) ---
ETHEREUM_RPC_ENDPOINT="https://mainnet.infura.io/v3/your_infura_project_id_here"
```

### 5. Run the Application

With your virtual environment active, run the main script:
```bash
python main.py
```
**First-time startup will be very long**, as the application needs to download several large AI models (50GB+). Subsequent startups will be much faster.

---

## 🤝 Contributing

Contributions are welcome! Whether it's adding a new agent, improving a tool, or fixing a bug, your help is appreciated. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide to get started.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

*   [Hugging Face](https://huggingface.co/) for their incredible work on the Transformers library and model hosting.
*   The teams behind [PyQt](https://www.riverbankcomputing.com/software/pyqt/) and [PyTorch](https://pytorch.org/).
*   [Mistral AI](https://mistral.ai/) for the powerful Mixtral model.
```

---

### **2. `requirements.txt`**

This file lists all the Python dependencies, making it easy for others to install everything with a single command.

```text
# Etherius AI Assistant - Core Dependencies

# --- GUI Framework ---
PyQt5==5.15.10
PyQtWebEngine==5.15.6
pyqtwebchannel==5.15.6

# --- AI & Machine Learning ---
# NOTE: This torch version is CPU-only for maximum compatibility.
# For GPU support (required for this project), install torch manually first.
# See README.md for GPU installation instructions.
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
transformers==4.36.2
sentence-transformers==2.2.2
accelerate==0.25.0
bitsandbytes==0.42.0

# --- Audio & Media ---
sounddevice==0.4.6
numpy==1.26.2
pyaudio==0.2.14
Pillow==10.1.0
speechrecognition==3.10.0
pyttsx3==2.90

# --- Web, API, and Server ---
requests==2.31.0
python-dotenv==1.0.0
beautifulsoup4==4.12.2
uvicorn==0.25.0
fastapi==0.108.0

# --- Database ---
chromadb==0.4.22
# Note: sqlite3 is part of the Python standard library.

# --- Specialized Agents & Tools ---
paho-mqtt==1.6.1
web3==6.12.0
```

---

### **3. `.gitignore`**

This file tells Git which files and directories to ignore. It's crucial for keeping your repository clean of temporary files, environment secrets, and large model caches.

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
.venv
venv/
VENV/
env.bak
venv.bak
ENV/

# Secrets
.env
*.env
.env.*
secrets.py
secrets.yml
*.pem
*.key

# Application Data & Caches
logs/
data/
database/
assets/
temp/
.huggingface/
.cache/
huggingface_hub/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# OS-specific
.DS_Store
Thumbs.db
```

---

### **4. `CONTRIBUTING.md`**

This file provides guidelines for developers who want to contribute to your project.

```markdown
# Contributing to Etherius AI

First off, thank you for considering contributing to Etherius! Your help is essential for keeping it great.

This document provides a set of guidelines for contributing to the project.

## How Can I Contribute?

There are many ways to contribute, from writing code and documentation to submitting bug reports and feature requests.

### 🐛 Reporting Bugs

*   Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/Misterioso76/Etherius/issues).
*   If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/Misterioso76/Etherius/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

### ✨ Suggesting Enhancements

*   Open a new issue to discuss your enhancement idea.
*   Clearly describe the proposed enhancement, why it's needed, and provide examples of how it would work.

### 💻 Code Contributions

We welcome pull requests!

1.  **Fork the repository** and create your branch from `main`.
2.  **Set up your development environment** by following the instructions in the `README.md`.
3.  **Make your changes.** Adhere to the existing code style.
4.  **Add tests** for your changes, if applicable.
5.  **Ensure your code lints.** We use standard Python style conventions.
6.  **Create a pull request.** Provide a clear description of the changes you have made. Link to any relevant issues.

## Styleguides

### Git Commit Messages

*   Use the present tense ("Add feature" not "Added feature").
*   Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
*   Limit the first line to 72 characters or less.
*   Reference issues and pull requests liberally after the first line.

### Python Styleguide

*   Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide.
*   Add comments to your code where necessary to explain complex logic.
*   Write clear, readable, and maintainable code.

We look forward to your contributions!
```

---

### **5. `LICENSE`**

A `LICENSE` file is important for open-source projects. The MIT license is a popular and permissive choice.

```text
MIT License

Copyright (c) 2025 [Pascal Aondover]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

With these five files (`main.py`, `README.md`, `requirements.txt`, `.gitignore`, `CONTRIBUTING.md`, and `LICENSE`), your GitHub repository will be professional, easy to navigate, and welcoming to other developers.
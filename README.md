# Bad Word Detection

A Flask server to detect Korean bad words in text using pre-trained SVM models and vectorizers.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.6+
- Flask
- joblib

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/developersung13/bad-word-detection.git
    cd bad-word-detection
    ```

2. Set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:
    ```bash
    pip install Flask joblib
    ```

### Running the Server

Start the Flask server:
```bash
python app.py

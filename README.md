# Smart Plant Care Assistant

A personalized plant care application that helps users understand their plants' watering needs and care requirements based on plant characteristics, environmental conditions, and soil moisture status.

The application combines a machine learning model, a plant care dataset, live weather information, and consultation history to provide practical plant care recommendations through an interactive Streamlit interface.

## ✨ Features

- **User Authentication:** Register and log in securely with password hashing.
- **Plant Selection:** Choose from a collection of plant species supported by the plant care dataset.
- **Watering Recommendations:** Get estimated watering intervals using a Random Forest regression model and environmental adjustment rules.
- **Environmental Awareness:** Consider temperature, humidity, sunlight exposure, and soil moisture when generating recommendations.
- **Live Weather Integration:** Retrieve current temperature and relative humidity from Open-Meteo using approximate regional coordinates.
- **Plant Care Guidance:** View plant-specific information, including watering methods, light preferences, drainage requirements, soil pH preferences, and care tips.
- **Aquatic Plant Support:** Provide separate care guidance for aquatic plants that require continuous water immersion.
- **Consultation History:** Save and review previous plant care consultations associated with the logged-in user.
- **Weather Caching:** Cache weather responses for 15 minutes to reduce repeated API requests.
- **Privacy-Conscious Location Handling:** Use coarsened coordinates for weather requests and avoid storing raw GPS coordinates in consultation history.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and data processing |
| Streamlit | Interactive web application interface |
| Pandas & NumPy | Dataset handling and numerical operations |
| Scikit-learn | Random Forest regression and feature preprocessing |
| Joblib | Loading serialized machine learning artifacts |
| MongoDB | User accounts and consultation history |
| PyMongo | Python integration with MongoDB |
| Bcrypt | Password hashing |
| Open-Meteo API | Current regional weather information |
| HTML & CSS | Custom interface styling |

## 📂 Project Structure

```text
Smart_Plant_Care_Assistant/
│
├── app.py
├── db_helper.py
├── train_model.py
├── requirements.txt
│
├── plant_watering_model.pkl
├── plant_encoder.pkl
├── plant_watering_model_backup.pkl
├── plant_encoder_backup.pkl
│
├── plantfolio_final_ml_dataset.csv
├── plantfolio_care_lookup.csv
│
├── style.css
├── plant_mascot.jpg
├── plant_mascot.png
├── vine_decoration.png
│
├── .env.example
└── .streamlit/
    └── secrets.toml.example
```

### Main Files

- `app.py` — Main Streamlit application, authentication interface, plant selection, recommendation logic, weather integration, and consultation history display.
- `db_helper.py` — MongoDB connection management, user registration and authentication, and consultation history operations.
- `train_model.py` — Trains and evaluates Random Forest models using conventional and plant-grouped train/test splits, and evaluates the existing model artifacts.
- `plant_watering_model.pkl` — Existing serialized watering-frequency model.
- `plant_encoder.pkl` — Existing categorical feature encoder.
- `plantfolio_final_ml_dataset.csv` — Dataset used for model training and evaluation.
- `plantfolio_care_lookup.csv` — Plant-specific care information used by the application.
- `requirements.txt` — Python dependency versions.
- `style.css` — Custom styling for the application.

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- MongoDB Atlas account or a locally running MongoDB instance
- A modern web browser

### 1. Clone the Repository

```bash
git clone https://github.com/Satyasri2006/Smart_Plant_Care_Assistant.git
cd Smart_Plant_Care_Assistant
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```cmd
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project pins `scikit-learn==1.6.1` to match the version used to serialize the existing model artifacts.

### 4. Configure MongoDB

Create a MongoDB Atlas cluster or use an existing MongoDB instance.

Create a database user and configure network access appropriately. Then set your MongoDB connection string.

**Option A — Using a `.env` file**

Copy `.env.example` to `.env` and add your connection string:

```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster-url>/plantcare_db?retryWrites=true&w=majority
```

Replace the placeholders with your actual MongoDB credentials and cluster details.

**Option B — Using Streamlit secrets**

Create `.streamlit/secrets.toml`:

```toml
MONGO_URI = "mongodb+srv://<username>:<password>@<cluster-url>/plantcare_db?retryWrites=true&w=majority"
```

The application checks Streamlit secrets first, followed by environment variables, and then attempts to use a local MongoDB instance if no connection string is configured.

**Important:** Never commit your `.env` or `secrets.toml` files containing real credentials to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

Streamlit will display a local URL in your terminal, typically `http://localhost:8501`. Open it in your browser to use the application.

## 🤖 Machine Learning

The project uses a `RandomForestRegressor` to estimate plant watering frequency.

**Model configuration:**
- Algorithm: Random Forest Regression
- Number of estimators: 100
- Random state: 42
- Categorical features: Plant name, season, humidity preference, light preference, drainage preference, and watering method
- Numerical features: Minimum and maximum preferred temperatures
- Target: `Watering_Frequency_days`

The `train_model.py` script evaluates performance using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

It evaluates both a conventional 80/20 random split and an 80/20 plant-grouped split to assess performance on plant species excluded from training.

**Note:** The existing model artifacts are used by the application. Running the evaluation script does not automatically replace them with newly trained artifacts. Model metrics reflect agreement with the available dataset labels and do not independently establish real-world watering accuracy.

To run the evaluation script:

```bash
python train_model.py
```

Run it from the project root so that the script can locate the CSV files and model artifacts.

## 🌦️ Weather Integration

The application retrieves current temperature and relative humidity through the Open-Meteo API.

- Browser geolocation can be used to obtain approximate regional weather.
- Coordinates are rounded to one decimal place before the weather request.
- Successful weather responses are cached for 15 minutes.
- Manual environmental settings are supported.
- Weather information is identified as approximate-region data rather than a verified local weather station.

Weather retrieval requires an internet connection. If weather services are unavailable, check the application's displayed status and use manual environmental inputs where available.

## 🗄️ Database

MongoDB stores application data, including:

- User account details and password hashes
- Plant care consultation records
- Recommendation details and consultation timestamps
- Submission identifiers used to prevent duplicate history records

The application uses a `plantcare_db` database with `users` and `history` collections.

## 🔒 Security and Privacy

- Passwords are hashed using Bcrypt before storage.
- MongoDB connection strings should be kept in environment variables or Streamlit secrets.
- Dynamic values rendered in HTML are escaped where appropriate.
- Consultation history is retrieved for the logged-in user's email.
- Raw GPS coordinates are not stored in consultation history.

Avoid sharing database credentials or uploading secret configuration files to a public repository.

## 🚀 Future Improvements

Potential enhancements include:

- Evaluating recommendations against independently collected plant-watering observations.
- Expanding the plant dataset and improving coverage across seasons.
- Adding automated tests for authentication, consultation history, and recommendation workflows.
- Improving weather failure handling and user feedback.
- Introducing plant growth tracking and reminders for routine care.

## 👩‍💻 Author

**Lehitha**
**Satyasri**  
**Sravani**
**Supraja**
B.Tech — Computer Science and Engineering

---

*Smart Plant Care Assistant — making everyday plant care more informed and accessible.*

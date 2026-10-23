# Local Docker deployment — Final_Project

This starter has only two Docker services: FastAPI backend and Streamlit frontend.
MySQL stays installed on your Windows host. No real passwords are included.

## 1. Prepare Windows

Install Docker Desktop and enable Linux containers / WSL 2. From PowerShell,
verify `docker --version` and `docker compose version`. Start your Windows MySQL
instance and confirm the `kkbox_dwh` schema includes all DWH tables, and
`churn_predictions` contains the completed batch predictions.

## 2. Copy files into Final_Project/

Place Dockerfile.backend, Dockerfile.frontend, docker-compose.yml,
requirements-backend.txt, requirements-frontend.txt, .dockerignore,
.env.example, README-DOCKER.md in the repository root.

Replace utils/config.py with the env-based version in this package.
Optionally replace utils/db.py with the compatible safe URL-building version.
Do not overwrite your existing dashboard/app.py, backend/, or artifacts/.

Verify artifacts/xgboost_model.pkl, artifacts/xgboost_info.pkl,
artifacts/inference_preprocessor.pkl and artifacts/best_shap_explainer.pkl exist.
They are required by the backend and must remain inside the Docker build context.

## 3. Make an application database user (run as MySQL admin)

    CREATE USER 'kkbox_app'@'%' IDENTIFIED BY 'YOUR_STRONG_PASSWORD';
    GRANT SELECT ON kkbox_dwh.* TO 'kkbox_app'@'%';

Do not expose MySQL port 3306 to the public Internet. Host `%` is convenient for
initial Docker Desktop development; narrow the allowed host where practicable.
Check Windows Firewall and the MySQL bind address if Docker cannot connect.

## 4. Create .env

PowerShell from Final_Project/:

    Copy-Item .env.example .env

Edit .env and set DB_PASSWORD to the database user's actual password. Verify
DB_NAME and DB_USER. DB_HOST stays localhost for running Python locally; Docker
Compose overrides DB_HOST=host.docker.internal inside the backend container.

Choose a Python minor version matching your trained artifacts, then check the
versions of scikit-learn, XGBoost, SHAP, pandas and NumPy in the original model
training environment. Update and pin requirements-backend.txt accordingly.
An unpinned requirement set can cause pickle/SHAP compatibility errors.

## 5. Start

Stop locally started Uvicorn/Streamlit processes using ports 8000/8501.
Open Docker Desktop and run in the repository root:

    docker compose config
    docker compose up --build -d
    docker compose ps

Wait for backend to become healthy and for frontend to start.

## 6. Check (PowerShell)

    Invoke-RestMethod http://localhost:8000/health
    Invoke-RestMethod http://localhost:8000/predictions/summary

Open http://localhost:8501 in the browser and test Overview, Users, User Analysis.
In Streamlit's sidebar, ensure FastAPI URL is `http://backend:8000`.

## 7. Troubleshooting

    docker compose logs -f backend
    docker compose logs -f frontend
    docker compose ps

* Cannot connect to MySQL: check Windows MySQL service, firewall and user grants.
  `localhost` *inside* a container is not Windows.
* Artifact / joblib / XGBoost / SHAP errors: match training Python and packages,
  confirm models were copied and rebuild with `docker compose build --no-cache`.
* Port in use: stop local Streamlit/Uvicorn or change host port mappings.
* Connection indicator shows failure: API Base URL must be `http://backend:8000`.

Stop the stack:

    docker compose down

Rebuild after changes:

    docker compose up --build -d

## Security

The example Compose maps 8501 and 8000 ONLY to the Windows loopback address.
This is LOCAL deployment, not yet public hosting. Before public deployment add
access control for API, HTTPS, safe secret management and a database/network
architecture suitable for the target server.

Keep utils/config.py as the safe, environment-based code in this starter.
Do not commit a real password in any file. If a password was exposed in older
commits, rotate it and review the repository history; `.gitignore` does not
remove secrets from Git history.

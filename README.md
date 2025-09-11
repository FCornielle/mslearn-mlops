# MLOps with Azure Machine Learning

This repository contains a complete MLOps (Machine Learning Operations) implementation using Azure Machine Learning for diabetes classification. The project demonstrates end-to-end machine learning workflows from experimentation to production deployment.

## 🎯 Project Overview

This project implements a diabetes classification model using logistic regression and demonstrates best practices for MLOps including:
- Model training and experimentation
- Model versioning with MLflow
- CI/CD pipelines with GitHub Actions
- Model deployment to Azure ML endpoints
- Automated testing and validation

## 📁 Repository Structure

```
mslearn-mlops/
├── experimentation/           # Jupyter notebooks for model experimentation
│   ├── data/
│   │   └── diabetes-dev.csv  # Development dataset
│   └── train-classification-model.ipynb
├── production/               # Production data and configurations
│   └── data/
│       └── diabetes-prod.csv # Production dataset
├── src/                     # Source code
│   ├── model/
│   │   └── train.py         # Training script
│   ├── score.py             # Model scoring script
│   └── job.yml              # Azure ML job configuration
├── tests/                   # Unit tests
│   ├── datasets/
│   └── test_train.py
├── documentation/           # Challenge documentation
├── mlruns/                  # MLflow experiment tracking
├── environment.yml          # Azure ML environment configuration
├── conda_inference.yml      # Inference environment dependencies
├── deployment.yml           # Model deployment configuration
├── endpoint.yml             # Endpoint configuration
└── requirements.txt         # Python dependencies
```

## 🚀 Features

- **Automated Model Training**: Parameterized training scripts with MLflow integration
- **Experiment Tracking**: Complete MLflow integration for model versioning and metrics tracking
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing and deployment
- **Model Deployment**: Azure ML managed endpoints for real-time inference
- **Environment Management**: Conda environments for both training and inference
- **Unit Testing**: Comprehensive test suite with pytest
- **Production Ready**: Separate development and production data handling

## 🛠️ Technologies Used

- **Azure Machine Learning**: Cloud-based ML platform
- **MLflow**: Model versioning and experiment tracking
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **GitHub Actions**: CI/CD automation
- **Python 3.8+**: Programming language
- **Conda**: Environment management

## 📋 Prerequisites

- Azure subscription
- Azure Machine Learning workspace
- GitHub account
- Python 3.8 or higher
- Git

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mslearn-mlops.git
cd mslearn-mlops
```

### 2. Set Up Environment

Create a conda environment:

```bash
conda env create -f environment.yml
conda activate diabetes-infer-env
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### 3. Configure Azure ML

1. Create an Azure Machine Learning workspace
2. Configure authentication (Azure CLI or service principal)
3. Update the job configuration in `src/job.yml` with your workspace details

### 4. Run Training

```bash
python src/model/train.py --training_data experimentation/data --reg_rate 0.01
```

### 5. Run Tests

```bash
pytest tests/
```

## 📊 Model Details

- **Algorithm**: Logistic Regression
- **Target**: Diabetes classification (binary)
- **Features**: 8 medical features (Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age)
- **Evaluation**: Accuracy, F1-score, Precision, Recall, ROC-AUC

## 🔄 MLOps Workflow

1. **Experimentation**: Jupyter notebooks for model development
2. **Code Conversion**: Convert notebooks to production-ready Python scripts
3. **Version Control**: Git for code versioning
4. **Experiment Tracking**: MLflow for model and experiment tracking
5. **Testing**: Automated unit tests with pytest
6. **CI/CD**: GitHub Actions for automated workflows
7. **Deployment**: Azure ML managed endpoints
8. **Monitoring**: Model performance and drift monitoring

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

The test suite includes:
- Data validation tests
- Model training tests
- Model scoring tests
- Integration tests

## 📈 Monitoring and Logging

- **MLflow**: Automatic logging of parameters, metrics, and artifacts
- **Azure ML**: Job run tracking and model registry
- **GitHub Actions**: Build and deployment logs

## 🚀 Deployment

### Deploy to Azure ML Endpoint

1. Register the model in Azure ML:
```bash
az ml model create --name diabetes_prod_model --version 1 --path mlruns/0/[run-id]/artifacts/model
```

2. Create deployment:
```bash
az ml online-endpoint create -f endpoint.yml
az ml online-deployment create -f deployment.yml
```

### Test the Endpoint

```python
import requests
import json

# Sample data
data = {
    "data": [[1, 85, 66, 29, 0, 26.6, 0.351, 31]]
}

response = requests.post(
    "https://your-endpoint.region.inference.ml.azure.com/score",
    headers={"Authorization": "Bearer your-key"},
    json=data
)
print(response.json())
```

## 📚 Documentation

Detailed challenge documentation is available in the `documentation/` folder:

- [Challenge 0: Convert notebook to production code](documentation/00-script.md)
- [Challenge 1: Azure ML job](documentation/01-aml-job.md)
- [Challenge 2: GitHub Actions](documentation/02-github-actions.md)
- [Challenge 3: Trigger workflow](documentation/03-trigger-workflow.md)
- [Challenge 4: Unit testing and linting](documentation/04-unit-test-linting.md)
- [Challenge 5: Environments](documentation/05-environments.md)
- [Challenge 6: Deploy model](documentation/06-deploy-model.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of Microsoft Learn MLOps challenges. See the [Microsoft Learn documentation](https://docs.microsoft.com/azure/machine-learning/) for more information.

## 🆘 Support

For questions and support:
- Check the [documentation](documentation/)
- Review [Azure ML documentation](https://docs.microsoft.com/azure/machine-learning/)
- Open an issue in this repository

## 🔗 Useful Links

- [Azure Machine Learning Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Note**: This repository is based on Microsoft Learn MLOps challenges and serves as a practical example of implementing MLOps best practices with Azure Machine Learning.
